extern crate cuda;
extern crate cudnn;
extern crate nn;

use nn::dataset::MNIST;
use nn::generic_value::values::*;

use cuda::memory::Repr;
use nn::generic_value::USize;
use nn::layer::Layer;
use nn::layer::UnaryLayer;

type BatchSize = U32;
type InputSize = U784;
type NClasses = U10;

fn main() {
    let mut context = nn::Context::new().unwrap();

    let mut x = nn::Tensor::<_, BatchSize, InputSize, _, _>::new().unwrap();
    let mut y = nn::Tensor::<_, _, NClasses, _, _>::new().unwrap();
    let mut z = nn::Tensor::<_, BatchSize, NClasses, U1, U1>::new().unwrap();

    let mut fc = nn::layer::Linear::<_, _, _>::new().unwrap();

    let mnist = MNIST::new("mnist").unwrap();
    let mut x_host = vec![0.; x.mem().len()];
    let mut t_host = vec![0.; z.mem().len()];
    for i in 0..BatchSize::VALUE {
        let (image, label) = mnist.train.get(i);
        for k in 0..InputSize::VALUE {
            x_host[i * InputSize::VALUE + k] = image[k] as f32;
            t_host[i * NClasses::VALUE + label as usize] = 1.;
        }
    }
    cuda::memory::memcpy(x.mem_mut(), &x_host).unwrap();

    for _ in 0..50 {
        fc.forward(&mut context, &x, &mut y).unwrap();
        cudnn::softmax::forward(&mut context.cudnn,
                                cudnn::softmax::Algorithm::Log,
                                cudnn::softmax::Mode::Channel,
                                1.,
                                y.cudnn(),
                                0.,
                                z.cudnn_mut())
                .unwrap();

        let mut z_host = vec![0.; z.mem().len()];
        cuda::memory::memcpy(&mut z_host, z.mem()).unwrap();
        {
            let loss = z_host
                .iter()
                .zip(&t_host)
                .map(|(z, t)| -z * t / (BatchSize::VALUE as f32))
                .sum::<f32>();
            println!("loss: {}", loss);
        }
        let dyz_host: Vec<_> = t_host
            .iter()
            .map(|t| -t / (BatchSize::VALUE as f32))
            .collect();
        let mut dyz = nn::Tensor::new().unwrap();
        cuda::memory::memcpy(dyz.mem_mut(), &dyz_host).unwrap();

        cudnn::softmax::backward(&mut context.cudnn,
                                 cudnn::softmax::Algorithm::Log,
                                 cudnn::softmax::Mode::Channel,
                                 1.,
                                 z.cudnn(),
                                 None as Option<(_, &cuda::memory::View<_>)>,
                                 0.,
                                 dyz.cudnn_mut())
                .unwrap();

        let mut dx = nn::Tensor::new().unwrap();
        fc.backward(&mut context, &x, &dyz, &mut dx).unwrap();

        fc.optimize(&mut context, 1e-5).unwrap();
    }
}
