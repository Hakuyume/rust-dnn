extern crate cuda;
extern crate cudnn;
extern crate nn;

use nn::dataset::MNIST;

use cuda::memory::Repr;
use nn::layer::Layer;

const BATCH_SIZE: usize = 32;
const N_CLASSES: usize = 10;

fn main() {
    let mut context = nn::Context::new().unwrap();

    let mut x = nn::Tensor::new((BATCH_SIZE, MNIST::SIZE * MNIST::SIZE, 1, 1)).unwrap();
    let mut y = nn::Tensor::new((BATCH_SIZE, N_CLASSES, 1, 1)).unwrap();
    let mut z = nn::Tensor::new((BATCH_SIZE, N_CLASSES, 1, 1)).unwrap();

    let mut conv = nn::layer::Convolution2D::new(x.shape(), N_CLASSES, 1, 0, 1, 1).unwrap();

    let mnist = MNIST::new("mnist").unwrap();
    let mut x_host = vec![0.; x.mem().len()];
    let mut t_host = vec![0.; z.mem().len()];
    for i in 0..BATCH_SIZE {
        let (image, label) = mnist.train.get(i);
        for k in 0..MNIST::SIZE * MNIST::SIZE {
            x_host[i * MNIST::SIZE * MNIST::SIZE + k] = image[k] as f32;
            t_host[i * N_CLASSES + label as usize] = 1.;
        }
    }
    cuda::memory::memcpy(x.mem_mut(), &x_host).unwrap();

    for _ in 0..50 {
        conv.forward(&mut context, &x, &mut y).unwrap();
        unsafe {
            cudnn::softmax::forward(&mut context.cudnn,
                                    cudnn::softmax::Algorithm::Log,
                                    cudnn::softmax::Mode::Channel,
                                    1.,
                                    y.cudnn(),
                                    0.,
                                    z.cudnn_mut())
                    .unwrap()
        }

        let mut z_host = vec![0.; z.mem().len()];
        cuda::memory::memcpy(&mut z_host, z.mem()).unwrap();
        {
            let loss = z_host
                .iter()
                .zip(&t_host)
                .map(|(z, t)| -z * t / (BATCH_SIZE as f32))
                .sum::<f32>();
            println!("loss: {}", loss);
        }
        let dyz_host: Vec<_> = t_host
            .iter()
            .map(|t| -t / (BATCH_SIZE as f32))
            .collect();
        let mut dyz = nn::Tensor::new((BATCH_SIZE, N_CLASSES, 1, 1)).unwrap();
        cuda::memory::memcpy(dyz.mem_mut(), &dyz_host).unwrap();

        unsafe {
            cudnn::softmax::backward(&mut context.cudnn,
                                     cudnn::softmax::Algorithm::Log,
                                     cudnn::softmax::Mode::Channel,
                                     1.,
                                     z.cudnn(),
                                     None as Option<(_, &cuda::memory::View<_>)>,
                                     0.,
                                     dyz.cudnn_mut())
                    .unwrap()
        }
        let mut dx = nn::Tensor::new(x.shape()).unwrap();
        conv.backward(&mut context, &x, &dyz, &mut dx, 0.9)
            .unwrap();

        conv.optimize(&mut context, 1e-5).unwrap();
    }
}
