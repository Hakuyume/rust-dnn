extern crate rand;
extern crate cuda;
extern crate nn;

use nn::dataset::MNIST;
use nn::generic_value::values::*;

use cuda::memory::Repr;
use nn::generic_value::USize;

type BatchSize = U32;
type InputSize = U784;
type NClasses = U10;

fn main() {
    let mut context = nn::Context::new().unwrap();
    let mut rng = rand::thread_rng();

    let mut x = nn::Tensor::<_, BatchSize, InputSize, _, _>::new().unwrap();
    let mut t = nn::Tensor::<_, _, NClasses, _, _>::new().unwrap();
    let mut y = nn::Tensor::new().unwrap();

    let mut dx = nn::Tensor::new().unwrap();
    let mut dy = nn::Tensor::new().unwrap();

    let mut fc = nn::layer::Linear::new().unwrap();
    let mut softmax_cross_entropy = nn::layer::SoftmaxCrossEntropy::new().unwrap();

    let mnist = MNIST::new("mnist").unwrap();

    let mut loss = 0.;

    for i in 0..10000 {
        {
            let indices = rand::seq::sample_indices(&mut rng, mnist.train.len(), BatchSize::VALUE);
            let mut x_host = vec![0.; x.mem().len()];
            let mut t_host = vec![0.; t.mem().len()];
            for (i, index) in indices.into_iter().enumerate() {
                let (image, label) = mnist.train.get(index);
                for k in 0..InputSize::VALUE {
                    x_host[i * InputSize::VALUE + k] = image[k] as f32;
                    t_host[i * NClasses::VALUE + label as usize] = 1.;
                }
            }
            cuda::memory::memcpy(x.mem_mut(), &x_host).unwrap();
            cuda::memory::memcpy(t.mem_mut(), &t_host).unwrap();
        }

        fc.forward(&mut context, &x, &mut y).unwrap();
        loss += softmax_cross_entropy
            .compute(&mut context, &y, &t, &mut dy)
            .unwrap();
        if i % 100 == 0 {
            println!("iteration: {}, loss: {}", i, loss / 100.);
            loss = 0.;
        }
        fc.backward(&mut context, &x, &dy, &mut dx).unwrap();

        fc.optimize(&mut context, 1e-5).unwrap();
    }
}
