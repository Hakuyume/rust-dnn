extern crate cuda;
extern crate cudnn;
extern crate nn;

use nn::dataset::MNIST;

const BATCH_SIZE: usize = 32;
const N_CLASSES: usize = 10;

fn main() {
    let mut context = cudnn::context::Context::new().unwrap();

    let x_desc = cudnn::tensor::Descriptor::new_4d(cudnn::tensor::Format::NCHW,
                                                   BATCH_SIZE,
                                                   MNIST::SIZE * MNIST::SIZE,
                                                   1,
                                                   1)
            .unwrap();
    let w_desc = cudnn::filter::Descriptor::new_4d(cudnn::tensor::Format::NCHW,
                                                   N_CLASSES,
                                                   MNIST::SIZE * MNIST::SIZE,
                                                   1,
                                                   1)
            .unwrap();
    let yz_desc =
        cudnn::tensor::Descriptor::new_4d(cudnn::tensor::Format::NCHW, BATCH_SIZE, N_CLASSES, 1, 1)
            .unwrap();
    let conv_desc = cudnn::convolution::Descriptor::new_2d(0,
                                                           0,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           cudnn::convolution::Mode::Convolution)
            .unwrap();
    let algo =
        cudnn::convolution::get_forward_algorithm(&mut context,
                                                  &x_desc,
                                                  &w_desc,
                                                  &conv_desc,
                                                  &yz_desc,
                                                  cudnn::convolution::FwdPreference::PreferFastest)
                .unwrap();
    let workspace_size = cudnn::convolution::get_forward_workspace_size(&mut context,
                                                                        &x_desc,
                                                                        &w_desc,
                                                                        &conv_desc,
                                                                        &yz_desc,
                                                                        algo)
            .unwrap();

    let mnist = MNIST::new("mnist").unwrap();
    let mut x = vec![0.; x_desc.len()];
    let mut t = vec![0.; yz_desc.len()];
    for i in 0..BATCH_SIZE {
        let (image, label) = mnist.train.get(i);
        for k in 0..MNIST::SIZE * MNIST::SIZE {
            x[i * MNIST::SIZE * MNIST::SIZE + k] = image[k] as f32;
            t[i * MNIST::SIZE * MNIST::SIZE + label as usize] = 1.;
        }
    }

    let mut x_dev = cuda::memory::Memory::new(x.len()).unwrap();
    let mut t_dev = cuda::memory::Memory::new(t.len()).unwrap();
    cuda::memory::memcpy(&mut x_dev, &x).unwrap();
    cuda::memory::memcpy(&mut t_dev, &t).unwrap();
    let w_dev = cuda::memory::Memory::new(w_desc.len()).unwrap();
    let mut y_dev = cuda::memory::Memory::new(yz_desc.len()).unwrap();
    let mut z_dev = cuda::memory::Memory::new(yz_desc.len()).unwrap();
    let mut workspace = cuda::memory::Memory::new(workspace_size).unwrap();

    cudnn::convolution::forward(&mut context,
                                1.,
                                cudnn::tensor::Tensor::new(&x_desc, &x_dev),
                                cudnn::filter::Filter::new(&w_desc, &w_dev),
                                &conv_desc,
                                algo,
                                &mut workspace,
                                0.,
                                cudnn::tensor::TensorMut::new(&yz_desc, &mut y_dev))
            .unwrap();
    cudnn::softmax::forward(&mut context,
                            cudnn::softmax::Algorithm::Log,
                            cudnn::softmax::Mode::Channel,
                            1.,
                            cudnn::tensor::Tensor::new(&yz_desc, &y_dev),
                            0.,
                            cudnn::tensor::TensorMut::new(&yz_desc, &mut z_dev))
            .unwrap();

    let mut z = vec![0.; yz_desc.len()];
    cuda::memory::memcpy(&mut z, &z_dev).unwrap();
    println!("{}",
             z.iter().zip(&t).map(|(z, t)| -z * t).sum::<f32>() / (BATCH_SIZE as f32));
}
