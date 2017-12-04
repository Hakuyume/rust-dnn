extern crate cuda;
extern crate cudnn;
extern crate nn;

use nn::dataset::MNIST;

use cuda::memory::Repr;

const BATCH_SIZE: usize = 32;
const N_CLASSES: usize = 10;

fn main() {
    let mut context = nn::Context::new().unwrap();

    let mut x = nn::Tensor::new((BATCH_SIZE, MNIST::SIZE * MNIST::SIZE, 1, 1)).unwrap();
    let mut y = nn::Tensor::new((BATCH_SIZE, N_CLASSES, 1, 1)).unwrap();
    let mut z = nn::Tensor::new((BATCH_SIZE, N_CLASSES, 1, 1)).unwrap();

    let mut w_desc = cudnn::filter::Descriptor::new().unwrap();
    w_desc
        .set_4d(cudnn::tensor::Format::NCHW,
                N_CLASSES,
                MNIST::SIZE * MNIST::SIZE,
                1,
                1)
        .unwrap();
    let mut conv_desc = cudnn::convolution::Descriptor::new().unwrap();
    conv_desc
        .set_2d(0, 0, 1, 1, 1, 1, cudnn::convolution::Mode::Convolution)
        .unwrap();

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
    let mut w = cuda::memory::Memory::new(N_CLASSES * MNIST::SIZE * MNIST::SIZE).unwrap();

    for _ in 0..50 {
        {
            let algo =
                cudnn::convolution::get_forward_algorithm(&mut context.cudnn,
                                                          x.cudnn().0,
                                                          &w_desc,
                                                          &conv_desc,
                                                          y.cudnn().0,
                                                          cudnn::convolution::FwdPreference::PreferFastest)
                .unwrap();
            let workspace_size =
                cudnn::convolution::get_forward_workspace_size(&mut context.cudnn,
                                                               x.cudnn().0,
                                                               &w_desc,
                                                               &conv_desc,
                                                               y.cudnn().0,
                                                               algo)
                        .unwrap();
            unsafe {
                cudnn::convolution::forward(&mut context.cudnn,
                                            1.,
                                            x.cudnn(),
                                            (&w_desc, &w),
                                            &conv_desc,
                                            algo,
                                            &mut context.workspace.get(workspace_size).unwrap(),
                                            0.,
                                            y.cudnn_mut())
                        .unwrap()
            }
        }
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
        {
            let algo =
                cudnn::convolution::get_backward_filter_algorithm(&mut context.cudnn,
                                                                  x.cudnn().0,
                                                                  dyz.cudnn().0,
                                                                  &conv_desc,
                                                                  &w_desc,
                                                                  cudnn::convolution::BwdFilterPreference::PreferFastest).unwrap();
            let workspace_size =
                cudnn::convolution::get_backward_filter_workspace_size(&mut context.cudnn,
                                                                       x.cudnn().0,
                                                                       dyz.cudnn().0,
                                                                       &conv_desc,
                                                                       &w_desc,
                                                                       algo)
                        .unwrap();
            unsafe {
                cudnn::convolution::backward_filter(&mut context.cudnn,
                                                    -1e-5,
                                                    x.cudnn(),
                                                    dyz.cudnn(),
                                                    &conv_desc,
                                                    algo,
                                                    &mut context
                                                             .workspace
                                                             .get(workspace_size)
                                                             .unwrap(),
                                                    1.,
                                                    (&w_desc, &mut w))
                        .unwrap()
            }
        }
    }
}
