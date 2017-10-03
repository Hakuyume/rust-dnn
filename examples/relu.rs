extern crate cuda;
extern crate cudnn;
extern crate nn;

use cuda::memory;
use cudnn::tensor;
use nn::context;
use nn::relu;

fn main() {
    let mut context = context::Context::new().unwrap();

    let (n, c, h, w) = (1, 16, 1, 1);

    let mut x_desc = tensor::Descriptor::new().unwrap();
    x_desc.set_4d(tensor::Format::NCHW, n, c, h, w).unwrap();
    let mut x = memory::Memory::new(x_desc.len()).unwrap();
    memory::memcpy(&mut x,
                   &[0.0, 0.1, -0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, 0.9, 1.0, -1.1, 1.2, -1.3,
                     1.4, 1.5])
            .unwrap();

    {
        let mut host = vec![0.; x.len()];
        memory::memcpy(&mut host, &x).unwrap();
        println!("{:?}", &host);
    }

    let relu = relu::ReLU::new();
    relu.forward_inplace(&mut context, tensor::TensorMut::new(&x_desc, &mut x))
        .unwrap();

    {
        let mut host = vec![0.; x.len()];
        memory::memcpy(&mut host, &x).unwrap();
        println!("{:?}", &host);
    }
}
