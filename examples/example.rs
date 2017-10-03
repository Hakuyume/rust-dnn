extern crate cuda;
extern crate cudnn;
extern crate nn;

use cuda::slice;
use cuda::memory;
use cudnn::tensor;

use nn::context;
use nn::convolution;
use nn::softmax;
use nn::activation;

fn dump(mem: &slice::Slice<f32>, len: usize) {
    let mut host = vec![0.; len];
    memory::memcpy(&mut host, &mem[..len]).unwrap();
    println!("{:?}", &host);
}

fn main() {
    let mut context = context::Context::new().unwrap();

    {
        let (n, h, w) = (32, 4, 4);
        let (c_in, c_out) = (256, 16);

        let mut x_desc = tensor::Descriptor::new().unwrap();
        x_desc
            .set_4d(tensor::Format::NCHW, n, c_in, h, w)
            .unwrap();
        let mut y_desc = tensor::Descriptor::new().unwrap();
        y_desc
            .set_4d(tensor::Format::NCHW, n, c_out, h, w)
            .unwrap();

        let mut conv = convolution::Convolution2D::new(c_out, c_in, 3, 1, 1, 1).unwrap();
        let softmax = softmax::Softmax::new(softmax::Algorithm::Accurate, softmax::Mode::Channel)
            .unwrap();
        let relu = activation::Activation::new(activation::Mode::Relu, true, 0.).unwrap();

        let x = memory::Memory::new(x_desc.len()).unwrap();
        let mut h = memory::Memory::new(y_desc.len()).unwrap();
        let mut y = memory::Memory::new(y_desc.len()).unwrap();

        conv.foward(&mut context,
                    tensor::Tensor::new(&x_desc, &x),
                    tensor::TensorMut::new(&y_desc, &mut h))
            .unwrap();
        dump(&h, 16);

        relu.foward_inplace(&mut context, tensor::TensorMut::new(&y_desc, &mut h))
            .unwrap();
        dump(&h, 16);

        softmax
            .foward(&mut context,
                    tensor::Tensor::new(&y_desc, &h),
                    tensor::TensorMut::new(&y_desc, &mut y))
            .unwrap();
        dump(&y, 16);
    }
}
