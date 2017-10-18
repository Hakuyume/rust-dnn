extern crate cuda;
extern crate cudnn;
extern crate nn;

use cuda::memory;
use cudnn::context;
use cudnn::tensor;
use nn::softmax;

mod utils;

fn main() {
    utils::bench(|| {
        let mut context = context::Context::new()?;

        let desc = {
            let mut desc = tensor::Descriptor::new()?;
            desc.set_4d(tensor::Format::NCHW, 1, 16, 1, 1)?;
            desc
        };
        let x = utils::random(desc.len())?;
        utils::dump(&x)?;

        let softmax = softmax::Softmax::new(softmax::Algorithm::Log, softmax::Mode::Channel);
        let mut y = memory::Memory::new(desc.len())?;
        softmax
            .foward(&mut context,
                    tensor::Tensor::new(&desc, &x),
                    tensor::TensorMut::new(&desc, &mut y))?;
        utils::dump(&y)?;

        let dy = utils::random(desc.len())?;
        let mut dx = memory::Memory::new(desc.len())?;
        softmax
            .backward(&mut context,
                      tensor::Tensor::new(&desc, &y),
                      tensor::Tensor::new(&desc, &dy),
                      tensor::TensorMut::new(&desc, &mut dx))?;
        utils::dump(&dx)?;

        Ok(())
    })
            .unwrap();
}
