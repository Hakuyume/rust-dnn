use std::marker;

use cuda::memory;

use cudnn::scalar;
use cudnn::tensor;

use cudnn::softmax;

use Result;
use context;

pub struct Softmax<T: scalar::Float> {
    algo: softmax::Algorithm,
    mode: softmax::Mode,
    _dummy: marker::PhantomData<T>,
}

pub struct SoftmaxCompiled<'a, T: 'a + scalar::Float> {
    algo: softmax::Algorithm,
    mode: softmax::Mode,
    x_desc: &'a tensor::Descriptor<T>,
    y_desc: &'a tensor::Descriptor<T>,
}

impl<T: scalar::Float> Softmax<T> {
    pub fn new(algo: softmax::Algorithm, mode: softmax::Mode) -> Result<Softmax<T>> {
        Ok(Softmax {
               algo,
               mode,
               _dummy: marker::PhantomData::default(),
           })
    }

    pub fn compile<'a>(self,
                       _: &mut context::Context,
                       x_desc: &'a tensor::Descriptor<T>,
                       y_desc: &'a tensor::Descriptor<T>)
                       -> Result<SoftmaxCompiled<'a, T>> {
        Ok(SoftmaxCompiled {
               algo: self.algo,
               mode: self.mode,
               x_desc: x_desc,
               y_desc: y_desc,
           })
    }
}

impl<'a, T: scalar::Float> SoftmaxCompiled<'a, T> {
    pub fn foward(&self,
                  context: &mut context::Context,
                  x: &memory::Slice<T>,
                  y: &mut memory::Slice<T>)
                  -> Result<()> {
        try!(softmax::forward(context.context(),
                              self.algo,
                              self.mode,
                              T::ONE,
                              tensor::Tensor::new(self.x_desc, x).unwrap(),
                              T::ZERO,
                              tensor::TensorMut::new(self.y_desc, y).unwrap()));
        Ok(())
    }
}
