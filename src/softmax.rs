use std::marker;

use cudnn::scalar;
use cudnn::tensor;

use cudnn::softmax;
pub use cudnn::softmax::{Algorithm, Mode};

use Result;
use context;

pub struct Softmax<T: scalar::Float> {
    algo: Algorithm,
    mode: Mode,
    _dummy: marker::PhantomData<T>,
}

impl<T: scalar::Float> Softmax<T> {
    pub fn new(algo: softmax::Algorithm, mode: softmax::Mode) -> Result<Softmax<T>> {
        Ok(Softmax {
               algo,
               mode,
               _dummy: marker::PhantomData::default(),
           })
    }

    pub fn foward<'a>(&mut self,
                      context: &mut context::Context,
                      x: tensor::Tensor<'a, T>,
                      y: tensor::TensorMut<'a, T>)
                      -> Result<()> {
        try!(softmax::forward(context.context(),
                              self.algo,
                              self.mode,
                              T::ONE,
                              x,
                              T::ZERO,
                              y));
        Ok(())
    }
}
