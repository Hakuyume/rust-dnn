use std::marker;

use cudnn::Result;
use cudnn::scalar;
use cudnn::context;
use cudnn::tensor;

use cudnn::softmax;
pub use cudnn::softmax::{Algorithm, Mode};

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

    pub fn foward<'a>(&self,
                      context: &mut context::Context,
                      x: tensor::Tensor<'a, T>,
                      y: tensor::TensorMut<'a, T>)
                      -> Result<()> {
        {
            softmax::forward(context, self.algo, self.mode, T::ONE, x, T::ZERO, y)?;
        }
        Ok(())
    }
}
