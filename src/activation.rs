use cudnn::scalar;
use cudnn::tensor;

use cudnn::activation;
pub use cudnn::activation::Mode;

use Result;
use context;

pub struct Activation<T: scalar::Float> {
    desc: activation::Descriptor<T>,
}

impl<T: scalar::Float> Activation<T> {
    pub fn new(mode: Mode, nan_prop: bool, coef: f64) -> Result<Activation<T>> {
        let mut desc = activation::Descriptor::new()?;
        desc.set(mode, nan_prop, coef)?;
        Ok(Activation { desc })
    }

    pub fn foward<'a>(&self,
                      context: &mut context::Context,
                      x: tensor::Tensor<'a, T>,
                      y: tensor::TensorMut<'a, T>)
                      -> Result<()> {
        activation::forward(context.context(), &self.desc, T::ONE, x, T::ZERO, y)?;
        Ok(())
    }

    pub fn foward_inplace<'a>(&self,
                              context: &mut context::Context,
                              x: tensor::TensorMut<'a, T>)
                              -> Result<()> {
        activation::forward_inplace(context.context(), &self.desc, T::ONE, x, T::ZERO)?;
        Ok(())
    }

    pub fn relu() -> Result<Activation<T>> {
        Activation::new(Mode::Relu, true, 0.)
    }
}
