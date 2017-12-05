use std::ops;

use num_traits;

use cudnn;

use generic_value::USize;
use generic_value::values::*;
use Result;
use misc;
use Context;
use Tensor;

use super::Layer;
use super::UnaryLayer;
use super::Convolution2D;

pub struct Linear<T, InC, OutC>(Convolution2D<T, InC, OutC, U1, U0, U1, U1>)
    where T: cudnn::scalar::Scalar,
          InC: USize,
          OutC: USize;

impl<T, InC, OutC> Linear<T, InC, OutC>
    where T: cudnn::scalar::Scalar,
          InC: USize,
          OutC: USize
{
    pub fn new() -> Result<Linear<T, InC, OutC>> {
        Ok(Linear(Convolution2D::new()?))
    }
}

impl<T, InC, OutC> Layer<T> for Linear<T, InC, OutC>
    where T: ops::Neg<Output = T> + cudnn::scalar::Scalar + misc::Scalar,
          InC: USize,
          OutC: USize
{
    fn optimize(&mut self, context: &mut Context, lr: T) -> Result<()> {
        self.0.optimize(context, lr)
    }
}

impl<T, S, N, InC, OutC> UnaryLayer<T, N, InC, U1, U1, N, OutC, U1, U1>
    for Linear<T, InC, OutC>
    where T: ops::Neg<Output = T> + cudnn::scalar::Scalar + cudnn::scalar::Scale<Scale = S> + misc::Scalar,
          S: From<T> + num_traits::Zero + num_traits::One,
          N: USize,
          InC: USize,
          OutC: USize,
{
    fn forward(&self,
               context: &mut Context,
               x: &Tensor<T, N, InC, U1, U1>,
               y: &mut Tensor<T, N, OutC, U1, U1>)
               -> Result<()> {
        self.0.forward(context, x, y)
    }

    fn backward(&mut self,
                context: &mut Context,
                x: &Tensor<T, N, InC, U1, U1>,
                dy: &Tensor<T, N, OutC, U1, U1>,
                dx: &mut Tensor<T, N, InC, U1, U1>)
                -> Result<()> {
        self.0.backward(context, x, dy, dx)
    }
}
