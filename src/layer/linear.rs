use num_traits;

use cublas;
use cudnn;

use generic_value::USize;
use generic_value::values::*;
use Result;
use Context;
use Tensor;

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

impl<T, InC, OutC> Linear<T, InC, OutC>
    where T: num_traits::Signed + cublas::scalar::Scalar + cudnn::scalar::Scalar,
          InC: USize,
          OutC: USize
{
    pub fn optimize(&mut self, context: &mut Context, lr: T) -> Result<()> {
        self.0.optimize(context, lr)
    }
}

impl<T, S, InC, OutC> Linear<T, InC, OutC>
    where T: cudnn::scalar::Scale<Scale = S>,
          S: num_traits::Signed,
          InC: USize,
          OutC: USize
{
    pub fn forward<N>(&mut self,
                      context: &mut Context,
                      x: &Tensor<T, N, InC, U1, U1>,
                      y: &mut Tensor<T, N, OutC, U1, U1>)
                      -> Result<()>
        where N: USize
    {
        self.0.forward(context, x, y)
    }

    pub fn backward<N>(&mut self,
                       context: &mut Context,
                       x: &Tensor<T, N, InC, U1, U1>,
                       dy: &Tensor<T, N, OutC, U1, U1>,
                       dx: &mut Tensor<T, N, InC, U1, U1>)
                       -> Result<()>
        where N: USize
    {
        self.0.backward(context, x, dy, dx)
    }
}
