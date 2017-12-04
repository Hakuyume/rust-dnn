use cudnn;

use Result;
use Context;
use Tensor;

pub trait Layer<T>
    where T: cudnn::scalar::Scalar
{
    fn in_shape(&self) -> (usize, usize, usize, usize);
    fn out_shape(&self) -> (usize, usize, usize, usize);
    fn forward(&self, context: &mut Context, x: &Tensor<T>, y: &mut Tensor<T>) -> Result<()>;
    fn backward(&mut self,
                context: &mut Context,
                x: &Tensor<T>,
                dy: &Tensor<T>,
                dx: &mut Tensor<T>,
                momentum: T)
                -> Result<()>;
    fn optimize(&mut self, context: &mut Context, lr: T) -> Result<()>;
}

mod convolution;
pub use self::convolution::Convolution2D;
