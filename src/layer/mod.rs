use cudnn;

use Result;
use Context;
use Tensor;

pub trait Layer<T>
    where T: cudnn::scalar::Scalar
{
    fn out_shape(&self,
                 in_shape: (usize, usize, usize, usize))
                 -> Result<(usize, usize, usize, usize)>;
    fn forward(&self, context: &mut Context, x: &Tensor<T>, y: &mut Tensor<T>) -> Result<()>;
}

mod convolution;
pub use self::convolution::Convolution2D;
