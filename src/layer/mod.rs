use Result;
use Context;
use Scalar;
use Tensor;

pub trait Layer<T: Scalar> {
    fn out_shape(&self,
                 in_shape: (usize, usize, usize, usize))
                 -> Result<(usize, usize, usize, usize)>;
    fn forward(&self, context: &mut Context, x: &Tensor<T>, y: &mut Tensor<T>) -> Result<()>;
}

mod convolution;
pub use self::convolution::Convolution2D;

pub trait InplaceLayer<T: Scalar> {
    fn forward(&self, context: &mut Context, x: &mut Tensor<T>) -> Result<()>;
}

mod relu;
pub use self::relu::ReLU;
