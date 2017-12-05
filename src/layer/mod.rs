use cudnn;

use generic_value::USize;
use Result;
use Context;
use Tensor;

pub trait Layer<T> {
    fn optimize(&mut self, context: &mut Context, lr: T) -> Result<()>;
}

pub trait UnaryLayer<T, InN, InC, InH, InW, OutN, OutC, OutH, OutW>: Layer<T>
    where T: cudnn::scalar::Scalar,
          InN: USize,
          InC: USize,
          InH: USize,
          InW: USize,
          OutN: USize,
          OutC: USize,
          OutH: USize,
          OutW: USize
{
    fn forward(&self,
               context: &mut Context,
               x: &Tensor<T, InN, InC, InH, InW>,
               y: &mut Tensor<T, OutN, OutC, OutH, OutW>)
               -> Result<()>;
    fn backward(&mut self,
                context: &mut Context,
                x: &Tensor<T, InN, InC, InH, InW>,
                dy: &Tensor<T, OutN, OutC, OutH, OutW>,
                dx: &mut Tensor<T, InN, InC, InH, InW>)
                -> Result<()>;
}

mod convolution;
pub use self::convolution::Convolution2D;
