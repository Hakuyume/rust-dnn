use num_traits;

use cuda;
use cublas;
use cudnn;

use generic_value::USize;
use generic_value::values::*;
use Result;
use Context;
use Tensor;

pub struct SoftmaxCrossEntropy<T, N, C>
    where T: cudnn::scalar::Scalar,
          N: USize,
          C: USize
{
    tmp: Tensor<T, N, C, U1, U1>,
}

impl<T, N, C> SoftmaxCrossEntropy<T, N, C>
    where T: cudnn::scalar::Scalar,
          N: USize,
          C: USize
{
    pub fn new() -> Result<SoftmaxCrossEntropy<T, N, C>> {
        Ok(SoftmaxCrossEntropy { tmp: Tensor::new()? })
    }
}


impl<T, S, N, C> SoftmaxCrossEntropy<T, N, C>
    where T: num_traits::Signed + num_traits::NumCast + cublas::scalar::Scalar + cudnn::scalar::Scale<Scale = S>,
          S: num_traits::Signed + num_traits::NumCast,
          N: USize,
          C: USize
{
    pub fn compute(&mut self,
                   context: &mut Context,
                   x: &Tensor<T, N, C, U1, U1>,
                   t: &Tensor<T, N, C, U1, U1>,
                   dx: &mut Tensor<T, N, C, U1, U1>)
                   -> Result<T> {
        cudnn::softmax::forward(&mut context.cudnn,
                                cudnn::softmax::Algorithm::Log,
                                cudnn::softmax::Mode::Channel,
                                &S::one(),
                                x.cudnn_mem(),
                                &S::zero(),
                                self.tmp.cudnn_mem_mut())?;
        cudnn::tensor::add(&mut context.cudnn,
                           &(-S::one() / S::from(N::VALUE).unwrap_or(S::one())),
                           t.cudnn_mem(),
                           &S::zero(),
                           dx.cudnn_mem_mut())?;
        cudnn::softmax::backward(&mut context.cudnn,
                                 cudnn::softmax::Algorithm::Log,
                                 cudnn::softmax::Mode::Channel,
                                 &S::one(),
                                 self.tmp.cudnn_mem(),
                                 None as Option<(_, &cuda::memory::View<_>)>,
                                 &S::zero(),
                                 dx.cudnn_mem_mut())?;
        let sum = cublas::dot(&mut context.cublas,
                              N::VALUE * C::VALUE,
                              self.tmp.mem(),
                              1,
                              t.mem(),
                              1)?;
        Ok(-sum / T::from(N::VALUE).unwrap_or(T::one()))
    }
}
