use std::marker;

use num;

use cuda::memory;
use cublas;
use cudnn;

use generic_value;
use generic_value::USize;
use generic_value::values::*;
use Result;
use Context;
use Tensor;

use cuda::memory::Repr;

pub struct Convolution2D<T, InC, OutC, KSize, Pad, Stride, Dilate>
    where T: cudnn::scalar::Scalar,
          InC: USize,
          OutC: USize,
          KSize: USize,
          Pad: USize,
          Stride: USize,
          Dilate: USize
{
    w_desc: cudnn::filter::Descriptor<T>,
    w: memory::Memory<T>,
    dw: memory::Memory<T>,
    conv_desc: cudnn::convolution::Descriptor<T>,
    _type: marker::PhantomData<(InC, OutC, KSize, Pad, Stride, Dilate)>,
}

impl<T, InC, OutC, KSize, Pad, Stride, Dilate> Convolution2D<T,
                                                             InC,
                                                             OutC,
                                                             KSize,
                                                             Pad,
                                                             Stride,
                                                             Dilate>
    where T: cudnn::scalar::Scalar,
          InC: USize,
          OutC: USize,
          KSize: USize,
          Pad: USize,
          Stride: USize,
          Dilate: USize
{
    pub fn new() -> Result<Convolution2D<T, InC, OutC, KSize, Pad, Stride, Dilate>> {
        let mut w_desc = cudnn::filter::Descriptor::new()?;
        w_desc
            .set_4d(cudnn::tensor::Format::NCHW,
                    OutC::VALUE,
                    InC::VALUE,
                    KSize::VALUE,
                    KSize::VALUE)?;
        let w = memory::Memory::new(OutC::VALUE * InC::VALUE * KSize::VALUE * KSize::VALUE)?;
        let dw = memory::Memory::new(w.len())?;
        let mut conv_desc = cudnn::convolution::Descriptor::new()?;
        conv_desc
            .set_2d(Pad::VALUE,
                    Pad::VALUE,
                    Stride::VALUE,
                    Stride::VALUE,
                    Dilate::VALUE,
                    Dilate::VALUE,
                    cudnn::convolution::Mode::Convolution)?;
        Ok(Convolution2D {
               w_desc,
               w,
               dw,
               conv_desc,
               _type: marker::PhantomData::default(),
           })
    }
}

impl<T, InC, OutC, KSize, Pad, Stride, Dilate> Convolution2D<T,
                                                             InC,
                                                             OutC,
                                                             KSize,
                                                             Pad,
                                                             Stride,
                                                             Dilate>
    where T: num::Float + cublas::scalar::Scalar + cudnn::scalar::Scalar,
          InC: USize,
          OutC: USize,
          KSize: USize,
          Pad: USize,
          Stride: USize,
          Dilate: USize
{
    pub fn optimize(&mut self, context: &mut Context, lr: T) -> Result<()> {
        cublas::axpy(&mut context.cublas,
                     self.w.len(),
                     &-lr,
                     &self.dw,
                     1,
                     &mut self.w,
                     1)?;
        Ok(())
    }
}

impl<T, S, InC, OutC, KSize> Convolution2D<T, InC, OutC, KSize, U0, U1, U1>
    where T: cudnn::scalar::Scale<Scale = S>,
          S: num::Float,
          InC: USize,
          OutC: USize,
          KSize: USize + generic_value::Sub<U1>
{
    pub fn forward<N, InH, InW, OutH, OutW>(&self,
                                            context: &mut Context,
                                            x: &Tensor<T, N, InC, InH, InW>,
                                            y: &mut Tensor<T, N, OutC, OutH, OutW>)
                                            -> Result<()>
        where N: USize,
              InH: USize + generic_value::Sub<<KSize as generic_value::Sub<U1>>::Output, Output = OutH>,
              InW: USize + generic_value::Sub<<KSize as generic_value::Sub<U1>>::Output, Output = OutW>,
              OutH: USize,
              OutW: USize
    {
        let algo =
            cudnn::convolution::get_forward_algorithm(&mut context.cudnn,
                                                      x.cudnn_desc(),
                                                      &self.w_desc,
                                                      &self.conv_desc,
                                                      y.cudnn_desc(),
                                                      cudnn::convolution::FwdPreference::PreferFastest)?;
        let workspace_size = cudnn::convolution::get_forward_workspace_size(&mut context.cudnn,
                                                                            x.cudnn_desc(),
                                                                            &self.w_desc,
                                                                            &self.conv_desc,
                                                                            y.cudnn_desc(),
                                                                            algo)?;
        cudnn::convolution::forward(&mut context.cudnn,
                                    &S::one(),
                                    x.cudnn_mem(),
                                    (&self.w_desc, &self.w),
                                    &self.conv_desc,
                                    algo,
                                    &mut context.workspace.get(workspace_size)?,
                                    &S::zero(),
                                    y.cudnn_mem_mut())?;
        Ok(())
    }

    pub fn backward<N, InH, InW,  OutH, OutW>(&mut self,
                                              context: &mut Context,
                                              x: &Tensor<T, N, InC, InH, InW>,
                                              dy: &Tensor<T, N, OutC, OutH, OutW>,
                                              _: &mut Tensor<T, N, InC, InH, InW>)
                                              -> Result<()>
        where N: USize,
              InC: USize,
              InH: USize + generic_value::Sub<<KSize as generic_value::Sub<U1>>::Output, Output = OutH>,
              InW: USize + generic_value::Sub<<KSize as generic_value::Sub<U1>>::Output, Output = OutW>,
              OutC: USize,
              OutH: USize,
              OutW: USize
    {
        let algo =
            cudnn::convolution::get_backward_filter_algorithm(&mut context.cudnn,
                                                              x.cudnn_desc(),
                                                              dy.cudnn_desc(),
                                                              &self.conv_desc,
                                                              &self.w_desc,
                                                              cudnn::convolution::BwdFilterPreference::PreferFastest)?;
        let workspace_size =
            cudnn::convolution::get_backward_filter_workspace_size(&mut context.cudnn,
                                                                   x.cudnn_desc(),
                                                                   dy.cudnn_desc(),
                                                                   &self.conv_desc,
                                                                   &self.w_desc,
                                                                   algo)?;
        cudnn::convolution::backward_filter(&mut context.cudnn,
                                            &S::one(),
                                            x.cudnn_mem(),
                                            dy.cudnn_mem(),
                                            &self.conv_desc,
                                            algo,
                                            &mut context.workspace.get(workspace_size)?,
                                            &S::zero(),
                                            (&self.w_desc, &mut self.dw))?;
        Ok(())
    }
}
