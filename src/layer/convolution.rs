use std::marker;
use std::ops;

use num_traits;

use cuda::memory;
use cudnn;

use generic_value;
use generic_value::USize;
use generic_value::values::*;
use Result;
use misc;
use Context;
use Tensor;

use super::Layer;
use super::UnaryLayer;

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
        let dw = memory::Memory::new(OutC::VALUE * InC::VALUE * KSize::VALUE * KSize::VALUE)?;
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

impl<T, InC, OutC, KSize, Pad, Stride, Dilate> Layer<T>
    for Convolution2D<T, InC, OutC, KSize, Pad, Stride, Dilate>
    where T: ops::Neg<Output = T> + cudnn::scalar::Scalar + misc::Scalar,
          InC: USize,
          OutC: USize,
          KSize: USize,
          Pad: USize,
          Stride: USize,
          Dilate: USize
{
    fn optimize(&mut self, _: &mut Context, lr: T) -> Result<()> {
        misc::axpy(-lr, &self.dw, &mut self.w)
    }
}

impl<T, S, KSize, N, InC, InH, InW, OutC, OutH, OutW> UnaryLayer<T, N, InC, InH, InW, N, OutC, OutH, OutW>
    for Convolution2D<T, InC, OutC, KSize, U0, U1, U1>
    where T: ops::Neg<Output = T> + cudnn::scalar::Scalar + cudnn::scalar::Scale<Scale = S> + misc::Scalar,
          S: From<T> + num_traits::Zero + num_traits::One,
          KSize: USize + generic_value::Sub<U1>,
          N: USize,
          InC: USize,
          InH: USize + generic_value::Sub<<KSize as generic_value::Sub<U1>>::Output, Output = OutH>,
          InW: USize + generic_value::Sub<<KSize as generic_value::Sub<U1>>::Output, Output = OutW>,
          OutC: USize,
          OutH: USize,
          OutW: USize
{
    fn forward(&self,
               context: &mut Context,
               x: &Tensor<T, N, InC, InH, InW>,
               y: &mut Tensor<T, N, OutC, OutH, OutW>)
               -> Result<()> {
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
                                        S::one(),
                                        x.cudnn(),
                                        (&self.w_desc, &self.w),
                                        &self.conv_desc,
                                        algo,
                                        &mut context.workspace.get(workspace_size)?,
                                        S::zero(),
                                        y.cudnn_mut())?;
        Ok(())
    }

    fn backward(&mut self,
                context: &mut Context,
                x: &Tensor<T, N, InC, InH, InW>,
                dy: &Tensor<T, N, OutC, OutH, OutW>,
                _: &mut Tensor<T, N, InC, InH, InW>)
                -> Result<()> {
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
                                                S::one(),
                                                x.cudnn(),
                                                dy.cudnn(),
                                                &self.conv_desc,
                                                algo,
                                                &mut context.workspace.get(workspace_size)?,
                                                S::zero(),
                                                (&self.w_desc, &mut self.dw))?;
        Ok(())
    }
}
