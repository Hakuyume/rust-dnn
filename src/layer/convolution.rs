use cuda::memory;
use cudnn;

use Result;
use misc;
use Context;
use Tensor;

use std::ops::{Sub, Neg};
use num_traits::{Zero, One};
use super::Layer;

pub struct Convolution2D<T>
    where T: cudnn::scalar::Scalar
{
    in_shape: (usize, usize, usize, usize),
    out_shape: (usize, usize, usize, usize),
    w_desc: cudnn::filter::Descriptor<T>,
    w: memory::Memory<T>,
    dw: memory::Memory<T>,
    conv_desc: cudnn::convolution::Descriptor<T>,
}

impl<T> Convolution2D<T>
    where T: cudnn::scalar::Scalar
{
    pub fn new(in_shape: (usize, usize, usize, usize),
               out_c: usize,
               ksize: usize,
               pad: usize,
               stride: usize,
               dilate: usize)
               -> Result<Convolution2D<T>> {
        let (in_n, in_c, in_h, in_w) = in_shape;
        let out_shape = (in_n,
                         out_c,
                         (in_h + pad * 2 - (ksize - 1) * dilate) / stride,
                         (in_w + pad * 2 - (ksize - 1) * dilate) / stride);

        let mut w_desc = cudnn::filter::Descriptor::new()?;
        w_desc
            .set_4d(cudnn::tensor::Format::NCHW, out_c, in_c, ksize, ksize)?;
        let w = memory::Memory::new(out_c * in_c * ksize * ksize)?;
        let dw = memory::Memory::new(out_c * in_c * ksize * ksize)?;
        let mut conv_desc = cudnn::convolution::Descriptor::new()?;
        conv_desc
            .set_2d(pad,
                    pad,
                    stride,
                    stride,
                    dilate,
                    dilate,
                    cudnn::convolution::Mode::Convolution)?;

        {
            let mut in_desc = cudnn::tensor::Descriptor::new()?;
            in_desc
                .set_4d(cudnn::tensor::Format::NCHW, in_n, in_c, in_h, in_w)?;
            assert_eq!(out_shape,
                       cudnn::convolution::get_2d_forward_output_dim(&conv_desc,
                                                                     &in_desc,
                                                                     &w_desc)?);
        }

        Ok(Convolution2D {
               in_shape,
               out_shape,
               w_desc,
               w,
               dw,
               conv_desc,
           })
    }
}

impl<T> Layer<T> for Convolution2D<T>
    where T: Copy + Neg<Output = T> + cudnn::scalar::Scalar + cudnn::scalar::Scale + misc::Scalar,
          T::Scale: From<T> + Sub<Output = T::Scale> + Zero + One
{
    fn in_shape(&self) -> (usize, usize, usize, usize) {
        self.in_shape
    }

    fn out_shape(&self) -> (usize, usize, usize, usize) {
        self.out_shape
    }

    fn forward(&self, context: &mut Context, x: &Tensor<T>, y: &mut Tensor<T>) -> Result<()> {
        assert_eq!(x.shape(), self.in_shape());
        assert_eq!(y.shape(), self.out_shape());

        let algo =
            cudnn::convolution::get_forward_algorithm(&mut context.cudnn,
                                                      x.cudnn().0,
                                                      &self.w_desc,
                                                      &self.conv_desc,
                                                      y.cudnn().0,
                                                      cudnn::convolution::FwdPreference::PreferFastest)?;
        let workspace_size = cudnn::convolution::get_forward_workspace_size(&mut context.cudnn,
                                                                            x.cudnn().0,
                                                                            &self.w_desc,
                                                                            &self.conv_desc,
                                                                            y.cudnn().0,
                                                                            algo)?;
        unsafe {
            cudnn::convolution::forward(&mut context.cudnn,
                                        T::Scale::one(),
                                        x.cudnn(),
                                        (&self.w_desc, &self.w),
                                        &self.conv_desc,
                                        algo,
                                        &mut context.workspace.get(workspace_size)?,
                                        T::Scale::zero(),
                                        y.cudnn_mut())?
        }
        Ok(())
    }

    fn backward(&mut self,
                context: &mut Context,
                x: &Tensor<T>,
                dy: &Tensor<T>,
                dx: &mut Tensor<T>,
                momentum: T)
                -> Result<()> {
        assert_eq!(x.shape(), self.in_shape());
        assert_eq!(dy.shape(), self.out_shape());
        assert_eq!(dx.shape(), self.in_shape());

        let algo =
                cudnn::convolution::get_backward_filter_algorithm(&mut context.cudnn,
                                                                  x.cudnn().0,
                                                                  dy.cudnn().0,
                                                                  &self.conv_desc,
                                                                  &self.w_desc,
                                                                  cudnn::convolution::BwdFilterPreference::PreferFastest)?;
        let workspace_size =
            cudnn::convolution::get_backward_filter_workspace_size(&mut context.cudnn,
                                                                   x.cudnn().0,
                                                                   dy.cudnn().0,
                                                                   &self.conv_desc,
                                                                   &self.w_desc,
                                                                   algo)?;
        unsafe {
            cudnn::convolution::backward_filter(&mut context.cudnn,
                                                T::Scale::one() - momentum.into(),
                                                x.cudnn(),
                                                dy.cudnn(),
                                                &self.conv_desc,
                                                algo,
                                                &mut context.workspace.get(workspace_size)?,
                                                momentum.into(),
                                                (&self.w_desc, &mut self.dw))?
        }
        Ok(())
    }

    fn optimize(&mut self, _: &mut Context, lr: T) -> Result<()> {
        misc::axpy(-lr, &self.dw, &mut self.w)
    }
}
