use num_traits;

use cuda::memory;
use cudnn;

use Result;
use Context;
use Tensor;

use super::Layer;

use num_traits::{Zero, One};

pub struct Convolution2D<T>
    where T: cudnn::scalar::Scalar
{
    w_desc: cudnn::filter::Descriptor<T>,
    w: memory::Memory<T>,
    conv_desc: cudnn::convolution::Descriptor<T>,
}

impl<T> Convolution2D<T>
    where T: cudnn::scalar::Scalar
{
    pub fn new(c_out: usize,
               c_in: usize,
               ksize: usize,
               pad: usize,
               stride: usize,
               dilate: usize)
               -> Result<Convolution2D<T>> {
        let w_desc = cudnn::filter::Descriptor::new_4d(cudnn::tensor::Format::NCHW,
                                                       c_out,
                                                       c_in,
                                                       ksize,
                                                       ksize)?;
        let w = memory::Memory::new(w_desc.len())?;
        let conv_desc =
            cudnn::convolution::Descriptor::new_2d(pad,
                                                   pad,
                                                   stride,
                                                   stride,
                                                   dilate,
                                                   dilate,
                                                   cudnn::convolution::Mode::Convolution)?;
        Ok(Convolution2D {
               w_desc,
               w,
               conv_desc,
           })
    }
}

impl<T> Layer<T> for Convolution2D<T>
    where T: cudnn::scalar::Scalar + cudnn::scalar::Scale,
          T::Scale: num_traits::Zero + num_traits::One
{
    fn out_shape(&self,
                 in_shape: (usize, usize, usize, usize))
                 -> Result<(usize, usize, usize, usize)> {
        let (n, c, h, w) = in_shape;
        let desc = cudnn::tensor::Descriptor::new_4d(cudnn::tensor::Format::NCHW, n, c, h, w)?;
        Ok(cudnn::convolution::get_2d_forward_output_dim(&self.conv_desc, &desc, &self.w_desc)?)
    }

    fn forward(&self, context: &mut Context, x: &Tensor<T>, y: &mut Tensor<T>) -> Result<()> {
        let algo =
            cudnn::convolution::get_forward_algorithm(context.cudnn(),
                                                      x.cudnn_desc(),
                                                      &self.w_desc,
                                                      &self.conv_desc,
                                                      y.cudnn_desc(),
                                                      cudnn::convolution::FwdPreference::PreferFastest)?;
        let workspace_size = cudnn::convolution::get_forward_workspace_size(context.cudnn(),
                                                                            x.cudnn_desc(),
                                                                            &self.w_desc,
                                                                            &self.conv_desc,
                                                                            y.cudnn_desc(),
                                                                            algo)?;
        let (context, mut workspace) = context.cudnn_with_workspace(workspace_size)?;
        cudnn::convolution::forward(context,
                                    T::Scale::one(),
                                    x.cudnn_tensor(),
                                    cudnn::filter::Filter::new(&self.w_desc, &self.w),
                                    &self.conv_desc,
                                    algo,
                                    &mut workspace,
                                    T::Scale::zero(),
                                    y.cudnn_tensor_mut())?;
        Ok(())
    }
}
