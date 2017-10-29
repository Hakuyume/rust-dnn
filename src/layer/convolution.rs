use cuda::memory;

use cudnn::scalar;
use cudnn::tensor;
use cudnn::filter;
use cudnn::convolution;

use Result;
use Context;
use Scalar;
use Tensor;
use layer::Layer;

pub struct Convolution2D<T>
    where T: scalar::Float
{
    w_desc: filter::Descriptor<T>,
    w: memory::Memory<T>,
    conv_desc: convolution::Descriptor<T>,
}

impl<T> Convolution2D<T>
    where T: scalar::Float
{
    pub fn new(c_out: usize,
               c_in: usize,
               ksize: usize,
               pad: usize,
               stride: usize,
               dilate: usize)
               -> Result<Convolution2D<T>> {
        let mut w_desc = filter::Descriptor::new()?;
        w_desc
            .set_4d(tensor::Format::NCHW, c_out, c_in, ksize, ksize)?;
        let w = memory::Memory::new(w_desc.len())?;
        let mut conv_desc = convolution::Descriptor::new()?;
        conv_desc
            .set_2d(pad,
                    pad,
                    stride,
                    stride,
                    dilate,
                    dilate,
                    convolution::Mode::Convolution)?;
        Ok(Convolution2D {
               w_desc,
               w,
               conv_desc,
           })
    }
}

impl<T> Layer<T> for Convolution2D<T>
    where T: Scalar + scalar::Float
{
    fn out_shape(&self,
                 in_shape: (usize, usize, usize, usize))
                 -> Result<(usize, usize, usize, usize)> {
        let (n, c, h, w) = in_shape;
        let mut desc = tensor::Descriptor::new()?;
        desc.set_4d(tensor::Format::NCHW, n, c, h, w)?;
        Ok(convolution::get_2d_forward_output_dim(&self.conv_desc, &desc, &self.w_desc)?)
    }

    fn forward(&self, context: &mut Context, x: &Tensor<T>, y: &mut Tensor<T>) -> Result<()> {
        let algo = convolution::get_forward_algorithm(context.cudnn(),
                                                      x.cudnn_desc(),
                                                      &self.w_desc,
                                                      &self.conv_desc,
                                                      y.cudnn_desc(),
                                                      convolution::FwdPreference::PreferFastest)?;
        let workspace_size = convolution::get_forward_workspace_size(context.cudnn(),
                                                                     x.cudnn_desc(),
                                                                     &self.w_desc,
                                                                     &self.conv_desc,
                                                                     y.cudnn_desc(),
                                                                     algo)?;
        let (context, mut workspace) = context.cudnn_with_workspace(workspace_size)?;
        convolution::forward(context,
                             T::ONE,
                             x.cudnn_tensor(),
                             filter::Filter::new(&self.w_desc, &self.w),
                             &self.conv_desc,
                             algo,
                             &mut workspace,
                             T::ZERO,
                             y.cudnn_tensor_mut())?;
        Ok(())
    }
}
