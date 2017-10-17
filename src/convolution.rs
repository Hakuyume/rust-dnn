use cuda::memory;

use cudnn::scalar;
use cudnn::context;
use cudnn::tensor;
use cudnn::filter;

use cudnn::convolution;

use Result;
use workspace;

pub struct Convolution2D<T: scalar::Float> {
    w_desc: filter::Descriptor<T>,
    w: memory::Memory<T>,
    conv_desc: convolution::Descriptor<T>,
}

impl<T: scalar::Float> Convolution2D<T> {
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

    pub fn foward<'a>(&self,
                      context: &mut context::Context,
                      workspace: &mut workspace::Workspace,
                      x: tensor::Tensor<'a, T>,
                      y: tensor::TensorMut<'a, T>)
                      -> Result<()> {
        let algo = convolution::get_forward_algorithm(context,
                                                      x.desc(),
                                                      &self.w_desc,
                                                      &self.conv_desc,
                                                      y.desc(),
                                                      convolution::FwdPreference::PreferFastest)?;
        let workspace_size = convolution::get_forward_workspace_size(context,
                                                                     x.desc(),
                                                                     &self.w_desc,
                                                                     &self.conv_desc,
                                                                     y.desc(),
                                                                     algo)?;
        convolution::forward(context,
                             T::ONE,
                             x,
                             filter::Filter::new(&self.w_desc, &self.w),
                             &self.conv_desc,
                             algo,
                             workspace.get(workspace_size)?,
                             T::ZERO,
                             y)?;
        Ok(())
    }
}
