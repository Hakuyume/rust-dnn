use cuda::memory;

use cudnn::scalar;
use cudnn::tensor;
use cudnn::filter;

use cudnn;
use cudnn::convolution;

use Result;
use context;

pub struct Convolution2D<T: scalar::Float> {
    w_desc: filter::Descriptor<T>,
    w: memory::Memory<T>,
    conv_desc: convolution::Descriptor<T>,
    forward_cache: Option<ForwardCache>,
}

struct ForwardCache {
    params: (tensor::Param4D, tensor::Param4D),
    algo: convolution::FwdAlgo,
    workspace_size: usize,
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
               forward_cache: None,
           })
    }

    fn find_forward_algorithm(&mut self,
                              context: &mut cudnn::context::Context,
                              x_desc: &tensor::Descriptor<T>,
                              y_desc: &tensor::Descriptor<T>)
                              -> Result<(convolution::FwdAlgo, usize)> {
        let params_new = (x_desc.get_4d()?, y_desc.get_4d()?);
        if let Some(ForwardCache {
                        ref params,
                        algo,
                        workspace_size,
                    }) = self.forward_cache {
            if &params_new == params {
                return Ok((algo, workspace_size));
            }
        }

        let perf_results = convolution::find_forward_algorithm(context,
                                                               x_desc,
                                                               &self.w_desc,
                                                               &self.conv_desc,
                                                               y_desc,
                                                               1)?;
        let convolution::FwdAlgoPerf {
            algo,
            memory: workspace_size,
            ..
        } = perf_results[0];
        self.forward_cache = Some(ForwardCache {
                                      params: params_new,
                                      algo,
                                      workspace_size,
                                  });
        Ok((algo, workspace_size))
    }

    pub fn foward<'a>(&mut self,
                      context: &mut context::Context,
                      x: tensor::Tensor<'a, T>,
                      y: tensor::TensorMut<'a, T>)
                      -> Result<()> {
        let (context, _) = context.cudnn(0)?;
        let (algo, workspace_size) = self.find_forward_algorithm(context, x.desc, y.desc)?;

        let (context, workspace) = context.cudnn(workspace_size)?;
        convolution::forward(context,
                             T::ONE,
                             x,
                             filter::Filter::new(&self.w_desc, &self.w),
                             &self.conv_desc,
                             algo,
                             workspace,
                             T::ZERO,
                             y)?;
        Ok(())
    }
}
