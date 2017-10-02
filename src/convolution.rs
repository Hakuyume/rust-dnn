use cuda::memory;

use cudnn::scalar;
use cudnn::tensor;
use cudnn::filter;

use cudnn::convolution;

use Result;
use context;

pub struct Convolution2D<T: scalar::Float> {
    w_desc: filter::Descriptor<T>,
    w: memory::Memory<T>,
    conv_desc: convolution::Descriptor<T>,
    algo: Option<convolution::FwdAlgo>,
    workspace_size: Option<usize>,
}

impl<T: scalar::Float + From<f32>> Convolution2D<T> {
    pub fn new(c_out: usize,
               c_in: usize,
               ksize: usize,
               pad: usize,
               stride: usize,
               dilate: usize)
               -> Result<Convolution2D<T>> {
        let mut w_desc = try!(filter::Descriptor::new());
        try!(w_desc.set_4d(tensor::Format::NCHW, c_out, c_in, ksize, ksize));
        let w = try!(memory::Memory::new(w_desc.len()));
        let mut conv_desc = try!(convolution::Descriptor::new());
        try!(conv_desc.set_2d(pad,
                              pad,
                              stride,
                              stride,
                              dilate,
                              dilate,
                              convolution::Mode::Convolution));
        Ok(Convolution2D {
               w_desc,
               w,
               conv_desc,
               algo: None,
               workspace_size: None,
           })
    }

    pub fn compile(&mut self,
                   context: &mut context::Context,
                   x_desc: &tensor::Descriptor<T>,
                   y_desc: &tensor::Descriptor<T>)
                   -> Result<()> {
        let perf_results = try!(convolution::find_forward_algorithm(context.context(),
                                                                    x_desc,
                                                                    &self.w_desc,
                                                                    &self.conv_desc,
                                                                    y_desc,
                                                                    1));
        let convolution::FwdAlgoPerf { algo, memory, .. } = perf_results[0];
        self.algo = Some(algo);
        self.workspace_size = Some(memory);
        Ok(())
    }

    pub fn foward<'a>(&self,
                      context: &mut context::Context,
                      x: tensor::Tensor<'a, T>,
                      y: tensor::TensorMut<'a, T>)
                      -> Result<()> {
        let (context, workspace) = try!(context.context_with_workspace(self.workspace_size
                                                                           .unwrap()));
        try!(convolution::forward(context,
                                  T::from(1.),
                                  x,
                                  filter::Filter::new(&self.w_desc, &self.w).unwrap(),
                                  &self.conv_desc,
                                  self.algo.unwrap(),
                                  workspace,
                                  T::from(0.),
                                  y));
        Ok(())
    }
}
