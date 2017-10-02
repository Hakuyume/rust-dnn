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
}

pub struct Convolution2DCompiled<'a, T: 'a + scalar::Float> {
    w_desc: filter::Descriptor<T>,
    w: memory::Memory<T>,
    conv_desc: convolution::Descriptor<T>,
    x_desc: &'a tensor::Descriptor<T>,
    y_desc: &'a tensor::Descriptor<T>,
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
           })
    }

    pub fn compile<'a>(self,
                       context: &mut context::Context,
                       x_desc: &'a tensor::Descriptor<T>,
                       y_desc: &'a tensor::Descriptor<T>)
                       -> Result<Convolution2DCompiled<'a, T>> {
        let perf_results = try!(convolution::find_forward_algorithm(context.context(),
                                                                    x_desc,
                                                                    &self.w_desc,
                                                                    &self.conv_desc,
                                                                    y_desc,
                                                                    1));
        let convolution::FwdAlgoPerf { algo, memory, .. } = perf_results[0];
        Ok(Convolution2DCompiled {
               w_desc: self.w_desc,
               w: self.w,
               conv_desc: self.conv_desc,
               x_desc: x_desc,
               y_desc: y_desc,
               algo,
               workspace_size: memory,
           })
    }
}

impl<'a, T: scalar::Float> Convolution2DCompiled<'a, T> {
    pub fn foward(&self,
                  context: &mut context::Context,
                  x: &memory::Slice<T>,
                  y: &mut memory::Slice<T>)
                  -> Result<()> {
        let (context, workspace) = try!(context.context_with_workspace(self.workspace_size));
        try!(convolution::forward(context,
                                  T::ONE,
                                  tensor::Tensor::new(self.x_desc, x).unwrap(),
                                  filter::Filter::new(&self.w_desc, &self.w).unwrap(),
                                  &self.conv_desc,
                                  self.algo,
                                  workspace,
                                  T::ZERO,
                                  tensor::TensorMut::new(self.y_desc, y).unwrap()));
        Ok(())
    }
}
