use cuda::memory;
use cudnn;

use Result;

pub struct Tensor<T>
    where T: cudnn::scalar::Scalar
{
    shape: (usize, usize, usize, usize),
    mem: memory::Memory<T>,
    cudnn_desc: cudnn::tensor::Descriptor<T>,
}

impl<T> Tensor<T>
    where T: cudnn::scalar::Scalar
{
    pub fn new(shape: (usize, usize, usize, usize)) -> Result<Tensor<T>> {
        let (n, c, h, w) = shape;
        let mem = memory::Memory::new(n * c * h * w)?;
        let cudnn_desc =
            cudnn::tensor::Descriptor::new_4d(cudnn::tensor::Format::NCHW, n, c, h, w)?;
        Ok(Tensor {
               shape,
               mem,
               cudnn_desc,
           })
    }

    pub fn shape(&self) -> (usize, usize, usize, usize) {
        self.shape
    }

    pub fn mem(&self) -> &memory::Memory<T> {
        &self.mem
    }

    pub fn mem_mut(&mut self) -> &mut memory::Memory<T> {
        &mut self.mem
    }

    pub fn cudnn_desc(&self) -> &cudnn::tensor::Descriptor<T> {
        &self.cudnn_desc
    }

    pub fn cudnn_tensor<'a>(&'a self) -> cudnn::tensor::Tensor<'a, T> {
        cudnn::tensor::Tensor::new(&self.cudnn_desc, &self.mem)
    }

    pub fn cudnn_tensor_mut<'a>(&'a mut self) -> cudnn::tensor::TensorMut<'a, T> {
        cudnn::tensor::TensorMut::new(&self.cudnn_desc, &mut self.mem)
    }
}
