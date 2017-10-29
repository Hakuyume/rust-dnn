use cuda::memory;
use cudnn;

use Result;
use Scalar;

pub struct Tensor<T>
    where T: Scalar
{
    shape: (usize, usize, usize, usize),
    mem: memory::Memory<T>,
    cudnn: cudnn::tensor::Descriptor<T>,
}

impl<T> Tensor<T>
    where T: Scalar
{
    pub fn new(shape: (usize, usize, usize, usize)) -> Result<Tensor<T>> {
        let (n, c, h, w) = shape;
        let mut cudnn = cudnn::tensor::Descriptor::new()?;
        cudnn.set_4d(cudnn::tensor::Format::NCHW, n, c, h, w)?;
        let mem = memory::Memory::new(cudnn.len())?;
        Ok(Tensor { shape, mem, cudnn })
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
        &self.cudnn
    }

    pub fn cudnn_tensor<'a>(&'a self) -> cudnn::tensor::Tensor<'a, T> {
        cudnn::tensor::Tensor::new(&self.cudnn, &self.mem)
    }

    pub fn cudnn_tensor_mut<'a>(&'a mut self) -> cudnn::tensor::TensorMut<'a, T> {
        cudnn::tensor::TensorMut::new(&self.cudnn, &mut self.mem)
    }
}
