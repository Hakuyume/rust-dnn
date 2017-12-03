use cuda::memory;
use cudnn;

use Result;

pub struct Tensor<T>
    where T: cudnn::scalar::Scalar
{
    shape: (usize, usize, usize, usize),
    mem: memory::Memory<T>,
    cudnn: cudnn::tensor::Descriptor<T>,
}

impl<T> Tensor<T>
    where T: cudnn::scalar::Scalar
{
    pub fn new(shape: (usize, usize, usize, usize)) -> Result<Tensor<T>> {
        let (n, c, h, w) = shape;
        let mem = memory::Memory::new(n * c * h * w)?;
        let mut cudnn = cudnn::tensor::Descriptor::new()?;
        cudnn.set_4d(cudnn::tensor::Format::NCHW, n, c, h, w)?;
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

    pub fn cudnn(&self) -> (&cudnn::tensor::Descriptor<T>, &memory::Memory<T>) {
        (&self.cudnn, &self.mem)
    }

    pub fn cudnn_mut(&mut self) -> (&cudnn::tensor::Descriptor<T>, &mut memory::Memory<T>) {
        (&self.cudnn, &mut self.mem)
    }
}
