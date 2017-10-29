use cuda::memory;
use cudnn::tensor;

use Result;
use Scalar;

pub struct Tensor<T>
    where T: Scalar
{
    shape: (usize, usize, usize, usize),
    mem: memory::Memory<T>,
    cudnn: tensor::Descriptor<T>,
}

impl<T> Tensor<T>
    where T: Scalar
{
    pub fn new(shape: (usize, usize, usize, usize)) -> Result<Tensor<T>> {
        let (n, c, h, w) = shape;
        let mut cudnn = tensor::Descriptor::new()?;
        cudnn.set_4d(tensor::Format::NCHW, n, c, h, w)?;
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

    pub fn cudnn_desc(&self) -> &tensor::Descriptor<T> {
        &self.cudnn
    }

    pub fn cudnn_tensor<'a>(&'a self) -> tensor::Tensor<'a, T> {
        tensor::Tensor::new(&self.cudnn, &self.mem)
    }

    pub fn cudnn_tensor_mut<'a>(&'a mut self) -> tensor::TensorMut<'a, T> {
        tensor::TensorMut::new(&self.cudnn, &mut self.mem)
    }
}
