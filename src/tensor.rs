use cuda::memory;
use cuda::slice;
use cudnn::tensor;

use Result;
use Scalar;

pub struct Tensor<T: Scalar> {
    mem: memory::Memory<T>,
    cudnn: tensor::Descriptor<T>,
}

impl<T: Scalar> Tensor<T> {
    pub fn new(shape: (usize, usize, usize, usize)) -> Result<Tensor<T>> {
        let (n, c, h, w) = shape;
        let mut cudnn = tensor::Descriptor::new()?;
        cudnn.set_4d(tensor::Format::NCHW, n, c, h, w)?;
        let mem = memory::Memory::new(cudnn.len())?;
        Ok(Tensor { mem, cudnn })
    }

    pub fn mem(&self) -> &slice::Slice<T> {
        &self.mem
    }

    pub fn mem_mut(&mut self) -> &mut slice::Slice<T> {
        &mut self.mem
    }

    pub fn cudnn_desc(&self) -> &tensor::Descriptor<T> {
        &self.cudnn
    }

    pub fn cudnn_tensor<'a>(&'a self) -> tensor::Tensor<'a, T> {
        tensor::Tensor::new(&self.cudnn, &self.mem)
    }

    pub fn cudnn_tensor_mut<'a>(&'a mut self) -> tensor::TensorMut<'a, T> {
        tensor::TensorMut::new(&mut self.cudnn, &mut self.mem)
    }
}
