use std::marker;

use cuda::memory;
use cudnn;

use generic_value::USize;
use Result;

pub struct Tensor<T, N, C, H, W>
    where T: cudnn::scalar::Scalar
{
    mem: memory::Memory<T>,
    cudnn: cudnn::tensor::Descriptor<T>,
    _type: marker::PhantomData<(N, C, H, W)>,
}

impl<T, N, C, H, W> Tensor<T, N, C, H, W>
    where T: cudnn::scalar::Scalar,
          N: USize,
          C: USize,
          H: USize,
          W: USize
{
    pub fn new() -> Result<Tensor<T, N, C, H, W>> {
        let mem = memory::Memory::new(N::VALUE * C::VALUE * H::VALUE * W::VALUE)?;
        let mut cudnn = cudnn::tensor::Descriptor::new()?;
        cudnn
            .set_4d(cudnn::tensor::Format::NCHW,
                    N::VALUE,
                    C::VALUE,
                    H::VALUE,
                    W::VALUE)?;
        Ok(Tensor {
               mem,
               cudnn,
               _type: marker::PhantomData::default(),
           })
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

    pub fn cudnn_mem(&self) -> (&cudnn::tensor::Descriptor<T>, &memory::Memory<T>) {
        (&self.cudnn, &self.mem)
    }

    pub fn cudnn_mem_mut(&mut self) -> (&cudnn::tensor::Descriptor<T>, &mut memory::Memory<T>) {
        (&self.cudnn, &mut self.mem)
    }
}
