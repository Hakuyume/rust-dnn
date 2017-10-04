use cuda::slice;
use cuda::memory;

use cudnn;

use Result;

pub struct Context {
    workspace: memory::Memory<u8>,
    cudnn: cudnn::context::Context,
}

impl Context {
    pub fn new() -> Result<Context> {
        let workspace = memory::Memory::new(0)?;
        let cudnn = context::Context::new()?;
        Ok(Context { workspace, cudnn })
    }

    fn alloc_workspace(&mut self, size: usize) -> Result<()> {
        if self.workspace.len() < size {
            self.workspace = memory::Memory::new(size)?;
        }
    }

    pub fn workspace(&mut self, size: usize) -> Result<&mut slice::Slice<u8>> {
        self.alloc_workspace(size)?;
        Ok(&mut self.workspace[..size])
    }

    pub fn cudnn(&mut self, size: usize) -> Result<(&mut context::Context, &mut slice::Slice<u8>)> {
        self.alloc_workspace(size)?;
        Ok((&mut self.context, &mut self.workspace[..size]))
    }
}
