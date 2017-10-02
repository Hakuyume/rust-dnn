use cuda::slice;
use cuda::memory;

use cudnn::context;

use Result;

pub struct Context {
    context: context::Context,
    workspace: memory::Memory<u8>,
}

impl Context {
    pub fn new() -> Result<Context> {
        let context = context::Context::new()?;
        let workspace = memory::Memory::new(0)?;
        Ok(Context { context, workspace })
    }

    pub fn context(&mut self) -> &mut context::Context {
        &mut self.context
    }

    pub fn context_with_workspace(&mut self,
                                  size: usize)
                                  -> Result<(&mut context::Context, &mut slice::Slice<u8>)> {
        if self.workspace.len() < size {
            self.workspace = memory::Memory::new(size)?;
        }
        Ok((&mut self.context, &mut self.workspace[..size]))
    }
}
