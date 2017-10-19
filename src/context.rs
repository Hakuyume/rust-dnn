use cuda::slice;
use cuda::memory;
use cudnn;

use Result;

pub struct Context {
    cudnn: cudnn::context::Context,
    workspace: Option<memory::Memory<u8>>,
}

impl Context {
    pub fn new() -> Result<Context> {
        Ok(Context {
               cudnn: cudnn::context::Context::new()?,
               workspace: None,
           })
    }

    pub fn cudnn(&mut self) -> &mut cudnn::context::Context {
        &mut self.cudnn
    }

    pub fn cudnn_with_workspace
        (&mut self,
         size: usize)
         -> Result<(&mut cudnn::context::Context, &mut slice::Slice<u8>)> {
        let workspace = {
            let workspace = match self.workspace.take() {
                Some(workspace) => {
                    if size <= workspace.len() {
                        workspace
                    } else {
                        drop(workspace);
                        memory::Memory::new(size)?
                    }
                }
                None => memory::Memory::new(size)?,
            };
            self.workspace.get_or_insert(workspace)
        };

        Ok((&mut self.cudnn, &mut workspace[..size]))
    }
}
