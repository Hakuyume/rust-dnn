use cuda::memory;
use cudnn;

use self::memory::Repr;
use self::memory::SliceMut;

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

    pub fn cudnn_with_workspace<'a>
        (&'a mut self,
         size: usize)
         -> Result<(&mut cudnn::context::Context, memory::ViewMut<'a, u8>)> {
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

        Ok((&mut self.cudnn, workspace.slice_mut(..size)))
    }
}
