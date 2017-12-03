use std::mem;

use cuda::memory;
use cudnn;

use Result;

use self::memory::{Repr, ReprMut};

pub struct Context {
    pub workspace: Workspace,
    pub cudnn: cudnn::context::Context,
}

impl Context {
    pub fn new() -> Result<Context> {
        Ok(Context {
               workspace: Workspace::new(),
               cudnn: cudnn::context::Context::new()?,
           })
    }
}

pub struct Workspace {
    mem: Option<memory::Memory<u8>>,
}

impl Workspace {
    pub fn new() -> Workspace {
        Workspace { mem: None }
    }

    pub fn get<'a, T>(&'a mut self, len: usize) -> Result<memory::ViewMut<'a, T>> {
        let size = mem::size_of::<T>() * len;
        let mem = match self.mem.take() {
            Some(mem) => {
                if size <= mem.len() {
                    mem
                } else {
                    drop(mem);
                    memory::Memory::new(size)?
                }
            }
            None => memory::Memory::new(size)?,
        };
        let mem = self.mem.get_or_insert(mem);
        Ok(unsafe { memory::from_raw_parts_mut(mem.as_mut_ptr() as *mut T, len) })
    }
}
