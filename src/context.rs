use std::mem;

use cuda::memory;
use cublas;
use cudnn;

use Result;

use self::memory::{Repr, ReprMut};

pub struct Context {
    pub workspace: Workspace,
    pub cublas: cublas::context::Context,
    pub cudnn: cudnn::context::Context,
}

impl Context {
    pub fn new() -> Result<Context> {
        let mut cublas = cublas::context::Context::new()?;
        cublas.set_pointer_mode(cublas::PointerMode::Host)?;

        Ok(Context {
               workspace: Workspace::new(),
               cublas,
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
