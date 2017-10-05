use std::mem;

use cuda::Result;
use cuda::slice;
use cuda::memory;

pub struct Workspace {
    mem: Option<memory::Memory<u8>>,
}

impl Workspace {
    pub fn new() -> Workspace {
        Workspace { mem: None }
    }

    pub fn get<T>(&mut self, len: usize) -> Result<&mut slice::Slice<T>> {
        let size = len * mem::size_of::<T>();

        let mem = {
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
            self.mem.get_or_insert(mem)
        };

        unsafe { Ok(slice::from_raw_parts_mut(mem.as_mut_ptr() as *mut T, len)) }
    }
}
