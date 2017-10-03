use std::marker;

use libc::{c_float, c_void, size_t};

use cuda::misc;
use cudnn::tensor;

use Result;
use context;

fn get_grid_block(len: usize) -> (misc::Dim3, misc::Dim3) {
    const GRID_MAX: usize = 65536;
    const BLOCK_MAX: usize = 512;

    let (grid, block) = {
        if len > GRID_MAX * BLOCK_MAX {
            (GRID_MAX, BLOCK_MAX)
        } else {
            if len > BLOCK_MAX {
                if len % BLOCK_MAX > 0 {
                    (len / BLOCK_MAX + 1, BLOCK_MAX)
                } else {
                    (len / BLOCK_MAX, BLOCK_MAX)
                }
            } else {
                (1, len)
            }
        }
    };

    (misc::Dim3 {
         x: grid,
         y: 1,
         z: 1,
     },
     misc::Dim3 {
         x: block,
         y: 1,
         z: 1,
     })
}

#[link(name = "custom_kernel")]
extern "C" {
    fn relu_forward_inplace_f(x: *mut c_float, len: size_t);
}

pub struct ReLU<T> {
    _dummy: marker::PhantomData<T>,
}

impl ReLU<c_float> {
    pub fn new() -> ReLU<c_float> {
        ReLU { _dummy: marker::PhantomData::default() }
    }

    pub fn forward_inplace<'a>(&self,
                               _: &mut context::Context,
                               x: tensor::TensorMut<'a, c_float>)
                               -> Result<()> {
        let (grid, block) = get_grid_block(x.mem.len());
        let x_ptr = x.mem.as_mut_ptr();
        let len = x.mem.len() as size_t;
        unsafe {
            misc::launch_kernel(relu_forward_inplace_f as *const c_void,
                                grid,
                                block,
                                &mut [&x_ptr as *const *mut c_float as *mut c_void,
                                      &len as *const size_t as *mut c_void],
                                0,
                                None)?
        }
        Ok(())
    }
}
