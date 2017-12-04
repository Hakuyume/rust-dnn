use libc::{c_float, c_void, size_t};
use std::cmp;

use cuda::memory;
use cuda::misc;

use Result;

fn ceil_div(x: usize, y: usize) -> usize {
    x / y + if x % y > 0 { 1 } else { 0 }
}

fn get_grid_block(len: usize) -> (misc::Dim3, misc::Dim3) {
    const GRID_MAX: usize = 65536;
    const BLOCK_MAX: usize = 512;

    (misc::Dim3 {
         x: cmp::min(ceil_div(len, BLOCK_MAX), GRID_MAX),
         y: 1,
         z: 1,
     },
     misc::Dim3 {
         x: cmp::min(len, BLOCK_MAX),
         y: 1,
         z: 1,
     })
}

pub trait Scalar {
    const AXPY: *const c_void;
}

extern "C" {
    fn axpy_f(len: size_t, alpha: c_float, x: *const c_float, y: *mut c_float);
}

impl Scalar for c_float {
    const AXPY: *const c_void = axpy_f as *const c_void;
}

pub fn axpy<T, X, Y>(alpha: T, x: &X, y: &mut Y) -> Result<()>
    where T: Scalar,
          X: memory::Repr<T>,
          Y: memory::ReprMut<T>
{
    assert_eq!(x.len(), y.len());
    let (grid, block) = get_grid_block(x.len());
    unsafe {
        misc::launch_kernel(T::AXPY,
                            grid,
                            block,
                            &[&(x.len() as size_t),
                              &alpha,
                              &(x.as_ptr()),
                              &(y.as_mut_ptr())],
                            0,
                            None)?
    }
    Ok(())

}
