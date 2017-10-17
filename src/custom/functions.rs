use libc::{c_float, c_void, size_t};

use cuda::Result;
use cuda::slice;
use cuda::misc;

use super::Scalar;
use super::ScalarType;
use super::calc_grid_block;

extern "C" {
    fn relu_forward_inplace_f(x: *mut c_float, len: size_t);
}

pub fn relu_forward_inplace<T: Scalar>(x: &mut slice::Slice<T>) -> Result<()> {
    let func = match T::TYPE {
        ScalarType::Float => relu_forward_inplace_f as *const c_void,
    };
    let (grid, block) = calc_grid_block(x.len());
    let (x, len) = (x.as_mut_ptr(), x.len() as size_t);
    unsafe {
        misc::launch_kernel(func,
                            grid,
                            block,
                            &mut [&x as *const *mut T as *mut c_void,
                                  &len as *const size_t as *mut c_void],
                            0,
                            None)?
    }
    Ok(())
}

extern "C" {
    fn relu_backward_inplace_f(x: *mut c_float, len: size_t);
}

pub fn relu_backward_inplace<T: Scalar>(y: &slice::Slice<T>,
                                        dy: &mut slice::Slice<T>)
                                        -> Result<()> {
    assert_eq!(y.len(), dy.len());

    let func = match T::TYPE {
        ScalarType::Float => relu_backward_inplace_f as *const c_void,
    };
    let (grid, block) = calc_grid_block(y.len());
    let (y, dy, len) = (y.as_ptr(), dy.as_mut_ptr(), y.len() as size_t);
    unsafe {
        misc::launch_kernel(func,
                            grid,
                            block,
                            &mut [&y as *const *const T as *mut c_void,
                                  &dy as *const *mut T as *mut c_void,
                                  &len as *const size_t as *mut c_void],
                            0,
                            None)?
    }
    Ok(())
}
