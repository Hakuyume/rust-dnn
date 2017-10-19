use libc::{c_float, c_void, size_t};

use cuda::Result;
use cuda::slice;
use cuda::misc;

use misc::get_grid_block;

extern "C" {
    fn relu_forward_f(len: size_t, x: *mut c_float);
    fn relu_backward_f(len: size_t, y: *const c_float, dy: *mut c_float);
}

pub trait Scalar {
    const FORWARD: *const c_void;
    const BACKWARD: *const c_void;
}

impl Scalar for c_float {
    const FORWARD: *const c_void = relu_forward_f as *const c_void;
    const BACKWARD: *const c_void = relu_backward_f as *const c_void;
}

pub fn forward<T: Scalar>(x: &mut slice::Slice<T>) -> Result<()> {
    let (grid, block) = get_grid_block(x.len());
    unsafe {
        misc::launch_kernel(T::FORWARD,
                            grid,
                            block,
                            &[&(x.len() as size_t), &(x.as_mut_ptr())],
                            0,
                            None)?
    }
    Ok(())
}

pub fn backward<T: Scalar>(y: &slice::Slice<T>, dy: &mut slice::Slice<T>) -> Result<()> {
    assert_eq!(y.len(), dy.len());
    let (grid, block) = get_grid_block(y.len());
    unsafe {
        misc::launch_kernel(T::BACKWARD,
                            grid,
                            block,
                            &[&(y.len() as size_t), &y.as_ptr(), &dy.as_mut_ptr()],
                            0,
                            None)?
    }
    Ok(())
}
