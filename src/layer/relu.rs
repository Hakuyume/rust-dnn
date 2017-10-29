use libc::{c_float, c_void, size_t};

use cuda::misc;

use cuda::memory::{Repr, ReprMut};

use Result;
use Context;
use scalar;
use Tensor;
use layer::InplaceLayer;
use misc::get_grid_block;

extern "C" {
    fn relu_forward_f(len: size_t, x: *mut c_float);
    fn relu_backward_f(len: size_t, y: *const c_float, dy: *mut c_float);
}

pub trait Scalar: scalar::Scalar {
    const FORWARD: *const c_void;
    const BACKWARD: *const c_void;
}

impl Scalar for c_float {
    const FORWARD: *const c_void = relu_forward_f as *const c_void;
    const BACKWARD: *const c_void = relu_backward_f as *const c_void;
}

pub struct ReLU {}

impl ReLU {
    pub fn new() -> ReLU {
        ReLU {}
    }
}

impl<T> InplaceLayer<T> for ReLU
    where T: Scalar
{
    fn forward(&self, _: &mut Context, x: &mut Tensor<T>) -> Result<()> {
        let (grid, block) = get_grid_block(x.mem().len());
        unsafe {
            misc::launch_kernel(T::FORWARD,
                                grid,
                                block,
                                &[&(x.mem().len() as size_t), &(x.mem_mut().as_mut_ptr())],
                                0,
                                None)?
        }
        Ok(())
    }
}
