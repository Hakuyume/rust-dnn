extern crate libc;
extern crate cuda;
extern crate cudnn;

use std::result;

mod error;
pub use error::Error;
pub type Result<T> = result::Result<T, Error>;

mod context;
pub use context::Context;

pub use cudnn::scalar::Scalar;

mod tensor;
pub use tensor::Tensor;

pub mod layer;

mod misc;
