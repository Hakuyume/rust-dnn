extern crate num_traits;
extern crate cuda;
extern crate cublas;
extern crate cudnn;

use std::result;

pub mod generic_value;

mod error;
pub use error::Error;
pub type Result<T> = result::Result<T, Error>;

mod context;
pub use context::Context;

mod tensor;
pub use tensor::Tensor;

pub mod layer;

pub mod dataset;
