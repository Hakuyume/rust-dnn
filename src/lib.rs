extern crate cuda;
extern crate cudnn;

mod error;
pub use error::Result;

pub mod context;

pub mod convolution;
pub mod softmax;
