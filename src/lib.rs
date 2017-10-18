extern crate libc;
extern crate cuda;
extern crate cudnn;

mod error;
pub use error::Result;

pub mod workspace;

pub mod convolution;
pub mod softmax;

mod misc;
pub mod relu;
