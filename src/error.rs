use std::error;
use std::fmt;
use std::result;

use cuda;
use cudnn;

#[derive(Clone, Copy, Debug)]
pub enum Error {
    Cuda(cuda::Error),
    Cudnn(cudnn::Error),
}

pub type Result<T> = result::Result<T, Error>;

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::Cuda(ref err) => err.description(),
            Error::Cudnn(ref err) => err.description(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Cuda(ref err) => err.fmt(f),
            Error::Cudnn(ref err) => err.fmt(f),
        }
    }
}

impl From<cuda::Error> for Error {
    fn from(err: cuda::Error) -> Error {
        Error::Cuda(err)
    }
}

impl From<cudnn::Error> for Error {
    fn from(err: cudnn::Error) -> Error {
        Error::Cudnn(err)
    }
}
