use std;

use cuda;
use cudnn;

#[derive(Clone, Copy, Debug)]
pub enum Error {
    Cuda(cuda::Error),
    Cudnn(cudnn::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::Cuda(ref err) => err.description(),
            Error::Cudnn(ref err) => err.description(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
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
