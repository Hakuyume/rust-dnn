use std::error;
use std::fmt;

use cuda;
use cublas;
use cudnn;

#[derive(Clone, Copy, Debug)]
pub enum Error {
    Cuda(cuda::Error),
    Cublas(cublas::Error),
    Cudnn(cudnn::Error),
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match self {
            &Error::Cuda(ref err) => err.description(),
            &Error::Cublas(ref err) => err.description(),
            &Error::Cudnn(ref err) => err.description(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Error::Cuda(ref err) => err.fmt(f),
            &Error::Cublas(ref err) => err.fmt(f),
            &Error::Cudnn(ref err) => err.fmt(f),
        }
    }
}

impl From<cuda::Error> for Error {
    fn from(err: cuda::Error) -> Error {
        Error::Cuda(err)
    }
}

impl From<cublas::Error> for Error {
    fn from(err: cublas::Error) -> Error {
        Error::Cublas(err)
    }
}

impl From<cudnn::Error> for Error {
    fn from(err: cudnn::Error) -> Error {
        Error::Cudnn(err)
    }
}
