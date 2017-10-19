use cudnn;

pub trait Scalar: cudnn::scalar::Scalar {}
impl Scalar for f32 {}
