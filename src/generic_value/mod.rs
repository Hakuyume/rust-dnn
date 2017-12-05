pub trait USize {
    const VALUE: usize;
}

pub trait Sub<Y> {
    type Output;
}

pub mod values;
mod impls;
