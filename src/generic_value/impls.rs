use super::traits::*;
use super::values::*;

impl USize for U0 {
    const VALUE: usize = 0;
}

impl USize for U1 {
    const VALUE: usize = 1;
}

impl USize for U10 {
    const VALUE: usize = 10;
}

impl USize for U32 {
    const VALUE: usize = 32;
}

impl USize for U784 {
    const VALUE: usize = 784;
}
