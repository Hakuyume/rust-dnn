extern crate cuda;
extern crate cudnn;
extern crate nn;

use nn::custom;

pub mod utils;

fn main() {
    let mut x = utils::random(16).unwrap();
    utils::dump(&x).unwrap();

    custom::relu_forward_inplace(&mut x).unwrap();
    utils::dump(&x).unwrap();

    let mut dy = utils::random(x.len()).unwrap();
    utils::dump(&dy).unwrap();

    custom::relu_backward_inplace(&x, &mut dy).unwrap();
    utils::dump(&dy).unwrap();
}
