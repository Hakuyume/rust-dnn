extern crate cuda;
extern crate cudnn;
extern crate nn;

use nn::custom;

mod utils;

fn main() {
    utils::bench(|| {
        let mut x = utils::random(16)?;
        utils::dump(&x)?;

        custom::relu_forward_inplace(&mut x)?;
        utils::dump(&x)?;

        let mut dy = utils::random(x.len())?;
        utils::dump(&dy)?;

        custom::relu_backward_inplace(&x, &mut dy)?;
        utils::dump(&dy)?;
        Ok(())
    })
            .unwrap();
}
