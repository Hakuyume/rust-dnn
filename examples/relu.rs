extern crate cuda;
extern crate nn;

use nn::relu;

mod utils;

fn main() {
    utils::bench(|| {
        let mut x = utils::random(16)?;
        utils::dump(&x)?;

        relu::forward(&mut x)?;
        utils::dump(&x)?;

        let mut dy = utils::random(x.len())?;
        utils::dump(&dy)?;

        relu::backward(&x, &mut dy)?;
        utils::dump(&dy)?;
        Ok(())
    })
            .unwrap();
}
