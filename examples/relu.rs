extern crate nn;

use nn::layer::InplaceLayer;

mod utils;

fn main() {
    utils::bench(|| {
        let mut context = nn::Context::new()?;

        let mut x = utils::random((1, 16, 1, 1))?;
        utils::dump(&x)?;

        let relu = nn::layer::ReLU::new();

        relu.forward(&mut context, &mut x)?;
        utils::dump(&x)?;
        Ok(())
    })
            .unwrap();
}
