extern crate nn;

use nn::dataset::MNIST;

fn main() {
    let mnist = MNIST::new("mnist").unwrap();
    println!("train: {}", mnist.train.len());
    println!("test: {}", mnist.test.len());

    let (image, label) = mnist.train.get(0);
    for y in 0..MNIST::SIZE {
        for x in 0..MNIST::SIZE {
            if image[y * MNIST::SIZE + x] > 128 {
                print!("xx");

            } else {
                print!("  ");
            }
        }
        println!();
    }
    println!("{}", label);
}
