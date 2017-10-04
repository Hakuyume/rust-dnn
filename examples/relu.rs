extern crate rand;
use rand::Rng;

extern crate cuda;
extern crate cudnn;
extern crate nn;

use cuda::memory;
use nn::custom;

fn main() {
    let mut x = memory::Memory::new(16).unwrap();
    {
        let mut rng = rand::thread_rng();
        for x in x.iter_mut() {
            *x = rng.gen::<f32>();
        }
    }

    {
        let mut host = vec![0.; x.len()];
        memory::memcpy(&mut host, &x).unwrap();
        println!("{:?}", &host);
    }

    custom::relu_forward_inplace(&mut x).unwrap();

    {
        let mut host = vec![0.; x.len()];
        memory::memcpy(&mut host, &x).unwrap();
        println!("{:?}", &host);
    }
}
