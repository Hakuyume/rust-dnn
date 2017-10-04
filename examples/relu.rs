extern crate rand;

use rand::Rng;
use rand::distributions::IndependentSample;

extern crate cuda;
extern crate cudnn;
extern crate nn;

use cuda::memory;
use nn::custom;

fn main() {
    let mut x = memory::Memory::new(16).unwrap();
    {
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Range::new(-1., 1.);
        let host: Vec<f32> = (0..x.len()).map(|_| dist.ind_sample(&mut rng)).collect();
        memory::memcpy(&mut x, &host).unwrap();
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
