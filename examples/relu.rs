extern crate cuda;
extern crate cudnn;
extern crate nn;
extern crate rand;

use cuda::memory;
use cuda::slice;
use nn::custom;

use rand::distributions::IndependentSample;

fn random(len: usize) -> memory::Memory<f32> {
    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Range::new(-1., 1.);
    let host: Vec<_> = (0..len).map(|_| dist.ind_sample(&mut rng)).collect();
    let mut device = memory::Memory::new(len).unwrap();
    memory::memcpy(&mut device, &host).unwrap();
    device
}

fn dump(mem: &slice::Slice<f32>) {
    let mut host = vec![0.; mem.len()];
    memory::memcpy(&mut host, &mem).unwrap();
    println!("{:?}", &host);
}

fn main() {
    let mut x = random(16);
    dump(&x);

    custom::relu_forward_inplace(&mut x).unwrap();
    dump(&x);

    let mut dy = random(x.len());
    dump(&dy);

    custom::relu_backward_inplace(&x, &mut dy).unwrap();
    dump(&dy);
}
