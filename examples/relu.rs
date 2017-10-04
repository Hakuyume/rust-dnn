extern crate cuda;
extern crate cudnn;
extern crate nn;

use cuda::memory;
use nn::custom;

fn main() {
    let mut x = memory::Memory::new(16).unwrap();
    memory::memcpy(&mut x,
                   &[0.0, 0.1, -0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, 0.9, 1.0, -1.1, 1.2, -1.3,
                     1.4, 1.5])
            .unwrap();

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
