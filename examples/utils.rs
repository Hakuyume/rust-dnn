extern crate cuda;
extern crate nn;
extern crate rand;

use std::cell;
use std::default;
use std::fmt;
use std::sync;
use std::time;

use self::cuda::memory;
use self::cuda::slice;
use self::nn::Result;

use self::rand::distributions::IndependentSample;

pub fn dump<T>(device: &slice::Slice<T>) -> Result<()>
    where T: Clone + default::Default + fmt::Debug
{
    let mut host = vec![T::default(); device.len()];
    memory::memcpy(&mut host, &device)?;
    println!("{:?}", &host);
    Ok(())
}

pub fn random(len: usize) -> Result<memory::Memory<f32>> {
    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Range::new(-1., 1.);
    let host: Vec<_> = (0..len).map(|_| dist.ind_sample(&mut rng)).collect();
    let mut device = memory::Memory::new(len)?;
    memory::memcpy(&mut device, &host)?;
    Ok(device)
}

struct MemoryUsage {
    current: usize,
    peak: usize,
}

pub fn bench<T, F: FnOnce() -> Result<T>>(f: F) -> Result<T> {
    let memory_usage = sync::Arc::new(cell::RefCell::new(MemoryUsage {
                                                             current: 0,
                                                             peak: 0,
                                                         }));
    {
        let usage = memory_usage.clone();
        memory::set_malloc_hook(move |_, size| {
                                    let mut usage = usage.borrow_mut();
                                    usage.current += size;
                                    if usage.peak < usage.current {
                                        usage.peak = usage.current;
                                    }
                                });
    }
    {
        let usage = memory_usage.clone();
        memory::set_free_hook(move |_, size| {
                                  let mut usage = usage.borrow_mut();
                                  usage.current -= size;
                              });
    }
    let time = time::SystemTime::now();
    let result = f();
    let elapsed = time.elapsed().unwrap();
    let elapsed = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 * 1e-9;
    println!("elapsed time: {:.2} secs", elapsed);
    println!("memory usage: {:.2} MiB",
             memory_usage.borrow().peak as f64 / (2.0_f64).powi(20));
    result
}
