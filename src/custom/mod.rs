use libc::c_float;

use cuda::misc;

pub enum Type {
    Float,
}

pub trait Scalar {
    const TYPE: Type;
}

impl Scalar for c_float {
    const TYPE: Type = Type::Float;
}

fn calc_grid_block(len: usize) -> (misc::Dim3, misc::Dim3) {
    const GRID_MAX: usize = 65536;
    const BLOCK_MAX: usize = 512;

    let (grid, block) = {
        if len > GRID_MAX * BLOCK_MAX {
            (GRID_MAX, BLOCK_MAX)
        } else {
            if len > BLOCK_MAX {
                if len % BLOCK_MAX > 0 {
                    (len / BLOCK_MAX + 1, BLOCK_MAX)
                } else {
                    (len / BLOCK_MAX, BLOCK_MAX)
                }
            } else {
                (1, len)
            }
        }
    };

    (misc::Dim3 {
         x: grid,
         y: 1,
         z: 1,
     },
     misc::Dim3 {
         x: block,
         y: 1,
         z: 1,
     })
}

mod functions;
pub use self::functions::*;
