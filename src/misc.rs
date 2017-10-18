use std::cmp;

use cuda::misc;

fn ceil_div(x: usize, y: usize) -> usize {
    x / y + if x % y > 0 { 1 } else { 0 }
}

pub fn get_grid_block(len: usize) -> (misc::Dim3, misc::Dim3) {
    const GRID_MAX: usize = 65536;
    const BLOCK_MAX: usize = 512;

    (misc::Dim3 {
         x: cmp::min(ceil_div(len, BLOCK_MAX), GRID_MAX),
         y: 1,
         z: 1,
     },
     misc::Dim3 {
         x: cmp::min(len, BLOCK_MAX),
         y: 1,
         z: 1,
     })
}
