extern crate nvcc;

fn main() {
    nvcc::compile_library("libcustom_kernel.a", &["src/custom/kernel.cu"]);
}
