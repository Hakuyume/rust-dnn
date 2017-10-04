extern crate nvcc;

fn main() {
    nvcc::compile_library("libcustom_kernel.a", &["src/custom_kernel.cu"]);
}
