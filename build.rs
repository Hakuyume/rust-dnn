use std::env;
use std::path;
use std::process;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    process::Command::new("nvcc")
        .args(&["-c", "src/custom_kernel.cu", "-Xcompiler", "-fPIC", "-o"])
        .arg(&format!("{}/custom_kernel.o", out_dir))
        .status()
        .unwrap();
    process::Command::new("ar")
        .args(&["crus", "libcustom_kernel.a", "custom_kernel.o"])
        .current_dir(&path::Path::new(&out_dir))
        .status()
        .unwrap();

    println!("cargo:rustc-link-search={}", out_dir);
}
