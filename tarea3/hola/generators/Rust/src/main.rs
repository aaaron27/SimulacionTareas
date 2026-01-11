use std::fs::File;
use std::io::Write;
use rand::prelude::*;

static FILEPATH: &str = "../../data/Rust.txt";

fn main() -> std::io::Result<()> {
    let mut rng = rand::thread_rng();
    let mut f = File::create(FILEPATH)?;
    let n = 1000000;

    for i in 0..n {
        if i < n-1 {
            writeln!(f, "{}", rng.gen::<f64>())?;
        } else {
            write!(f, "{}", rng.gen::<f64>())?;
        }
    }

    Ok(())
}
