use std::fs::File;
use std::io::Write;
use rand::prelude::*;

static FILEPATH: &str = "../../data/Rust.txt";

fn main() -> std::io::Result<()> {
    let mut rng = rand::thread_rng();
    let mut f = File::create(FILEPATH)?;

    for _ in 0..1000000 {
        writeln!(f, "{}", rng.gen::<f64>())?;
    }

    Ok(())
}
