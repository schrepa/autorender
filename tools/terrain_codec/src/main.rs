use std::io::{self, BufRead};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for line in io::stdin().lock().lines() {
        println!("{}", engine_terrain_codec::run(&line?));
    }
    Ok(())
}
