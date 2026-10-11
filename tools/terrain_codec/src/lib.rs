//! Standalone acceptance fixtures. The game never links this crate.

use std::sync::Arc;

use engine_terrain::codec;
use serde::Deserialize;
use serde_json::{Value, json};

mod terrain;

#[derive(Deserialize)]
struct Request {
    op: String,
    #[serde(default)]
    container: String,
    #[serde(default)]
    bits: Vec<u32>,
    #[serde(default)]
    bytes: Vec<u8>,
}

fn execute(request: Request) -> Result<Value, String> {
    match request.op.as_str() {
        "production" => terrain::run(),
        #[cfg(not(target_arch = "wasm32"))]
        "profile" => {
            let floats: Vec<_> = request.bits.iter().copied().map(f32::from_bits).collect();
            let mut encoded = Vec::new();
            let iterations = 8;
            let started = std::time::Instant::now();
            for _ in 0..iterations {
                encoded = codec::encode(std::hint::black_box(&floats));
            }
            let encode_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = std::time::Instant::now();
            for _ in 0..iterations {
                let decoded: Vec<f32> = codec::decode(std::hint::black_box(&encoded))
                    .map_err(|error| error.to_string())?;
                std::hint::black_box(decoded);
            }
            let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
            Ok(
                json!({"iterations": iterations, "encode_ms": encode_ms, "decode_ms": decode_ms,
                "bytes": encoded.len(), "words": floats.len(), "profile": "dev opt-level=1; local CPU context only"}),
            )
        }
        "encode" => {
            let floats: Vec<_> = request.bits.iter().copied().map(f32::from_bits).collect();
            let bytes = match request.container.as_str() {
                "box" => codec::encode(&floats.into_boxed_slice()),
                "arc" => codec::encode(&Arc::<[f32]>::from(floats)),
                "vec" => codec::encode(&floats),
                _ => return Err("unknown container".into()),
            };
            Ok(json!({"bytes": bytes}))
        }
        "decode" => {
            let floats = match request.container.as_str() {
                "box" => codec::decode::<Box<[f32]>>(&request.bytes).map(Vec::from),
                "arc" => codec::decode::<Arc<[f32]>>(&request.bytes).map(|values| values.to_vec()),
                "vec" => codec::decode::<Vec<f32>>(&request.bytes),
                _ => return Err("unknown container".into()),
            }
            .map_err(|error| error.to_string())?;
            Ok(json!({"bits": floats.iter().map(|value| value.to_bits()).collect::<Vec<_>>()}))
        }
        _ => Err("unknown operation".into()),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen)]
pub fn run(input: &str) -> String {
    let result = serde_json::from_str::<Request>(input)
        .map_err(|error| error.to_string())
        .and_then(execute);
    match result {
        Ok(value) => value.to_string(),
        Err(error) => json!({"error": error}).to_string(),
    }
}
