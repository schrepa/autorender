//! Native development timing for the transform/extraction pipeline.
//!
//! Ignored by default. Run with:
//! `cargo test -p stress-lab --release --features profiling --test pipeline_timing -- --ignored --nocapture`
//!
//! Native x86 code, SIMD backend and memory behavior differ from browser Wasm.
//! Use this to compare local revisions quickly; performance claims still come
//! from matched browser workloads.
#![cfg(feature = "profiling")]

use std::{collections::BTreeMap, sync::OnceLock, time::Instant};

use engine::prelude::*;
use stress_lab::{StressConfig, create_workload};

fn now_ms() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// The user's 90,000-object, all-moving capture configuration and camera.
/// `PIPELINE_RENDERABLES` scales the object count, e.g. to compare per-node
/// cost with a cache-resident working set.
fn workload() -> Engine {
    let renderables = std::env::var("PIPELINE_RENDERABLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(90_000);
    let mut engine = create_workload(StressConfig {
        renderables,
        moving_percent: 100,
        ..StressConfig::default()
    })
    .expect("stress workload");
    engine.app.insert_resource(Camera {
        position: Vec3::new(331.2, 446.4, 576.0),
        target: Vec3::new(330.786_25, 445.842_3, 575.280_4),
        vertical_fov_radians: 0.959_931_1,
        near: 0.1,
        far: 5000.0,
    });
    engine
}

#[test]
#[ignore = "native timing aid; run explicitly with --ignored --nocapture"]
fn pipeline_stage_timing() {
    engine::profiling::set_clock(now_ms);
    let mut engine = workload();
    let aspect = 1812.0 / 843.0;
    let frames = 24;
    let rounds = 5;
    // PIPELINE_TICKS fixed ticks per displayed frame (default 2, as in the 29
    // FPS capture); PIPELINE_GPU=1 gathers GPU presentation inputs instead of
    // preparing CPU presented poses.
    let ticks: u32 = std::env::var("PIPELINE_TICKS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2);
    let gpu = std::env::var("PIPELINE_GPU").is_ok_and(|value| value == "1");
    let mut best: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut counters: BTreeMap<&'static str, f64> = BTreeMap::new();
    for round in 0..=rounds {
        engine::profiling::start(65_536).expect("capture starts");
        for frame in 0..frames {
            engine::profiling::set_frame((round * frames + frame) as u64);
            engine.advance(f64::from(ticks) / 60.0);
            let alpha = 0.25 + 0.5 * (frame % 2) as f32;
            if gpu {
                engine
                    .extract_gpu_frame(aspect, alpha)
                    .expect("valid presentation");
            } else {
                engine
                    .extract_frame(aspect, PresentationMode::Interpolated { alpha })
                    .expect("valid presentation");
            }
        }
        let capture = engine::profiling::finish();
        assert!(capture.complete, "bounded capture completes");
        if round == 0 {
            continue; // warm-up
        }
        let mut totals: BTreeMap<&'static str, f64> = BTreeMap::new();
        for event in &capture.events {
            match event.kind {
                "span" => *totals.entry(event.name).or_default() += event.duration_ms,
                "counter" => {
                    counters.insert(event.name, event.value.unwrap_or_default());
                }
                _ => {}
            }
        }
        for (name, total) in totals {
            let per_frame = total / frames as f64;
            best.entry(name)
                .and_modify(|value| *value = value.min(per_frame))
                .or_insert(per_frame);
        }
    }
    println!(
        "minimum over {rounds} rounds of per-frame span totals (ms, {ticks} ticks/frame, gpu {gpu}):"
    );
    for (name, value) in &best {
        println!("  {name:<40} {value:>8.3}");
    }
    for name in [
        "extraction.culling_interior",
        "extraction.culling_tests",
        "hierarchy.evaluated_nodes",
        "interpolation.evaluated_nodes",
        "interpolation.blend_blocks",
    ] {
        if let Some(value) = counters.get(name) {
            println!("  last {name:<35} {value:>8.0}");
        }
    }
}
