//! The tables corpus: the renderer's static tables (`engine_tables`) for
//! the default settings. Falling water's map, the physically based sky's
//! tables for its default (Earth's) air, and a slab of each of the cloud
//! noise's volumes. Each job runs as hosts run it in place and as workers
//! run it, a slice at a time, which must be the same bits, and each record
//! must read back. Every job and its output's hash go into a file `cargo
//! xtask pure --wasm` replays in the browser worker's Wasm module.

use std::path::Path;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_core::Environment;
use engine_core::table_data::{TableKind, TableWorker, clouds, f16_value, falls, sky};
use serde_json::json;

use super::{Observations, hash};

/// Every job the corpus ran, each output by its hash: what `cargo xtask
/// pure --wasm` replays.
pub const JOBS: &str = "target/pure/tables-jobs.json";
/// Work units each slice of a job gets: few enough that a slab of noise
/// takes many slices, as workers slice them.
const SLICE: u64 = 4096;

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let environment = Environment::default();
    let mut worker = Worker::default();

    let map = worker.both(TableKind::FallMap, &[], observed)?;
    observed.hash("fall map", &map);
    let levels = falls::decode(&map)?.levels.len();
    observed.check(levels == falls::LEVELS as usize, || {
        format!("the fall map holds {levels} levels")
    });

    let settings = environment.sky;
    let air = sky::Air {
        rayleigh: settings.rayleigh,
        rayleigh_height: settings.rayleigh_height,
        mie: settings.mie,
        mie_absorption: settings.mie_absorption,
        mie_height: settings.mie_height,
        ozone: settings.ozone,
        ground: settings.ground_albedo,
        bottom: settings.planet_radius,
        top: settings.planet_radius + settings.atmosphere_height,
    };
    let tables = worker.both(TableKind::SkyTables, &air.identity(), observed)?;
    observed.hash("sky tables", &tables);
    // Light passes the air only partly, all of it at most.
    let passing = sky::decode(&tables)?
        .transmittance
        .iter()
        .all(|&bits| (0.0..=1.0).contains(&f16_value(bits)));
    observed.check(passing, || {
        "the transmittance table holds shares beyond 0 to 1".into()
    });

    let mut noise = clouds::CloudNoise::empty();
    for part in clouds::parts().iter().filter(|part| part.first == 0) {
        // The detail slab is the shape's code at another size: once whole.
        let record = if part.volume == clouds::Volume::Shape {
            worker.both(TableKind::CloudNoise, &part.identity(), observed)?
        } else {
            worker.run(TableKind::CloudNoise, &part.identity())?
        };
        observed.hash(format!("cloud noise {:?} slab", part.volume), &record);
        noise.insert(&record)?;
    }
    observed.hash(
        "cloud noise Shape slab texels",
        &noise.shape[..slab(clouds::Volume::Shape)],
    );
    observed.hash(
        "cloud noise Detail slab texels",
        &noise.detail[..slab(clouds::Volume::Detail)],
    );
    worker.write(root)
}

/// Texels of a slab of `volume`.
fn slab(volume: clouds::Volume) -> usize {
    let side = volume.size() as usize;
    side * side * volume.layers() as usize
}

/// Runs jobs as hosts and workers run them, recording every job (with its
/// output's hash) for the replay in Wasm.
#[derive(Default)]
struct Worker {
    worker: TableWorker,
    next: u64,
    log: Vec<serde_json::Value>,
}

impl Worker {
    /// Job `kind` on `input` run in place and a slice at a time, which must
    /// be the same bits: its record.
    fn both(
        &mut self,
        kind: TableKind,
        input: &[u8],
        observed: &mut Observations,
    ) -> Result<Vec<u8>, String> {
        let whole = self.run(kind, input)?;
        let sliced = self.sliced(kind, input)?;
        observed.check(sliced == whole, || {
            format!(
                "{}: the job run a slice at a time differs from the job run in place",
                kind.name()
            )
        });
        Ok(whole)
    }

    /// Job `kind` on `input`, run in place as hosts run it.
    fn run(&mut self, kind: TableKind, input: &[u8]) -> Result<Vec<u8>, String> {
        let output = engine_core::table_data::run(kind.name(), input)
            .ok_or("a table job's kind names no table")??;
        self.log.push(json!({
            "job": kind.name(),
            "input": STANDARD.encode(input),
            "output": hash(&output),
        }));
        Ok(output)
    }

    /// Job `kind` on `input`, run a slice at a time as workers run it.
    fn sliced(&mut self, kind: TableKind, input: &[u8]) -> Result<Vec<u8>, String> {
        self.next += 1;
        self.worker.start(self.next, kind.name(), input)?;
        loop {
            if let Some(output) = self.worker.step(self.next, SLICE)? {
                self.log.push(json!({
                    "job": kind.name(),
                    "input": STANDARD.encode(input),
                    "slice": SLICE,
                    "output": hash(&output),
                }));
                return Ok(output);
            }
        }
    }

    /// Writes what it ran to [`JOBS`].
    fn write(&self, root: &Path) -> Result<(), String> {
        let path = root.join(JOBS);
        if let Some(folder) = path.parent() {
            std::fs::create_dir_all(folder)
                .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
        }
        std::fs::write(
            &path,
            serde_json::Value::Array(self.log.clone()).to_string(),
        )
        .map_err(|error| format!("cannot write {JOBS}: {error}"))
    }
}
