use std::path::Path;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_core::Environment;
use engine_core::table_data::{TableKind, TableWorker, clouds, f16_value, sky};
use serde_json::json;

use super::{Observations, hash};

pub const JOBS: &str = "target/pure/tables-jobs.json";
const SLICE: u64 = 4096;

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let environment = Environment::default();
    let mut worker = Worker::default();

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
    let passing = sky::decode(&tables)?
        .transmittance
        .iter()
        .all(|&bits| (0.0..=1.0).contains(&f16_value(bits)));
    observed.check(passing, || {
        "the transmittance table holds shares beyond 0 to 1".into()
    });

    let mut noise = clouds::CloudNoise::empty();
    for part in clouds::parts().iter().filter(|part| part.first == 0) {
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

fn slab(volume: clouds::Volume) -> usize {
    let side = volume.size() as usize;
    side * side * volume.layers() as usize
}

#[derive(Default)]
struct Worker {
    worker: TableWorker,
    next: u64,
    log: Vec<serde_json::Value>,
}

impl Worker {
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
