use std::path::Path;

use engine_core::JobKind;
use engine_sky::Sky;
use engine_sky::tables::{CLOUD_TABLES, SKY_TABLES, clouds, f16_value, sky};

use super::{JobLog, Observations};

pub const JOBS: &str = "target/pure/tables-jobs.json";
const SLICE: u64 = 4096;

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let mut worker = JobLog::new()?;

    let settings = Sky::default();
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
    let tables = both(&mut worker, &SKY_TABLES, &air.identity(), observed)?;
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
            both(&mut worker, &CLOUD_TABLES, &part.identity(), observed)?
        } else {
            worker.run(&CLOUD_TABLES, &part.identity())?
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
    worker.write(root, JOBS)
}

fn slab(volume: clouds::Volume) -> usize {
    let side = volume.size() as usize;
    side * side * volume.layers() as usize
}

fn both(
    worker: &mut JobLog,
    kind: &'static JobKind,
    input: &[u8],
    observed: &mut Observations,
) -> Result<Vec<u8>, String> {
    let whole = worker.run(kind, input)?;
    let sliced = worker.sliced(kind, input, SLICE)?;
    observed.check(sliced == whole, || {
        format!(
            "{}: the job run a slice at a time differs from the job run in place",
            kind.name()
        )
    });
    Ok(whole)
}
