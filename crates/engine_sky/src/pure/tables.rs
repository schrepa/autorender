use engine_core::{JobKind, PureSession};

use crate::Sky;
use crate::tables::{CLOUD_TABLES, SKY_TABLES, clouds, f16_value, sky};

const SLICE: u64 = 4096;

pub fn run(session: &mut dyn PureSession) -> Result<(), String> {
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
    let tables = both(session, &SKY_TABLES, &air.identity())?;
    session.hash("sky tables", &tables);
    let passing = sky::decode(&tables)?
        .transmittance
        .iter()
        .all(|&bits| (0.0..=1.0).contains(&f16_value(bits)));
    session.check(passing, &|| {
        "the transmittance table holds shares beyond 0 to 1".into()
    });

    let mut noise = clouds::CloudNoise::empty();
    for part in clouds::parts().iter().filter(|part| part.first == 0) {
        let record = if part.volume == clouds::Volume::Shape {
            both(session, &CLOUD_TABLES, &part.identity())?
        } else {
            session.run(&CLOUD_TABLES, &part.identity())?
        };
        session.hash(&format!("cloud noise {:?} slab", part.volume), &record);
        noise.insert(&record)?;
    }
    session.hash(
        "cloud noise Shape slab texels",
        &noise.shape[..slab(clouds::Volume::Shape)],
    );
    session.hash(
        "cloud noise Detail slab texels",
        &noise.detail[..slab(clouds::Volume::Detail)],
    );
    Ok(())
}

fn slab(volume: clouds::Volume) -> usize {
    let side = volume.size() as usize;
    side * side * volume.layers() as usize
}

fn both(
    session: &mut dyn PureSession,
    kind: &'static JobKind,
    input: &[u8],
) -> Result<Vec<u8>, String> {
    let whole = session.run(kind, input)?;
    let sliced = session.sliced(kind, input, SLICE)?;
    session.check(sliced == whole, &|| {
        format!(
            "{}: the job run a slice at a time differs from the job run in place",
            kind.name()
        )
    });
    Ok(whole)
}
