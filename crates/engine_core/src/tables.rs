//! The renderer's static tables as records ([`engine_tables`]): the noise
//! clouds are shaped from while the scene has clouds, the physically
//! based sky's tables for its air, and falling water's map and the spectra
//! of water's waves while water can draw. Each frame [`StaticTables`] asks
//! for those the scene's settings want, as jobs whose records the host's
//! packs and store hand back, and a worker (or the host, in place) makes
//! where they lack them; frames carry the latest of each table that
//! arrived whole to the renderer, which uploads it ([`RenderTables`]). A
//! table wanted for something new (another air, another wind) leaves the
//! last one showing until the new one arrives. A session that cooks makes
//! every table its scene wants, which then cooks with it. An engine no host
//! serves draws no frames, so asks for none.

use std::sync::Arc;

use bevy_ecs::prelude::*;

use engine_tables::TableKind;
use engine_tables::clouds::CloudNoise;
use engine_tables::falls::FallMap;
use engine_tables::sky::{Air, SkyTables};
use engine_tables::waves::{WaveShape, WaveSpectra};

use crate::{
    Cooking, Environment, JobBudget, JobFailures, JobId, Jobs, RecordKey, URGENT_JOBS, job_priority,
};

/// Frames after a part's job failed before it is asked for again.
const RETRY_AFTER: u64 = 120;

/// A table as frames carry it: a number no other table of the engine has,
/// which renderers tell what they uploaded by, and the table.
#[derive(Clone, Debug)]
pub struct StaticTable<T> {
    pub id: u64,
    pub table: Arc<T>,
}

/// The tables a frame's renderer reads: the latest of each that arrived
/// whole, none before the first.
#[derive(Clone, Debug, Default)]
pub struct RenderTables {
    pub cloud_noise: Option<StaticTable<CloudNoise>>,
    pub sky: Option<StaticTable<SkyTables>>,
    pub fall_map: Option<StaticTable<FallMap>>,
    pub waves: Option<StaticTable<WaveSpectra>>,
}

/// What a table is assembled from: its parts' records, taken as they
/// arrive.
trait Assembly: Sized {
    /// Takes `record`, one of its parts', into `assembling`.
    fn take(assembling: &mut Option<Self>, record: &[u8]) -> Result<(), String>;
}

impl Assembly for CloudNoise {
    fn take(assembling: &mut Option<Self>, record: &[u8]) -> Result<(), String> {
        assembling
            .get_or_insert_with(CloudNoise::empty)
            .insert(record)
    }
}

impl Assembly for SkyTables {
    fn take(assembling: &mut Option<Self>, record: &[u8]) -> Result<(), String> {
        *assembling = Some(engine_tables::sky::decode(record)?);
        Ok(())
    }
}

impl Assembly for FallMap {
    fn take(assembling: &mut Option<Self>, record: &[u8]) -> Result<(), String> {
        *assembling = Some(engine_tables::falls::decode(record)?);
        Ok(())
    }
}

impl Assembly for WaveSpectra {
    fn take(assembling: &mut Option<Self>, record: &[u8]) -> Result<(), String> {
        *assembling = Some(engine_tables::waves::decode(record)?);
        Ok(())
    }
}

/// A part of a table: its job's identity, its job while asked for, and
/// whether its record arrived; when its job last failed, if it did.
struct Part {
    identity: Vec<u8>,
    job: Option<JobId>,
    arrived: bool,
    failed: Option<u64>,
}

/// One table: what it is wanted for (its parts' identities), its parts and
/// what they assembled so far, and the latest that arrived whole.
struct TableState<T> {
    kind: TableKind,
    wanted: Option<Vec<Vec<u8>>>,
    parts: Vec<Part>,
    assembling: Option<T>,
    ready: Option<StaticTable<T>>,
}

impl<T: Assembly> TableState<T> {
    fn new(kind: TableKind) -> Self {
        Self {
            kind,
            wanted: None,
            parts: Vec::new(),
            assembling: None,
            ready: None,
        }
    }

    /// Follows what the scene wants it for: each part's identity, none
    /// while it wants none. A change drops the parts asked for, whose jobs
    /// stop.
    fn want(&mut self, wanted: Option<Vec<Vec<u8>>>, jobs: &mut Jobs) {
        if wanted == self.wanted {
            return;
        }
        for part in &self.parts {
            if let Some(id) = part.job {
                jobs.cancel(id);
            }
        }
        self.parts = wanted
            .iter()
            .flatten()
            .map(|identity| Part {
                identity: identity.clone(),
                job: None,
                arrived: false,
                failed: None,
            })
            .collect();
        self.assembling = None;
        self.wanted = wanted;
    }

    /// Asks for the parts not yet asked for as `slots` admits them, and
    /// takes those that arrived; once all did, the table is the latest,
    /// numbered `next`. What failed counts in `failures`.
    fn update(
        &mut self,
        slots: &mut JobBudget<'_>,
        (frame, next): (u64, &mut u64),
        failures: &mut JobFailures,
    ) {
        if self.parts.is_empty() || self.parts.iter().all(|part| part.arrived) {
            return;
        }
        let name = self.kind.name();
        for (place, part) in self.parts.iter_mut().enumerate() {
            if part.arrived {
                continue;
            }
            let Some(id) = part.job else {
                let waiting = part.failed.is_some_and(|at| frame < at + RETRY_AFTER);
                let priority = job_priority(URGENT_JOBS, place as u32);
                if !waiting && slots.take(priority) {
                    let record = RecordKey::new(&engine_tables::SOURCE, name, &part.identity);
                    part.job = Some(slots.jobs.submit_record(
                        name,
                        part.identity.clone(),
                        priority,
                        record,
                    ));
                }
                continue;
            };
            let Some(output) = slots.jobs.take(id) else {
                continue;
            };
            part.job = None;
            let taken = crate::jobs::job_output(name, output, |record| {
                T::take(&mut self.assembling, record)
            });
            match taken {
                Ok(()) => part.arrived = true,
                Err(error) => {
                    failures.record(frame, error);
                    part.failed = Some(frame);
                }
            }
        }
        if self.parts.iter().all(|part| part.arrived)
            && let Some(table) = self.assembling.take()
        {
            *next += 1;
            self.ready = Some(StaticTable {
                id: *next,
                table: Arc::new(table),
            });
        }
    }

    /// Parts wanted that have yet to arrive.
    fn left(&self) -> u64 {
        self.parts.iter().filter(|part| !part.arrived).count() as u64
    }

    /// Its kind's name, its parts wanted and those that arrived, and
    /// whether one arrived whole.
    fn row(&self) -> (&'static str, u64, u64, bool) {
        let parts = self.parts.len() as u64;
        (
            self.kind.name(),
            parts,
            parts - self.left(),
            self.ready.is_some(),
        )
    }
}

/// The renderer's static tables: each one's records asked for as the
/// scene's settings want them, and the latest of each that arrived whole.
#[derive(Resource)]
pub struct StaticTables {
    cloud_noise: TableState<CloudNoise>,
    sky: TableState<SkyTables>,
    fall_map: TableState<FallMap>,
    waves: TableState<WaveSpectra>,
    // What the scene's settings wanted the tables for at the last frame.
    wants: Option<Wants>,
    frame: u64,
    // The latest table's number.
    made: u64,
    failures: JobFailures,
}

impl Default for StaticTables {
    fn default() -> Self {
        Self {
            cloud_noise: TableState::new(TableKind::CloudNoise),
            sky: TableState::new(TableKind::SkyTables),
            fall_map: TableState::new(TableKind::FallMap),
            waves: TableState::new(TableKind::WaveSpectra),
            wants: None,
            frame: 0,
            made: 0,
            failures: JobFailures::default(),
        }
    }
}

impl StaticTables {
    /// Whether a table the scene wants has yet to arrive: hosts keep
    /// preparing meanwhile.
    pub fn loading(&self) -> bool {
        self.left() > 0
    }

    /// Parts of the tables wanted that have yet to arrive.
    fn left(&self) -> u64 {
        self.cloud_noise.left() + self.sky.left() + self.fall_map.left() + self.waves.left()
    }

    /// Each table: its kind's name, the parts the scene's settings want of
    /// it and those that arrived, and whether one arrived whole (the
    /// latest, which frames carry).
    pub fn tables(&self) -> [(&'static str, u64, u64, bool); 4] {
        [
            self.cloud_noise.row(),
            self.sky.row(),
            self.fall_map.row(),
            self.waves.row(),
        ]
    }

    /// Jobs of its tables that failed, each asked for again a while later,
    /// and the latest one's error.
    pub fn job_failures(&self) -> (u64, Option<&str>) {
        let last = self.failures.last.as_ref();
        (self.failures.count, last.map(|(_, error)| error.as_str()))
    }

    /// The latest of each table, as frames carry them.
    pub fn render(&self) -> RenderTables {
        RenderTables {
            cloud_noise: self.cloud_noise.ready.clone(),
            sky: self.sky.ready.clone(),
            fall_map: self.fall_map.ready.clone(),
            waves: self.waves.ready.clone(),
        }
    }
}

/// What the scene's settings want the tables for: the cloud noise while
/// clouds cover any sky; the sky's tables for the physically based sky's
/// air; and the fall map and the wave spectra for the wind's heading while
/// the sea draws or a terrain's bakes give it water.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Wants {
    clouds: bool,
    sky: Option<Air>,
    water: Option<WaveShape>,
}

impl Wants {
    /// What `environment` wants, `water` saying whether a terrain's bakes
    /// give it water.
    fn of(environment: &Environment, water: bool) -> Self {
        let (clouds, sky) = (environment.clouds, environment.sky);
        Self {
            clouds: clouds.enabled && clouds.coverage > 0.0,
            sky: sky.enabled.then_some(Air {
                rayleigh: sky.rayleigh,
                rayleigh_height: sky.rayleigh_height,
                mie: sky.mie,
                mie_absorption: sky.mie_absorption,
                mie_height: sky.mie_height,
                ozone: sky.ozone,
                ground: sky.ground_albedo,
                bottom: sky.planet_radius,
                top: sky.planet_radius + sky.atmosphere_height,
            }),
            water: (environment.ocean.enabled || (environment.water.enabled && water))
                .then_some(WaveShape::blown(environment.wind.velocity)),
        }
    }

    /// Each table's parts' identities: the cloud noise's, the sky's
    /// tables', the fall map's and the wave spectra's.
    fn identities(&self) -> [Option<Vec<Vec<u8>>>; 4] {
        [
            self.clouds.then(|| {
                engine_tables::clouds::parts()
                    .iter()
                    .map(engine_tables::clouds::Part::identity)
                    .collect()
            }),
            self.sky.map(|air| vec![air.identity()]),
            self.water.map(|_| vec![Vec::new()]),
            self.water.map(|shape| vec![shape.identity()]),
        ]
    }
}

/// Asks for the tables the scene's settings want and takes those that
/// arrived, `water` saying whether a terrain's bakes give it water; while
/// the session cooks, says how many parts are left.
pub(crate) fn update_tables(world: &mut World, water: bool) {
    if !world.get_resource::<Jobs>().is_some_and(Jobs::hosted) {
        return;
    }
    let environment = Environment::current(world).sanitized();
    let wants = Wants::of(&environment, water);
    world.resource_scope(|world, mut tables: Mut<StaticTables>| {
        let tables = tables.as_mut();
        if let Some(mut jobs) = world.get_resource_mut::<Jobs>() {
            tables.frame += 1;
            if tables.wants != Some(wants) {
                tables.wants = Some(wants);
                let [cloud_noise, sky, fall_map, waves] = wants.identities();
                tables.cloud_noise.want(cloud_noise, &mut jobs);
                tables.sky.want(sky, &mut jobs);
                tables.fall_map.want(fall_map, &mut jobs);
                tables.waves.want(waves, &mut jobs);
            }
            let mut slots = JobBudget::of(&mut jobs);
            let frame = tables.frame;
            let (made, failures) = (&mut tables.made, &mut tables.failures);
            tables
                .cloud_noise
                .update(&mut slots, (frame, &mut *made), &mut *failures);
            tables
                .sky
                .update(&mut slots, (frame, &mut *made), &mut *failures);
            tables
                .fall_map
                .update(&mut slots, (frame, &mut *made), &mut *failures);
            tables
                .waves
                .update(&mut slots, (frame, &mut *made), &mut *failures);
        }
        if let Some(mut cooking) = world.get_resource_mut::<Cooking>() {
            cooking.left("tables", tables.left());
        }
    });
}
