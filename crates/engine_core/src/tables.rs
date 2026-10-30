use std::sync::Arc;

use bevy_ecs::prelude::*;

use engine_tables::TableKind;
use engine_tables::clouds::CloudNoise;
use engine_tables::sky::{Air, SkyTables};

use crate::{
    Clouds, Cooking, JobBudget, JobFailures, JobId, Jobs, RecordKey, Sky, URGENT_JOBS, job_priority,
    section_in,
};

const RETRY_AFTER: u64 = 120;

#[derive(Clone, Debug)]
pub struct StaticTable<T> {
    pub id: u64,
    pub table: Arc<T>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderTables {
    pub cloud_noise: Option<StaticTable<CloudNoise>>,
    pub sky: Option<StaticTable<SkyTables>>,
}

trait Assembly: Sized {
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

struct Part {
    identity: Vec<u8>,
    job: Option<JobId>,
    arrived: bool,
    failed: Option<u64>,
}

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

    fn left(&self) -> u64 {
        self.parts.iter().filter(|part| !part.arrived).count() as u64
    }

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

#[derive(Resource)]
pub struct StaticTables {
    cloud_noise: TableState<CloudNoise>,
    sky: TableState<SkyTables>,
    wants: Option<Wants>,
    frame: u64,
    made: u64,
    failures: JobFailures,
}

impl Default for StaticTables {
    fn default() -> Self {
        Self {
            cloud_noise: TableState::new(TableKind::CloudNoise),
            sky: TableState::new(TableKind::SkyTables),
            wants: None,
            frame: 0,
            made: 0,
            failures: JobFailures::default(),
        }
    }
}

impl StaticTables {
    pub fn loading(&self) -> bool {
        self.left() > 0
    }

    fn left(&self) -> u64 {
        self.cloud_noise.left() + self.sky.left()
    }

    pub fn tables(&self) -> [(&'static str, u64, u64, bool); 2] {
        [self.cloud_noise.row(), self.sky.row()]
    }

    pub fn job_failures(&self) -> (u64, Option<&str>) {
        let last = self.failures.last.as_ref();
        (self.failures.count, last.map(|(_, error)| error.as_str()))
    }

    pub fn render(&self) -> RenderTables {
        RenderTables {
            cloud_noise: self.cloud_noise.ready.clone(),
            sky: self.sky.ready.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Wants {
    clouds: bool,
    sky: Option<Air>,
}

impl Wants {
    fn of(clouds: Clouds, sky: Sky) -> Self {
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
        }
    }

    fn identities(&self) -> [Option<Vec<Vec<u8>>>; 2] {
        [
            self.clouds.then(|| {
                engine_tables::clouds::parts()
                    .iter()
                    .map(engine_tables::clouds::Part::identity)
                    .collect()
            }),
            self.sky.map(|air| vec![air.identity()]),
        ]
    }
}

pub(crate) fn update_tables(world: &mut World) {
    if !world.get_resource::<Jobs>().is_some_and(Jobs::hosted) {
        return;
    }
    let wants = Wants::of(section_in::<Clouds>(world), section_in::<Sky>(world));
    world.resource_scope(|world, mut tables: Mut<StaticTables>| {
        let tables = tables.as_mut();
        if let Some(mut jobs) = world.get_resource_mut::<Jobs>() {
            tables.frame += 1;
            if tables.wants != Some(wants) {
                tables.wants = Some(wants);
                let [cloud_noise, sky] = wants.identities();
                tables.cloud_noise.want(cloud_noise, &mut jobs);
                tables.sky.want(sky, &mut jobs);
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
        }
        if let Some(mut cooking) = world.get_resource_mut::<Cooking>() {
            cooking.left("tables", tables.left());
        }
    });
}
