mod bytes;
pub use bytes::f16_value;
pub mod clouds;
pub mod sky;

use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    Error, ErrorCode, Job, JobCx, JobKind, Jobs, Records, Ticket, URGENT_JOBS, job_priority,
};

use crate::SOURCE;
use crate::settings::{Clouds, Sky};
use crate::views::{StaticTableRow, StaticTablesView};
use clouds::{CloudNoise, NoisePart};
use sky::{Air, SkyTables};

const SKY_TABLES_KIND: &str = "sky::sky_tables";
const CLOUD_NOISE_KIND: &str = "sky::cloud_noise";

pub static SKY_TABLES: JobKind = JobKind::new(SKY_TABLES_KIND, &SOURCE, start_sky_tables);
pub static CLOUD_TABLES: JobKind = JobKind::new(CLOUD_NOISE_KIND, &SOURCE, start_cloud_noise);

const SKY_TABLES_NAME: &str = "sky_tables";
const CLOUD_NOISE_NAME: &str = "cloud_noise";
const RETRY_AFTER: u64 = 120;

#[derive(Clone, Debug)]
pub(crate) struct StaticTable<T> {
    pub(crate) id: u64,
    pub(crate) table: Arc<T>,
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
        *assembling = Some(sky::decode(record)?);
        Ok(())
    }
}

#[derive(Default)]
struct Failures {
    count: u64,
    last: Option<String>,
}

impl Failures {
    fn record(&mut self, error: String) {
        self.count += 1;
        self.last = Some(error);
    }
}

struct Part {
    identity: Vec<u8>,
    ticket: Option<Ticket>,
    arrived: bool,
    failed: Option<u64>,
}

struct TableState<T> {
    kind: &'static JobKind,
    name: &'static str,
    wanted: Option<Vec<Vec<u8>>>,
    parts: Vec<Part>,
    assembling: Option<T>,
    ready: Option<StaticTable<T>>,
}

impl<T: Assembly> TableState<T> {
    fn new(kind: &'static JobKind, name: &'static str) -> Self {
        Self {
            kind,
            name,
            wanted: None,
            parts: Vec::new(),
            assembling: None,
            ready: None,
        }
    }

    fn want(&mut self, wanted: Option<Vec<Vec<u8>>>, jobs: &mut Jobs<'_, '_>) {
        if wanted == self.wanted {
            return;
        }
        for part in &mut self.parts {
            if let Some(ticket) = part.ticket.take() {
                jobs.cancel(ticket);
            }
        }
        self.parts = wanted
            .iter()
            .flatten()
            .map(|identity| Part {
                identity: identity.clone(),
                ticket: None,
                arrived: false,
                failed: None,
            })
            .collect();
        self.assembling = None;
        self.wanted = wanted;
    }

    fn update(
        &mut self,
        (jobs, records): (&mut Jobs<'_, '_>, &Records),
        (frame, next): (u64, &mut u64),
        failures: &mut Failures,
    ) {
        if self.parts.is_empty() || self.parts.iter().all(|part| part.arrived) {
            return;
        }
        let name = self.name;
        for (place, part) in self.parts.iter_mut().enumerate() {
            if part.arrived {
                continue;
            }
            if part.ticket.is_none() {
                let waiting = part.failed.is_some_and(|at| frame < at + RETRY_AFTER);
                if !waiting {
                    let priority = job_priority(URGENT_JOBS, place as u32);
                    part.ticket = Some(jobs.fetch(self.kind, &part.identity, priority));
                }
                continue;
            }
            let Some(output) = part.ticket.as_ref().and_then(|ticket| records.get(ticket)) else {
                continue;
            };
            if let Some(ticket) = part.ticket.take() {
                jobs.cancel(ticket);
            }
            let taken = output
                .map_err(|error| format!("{name}: {error}"))
                .and_then(|bytes| {
                    T::take(&mut self.assembling, &bytes)
                        .map_err(|error| format!("{name}: its output does not decode: {error}"))
                });
            match taken {
                Ok(()) => part.arrived = true,
                Err(error) => {
                    failures.record(error);
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

    fn row(&self) -> StaticTableRow {
        let parts = self.parts.len() as u64;
        StaticTableRow {
            kind: self.name.to_owned(),
            parts,
            arrived: parts - self.left(),
            showing: self.ready.is_some(),
        }
    }
}

#[derive(Resource)]
pub(crate) struct StaticTables {
    cloud_noise: TableState<CloudNoise>,
    sky: TableState<SkyTables>,
    wants: Option<Wants>,
    frame: u64,
    made: u64,
    failures: Failures,
}

impl Default for StaticTables {
    fn default() -> Self {
        Self {
            cloud_noise: TableState::new(&CLOUD_TABLES, CLOUD_NOISE_NAME),
            sky: TableState::new(&SKY_TABLES, SKY_TABLES_NAME),
            wants: None,
            frame: 0,
            made: 0,
            failures: Failures::default(),
        }
    }
}

impl StaticTables {
    pub(crate) fn cloud_noise(&self) -> Option<&StaticTable<CloudNoise>> {
        self.cloud_noise.ready.as_ref()
    }

    pub(crate) fn sky(&self) -> Option<&StaticTable<SkyTables>> {
        self.sky.ready.as_ref()
    }

    fn left(&self) -> u64 {
        self.cloud_noise.left() + self.sky.left()
    }

    fn view(&self) -> StaticTablesView {
        StaticTablesView {
            loading: u32::try_from(self.left()).unwrap_or(u32::MAX),
            tables: vec![self.cloud_noise.row(), self.sky.row()],
            failed_jobs: self.failures.count,
            latest_job_failure: self.failures.last.clone(),
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
            clouds: sky.enabled && clouds.enabled && clouds.coverage > 0.0,
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
            self.clouds
                .then(|| clouds::parts().iter().map(clouds::Part::identity).collect()),
            self.sky.map(|air| vec![air.identity()]),
        ]
    }
}

pub(crate) fn ask_tables(
    sky: Res<Sky>,
    clouds: Res<Clouds>,
    records: Res<Records>,
    mut tables: ResMut<StaticTables>,
    mut jobs: Jobs,
) {
    let wants = Wants::of(*clouds, *sky);
    let tables = &mut *tables;
    tables.frame += 1;
    if tables.wants != Some(wants) {
        tables.wants = Some(wants);
        let [cloud_noise, sky] = wants.identities();
        tables.cloud_noise.want(cloud_noise, &mut jobs);
        tables.sky.want(sky, &mut jobs);
    }
    let frame = tables.frame;
    let (made, failures) = (&mut tables.made, &mut tables.failures);
    tables
        .cloud_noise
        .update((&mut jobs, &records), (frame, &mut *made), &mut *failures);
    tables
        .sky
        .update((&mut jobs, &records), (frame, &mut *made), &mut *failures);
}

pub(crate) fn table_stats(tables: Res<StaticTables>, mut stats: ResMut<StaticTablesView>) {
    stats.set_if_neq(tables.view());
}

fn table_error(kind: &'static str, error: String) -> Error {
    Error::new(ErrorCode::InvalidRecord, format!("$.jobs.{kind}")).got(error)
}

struct SkyTablesJob {
    input: Vec<u8>,
}

impl Job for SkyTablesJob {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        sky::record(&self.input)
            .map(Some)
            .map_err(|error| table_error(SKY_TABLES_KIND, error))
    }
}

fn start_sky_tables(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Ok(Box::new(SkyTablesJob {
        input: input.to_vec(),
    }))
}

struct CloudNoiseJob {
    part: Box<NoisePart>,
}

impl Job for CloudNoiseJob {
    fn step(&mut self, budget: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let mut budget = budget.max(1);
        Ok(self.part.advance(&mut budget))
    }
}

fn start_cloud_noise(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let part = NoisePart::new(input).map_err(|error| table_error(CLOUD_NOISE_KIND, error))?;
    Ok(Box::new(CloudNoiseJob {
        part: Box::new(part),
    }))
}
