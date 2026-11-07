mod bytes;
pub use bytes::f16_value;
pub mod clouds;
pub mod sky;

use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    Error, ErrorCode, Interest, Job, JobCx, JobKind, Jobs, RecordKey, Records, URGENT_JOBS,
    job_priority,
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

enum Part {
    Asked(Interest<RecordKey>),
    Arrived,
    Failed { _held: Interest<RecordKey> },
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
        let kind = self.kind;
        self.parts = wanted
            .iter()
            .flatten()
            .enumerate()
            .map(|(place, identity)| {
                let priority = job_priority(URGENT_JOBS, place as u32);
                Part::Asked(jobs.fetch(kind, identity, priority))
            })
            .collect();
        self.assembling = None;
        self.wanted = wanted;
    }

    fn update(&mut self, records: &Records, next: &mut u64, failures: &mut Failures) {
        let name = self.name;
        for part in &mut self.parts {
            let output = match &*part {
                Part::Asked(interest) => records.get(interest),
                Part::Arrived | Part::Failed { .. } => None,
            };
            let Some(output) = output else {
                continue;
            };
            let taken = output
                .map_err(|error| format!("{name}: {error}"))
                .and_then(|bytes| {
                    T::take(&mut self.assembling, &bytes)
                        .map_err(|error| format!("{name}: its output does not decode: {error}"))
                });
            let asked = std::mem::replace(part, Part::Arrived);
            if let (Err(error), Part::Asked(interest)) = (taken, asked) {
                failures.record(error);
                *part = Part::Failed { _held: interest };
            }
        }
        if !self.parts.is_empty()
            && self.parts.iter().all(|part| matches!(part, Part::Arrived))
            && let Some(table) = self.assembling.take()
        {
            *next += 1;
            self.ready = Some(StaticTable {
                id: *next,
                table: Arc::new(table),
            });
        }
    }

    fn count(&self, state: fn(&Part) -> bool) -> u64 {
        self.parts.iter().filter(|part| state(part)).count() as u64
    }

    fn left(&self) -> u64 {
        self.count(|part| matches!(part, Part::Asked(_)))
    }

    fn row(&self) -> StaticTableRow {
        StaticTableRow {
            kind: self.name.to_owned(),
            parts: self.parts.len() as u64,
            arrived: self.count(|part| matches!(part, Part::Arrived)),
            showing: self.ready.is_some(),
        }
    }
}

#[derive(Resource)]
pub(crate) struct StaticTables {
    cloud_noise: TableState<CloudNoise>,
    sky: TableState<SkyTables>,
    wants: Option<Wants>,
    made: u64,
    failures: Failures,
}

impl Default for StaticTables {
    fn default() -> Self {
        Self {
            cloud_noise: TableState::new(&CLOUD_TABLES, CLOUD_NOISE_NAME),
            sky: TableState::new(&SKY_TABLES, SKY_TABLES_NAME),
            wants: None,
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
    if tables.wants != Some(wants) {
        tables.wants = Some(wants);
        let [cloud_noise, sky] = wants.identities();
        tables.cloud_noise.want(cloud_noise, &mut jobs);
        tables.sky.want(sky, &mut jobs);
    }
    let (made, failures) = (&mut tables.made, &mut tables.failures);
    tables.cloud_noise.update(&records, made, failures);
    tables.sky.update(&records, made, failures);
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
