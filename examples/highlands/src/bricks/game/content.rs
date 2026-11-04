use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, Error, ErrorCode, Job, JobCx, JobKind, Jobs, Mesh, MeshHandle, ProvidedAssets,
    RecordExchange, Records, Ticket, URGENT_JOBS, job_priority,
};
use glam::Vec3;

use crate::bricks::game::cave::{CAVE_RADIUS, Cave};
use crate::bricks::game::{HighlandsStats, SOURCE, caves, rocks, trees};

pub(crate) const SEED: u32 = 1337;

const BOULDERS: u32 = 5;

const SPRUCES: u32 = 3;

const CAVE_SEED: u32 = 79;

pub(crate) const CAVE_POINTS: usize = 5;

pub(crate) static BOULDER: JobKind = JobKind::new("highlands::boulder", &SOURCE, start_boulder);

pub(crate) static CONIFER: JobKind = JobKind::new("highlands::conifer", &SOURCE, start_conifer);

pub(crate) static CONIFER_SHADOW: JobKind =
    JobKind::new("highlands::conifer_shadow", &SOURCE, start_conifer_shadow);

pub(crate) static CAVE: JobKind = JobKind::new("highlands::cave", &SOURCE, start_cave);

#[derive(Resource, Default)]
pub(crate) struct Content {
    asked: Vec<(Ticket, MeshHandle, &'static JobKind)>,
    pub(crate) boulders: Vec<MeshHandle>,
    failures: u64,
    latest: Option<String>,
}

impl Content {
    pub(crate) fn ask(
        &mut self,
        (jobs, assets): (&mut Jobs<'_, '_>, &mut AssetLibrary),
        kind: &'static JobKind,
        input: &[u8],
    ) -> Result<MeshHandle, Error> {
        let handle = assets.reserve_mesh()?;
        let ticket = jobs.fetch(kind, input, self.priority());
        self.asked.push((ticket, handle, kind));
        Ok(handle)
    }

    pub(crate) fn priority(&self) -> u32 {
        job_priority(URGENT_JOBS, self.asked.len() as u32)
    }

    pub(crate) fn failed(&mut self, kind: &JobKind, error: &Error) {
        self.failures += 1;
        self.latest = Some(format!("{}: {error}", kind.name()));
    }
}

pub(crate) fn provide_land(
    (assets, records, provided): (&mut AssetLibrary, &mut RecordExchange, &mut ProvidedAssets),
    (content, jobs): (&mut Content, &mut Jobs<'_, '_>),
) -> Result<(), Error> {
    let boulders = (0..BOULDERS)
        .map(|seed| content.ask((&mut *jobs, &mut *assets), &BOULDER, &seed.to_le_bytes()))
        .collect::<Result<Vec<_>, Error>>()?;
    for (seed, &mesh) in boulders.iter().enumerate() {
        provided.provide_mesh(format!("boulder{seed}"), mesh);
    }
    content.boulders = boulders;
    provided.provide_texture("bark_atlas", trees::bark_atlas(records, assets)?);
    for seed in 0..SPRUCES {
        for detail in 0..trees::DETAIL_LEVELS {
            let mut identity = seed.to_le_bytes().to_vec();
            identity.extend_from_slice(&(detail as u32).to_le_bytes());
            let mesh = content.ask((&mut *jobs, &mut *assets), &CONIFER, &identity)?;
            let name = match detail {
                0 => format!("spruce{seed}"),
                _ => format!("spruce{seed}_lod{detail}"),
            };
            provided.provide_mesh(name, mesh);
        }
        let shadow = content.ask(
            (&mut *jobs, &mut *assets),
            &CONIFER_SHADOW,
            &seed.to_le_bytes(),
        )?;
        provided.provide_mesh(format!("spruce{seed}_shadow"), shadow);
    }
    Ok(())
}

pub(crate) fn bank_boulder(
    content: &mut Content,
    (jobs, assets): (&mut Jobs<'_, '_>, &mut AssetLibrary),
) -> Result<MeshHandle, Error> {
    content.ask((jobs, assets), &BOULDER, &(SEED ^ 0x5eed).to_le_bytes())
}

pub(crate) fn cave_input(points: &[Vec3; CAVE_POINTS]) -> Vec<u8> {
    points
        .iter()
        .flat_map(|point| point.to_array())
        .flat_map(f32::to_le_bytes)
        .collect()
}

pub(crate) fn arrive(
    records: Res<Records>,
    mut assets: ResMut<AssetLibrary>,
    mut content: ResMut<Content>,
    mut jobs: Jobs,
) {
    let content = &mut *content;
    let asked = std::mem::take(&mut content.asked);
    for (ticket, handle, kind) in asked {
        let Some(record) = records.get(&ticket) else {
            content.asked.push((ticket, handle, kind));
            continue;
        };
        jobs.cancel(ticket);
        let filled = record
            .and_then(|bytes| Mesh::from_record(&bytes))
            .and_then(|mesh| assets.fill_mesh(handle, mesh));
        if let Err(error) = filled {
            content.failed(kind, &error);
        }
    }
}

pub(crate) fn publish(content: Res<Content>, cave: Res<Cave>, mut stats: ResMut<HighlandsStats>) {
    stats.set_if_neq(HighlandsStats {
        content: (content.asked.len() + usize::from(cave.asking())) as u32,
        failed_jobs: content.failures,
        latest_job_failure: content.latest.clone(),
    });
}

struct Making {
    input: Vec<u8>,
    make: fn(&[u8]) -> Result<Mesh, Error>,
}

impl Job for Making {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        (self.make)(&self.input).map(|mesh| Some(mesh.to_record()))
    }
}

fn making(input: &[u8], make: fn(&[u8]) -> Result<Mesh, Error>) -> Result<Box<dyn Job>, Error> {
    Ok(Box::new(Making {
        input: input.to_vec(),
        make,
    }))
}

fn words<const N: usize>(input: &[u8], kind: &JobKind) -> Result<[u32; N], Error> {
    if input.len() != N * 4 {
        return Err(
            Error::new(ErrorCode::InvalidRecord, format!("$.jobs.{}", kind.name()))
                .range((N * 4) as f64, (N * 4) as f64)
                .got(input.len()),
        );
    }
    let mut words = [0; N];
    for (word, bytes) in words.iter_mut().zip(input.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*bytes);
    }
    Ok(words)
}

fn start_boulder(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    making(input, |input| {
        let [seed] = words(input, &BOULDER)?;
        rocks::boulder(seed)
    })
}

fn start_conifer(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    making(input, |input| {
        let [seed, detail] = words(input, &CONIFER)?;
        let detail = detail as usize;
        if detail >= trees::DETAIL_LEVELS {
            return Err(Error::new(
                ErrorCode::InvalidRecord,
                format!("$.jobs.{}", CONIFER.name()),
            )
            .at_most(trees::DETAIL_LEVELS as u64 - 1)
            .got(detail));
        }
        trees::conifer(seed, detail)
    })
}

fn start_conifer_shadow(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    making(input, |input| {
        let [seed] = words(input, &CONIFER_SHADOW)?;
        trees::shadow_proxy(seed)
    })
}

fn start_cave(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    making(input, |input| {
        let bits: [u32; CAVE_POINTS * 3] = words(input, &CAVE)?;
        let points: Vec<Vec3> = bits
            .as_chunks::<3>()
            .0
            .iter()
            .map(|point| {
                Vec3::new(
                    f32::from_bits(point[0]),
                    f32::from_bits(point[1]),
                    f32::from_bits(point[2]),
                )
            })
            .collect();
        caves::cave(&points, CAVE_RADIUS, CAVE_SEED)
    })
}
