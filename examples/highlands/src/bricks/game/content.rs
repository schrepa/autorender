use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, Error, Interest, JobKind, Jobs, Mesh, MeshHandle, ProvidedAssets, RecordExchange,
    RecordKey, Records, Requests, URGENT_JOBS, job_priority,
};
use glam::Vec3;

use crate::bricks::game::HighlandsStats;
use crate::jobs::{BOULDER, CAVE_POINTS, CONIFER, CONIFER_SHADOW, trees};

const BOULDERS: u32 = 5;

const SPRUCES: u32 = 3;

#[derive(Resource, Default)]
pub(crate) struct Content {
    asked: Vec<(Interest<RecordKey>, MeshHandle, &'static JobKind)>,
    fills: Requests<MeshHandle, ()>,
    filling: Vec<Interest<MeshHandle>>,
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
        self.hold(handle);
        Ok(handle)
    }

    pub(crate) fn hold(&mut self, mesh: MeshHandle) {
        self.filling.push(self.fills.ask(mesh));
    }

    pub(crate) fn filled(&mut self, kind: &JobKind, mesh: MeshHandle, filled: Result<(), Error>) {
        if let Err(error) = &filled {
            self.failed(kind, error);
        }
        let Some(ticket) = self.fills.ticket(&mesh) else {
            return;
        };
        match filled {
            Ok(()) => self.fills.ready(&mesh, ticket, ()),
            Err(error) => self.fills.fail(&mesh, ticket, error),
        };
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
) {
    let content = &mut *content;
    let asked = std::mem::take(&mut content.asked);
    for (ticket, handle, kind) in asked {
        let Some(record) = records.get(&ticket) else {
            content.asked.push((ticket, handle, kind));
            continue;
        };
        let filled = record
            .and_then(|bytes| Mesh::from_record(&bytes))
            .and_then(|mesh| assets.fill_mesh(handle, mesh));
        content.filled(kind, handle, filled);
    }
}

pub(crate) fn publish(content: Res<Content>, mut stats: ResMut<HighlandsStats>) {
    stats.set_if_neq(HighlandsStats {
        failed_jobs: content.failures,
        latest_job_failure: content.latest.clone(),
    });
}
