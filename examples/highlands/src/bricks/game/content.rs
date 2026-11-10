use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, Background, Error, Hold, JobKind, Jobs, Landed, Mesh, MeshHandle, ProvidedAssets,
    RecordExchange, Records, Requests, URGENT_JOBS, job_priority,
};
use glam::Vec3;

use crate::bricks::game::HighlandsStats;
use crate::jobs::{BOULDER, CAVE_POINTS, CONIFER, CONIFER_SHADOW, trees};

const BOULDERS: u32 = 5;

const SPRUCES: u32 = 3;

#[derive(Resource, Default)]
pub(crate) struct Content {
    asked: Vec<(Background<Mesh>, MeshHandle, &'static JobKind)>,
    fills: Requests<MeshHandle, ()>,
    filling: Vec<Hold>,
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
        let work = Background::asked(jobs.fetch(kind, input, self.priority()), Mesh::from_record);
        self.asked.push((work, handle, kind));
        self.hold(handle);
        Ok(handle)
    }

    pub(crate) fn hold(&mut self, mesh: MeshHandle) {
        self.filling.push(self.fills.ask(mesh).into());
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
    for (mut work, handle, kind) in asked {
        let landed = match work.poll(&*records) {
            Landed::Waiting => None,
            Landed::Ready(_) => Some(None),
            Landed::Failed(error) => Some(Some(error.clone())),
        };
        let Some(failed) = landed else {
            content.asked.push((work, handle, kind));
            continue;
        };
        let filled = match (failed, work.into_ready()) {
            (Some(error), _) => Err(error),
            (None, Some(mesh)) => assets.fill_mesh(handle, mesh),
            (None, None) => continue,
        };
        content.filled(kind, handle, filled);
    }
}

pub(crate) fn publish(content: Res<Content>, mut stats: ResMut<HighlandsStats>) {
    stats.set_if_neq(HighlandsStats {
        failed_jobs: content.failures,
        latest_job_failure: content.latest.clone(),
    });
}
