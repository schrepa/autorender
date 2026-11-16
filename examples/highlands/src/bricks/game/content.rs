use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, Background, Error, GeometryHandle, Hold, JobKind, Jobs, Landed, Mesh,
    ProvidedAssets, RecordExchange, Records, Requests, URGENT_JOBS, job_priority,
};
use glam::Vec3;

use crate::bricks::game::HighlandsStats;
use crate::jobs::photographs::{ATLAS_MAPS, BARK_MAPS, Photographs};
use crate::jobs::trees::{self, PLANTS};
use crate::jobs::{BOULDER, CAVE_POINTS, PLANT, sprays};

const BOULDERS: u32 = 5;

const ROCK_SHAPE: u32 = 8;

const SHAPED_ROCKS: [(&str, u32, u32); 3] = [("outcrop", 1, 4), ("ledge", 2, 4), ("block", 3, 6)];

const ATLASES: [&str; 3] = ["spruce_twigs", "pine_twigs", "undergrowth"];

const TWIGS: [&str; 2] = ["spruce_twigs", "pine_twigs"];

const LEAVES: &str = "birch_leaves";

const BARKS: [&str; 3] = ["spruce", "pine", "birch"];

#[derive(Resource, Default)]
pub(crate) struct Content {
    asked: Vec<(Background<Mesh>, GeometryHandle, &'static JobKind)>,
    fills: Requests<GeometryHandle, ()>,
    filling: Vec<Hold>,
    pub(crate) boulders: Vec<GeometryHandle>,
    failures: u64,
    latest: Option<String>,
}

impl Content {
    pub(crate) fn ask(
        &mut self,
        (jobs, assets): (&mut Jobs<'_, '_>, &mut AssetLibrary),
        kind: &'static JobKind,
        input: &[u8],
    ) -> Result<GeometryHandle, Error> {
        let handle = assets.reserve_mesh()?;
        let work = jobs.fetch(kind, input, self.priority(), |bytes| {
            Mesh::from_record(&bytes)
        });
        self.asked.push((work, handle, kind));
        self.hold(handle);
        Ok(handle)
    }

    pub(crate) fn hold(&mut self, mesh: GeometryHandle) {
        self.filling.push(self.fills.ask(mesh).into());
    }

    pub(crate) fn filled(
        &mut self,
        kind: &JobKind,
        mesh: GeometryHandle,
        filled: Result<(), Error>,
    ) {
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
    photographs: &Photographs,
) -> Result<(), Error> {
    let boulders = (0..BOULDERS)
        .map(|seed| content.ask((&mut *jobs, &mut *assets), &BOULDER, &seed.to_le_bytes()))
        .collect::<Result<Vec<_>, Error>>()?;
    for (seed, &mesh) in boulders.iter().enumerate() {
        provided.provide_mesh(format!("boulder{seed}"), mesh);
    }
    content.boulders = boulders;
    for (name, shape, count) in SHAPED_ROCKS {
        for seed in 0..count {
            let rock = (shape << ROCK_SHAPE) | seed;
            let mesh = content.ask((&mut *jobs, &mut *assets), &BOULDER, &rock.to_le_bytes())?;
            provided.provide_mesh(format!("{name}{seed}"), mesh);
        }
    }
    provide_plants((assets, records, provided), (content, jobs), photographs)
}

fn provide_plants(
    (assets, records, provided): (&mut AssetLibrary, &mut RecordExchange, &mut ProvidedAssets),
    (content, jobs): (&mut Content, &mut Jobs<'_, '_>),
    photographs: &Photographs,
) -> Result<(), Error> {
    for bark in BARKS {
        let textures =
            photographs.textures((&mut *records, &mut *assets), ("bark", bark), &BARK_MAPS)?;
        for (&(map, _), texture) in BARK_MAPS.iter().zip(textures) {
            provided.provide_texture(format!("{bark}_bark_{map}"), texture);
        }
    }
    for atlas in ATLASES {
        let textures =
            photographs.textures((&mut *records, &mut *assets), ("plants", atlas), &ATLAS_MAPS)?;
        for (&(map, _), texture) in ATLAS_MAPS.iter().zip(textures) {
            provided.provide_texture(format!("{atlas}_{map}"), texture);
        }
    }
    let mut atlases = Vec::with_capacity(TWIGS.len() + 1);
    for twigs in TWIGS {
        atlases.push((twigs, photographs.atlas(twigs)?));
    }
    let textures = photographs.sprays((&mut *records, &mut *assets), LEAVES)?;
    for (&(map, _), texture) in ATLAS_MAPS.iter().zip(textures) {
        provided.provide_texture(format!("{LEAVES}_{map}"), texture);
    }
    atlases.push((LEAVES, sprays::atlas()));
    for (index, plant) in PLANTS.iter().enumerate() {
        let atlas = atlases
            .iter()
            .find(|(twigs, _)| plant.atlas() == Some(*twigs))
            .map(|(_, atlas)| atlas);
        for &piece in plant.pieces() {
            for detail in 0..plant.details() {
                let input = trees::input(index, detail, piece, atlas);
                let mesh = content.ask((&mut *jobs, &mut *assets), &PLANT, &input)?;
                provided.provide_mesh(plant.mesh(piece, detail), mesh);
            }
        }
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
