use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_ecs::world::World;
use engine_core::{
    AssetLibrary, Background, CookAsks, FrameTime, Jobs, Landed, Records, Texture, TextureFormat,
    VIEW_JOBS, codec, content_hash, job_priority, record_missing,
};
use engine_open_world_kit::ground::{SurfaceMaps, SurfaceMaterial, TerrainTiles};

use crate::jobs::{ROUGH, RoughJob, RoughKey, rough_output};
use crate::state::Bakes;
use crate::terrain_maps::TerrainMaps;

const BAKED_SIDE: u32 = 1024;

type Work = Background<Arc<[u8]>>;

enum Stage {
    Looking(Work),
    Making(Work),
    Made(Arc<[u32]>),
    Failed,
}

struct Held {
    key: RoughKey,
    stage: Stage,
}

#[derive(Resource, Default)]
pub(crate) struct RoughLights {
    held: Vec<Held>,
}

impl RoughLights {
    pub(crate) fn made(&self, key: &RoughKey) -> Option<&Arc<[u32]>> {
        self.held
            .iter()
            .find(|held| held.key == *key)
            .and_then(|held| match &held.stage {
                Stage::Made(shares) => Some(shares),
                Stage::Looking(_) | Stage::Making(_) | Stage::Failed => None,
            })
    }
}

fn level_of(texture: &Texture) -> Option<u32> {
    let levels = texture.levels().len() as u32;
    let level = (0..levels)
        .find(|&level| (texture.width().max(texture.height()) >> level) <= BAKED_SIDE)
        .or(levels.checked_sub(1))?;
    let whole = (texture.width() >> level).is_multiple_of(4)
        && (texture.height() >> level).is_multiple_of(4);
    (whole || !texture.format().is_compressed()).then_some(level)
}

fn heights<'a>(maps: &SurfaceMaps, assets: &'a AssetLibrary) -> Option<&'a Texture> {
    let texture = assets.texture(maps.metallic_roughness?)?;
    match texture.format() {
        TextureFormat::Rgba8 | TextureFormat::Bc7 => Some(texture),
        TextureFormat::Etc2Rgba8 | TextureFormat::Astc4x4 => None,
    }
}

pub(crate) fn key(
    look: &SurfaceMaterial,
    maps: &SurfaceMaps,
    assets: &AssetLibrary,
) -> Option<RoughKey> {
    if look.depth <= 0.0 || look.tiling <= 0.0 {
        return None;
    }
    let texture = heights(maps, assets)?;
    let identity = texture.identity(content_hash);
    Some(RoughKey {
        texture: [identity as u64, (identity >> 64) as u64],
        level: level_of(texture)?,
        tile: look.tiling,
        depth: look.depth,
        roughness: TerrainMaps::material(maps, assets)?.roughness * texture.summary()?.mean[1],
    })
}

fn job(key: RoughKey, texture: &Texture) -> RoughJob {
    let bytes = &texture.levels()[key.level as usize];
    let blocks = texture.format() == TextureFormat::Bc7;
    RoughJob {
        key,
        width: (texture.width() >> key.level).max(1),
        height: (texture.height() >> key.level).max(1),
        blocks,
        texels: if blocks {
            bytes.to_vec()
        } else {
            bytes.as_chunks::<4>().0.iter().map(|texel| texel[3]).collect()
        },
    }
}

fn wanted<'a>(
    tiles: &'a TerrainTiles,
    assets: &'a AssetLibrary,
) -> impl Iterator<Item = (RoughKey, &'a Texture)> + 'a {
    tiles.terrains.iter().flat_map(move |terrain| {
        terrain
            .materials
            .iter()
            .zip(terrain.maps.iter())
            .filter_map(move |(look, maps)| {
                let maps = maps.as_ref()?;
                Some((key(look, maps, assets)?, heights(maps, assets)?))
            })
    })
}

enum Step {
    Wait,
    Make,
    Landed(Result<(Arc<[u8]>, bool), String>),
}

fn step(work: &mut Work, records: &Records, made: bool) -> Step {
    match work.poll(records) {
        Landed::Waiting => Step::Wait,
        Landed::Ready(bytes) => Step::Landed(Ok((Arc::clone(bytes), made))),
        Landed::Failed(error) if !made && record_missing(error) => Step::Make,
        Landed::Failed(error) => Step::Landed(Err(error.to_string())),
    }
}

pub(crate) fn roughen(
    (tiles, assets, records, time): (
        Res<TerrainTiles>,
        Res<AssetLibrary>,
        Res<Records>,
        Res<FrameTime>,
    ),
    mut jobs: Jobs,
    mut lights: ResMut<RoughLights>,
    mut bakes: ResMut<Bakes>,
) {
    let lights = &mut *lights;
    let mut keys: Vec<RoughKey> = Vec::new();
    for (place, (key, texture)) in wanted(&tiles, &assets).enumerate() {
        if keys.contains(&key) {
            continue;
        }
        keys.push(key);
        let priority = job_priority(VIEW_JOBS, place as u32);
        let Some(held) = lights.held.iter_mut().find(|held| held.key == key) else {
            let work = jobs.look(&ROUGH, &codec::encode(&key), priority, Ok);
            lights.held.push(Held {
                key,
                stage: Stage::Looking(work),
            });
            continue;
        };
        let landed = match &mut held.stage {
            Stage::Looking(work) => step(work, &records, false),
            Stage::Making(work) => step(work, &records, true),
            Stage::Made(_) | Stage::Failed => Step::Wait,
        };
        let shares = match landed {
            Step::Wait => continue,
            Step::Make => {
                let input = codec::encode(&job(key, texture));
                held.stage = Stage::Making(jobs.run(&ROUGH, input, priority, Ok));
                continue;
            }
            Step::Landed(landed) => landed.and_then(|(bytes, made)| {
                let light =
                    rough_output(&bytes).map_err(|error| ROUGH.undecoded(error).to_string())?;
                if made {
                    jobs.keep(&ROUGH, &codec::encode(&key), bytes.to_vec());
                }
                Ok(Arc::<[u32]>::from(light.shares))
            }),
        };
        held.stage = match shares {
            Ok(shares) => Stage::Made(shares),
            Err(error) => {
                bakes
                    .failures
                    .record(time.frame, format!("{}: {error}", ROUGH.name()));
                Stage::Failed
            }
        };
    }
    lights.held.retain(|held| keys.contains(&held.key));
}

pub(crate) fn cook(world: &World, asks: &mut CookAsks<'_>) {
    let (Some(tiles), Some(assets)) = (
        world.get_resource::<TerrainTiles>(),
        world.get_resource::<AssetLibrary>(),
    ) else {
        return;
    };
    for terrain in tiles.terrains.iter() {
        for (look, maps) in terrain.materials.iter().zip(terrain.maps.iter()) {
            let unfilled = maps
                .as_ref()
                .and_then(|maps| maps.metallic_roughness)
                .is_some_and(|handle| assets.texture(handle).is_none());
            if unfilled && look.depth > 0.0 {
                asks.left(1);
            }
        }
    }
    for (key, texture) in wanted(tiles, assets) {
        let record = key.record();
        if !asks.asked(&record) {
            asks.ask_as(record, &ROUGH, codec::encode(&job(key, texture)));
        }
    }
}
