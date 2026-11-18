use engine::math;
use engine::prelude::*;
use engine_core::{AssetLibrary, DrawnTable};

use super::numbers::Antialias;
use super::numbers::{self, LightingNumbers};
use super::{LevelStats, SKY};
use crate::bricks::figures::{LEVEL, LIGHTMAP_MAP};
use crate::play::actors::{Alive, Body};
use crate::play::effects::{Blob, GroundMarks};
use crate::play::light::{Ambient, Fog, Grade, Light, Lighting, SkyLook};
use crate::play::numbers::Numbers;
use crate::play::world::Arena;

#[derive(Resource, Default)]
pub(super) struct Setup {
    revision: Option<u64>,
    parsed: Option<LightingNumbers>,
    refusal: Option<String>,
    radius_m: f32,
}

pub(super) fn setup(numbers: Res<Numbers>, mut setup: ResMut<Setup>) {
    let Some(game) = numbers.get() else {
        if setup.refusal.is_none() {
            setup.refusal = Some("numbers.json not read yet".into());
        }
        return;
    };
    if setup.revision == Some(numbers.revision()) {
        return;
    }
    setup.revision = Some(numbers.revision());
    setup.radius_m = game.player.moving.radius_m;
    match numbers::parse(&game.text) {
        Ok(parsed) => {
            setup.parsed = Some(parsed);
            setup.refusal = None;
        }
        Err(error) => {
            setup.parsed = None;
            setup.refusal = Some(error);
        }
    }
}

fn scaled(color: [f32; 3], by: f32) -> [f32; 3] {
    color.map(|value| value * by)
}

fn sky_look(light: &Light, numbers: &LightingNumbers) -> SkyLook {
    let sky = &numbers.sky;
    let inner = sky.disc_degrees.to_radians();
    let outer = (sky.disc_degrees + sky.disc_soft_degrees).to_radians();
    SkyLook {
        zenith: scaled(light.sky, sky.zenith_scale),
        horizon: sky.horizon,
        below: sky.below,
        haze_height: sky.haze_height,
        disc_cos_inner: math::cos(inner),
        disc_cos_outer: math::cos(outer),
        disc_strength: sky.disc_strength,
        glow_power: sky.glow_power,
        glow_strength: sky.glow_strength,
    }
}

pub(super) fn light(setup: Res<Setup>, lights: Query<&Light>, mut lighting: ResMut<Lighting>) {
    let numbers = setup.parsed.as_ref();
    let unlit = Lighting {
        grade: numbers.map_or_else(Grade::default, |numbers| Grade {
            exposure: numbers.grade.exposure,
            contrast: numbers.grade.contrast,
            saturation: numbers.grade.saturation,
        }),
        antialias: numbers.and_then(|numbers| numbers.antialias) == Some(Antialias::Fxaa),
        ..Lighting::default()
    };
    let mut listed: Vec<&Light> = lights.iter().collect();
    listed.truncate(2);
    let next = match listed.as_slice() {
        [only] => Lighting {
            lit: true,
            to_sun: Vec3::from_array(only.direction).normalize_or_zero(),
            sun: only.sun,
            sky: only.sky,
            ground: only.ground,
            wrap: numbers.map_or(0.0, |numbers| numbers.sun_wrap),
            sky_look: numbers.map(|numbers| sky_look(only, numbers)),
            fog: numbers.map(|numbers| Fog {
                start_m: numbers.fog.start_m,
                density_per_m: numbers.fog.density_per_m,
                most: numbers.fog.most,
            }),
            ambient: numbers.and_then(|numbers| numbers.ambient).map_or_else(
                Ambient::default,
                |ambient| Ambient {
                    sky_scale: ambient.sky_scale,
                    ground_scale: ambient.ground_scale,
                    base_occlusion: ambient.base_occlusion,
                    base_height_m: ambient.base_height_m,
                    floor_m: ambient.floor_m,
                    black_floor: ambient.black_floor,
                },
            ),
            ..unlit
        },
        _ => unlit,
    };
    lighting.set_if_neq(next);
}

pub(super) fn blobs(
    time: Res<FrameTime>,
    setup: Res<Setup>,
    arena: Option<Res<Arena>>,
    actors: Query<&Body, With<Alive>>,
    mut marks: ResMut<GroundMarks>,
) {
    let (Some(numbers), Some(arena)) = (setup.parsed.as_ref(), arena.as_deref()) else {
        if !marks.blobs.is_empty() {
            marks.blobs.clear();
        }
        return;
    };
    let blob = &numbers.blob;
    let mut blobs = Vec::new();
    if arena.ready() {
        for body in &actors {
            let drawn_eye = body.eye_before.lerp(body.eye, time.alpha);
            let feet = body.feet + (drawn_eye - body.eye);
            let lift = Vec3::Y * blob.depth_m;
            let Some(hit) = arena.ground_under(feet + lift, blob.fade_height_m + blob.depth_m)
            else {
                continue;
            };
            let height = math::max(feet.y - hit.point.y, 0.0);
            let fade = math::clamp(1.0 - height / blob.fade_height_m, 0.0, 1.0);
            blobs.push(Blob {
                at: hit.point,
                radius: setup.radius_m * blob.radius_scale,
                height: blob.depth_m,
                strength: blob.opacity * fade,
            });
        }
    }
    if marks.blobs != blobs {
        marks.blobs = blobs;
    }
}

pub(super) fn count(
    setup: Res<Setup>,
    lighting: Res<Lighting>,
    lights: Query<&Light>,
    marks: Res<GroundMarks>,
    table: Res<DrawnTable>,
    library: Option<Res<AssetLibrary>>,
    mut stats: ResMut<LevelStats>,
) {
    let mut next = LevelStats {
        refusal: setup.refusal.clone(),
        lights: lights.iter().count() as u32,
        numbers: setup.parsed,
        zenith: lighting.sky_look.map_or(lighting.sky, |sky| sky.zenith),
        blobs: marks
            .blobs
            .iter()
            .filter(|blob| blob.strength > 0.0)
            .count() as u32,
        ..LevelStats::default()
    };
    let mut materials = Vec::new();
    for batch in table.batches(LEVEL) {
        next.draws += 1;
        let instances = batch.instances.len() as u64;
        let library = library.as_deref();
        if let Some(mesh) = batch
            .geometry
            .zip(library)
            .and_then(|(geometry, library)| library.mesh(geometry))
        {
            next.triangles += mesh.indices().len() as u64 / 3 * instances;
        }
        if let Some(handle) = batch.material
            && !materials.contains(&handle)
        {
            materials.push(handle);
            let baked = library
                .and_then(|library| library.material(handle))
                .is_some_and(|material| material.map(LIGHTMAP_MAP).is_some());
            next.baked += u32::from(baked);
        }
    }
    next.materials = materials.len() as u32;
    next.lightmap = next.baked > 0;
    next.sky = table.batches(SKY).next().is_some();
    next.draws += u32::from(next.sky);
    stats.set_if_neq(next);
}
