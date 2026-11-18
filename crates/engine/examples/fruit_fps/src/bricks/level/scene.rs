use engine::math;
use engine::prelude::*;
use std::collections::BTreeMap;

use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, DrawnTable, Error, ErrorCode, Expected, Material,
    MaterialHandle, SceneContext, Views,
};

use super::chunks::{self, PieceSizes, Sized};
use super::gpu::problems_of;
use super::numbers::Antialias;
use super::numbers::{self, LightingNumbers};
use super::sets::VisibleSets;
use super::{
    FIELD_HEIGHT, FIELD_MAP, FIELD_RECT, LEVEL_DETAIL, LIGHT_FIELD, LevelStats, MACRO_COLOR,
    MACRO_ROUGH, MIRROR, PICTURE_MAP, SKY,
};
use crate::bricks::figures::{LEVEL, LIGHTMAP_MAP};
use crate::play::actors::{Alive, Body};
use crate::play::effects::{Blob, GroundMarks};
use crate::play::light::{Ambient, DetailLimit, Fog, Grade, Light, LightField, Lighting, SkyLook};
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

fn sky_look(numbers: &LightingNumbers) -> SkyLook {
    let sky = &numbers.sky;
    let inner = sky.disc_degrees.to_radians();
    let outer = (sky.disc_degrees + sky.disc_soft_degrees).to_radians();
    SkyLook {
        zenith: sky.zenith,
        horizon: sky.horizon,
        below: sky.below,
        haze_height: sky.haze_height,
        disc_cos_inner: math::cos(inner),
        disc_cos_outer: math::cos(outer),
        disc_strength: sky.disc_strength,
        glow_power: sky.glow_power,
        glow_strength: sky.glow_strength,
        mirror_roughness: sky.mirror_roughness,
    }
}

fn field_value(material: &Material, path: &str, name: &str) -> Result<f32, Error> {
    let value = material.value(name).ok_or_else(|| {
        Error::new(ErrorCode::MissingField, format!("{path}.values.{name}"))
            .expected(Expected::Present)
    })?;
    if !value.is_finite() {
        return Err(
            Error::new(ErrorCode::InvalidValue, format!("{path}.values.{name}"))
                .expected(Expected::Finite)
                .got(value),
        );
    }
    Ok(value)
}

fn field_of(material: &Material, path: &str) -> Result<LightField, Error> {
    let map = material.map(FIELD_MAP).ok_or_else(|| {
        Error::new(ErrorCode::MissingField, format!("{path}.maps.{FIELD_MAP}"))
            .expected(Expected::Present)
    })?;
    let mut rect = [0.0_f32; 4];
    for (at, name) in FIELD_RECT.into_iter().enumerate() {
        rect[at] = field_value(material, path, name)?;
    }
    for (low, high) in [(0, 2), (1, 3)] {
        if rect[high] <= rect[low] {
            return Err(Error::new(
                ErrorCode::OutOfRange,
                format!("{path}.values.{}", FIELD_RECT[high]),
            )
            .expected(Expected::Above)
            .got(rect[high]));
        }
    }
    let scale = field_value(material, path, super::LIGHTMAP_SCALE)?;
    if scale <= 0.0 {
        return Err(Error::new(
            ErrorCode::OutOfRange,
            format!("{path}.values.{}", super::LIGHTMAP_SCALE),
        )
        .expected(Expected::Above)
        .got(scale));
    }
    Ok(LightField {
        map,
        rect,
        height_m: field_value(material, path, FIELD_HEIGHT)?,
        scale,
    })
}

fn field_in(
    table: &DrawnTable,
    library: Option<&AssetLibrary>,
) -> Option<Result<LightField, Error>> {
    let handle = table
        .batches(LIGHT_FIELD)
        .find_map(|batch| batch.material)?;
    let material = library?.material(handle)?;
    Some(field_of(
        material,
        &format!("$.materials.{}", handle.slot()),
    ))
}

pub(super) fn light(
    setup: Res<Setup>,
    lights: Query<&Light>,
    table: Res<DrawnTable>,
    library: Option<Res<AssetLibrary>>,
    mut lighting: ResMut<Lighting>,
) {
    let numbers = setup.parsed.as_ref();
    let field = field_in(&table, library.as_deref()).and_then(Result::ok);
    let unlit = Lighting {
        grade: numbers.map_or_else(Grade::default, |numbers| Grade {
            exposure: numbers.grade.exposure,
            contrast: numbers.grade.contrast,
            saturation: numbers.grade.saturation,
        }),
        antialias: numbers.and_then(|numbers| numbers.antialias) == Some(Antialias::Fxaa),
        filtering: numbers.map_or_else(Default::default, |numbers| numbers.filtering),
        detail: numbers.map_or_else(Default::default, |numbers| numbers.detail),
        visible_sets: numbers.is_some_and(|numbers| numbers.visible_sets),
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
            sky_look: numbers.map(sky_look),
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
            field,
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

type LevelAssets<'w> = (
    Option<Res<'w, AssetLibrary>>,
    Option<Res<'w, AssetResidency>>,
    Option<Res<'w, SceneContext>>,
);

pub(super) fn count(
    setup: Res<Setup>,
    lighting: Res<Lighting>,
    lights: Query<&Light>,
    marks: Res<GroundMarks>,
    (table, views, pieces, sets): (
        Res<DrawnTable>,
        Res<Views>,
        Res<PieceSizes>,
        Res<VisibleSets>,
    ),
    (library, residency, scene): LevelAssets<'_>,
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
    let library = library.as_deref();
    let mut materials = Vec::new();
    let batches = table
        .batches(LEVEL)
        .map(|batch| (batch, false))
        .chain(table.batches(LEVEL_DETAIL).map(|batch| (batch, true)));
    for (batch, detail) in batches {
        next.draws += 1;
        next.detail += u32::from(detail);
        let instances = batch.instances.len() as u64;
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
            if let Some(material) = library.and_then(|library| library.material(handle)) {
                tally(&mut next, material);
            }
        }
    }
    next.materials = materials.len() as u32;
    let judged = judged(
        &table,
        library,
        residency.as_deref(),
        scene.as_deref(),
        &mut next,
    );
    let sources = (&*table, library, &*pieces);
    next.sets_loaded = sets.loaded();
    next.sets_unknown = sets.unknown;
    let set = (lighting.visible_sets, &*sets);
    submitted(&mut next, sources, (&views, &judged, set), lighting.detail);
    next.lightmap = next.baked > 0;
    let sky = table.batches(SKY).next();
    next.sky = sky.is_some();
    next.picture = sky
        .and_then(|batch| batch.material)
        .zip(library)
        .and_then(|(handle, library)| library.material(handle))
        .is_some_and(|material| material.map(PICTURE_MAP).is_some());
    next.draws += u32::from(next.sky);
    match field_in(&table, library) {
        Some(Ok(_)) => next.field = true,
        Some(Err(problem)) => next.field_refusal = Some(problem.to_string()),
        None => {}
    }
    stats.set_if_neq(next);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Judged {
    Ready,
    Broken,
    Waiting,
}

fn name_of(scene: Option<&SceneContext>, handle: MaterialHandle) -> String {
    scene
        .and_then(|scene| {
            scene
                .assets
                .materials
                .iter()
                .find(|(_, known)| **known == handle)
                .map(|(name, _)| name.clone())
        })
        .unwrap_or_else(|| format!("$.materials.{}", handle.slot()))
}

fn judged(
    table: &DrawnTable,
    library: Option<&AssetLibrary>,
    residency: Option<&AssetResidency>,
    scene: Option<&SceneContext>,
    next: &mut LevelStats,
) -> BTreeMap<MaterialHandle, Judged> {
    let mut judged = BTreeMap::new();
    let used = table
        .batches(LEVEL_DETAIL)
        .map(|batch| (batch.material, true))
        .chain(table.batches(LEVEL).map(|batch| (batch.material, false)));
    for (handle, detail) in used {
        let Some(handle) = handle else {
            continue;
        };
        if judged.contains_key(&handle) {
            continue;
        }
        let material = library.and_then(|library| library.material(handle));
        let problems = material.map_or_else(Vec::new, |material| {
            problems_of(material, &format!("$.materials.{}", handle.slot()), detail)
        });
        let resident = residency.is_some_and(|residency| residency.material_resident(handle));
        let verdict = if !problems.is_empty() {
            Judged::Broken
        } else if resident && material.is_some() {
            Judged::Ready
        } else {
            Judged::Waiting
        };
        if !problems.is_empty() {
            let name = name_of(scene, handle);
            next.problems.extend(
                problems
                    .iter()
                    .map(|problem| format!("level material {name}: {problem}")),
            );
        }
        judged.insert(handle, verdict);
    }
    judged
}

fn submitted(
    next: &mut LevelStats,
    sources: (&DrawnTable, Option<&AssetLibrary>, &PieceSizes),
    (views, judged, (on, sets)): (
        &Views,
        &BTreeMap<MaterialHandle, Judged>,
        (bool, &VisibleSets),
    ),
    limit: DetailLimit,
) {
    let mut listed = Vec::new();
    let mut order = Vec::new();
    let set = views
        .main()
        .filter(|_| on)
        .and_then(|facts| sets.at(facts.eye));
    next.sets = set.is_some();
    next.set_cell = set.map(|set| set.cell);
    let mut sized = Sized {
        limit,
        held: None,
        set,
    };
    for kind in [LEVEL, LEVEL_DETAIL] {
        chunks::listed(sources, kind, &mut listed);
        next.chunks += listed.len() as u32;
        let Some(facts) = views.main() else {
            continue;
        };
        let counts = chunks::ordered(&listed, facts, &mut sized, &mut order);
        next.skipped_small += counts.small;
        next.kept_by_frustum += counts.framed;
        next.kept_by_set += counts.framed - counts.hidden;
        for chunk in order.iter().map(|&at| &listed[at as usize]) {
            let verdict = chunk.material.map_or(Judged::Ready, |handle| {
                judged.get(&handle).copied().unwrap_or(Judged::Waiting)
            });
            match verdict {
                Judged::Ready => {
                    next.submitted += 1;
                    next.vertices += chunk.vertices;
                    next.triangles_submitted += chunk.vertices / 3;
                }
                Judged::Broken => next.skipped += 1,
                Judged::Waiting => next.waiting += 1,
            }
        }
    }
}

fn tally(next: &mut LevelStats, material: &Material) {
    let strength = |name| material.value(name).unwrap_or(0.0);
    next.baked += u32::from(material.map(LIGHTMAP_MAP).is_some());
    next.cutout += u32::from(matches!(material.alpha_mode(), AlphaMode::Mask { .. }));
    next.glowing += u32::from(material.emissive().iter().any(|value| *value > 0.0));
    next.macro_strengths += u32::from(strength(MACRO_COLOR) + strength(MACRO_ROUGH) > 0.0);
    next.mirrors += u32::from(strength(MIRROR) > 0.0);
    next.plain += u32::from(
        super::SURFACE_MAPS
            .iter()
            .all(|name| material.map(name).is_none()),
    );
}
