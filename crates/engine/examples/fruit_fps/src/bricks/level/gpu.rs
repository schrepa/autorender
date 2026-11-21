use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::system::SystemParamItem;
use engine::prelude::*;
use engine_core::{
    AlphaMode, Asset, Assets, ColorSpace, Cull, DrawnTable, Error, ErrorCode, Expected, Material,
    MeshFile, PassBit, TextureRef, Var, VertexStream, ViewFacts, gpu_struct,
};
use engine_drawing::{DetailLimit, MaterialBinds, SizeSkip, StandIn, pixel_scale};
use engine_meshpack::{BASE, COLOR, LIGHTMAP};
use engine_render::{BatchGroup, BatchOf, DrawOrder, GroupEntry, PerBatch, Scene, Variant, wgpu};

use super::chunks::{self, Pieces};
use super::sets::SetIndex;
use super::shaders::level::{light, map, material, picture, sky};
use super::{
    EMISSIVE_STRENGTH, LEVEL, LEVEL_DETAIL, LIGHTMAP_CURVE, LIGHTMAP_MAP, LIGHTMAP_SCALE,
    MACRO_COLOR, MACRO_MAP, MACRO_ROUGH, MACRO_TILE, MIRROR, PICTURE_MAP, SKY, SKY_ABOVE,
    SKY_BELOW, SKY_FADE, SKY_HAZE, SKY_TURNS, SURFACE_MAPS,
};
use crate::names::{COLOR_PASS, DEPTH_PASS};
use crate::play::light::{CLAMP, Filtering, Lighting};

gpu_struct! {
    pub struct LevelLight {
        pub to_sun: [f32; 4],
        pub sun: [f32; 4],
        pub zenith: [f32; 4],
        pub ground: [f32; 4],
        pub horizon: [f32; 4],
        pub below: [f32; 4],
        pub fog: [f32; 4],
        pub glow: [f32; 4],
        pub sky_light: [f32; 4],
        pub ambient: [f32; 4],
        pub picture: [f32; 4],
        pub reflect: [f32; 4],
    }
}

gpu_struct! {
    pub struct LevelSurface {
        pub tint: [f32; 4],
        pub glow: [f32; 4],
        pub baked: [f32; 4],
        pub broken: [f32; 4],
        pub cut: [f32; 4],
    }
}

const BOX_VERTICES: u32 = 36;

const RANGE_ALIGN: u64 = 256;

const DEPTH_STREAMS: [VertexStream; 1] = [BASE];

const LEVEL_STREAMS: [VertexStream; 2] = [BASE, LIGHTMAP];

const DETAIL_STREAMS: [VertexStream; 2] = [BASE, COLOR];

const MAPS: [(&str, Var, Var); 3] = [
    (
        SURFACE_MAPS[0],
        material::LEVEL_BASE_COLOR_MAP,
        material::LEVEL_BASE_COLOR_SAMPLER,
    ),
    (
        SURFACE_MAPS[1],
        material::LEVEL_NORMAL_MAP,
        material::LEVEL_NORMAL_SAMPLER,
    ),
    (
        SURFACE_MAPS[2],
        material::LEVEL_ORM_MAP,
        material::LEVEL_ORM_SAMPLER,
    ),
];

const PLAIN: [(&str, [u8; 4]); 3] = [
    ("level_plain_base", [255, 255, 255, 255]),
    ("level_plain_normal", [128, 128, 255, 255]),
    ("level_plain_orm", [255, 128, 0, 255]),
];

struct Bound {
    slot: u32,
    mapped: Mapped,
}

struct Mapped {
    plain: bool,
    maps: Vec<TextureRef>,
    baked: Option<TextureRef>,
    broken: Option<TextureRef>,
    surface: LevelSurface,
    cutout: bool,
    cull: Option<wgpu::Face>,
}

#[derive(Clone, Copy, PartialEq)]
struct Shaped {
    map: TextureRef,
    shape: [f32; 4],
    haze: f32,
}

pub(super) struct LevelGpu {
    depth_pass: Option<PassBit>,
    depth_back: Pipeline<Self, Scene>,
    depth_double: Pipeline<Self, Scene>,
    level: Pipeline<Self, Scene>,
    detail: Pipeline<Self, Scene>,
    sky: Pipeline<Self, Scene>,
    picture: Pipeline<Self, Scene>,
    group: BatchGroup<Self>,
    sky_group: BatchGroup<Self>,
    light: Buffer<Self>,
    surfaces: Buffer<Self>,
    unbaked: StandIn<Self>,
    plain: [StandIn<Self>; 3],
    sent: Option<LevelLight>,
    binds: MaterialBinds<bool, Bound>,
    free: Vec<u32>,
    slots: u32,
    pictured: MaterialBinds<(), Shaped>,
    ready: bool,
    pieces: Pieces,
    filtering: Filtering,
    detail_limit: DetailLimit,
    held_small: BTreeSet<Asset<MeshFile>>,
    sets: SetIndex,
    sets_on: bool,
    batched: Vec<(bool, Option<Asset<Material>>)>,
    used: BTreeMap<Asset<Material>, bool>,
    sky_bound: Option<Asset<Material>>,
    has_double_depth: bool,
    again: bool,
}

fn face(cull: Cull) -> Option<wgpu::Face> {
    match cull {
        Cull::Back => Some(wgpu::Face::Back),
        Cull::Front => Some(wgpu::Face::Front),
        Cull::None => None,
    }
}

fn depth_described(cull_mode: Option<wgpu::Face>) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: map::VS_LEVEL_DEPTH,
        fragment: Some(map::FS_LEVEL_DEPTH),
        vertices: &DEPTH_STREAMS,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode,
            ..wgpu::PrimitiveState::default()
        },
        depth: DepthUse::Compare {
            compare: wgpu::CompareFunction::GreaterEqual,
            write: true,
            bias: wgpu::DepthBiasState::default(),
        },
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets: &[],
        constants: &[],
        per_batch: PerBatch::NONE,
    }
}

fn depth_pass_cull(mapped: &Mapped) -> Option<Option<wgpu::Face>> {
    if mapped.cutout || mapped.cull == Some(wgpu::Face::Front) {
        return None;
    }
    Some(mapped.cull)
}

fn described<'a>(
    (vertex, fragment): (engine_core::Entry, engine_core::Entry),
    vertices: &'a [VertexStream],
    write: bool,
    per_batch: PerBatch,
    constants: &'a [(&'static str, f64)],
) -> RenderPipeline<'a> {
    RenderPipeline {
        vertex,
        fragment: Some(fragment),
        vertices,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: if write { Some(wgpu::Face::Back) } else { None },
            ..wgpu::PrimitiveState::default()
        },
        depth: DepthUse::Compare {
            compare: wgpu::CompareFunction::GreaterEqual,
            write,
            bias: wgpu::DepthBiasState::default(),
        },
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets: &[(None, wgpu::ColorWrites::ALL)],
        constants,
        per_batch,
    }
}

fn rgb(color: [f32; 3], w: f32) -> [f32; 4] {
    [color[0], color[1], color[2], w]
}

fn light_of(lighting: &Lighting, shown: Option<Shaped>) -> LevelLight {
    let to_sun = lighting.to_sun;
    let sky = lighting.sky_look;
    let fog = lighting.fog;
    let haze_height = sky.map_or(1.0, |sky| sky.haze_height);
    let inv_haze = 1.0 / if haze_height > 1.0e-3 { haze_height } else { 1.0e-3 };
    let inv_span = shown.map_or(0.0, |shown| 1.0 / (shown.shape[1] - shown.shape[0]));
    let inv_wrap = 1.0 / (1.0 + lighting.wrap);
    let sky_luma = (lighting.sky[0] + lighting.sky[1] + lighting.sky[2]) * 0.333;
    let inv_sky_luma = 0.333 / if sky_luma > 1.0e-3 { sky_luma } else { 1.0e-3 };
    LevelLight {
        to_sun: [to_sun.x, to_sun.y, to_sun.z, lighting.wrap],
        sun: rgb(lighting.sun, f32::from(u8::from(lighting.lit))),
        zenith: rgb(sky.map_or(lighting.sky, |sky| sky.zenith), haze_height),
        ground: rgb(lighting.ground, lighting.ambient.black_floor),
        horizon: rgb(
            sky.map_or(lighting.sky, |sky| sky.horizon),
            fog.map_or(0.0, |fog| fog.most),
        ),
        below: rgb(sky.map_or(lighting.ground, |sky| sky.below), inv_haze),
        fog: [
            fog.map_or(0.0, |fog| fog.start_m),
            fog.map_or(0.0, |fog| fog.density_per_m),
            sky.map_or(1.0, |sky| sky.disc_cos_inner),
            sky.map_or(1.0, |sky| sky.disc_cos_outer),
        ],
        glow: [
            sky.map_or(0.0, |sky| sky.disc_strength),
            sky.map_or(1.0, |sky| sky.glow_power),
            sky.map_or(0.0, |sky| sky.glow_strength),
            inv_span,
        ],
        sky_light: rgb(lighting.sky, lighting.ambient.floor_m),
        ambient: [
            lighting.ambient.sky_scale,
            lighting.ambient.ground_scale,
            lighting.ambient.base_occlusion,
            lighting.ambient.base_height_m,
        ],
        picture: shown.map_or([0.0; 4], |shown| shown.shape),
        reflect: [
            sky.map_or(0.0, |sky| sky.mirror_roughness),
            shown.map_or(0.0, |shown| shown.haze),
            inv_wrap,
            inv_sky_luma,
        ],
    }
}

fn missing(path: &str, name: &str) -> Error {
    Error::new(ErrorCode::MissingField, format!("{path}.{name}")).expected(Expected::Present)
}

fn value_at_least(material: &Material, path: &str, name: &str, least: f32) -> Result<f32, Error> {
    match material.value(name) {
        None => Ok(least),
        Some(value) if value.is_finite() && value >= least => Ok(value),
        Some(value) => Err(
            Error::new(ErrorCode::OutOfRange, format!("{path}.values.{name}"))
                .range(f64::from(least), f64::from(f32::MAX))
                .got(value),
        ),
    }
}

fn lightmap_of(material: &Material, path: &str) -> Result<Option<TextureRef>, Error> {
    let Some(map) = material.map(LIGHTMAP_MAP) else {
        return Ok(None);
    };
    if map.tex_coord != 1 {
        return Err(Error::new(
            ErrorCode::InvalidMaterial,
            format!("{path}.maps.{LIGHTMAP_MAP}.tex_coord"),
        )
        .got(map.tex_coord));
    }
    Ok(Some(map))
}

fn curve_of(material: &Material, path: &str) -> Result<f32, Error> {
    let curve = material.value(LIGHTMAP_CURVE).unwrap_or(1.0);
    if curve != 1.0 && curve != 2.0 {
        return Err(Error::new(
            ErrorCode::InvalidValue,
            format!("{path}.values.{LIGHTMAP_CURVE}"),
        )
        .variants(["1", "2"])
        .got(curve));
    }
    Ok(curve)
}

fn kept<T>(problems: &mut Vec<Error>, found: Result<T, Error>, instead: T) -> T {
    found.unwrap_or_else(|problem| {
        problems.push(problem);
        instead
    })
}

fn broken_of(
    material: &Material,
    path: &str,
    problems: &mut Vec<Error>,
) -> (Option<TextureRef>, [f32; 4]) {
    let color = kept(
        problems,
        value_at_least(material, path, MACRO_COLOR, 0.0),
        0.0,
    );
    let rough = kept(
        problems,
        value_at_least(material, path, MACRO_ROUGH, 0.0),
        0.0,
    );
    if color + rough <= 0.0 {
        return (None, [0.0; 4]);
    }
    let map = material.map(MACRO_MAP);
    if map.is_none() {
        problems.push(missing(path, &format!("maps.{MACRO_MAP}")));
    }
    let tile = match material.value(MACRO_TILE) {
        None => {
            problems.push(missing(path, &format!("values.{MACRO_TILE}")));
            1.0
        }
        Some(tile) if !(tile.is_finite() && tile > 0.0) => {
            problems.push(
                Error::new(ErrorCode::OutOfRange, format!("{path}.values.{MACRO_TILE}"))
                    .expected(Expected::Above)
                    .got(tile),
            );
            1.0
        }
        Some(tile) => tile,
    };
    (map, [color, rough, 1.0 / tile, 0.0])
}

fn cutoff_of(material: &Material, path: &str) -> Result<Option<f32>, Error> {
    match material.alpha_mode() {
        AlphaMode::Opaque => Ok(None),
        AlphaMode::Mask { cutoff } => Ok(Some(cutoff)),
        AlphaMode::Blend | AlphaMode::Additive => Err(Error::new(
            ErrorCode::InvalidMaterial,
            format!("{path}.alpha_mode"),
        )
        .variants(["opaque", "mask"])),
    }
}

pub(super) fn problems_of(material: &Material, path: &str, detail: bool) -> Vec<Error> {
    mapped(material, path, detail).err().unwrap_or_default()
}

fn mapped(material: &Material, path: &str, detail: bool) -> Result<Mapped, Vec<Error>> {
    let mut problems = Vec::new();
    let plain = SURFACE_MAPS.iter().all(|name| material.map(name).is_none());
    let mut maps = Vec::with_capacity(MAPS.len());
    for (name, _, _) in MAPS.iter().filter(|_| !plain) {
        match material.map(name) {
            None => problems.push(
                missing(path, &format!("maps.{name}"))
                    .variants(material.maps().iter().map(|(known, _)| known.clone())),
            ),
            Some(map) if map.tex_coord != 0 => problems.push(
                Error::new(
                    ErrorCode::InvalidMaterial,
                    format!("{path}.maps.{name}.tex_coord"),
                )
                .got(map.tex_coord),
            ),
            Some(map) => maps.push(map),
        }
    }
    let baked = kept(&mut problems, lightmap_of(material, path), None);
    let curve = kept(&mut problems, curve_of(material, path), 1.0);
    let scale = match material.value(LIGHTMAP_SCALE) {
        Some(scale) => scale,
        None if baked.is_none() && !detail => 1.0,
        None => {
            problems.push(missing(path, &format!("values.{LIGHTMAP_SCALE}")));
            1.0
        }
    };
    let (broken, strengths) = broken_of(material, path, &mut problems);
    let cutoff = kept(&mut problems, cutoff_of(material, path), None);
    let strength = match material.value(EMISSIVE_STRENGTH) {
        None => 1.0,
        Some(_) => kept(
            &mut problems,
            value_at_least(material, path, EMISSIVE_STRENGTH, 0.0),
            1.0,
        ),
    };
    let emissive = material.emissive().map(|value| value * strength);
    let mirror = kept(
        &mut problems,
        value_at_least(material, path, MIRROR, 0.0),
        0.0,
    );
    if mirror > 1.0 {
        problems.push(
            Error::new(ErrorCode::OutOfRange, format!("{path}.values.{MIRROR}"))
                .range(0.0, 1.0)
                .got(mirror),
        );
    }
    if !problems.is_empty() {
        return Err(problems);
    }
    Ok(Mapped {
        plain,
        maps,
        baked,
        broken,
        surface: LevelSurface {
            tint: material.color(),
            glow: rgb(emissive, 0.0),
            baked: [scale, curve, material.normal_scale(), 0.0],
            broken: strengths,
            cut: [
                cutoff.unwrap_or(0.0),
                mirror,
                if plain || material.normal_scale() == 0.0 {
                    1.0
                } else {
                    0.0
                },
                0.0,
            ],
        },
        cutout: cutoff.is_some(),
        cull: face(material.cull()),
    })
}

fn picture_shape(material: &Material, path: &str) -> Result<([f32; 4], f32), Error> {
    let mut found = [0.0_f32; 3];
    for (at, name) in [SKY_BELOW, SKY_ABOVE, SKY_FADE].into_iter().enumerate() {
        let value = material
            .value(name)
            .ok_or_else(|| missing(path, &format!("values.{name}")))?;
        found[at] = value;
    }
    let [below, above, fade] = found;
    let haze = material.value(SKY_HAZE).unwrap_or(0.0);
    let span = above - below;
    if !(below >= -90.0 && above <= 90.0 && span > 0.0) {
        return Err(
            Error::new(ErrorCode::OutOfRange, format!("{path}.values.{SKY_ABOVE}"))
                .range(f64::from(below), 90.0)
                .got(above),
        );
    }
    if !(fade > 0.0 && fade <= span * 0.5) {
        return Err(
            Error::new(ErrorCode::OutOfRange, format!("{path}.values.{SKY_FADE}"))
                .range(0.0, f64::from(span * 0.5))
                .got(fade),
        );
    }
    if !(0.0..=1.0).contains(&haze) {
        return Err(
            Error::new(ErrorCode::OutOfRange, format!("{path}.values.{SKY_HAZE}"))
                .range(0.0, 1.0)
                .got(haze),
        );
    }
    let turns = material.value(SKY_TURNS).unwrap_or(1.0);
    if !(turns >= 1.0 && turns.fract() == 0.0) {
        return Err(
            Error::new(ErrorCode::OutOfRange, format!("{path}.values.{SKY_TURNS}"))
                .expected(Expected::Integer)
                .got(turns),
        );
    }
    Ok(([below, above, fade, turns], haze))
}

pub(super) fn srgb_in_linear(
    assets: &Assets,
    map: TextureRef,
    path: &str,
    name: &str,
) -> Option<Error> {
    let space = assets.get(map.texture).ok()?.color_space();
    (space == ColorSpace::Srgb).then(|| {
        Error::new(ErrorCode::InvalidTexture, format!("{path}.maps.{name}"))
            .expected(Expected::Matching)
            .got("the file states Srgb, the slot reads linear")
    })
}

fn linear_refused(assets: &Assets, mapped: &Mapped, path: &str) -> Vec<Error> {
    let surface = SURFACE_MAPS.iter().zip(&mapped.maps).skip(1);
    let extra = [(LIGHTMAP_MAP, mapped.baked), (MACRO_MAP, mapped.broken)];
    surface
        .map(|(name, map)| (*name, *map))
        .chain(extra.into_iter().filter_map(|(name, map)| map.map(|map| (name, map))))
        .filter_map(|(name, map)| srgb_in_linear(assets, map, path, name))
        .collect()
}

impl GpuPart for LevelGpu {
    type Reads = (
        Res<'static, Lighting>,
        Res<'static, DrawnTable>,
        Res<'static, Assets>,
        (Res<'static, Pieces>, Res<'static, SetIndex>),
    );
    const NAME: &'static str = "level";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let light = cx.buffer("level_light", Lifetime::Kept);
        let surfaces = cx.buffer("level_surfaces", Lifetime::Kept);
        let unbaked = StandIn::install(cx, "level_unbaked", [0; 4]);
        let plain = PLAIN.map(|(name, texel)| StandIn::install(cx, name, texel));
        let group = cx.batch_group(material::MODULE, 1, &[(material::LEVEL_SURFACE, surfaces)])?;
        let sky_group = cx.batch_group(picture::MODULE, 1, &[])?;
        let depth_scene = Scene::named(DEPTH_PASS);
        let depth_pass = cx.scene_bit(depth_scene);
        let depth_back =
            cx.render_pipeline(depth_scene, &depth_described(Some(wgpu::Face::Back)), &[])?;
        let depth_double = cx.render_pipeline(depth_scene, &depth_described(None), &[])?;
        let scene = Scene::named(COLOR_PASS);
        let lit = [(light::LEVEL_LIGHT, Bind::Buffer(light))];
        let per_batch = PerBatch {
            state: true,
            cull: true,
            ..PerBatch::NONE
        };
        let level = cx.render_pipeline(
            scene,
            &described(
                (map::VS_LEVEL, map::FS_LEVEL),
                &LEVEL_STREAMS,
                true,
                per_batch,
                &[],
            ),
            &lit,
        )?;
        let detail = cx.render_pipeline(
            scene,
            &described(
                (map::VS_LEVEL_DETAIL, map::FS_LEVEL),
                &DETAIL_STREAMS,
                true,
                per_batch,
                &[("LEVEL_DETAIL", 1.0)],
            ),
            &lit,
        )?;
        let sky = cx.render_pipeline(
            scene,
            &described((sky::VS_SKY, sky::FS_SKY), &[], false, PerBatch::NONE, &[]),
            &lit,
        )?;
        let picture = cx.render_pipeline(
            scene,
            &described(
                (picture::VS_SKY_PICTURE, picture::FS_SKY_PICTURE),
                &[],
                false,
                PerBatch::NONE,
                &[],
            ),
            &lit,
        )?;
        Ok(Self {
            depth_pass,
            depth_back,
            depth_double,
            level,
            detail,
            sky,
            picture,
            group,
            sky_group,
            light,
            surfaces,
            unbaked,
            plain,
            sent: None,
            binds: MaterialBinds::default(),
            free: Vec::new(),
            slots: 0,
            pictured: MaterialBinds::default(),
            ready: false,
            pieces: Pieces::default(),
            filtering: Filtering::default(),
            detail_limit: DetailLimit::default(),
            held_small: BTreeSet::new(),
            sets: SetIndex::default(),
            sets_on: false,
            batched: Vec::new(),
            used: BTreeMap::new(),
            sky_bound: None,
            has_double_depth: false,
            again: false,
        })
    }

    fn prepare(
        &mut self,
        (lighting, table, assets, (pieces, sets)): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.detail_limit = lighting.detail;
        self.sets_on = lighting.visible_sets;
        if sets.is_changed() {
            self.sets = sets.clone();
        }
        if pieces.is_changed() {
            self.pieces = pieces.clone();
        }
        let mut fresh = self.unbaked.written(cx);
        for plain in &self.plain {
            fresh |= plain.written(cx);
        }
        if fresh {
            self.binds.clear();
            self.free.clear();
            self.slots = 0;
            self.pictured.clear();
            self.ready = true;
            self.has_double_depth = false;
            self.again = true;
        }
        let mut changed = Vec::new();
        if self.filtering != lighting.filtering {
            self.filtering = lighting.filtering;
            self.pictured.clear();
            self.again = true;
            changed.extend(self.binds.each_bound().map(|(handle, _)| handle));
        }
        if self.ready {
            let again = std::mem::take(&mut self.again);
            self.sync(&table, &assets, cx, (&mut changed, again));
            let before = self.shown().map(|shown| shown.map);
            self.sync_sky(&table, &assets, cx, again);
            if self.shown().map(|shown| shown.map) != before {
                changed.extend(self.binds.each_bound().map(|(handle, _)| handle));
                changed.sort_unstable();
                changed.dedup();
            }
        }
        let stride = (size_of::<LevelSurface>() as u64).next_multiple_of(RANGE_ALIGN);
        let bytes = u64::from(self.slots.max(1).next_power_of_two()) * stride;
        if cx.allocate_buffer(self.surfaces, bytes) == Allocation::Fresh {
            changed = self.binds.each_bound().map(|(handle, _)| handle).collect();
        }
        for handle in changed {
            self.send(handle, stride, cx);
        }
        let shown = light_of(&lighting, self.shown());
        let bytes = size_of::<LevelLight>() as u64;
        if cx.allocate_buffer(self.light, bytes) == Allocation::Fresh || self.sent != Some(shown) {
            cx.write(self.light, 0, &[shown]);
            self.sent = Some(shown);
        }
    }

    fn view(&mut self, facts: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        let set = self.sets_on.then(|| self.sets.at(facts.eye)).flatten();
        let scale = pixel_scale(facts);
        let pieces = &self.pieces;
        let mut sized = SizeSkip {
            limit: self.detail_limit,
            held: Some(&mut self.held_small),
        };
        let mut verdict = |batch: BatchOf<'_>, _| {
            let piece = pieces.single(batch.geometry, batch.instances);
            let near = chunks::near(piece, batch.geometry, facts, set)?;
            let small = (batch.geometry)
                .zip(piece.and_then(|piece| piece.largest_m))
                .is_some_and(|sized_piece| !sized.shown(sized_piece, near, scale));
            (!small).then_some(near.to_bits())
        };
        cx.judge(LEVEL, &mut verdict);
        cx.judge(LEVEL_DETAIL, &mut verdict);
    }
}

impl LevelGpu {
    fn depth_cull(&self, material: Option<Asset<Material>>) -> Option<Option<wgpu::Face>> {
        let bound = self.binds.bound(material?)?;
        depth_pass_cull(&bound.mapped)
    }

    fn draw_depth(&self, draw: &mut Draw<'_, Self, Scene>) {
        draw.batches_matching(&self.depth_back, 0..0, DrawOrder::Sorted, |batch| {
            self.depth_cull(batch.material) == Some(Some(wgpu::Face::Back))
        });
        if self.has_double_depth {
            draw.batches_matching(&self.depth_double, 0..0, DrawOrder::Sorted, |batch| {
                self.depth_cull(batch.material) == Some(None)
            });
        }
    }

    pub(super) fn draw_level(&self, draw: &mut Draw<'_, Self, Scene>) {
        if draw.pass() == self.depth_pass {
            self.draw_depth(draw);
        } else {
            draw.batches(&self.level, 0..0, DrawOrder::Grouped);
        }
    }

    pub(super) fn draw_detail(&self, draw: &mut Draw<'_, Self, Scene>) {
        if draw.pass() == self.depth_pass {
            self.draw_depth(draw);
        } else {
            draw.batches(&self.detail, 0..0, DrawOrder::Grouped);
        }
    }

    pub(super) fn draw_field(&self, _draw: &mut Draw<'_, Self, Scene>) {}

    fn shown(&self) -> Option<Shaped> {
        self.pictured.each_bound().next().map(|(_, shown)| *shown)
    }

    pub(super) fn draw_sky(&self, draw: &mut Draw<'_, Self, Scene>) {
        let shown = self.pictured.each_bound().next().map(|(material, _)| material);
        let pipeline = if shown.is_some() {
            &self.picture
        } else {
            &self.sky
        };
        draw.draw(
            pipeline,
            Shape::Pulled {
                vertices: 0..BOX_VERTICES,
                instances: 0..1,
                key: shown,
            },
        );
    }

    fn sync(
        &mut self,
        table: &DrawnTable,
        assets: &Assets,
        cx: &mut Prepare<'_, Self>,
        (changed, again): (&mut Vec<Asset<Material>>, bool),
    ) {
        let batched = || {
            table
                .batches(LEVEL)
                .chain(table.batches(LEVEL_DETAIL))
                .map(|batch| (batch.kind == LEVEL_DETAIL, batch.material))
        };
        let moved = !batched().eq(self.batched.iter().copied());
        if moved {
            self.batched = batched().collect();
            let detail: BTreeSet<Asset<Material>> = self
                .batched
                .iter()
                .filter_map(|&(detail, material)| material.filter(|_| detail))
                .collect();
            self.used = self
                .batched
                .iter()
                .filter_map(|&(_, material)| material)
                .map(|handle| (handle, detail.contains(&handle)))
                .collect();
        }
        if !(self.binds.changed(assets) | moved | again) {
            return;
        }
        for (handle, bound) in self.binds.dropped(assets, &self.used) {
            if let Some(Bound { slot, .. }) = bound {
                self.free.push(slot);
            }
            cx.clear_batch(self.group, Some(handle));
        }
        for (handle, revision, detail, material) in self.binds.fresh(assets, &self.used) {
            self.bind(handle, (revision, detail), material, assets, cx, changed);
        }
        self.has_double_depth = self
            .binds
            .each_bound()
            .any(|(_, bound)| depth_pass_cull(&bound.mapped) == Some(None));
    }

    fn bind(
        &mut self,
        handle: Asset<Material>,
        (revision, detail): (u64, bool),
        material: &Material,
        assets: &Assets,
        cx: &mut Prepare<'_, Self>,
        changed: &mut Vec<Asset<Material>>,
    ) {
        let path = format!("$.materials.{}", handle.slot());
        let mapped = match mapped(material, &path, detail) {
            Ok(mapped) => mapped,
            Err(problems) => {
                self.refuse((handle, revision, detail), (problems, &[]), assets, cx);
                return;
            }
        };
        let textures: Vec<TextureRef> = mapped
            .maps
            .iter()
            .copied()
            .chain(mapped.baked)
            .chain(mapped.broken)
            .collect();
        if textures.iter().any(|map| !cx.texture_resident(map.texture)) {
            self.again = true;
            return;
        }
        let refused = linear_refused(assets, &mapped, &path);
        if !refused.is_empty() {
            self.refuse((handle, revision, detail), (refused, &textures), assets, cx);
            return;
        }
        let slot = match self.binds.bound(handle) {
            Some(bound) => bound.slot,
            None => self.free.pop().unwrap_or_else(|| {
                self.slots += 1;
                self.slots - 1
            }),
        };
        let held = (handle, revision, detail);
        self.binds
            .keep(assets, held, &textures, Some(Bound { slot, mapped }));
        changed.push(handle);
    }

    fn refuse(
        &mut self,
        held: (Asset<Material>, u64, bool),
        (problems, textures): (Vec<Error>, &[TextureRef]),
        assets: &Assets,
        cx: &mut Prepare<'_, Self>,
    ) {
        for problem in problems {
            cx.report(problem);
        }
        if let Some(Bound { slot, .. }) = self.binds.keep(assets, held, textures, None) {
            self.free.push(slot);
        }
        cx.clear_batch(self.group, Some(held.0));
    }

    fn send(&self, handle: Asset<Material>, stride: u64, cx: &mut Prepare<'_, Self>) {
        let Some(Bound { slot, mapped }) = self.binds.bound(handle) else {
            return;
        };
        let offset = u64::from(*slot) * stride;
        cx.write(self.surfaces, offset, &[mapped.surface]);
        let mut entries = Vec::with_capacity(2 * (MAPS.len() + 3) + 1);
        if mapped.plain {
            for ((_, texture, sampler), plain) in MAPS.iter().zip(&self.plain) {
                entries.push((*texture, GroupEntry::Texture(plain.texture())));
                entries.push((*sampler, GroupEntry::AssetSampler(CLAMP)));
            }
        }
        let filtering = &self.filtering;
        let surface = [filtering.base_color, filtering.normal, filtering.orm];
        for (((_, texture, sampler), map), filter) in MAPS.iter().zip(&mapped.maps).zip(surface) {
            entries.push((*texture, GroupEntry::Asset(map.texture)));
            entries.push((
                *sampler,
                GroupEntry::AssetSampler(filter.applied(map.sampler)),
            ));
        }
        let extra = [
            (
                mapped.baked,
                material::LEVEL_LIGHTMAP,
                material::LEVEL_LIGHTMAP_SAMPLER,
                filtering.lightmap,
            ),
            (
                mapped.broken,
                material::LEVEL_MACRO_MAP,
                material::LEVEL_MACRO_SAMPLER,
                filtering.macro_map,
            ),
            (
                self.shown().map(|shown| shown.map),
                material::LEVEL_SKY_MAP,
                material::LEVEL_SKY_SAMPLER,
                filtering.sky_picture,
            ),
        ];
        for (map, texture, sampler, chosen) in extra {
            let (bound, filter) = match map {
                Some(map) => (GroupEntry::Asset(map.texture), chosen.applied(map.sampler)),
                None => (GroupEntry::Texture(self.unbaked.texture()), CLAMP),
            };
            entries.push((texture, bound));
            entries.push((sampler, GroupEntry::AssetSampler(filter)));
        }
        entries.push((
            material::LEVEL_SURFACE,
            GroupEntry::Range {
                buffer: self.surfaces,
                offset,
                size: size_of::<LevelSurface>() as u64,
            },
        ));
        let prepassed = self.depth_pass.is_some() && depth_pass_cull(mapped).is_some();
        let variant = Variant {
            constants: vec![
                (
                    "LEVEL_BAKED".to_owned(),
                    f64::from(u8::from(mapped.baked.is_some())),
                ),
                (
                    "LEVEL_CUTOUT".to_owned(),
                    f64::from(u8::from(mapped.cutout)),
                ),
                (
                    "LEVEL_BROKEN".to_owned(),
                    f64::from(u8::from(mapped.broken.is_some())),
                ),
                (
                    "LEVEL_FLAT".to_owned(),
                    f64::from(u8::from(mapped.surface.cut[2] > 0.5)),
                ),
                (
                    "LEVEL_MIRROR".to_owned(),
                    f64::from(u8::from(mapped.surface.cut[1] > 0.0)),
                ),
                (
                    "LEVEL_BAKED_SQUARED".to_owned(),
                    f64::from(u8::from(mapped.surface.baked[1] > 1.5)),
                ),
                (
                    "LEVEL_GLOW".to_owned(),
                    f64::from(u8::from(
                        mapped.surface.glow[0] != 0.0
                            || mapped.surface.glow[1] != 0.0
                            || mapped.surface.glow[2] != 0.0,
                    )),
                ),
            ],
            depth_write: prepassed.then_some(false),
            depth_compare: prepassed.then_some(wgpu::CompareFunction::Equal),
            cull: mapped.cull,
            ..Variant::default()
        };
        if let Err(error) = cx.set_batch(self.group, Some(handle), &entries, Some(variant)) {
            cx.report(error);
        }
    }

    fn sync_sky(
        &mut self,
        table: &DrawnTable,
        assets: &Assets,
        cx: &mut Prepare<'_, Self>,
        again: bool,
    ) {
        let sky = table.batches(SKY).find_map(|batch| batch.material);
        let moved = std::mem::replace(&mut self.sky_bound, sky) != sky;
        if !(self.pictured.changed(assets) | moved | again) {
            return;
        }
        let wanted: BTreeMap<Asset<Material>, ()> =
            sky.map(|handle| (handle, ())).into_iter().collect();
        for (old, _) in self.pictured.dropped(assets, &wanted) {
            cx.clear_batch(self.sky_group, Some(old));
        }
        let Some((handle, revision, (), material)) =
            self.pictured.fresh(assets, &wanted).into_iter().next()
        else {
            return;
        };
        if self.pictured.forget(handle) {
            cx.clear_batch(self.sky_group, Some(handle));
        }
        let path = format!("$.materials.{}", handle.slot());
        let made = material
            .map(PICTURE_MAP)
            .ok_or_else(|| missing(&path, &format!("maps.{PICTURE_MAP}")))
            .and_then(|map| picture_shape(material, &path).map(|shape| (map, shape)));
        let (map, (shape, haze)) = match made {
            Ok(made) => made,
            Err(problem) => {
                cx.report(problem);
                self.pictured.keep(assets, (handle, revision, ()), &[], None);
                return;
            }
        };
        if !cx.texture_resident(map.texture) {
            self.again = true;
            return;
        }
        let entries = [
            (picture::SKY_PICTURE_MAP, GroupEntry::Asset(map.texture)),
            (
                picture::SKY_PICTURE_SAMPLER,
                GroupEntry::AssetSampler(self.filtering.sky_picture.applied(map.sampler)),
            ),
        ];
        let shown = match cx.set_batch(self.sky_group, Some(handle), &entries, None) {
            Ok(()) => Some(Shaped { map, shape, haze }),
            Err(error) => {
                cx.report(error);
                None
            }
        };
        self.pictured
            .keep(assets, (handle, revision, ()), &[map], shown);
    }
}
