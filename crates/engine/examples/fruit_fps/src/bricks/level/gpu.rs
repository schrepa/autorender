use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::system::SystemParamItem;
use engine::prelude::*;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, Cull, DrawnTable, Error, ErrorCode, Expected,
    GeometryHandle, Material, MaterialHandle, Sampler, TextureHandle, TextureRef, Var, ViewFacts,
    gpu_struct,
};
use engine_render::{
    BatchGroup, GroupEntry, PerBatch, Scene, Texture, Variant, VertexLayout, wgpu,
};

use super::chunks::{self, Chunk, PieceSizes, Sized};
use super::sets::VisibleSets;
use super::shaders::level::{light, map, material, picture, sky};
use super::{
    COLOR_STREAM, EMISSIVE_STRENGTH, LEVEL_DETAIL, LIGHT_FIELD, LIGHTMAP_CURVE, LIGHTMAP_SCALE,
    MACRO_COLOR, MACRO_MAP, MACRO_ROUGH, MACRO_TILE, MIRROR, PICTURE_MAP, SKY, SKY_ABOVE,
    SKY_BELOW, SKY_FADE, SKY_HAZE, SKY_TURNS, SURFACE_MAPS,
};
use crate::bricks::figures::{LEVEL, LIGHTMAP_MAP};
use crate::names::COLOR_PASS;
use crate::play::light::{DetailLimit, Filtering, Lighting};
use crate::play::screen::Covered;

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

const TEXEL: wgpu::Extent3d = wgpu::Extent3d {
    width: 1,
    height: 1,
    depth_or_array_layers: 1,
};

const LEVEL_STREAMS: [VertexLayout; 2] = [
    VertexLayout::Geometry(engine_core::GeometryStream::Vertex),
    VertexLayout::Geometry(engine_core::GeometryStream::Extra),
];

const DETAIL_STREAMS: [VertexLayout; 2] = [
    VertexLayout::Geometry(engine_core::GeometryStream::Vertex),
    VertexLayout::Geometry(engine_core::GeometryStream::Named {
        shape: COLOR_STREAM,
        location: 6,
    }),
];

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

const PLAIN_TEXELS: [[u8; 4]; 3] = [
    [255, 255, 255, 255],
    [128, 128, 255, 255],
    [255, 128, 0, 255],
];

struct Held {
    revision: u64,
    detail: bool,
    textures: Vec<(TextureHandle, u64)>,
    bound: Option<Bound>,
}

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

struct Pictured {
    material: MaterialHandle,
    revision: u64,
    textures: Vec<(TextureHandle, u64)>,
    shown: Option<Shaped>,
}

pub(super) struct LevelGpu {
    level: Pipeline<Self, Scene>,
    detail: Pipeline<Self, Scene>,
    sky: Pipeline<Self, Scene>,
    picture: Pipeline<Self, Scene>,
    group: BatchGroup<Self>,
    sky_group: BatchGroup<Self>,
    light: Buffer<Self>,
    surfaces: Buffer<Self>,
    unbaked: Texture<Self>,
    plain: [Texture<Self>; 3],
    sent: Option<LevelLight>,
    held: BTreeMap<MaterialHandle, Held>,
    free: Vec<u32>,
    slots: u32,
    pictured: Option<Pictured>,
    ready: bool,
    chunks: [Vec<Chunk>; 2],
    order: [Vec<u32>; 2],
    filtering: Filtering,
    covered: bool,
    detail_limit: DetailLimit,
    held_small: BTreeSet<GeometryHandle>,
    sets: VisibleSets,
    sets_on: bool,
}

fn face(cull: Cull) -> Option<wgpu::Face> {
    match cull {
        Cull::Back => Some(wgpu::Face::Back),
        Cull::Front => Some(wgpu::Face::Front),
        Cull::None => None,
    }
}

fn described<'a>(
    (vertex, fragment): (engine_core::Entry, engine_core::Entry),
    vertices: &'a [VertexLayout],
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
    LevelLight {
        to_sun: [to_sun.x, to_sun.y, to_sun.z, lighting.wrap],
        sun: rgb(lighting.sun, f32::from(u8::from(lighting.lit))),
        zenith: rgb(
            sky.map_or(lighting.sky, |sky| sky.zenith),
            sky.map_or(1.0, |sky| sky.haze_height),
        ),
        ground: rgb(lighting.ground, lighting.ambient.black_floor),
        horizon: rgb(
            sky.map_or(lighting.sky, |sky| sky.horizon),
            fog.map_or(0.0, |fog| fog.most),
        ),
        below: rgb(sky.map_or(lighting.ground, |sky| sky.below), 0.0),
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
            0.0,
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
            0.0,
            0.0,
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

fn revisions(library: &AssetLibrary, maps: &[TextureRef]) -> Vec<(TextureHandle, u64)> {
    maps.iter()
        .map(|map| {
            let known = library.revision_of(map.texture).unwrap_or_default();
            (map.texture, known)
        })
        .collect()
}

fn stale(library: &AssetLibrary, textures: &[(TextureHandle, u64)]) -> bool {
    textures
        .iter()
        .any(|(texture, revision)| library.revision_of(*texture) != Some(*revision))
}

impl GpuPart for LevelGpu {
    type Reads = (
        Res<'static, Lighting>,
        Res<'static, DrawnTable>,
        Option<Res<'static, AssetResidency>>,
        Option<Res<'static, AssetLibrary>>,
        (
            Option<Res<'static, Covered>>,
            Res<'static, PieceSizes>,
            Res<'static, VisibleSets>,
        ),
    );
    const NAME: &'static str = "level";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let light = cx.buffer("level_light", Lifetime::Kept);
        let surfaces = cx.buffer("level_surfaces", Lifetime::Kept);
        let unbaked = cx.texture(
            "level_unbaked",
            wgpu::TextureFormat::Rgba8Unorm,
            Lifetime::Kept,
        );
        let plain = ["level_plain_base", "level_plain_normal", "level_plain_orm"]
            .map(|name| cx.texture(name, wgpu::TextureFormat::Rgba8Unorm, Lifetime::Kept));
        let group = cx.batch_group(material::MODULE, 1, &[(material::LEVEL_SURFACE, surfaces)])?;
        let sky_group = cx.batch_group(picture::MODULE, 1, &[])?;
        let scene = Scene::named(COLOR_PASS);
        let lit = [(light::LEVEL_LIGHT, Bind::Buffer(light))];
        let per_batch = PerBatch {
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
            held: BTreeMap::new(),
            free: Vec::new(),
            slots: 0,
            pictured: None,
            ready: false,
            chunks: [Vec::new(), Vec::new()],
            order: [Vec::new(), Vec::new()],
            filtering: Filtering::default(),
            covered: false,
            detail_limit: DetailLimit::default(),
            held_small: BTreeSet::new(),
            sets: VisibleSets::default(),
            sets_on: false,
        })
    }

    fn prepare(
        &mut self,
        (lighting, table, residency, library, (covered, pieces, sets)): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.covered = covered.is_some_and(|covered| covered.0);
        if self.covered {
            for kind in [LEVEL, LEVEL_DETAIL, SKY, LIGHT_FIELD] {
                cx.skip_kind(kind, "covered");
            }
        }
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        };
        self.detail_limit = lighting.detail;
        self.sets_on = lighting.visible_sets;
        if self.sets != *sets {
            self.sets = sets.clone();
        }
        for (kind, into) in [LEVEL, LEVEL_DETAIL].into_iter().zip(&mut self.chunks) {
            chunks::listed((&table, library.as_deref(), &pieces), kind, into);
        }
        let mut fresh = cx.allocate(self.unbaked, TEXEL, 1) == Allocation::Fresh;
        for texture in self.plain {
            fresh |= cx.allocate(texture, TEXEL, 1) == Allocation::Fresh;
        }
        if fresh {
            cx.write_texture(
                self.unbaked,
                wgpu::Origin3d::ZERO,
                &[0_u8; 4],
                layout,
                TEXEL,
            );
            for (texture, texel) in self.plain.into_iter().zip(PLAIN_TEXELS) {
                cx.write_texture(texture, wgpu::Origin3d::ZERO, &texel, layout, TEXEL);
            }
            self.held.clear();
            self.free.clear();
            self.slots = 0;
            self.pictured = None;
            self.ready = true;
        }
        let mut changed = Vec::new();
        if self.filtering != lighting.filtering {
            self.filtering = lighting.filtering;
            self.pictured = None;
            changed.extend(
                self.held
                    .iter()
                    .filter(|(_, held)| held.bound.is_some())
                    .map(|(handle, _)| *handle),
            );
        }
        if self.ready
            && let (Some(residency), Some(library)) = (residency, library)
        {
            self.sync(&table, &residency, &library, cx, &mut changed);
            let before = self.shown().map(|shown| shown.map);
            self.sync_sky(&table, &residency, &library, cx);
            if self.shown().map(|shown| shown.map) != before {
                changed.extend(
                    self.held
                        .iter()
                        .filter(|(_, held)| held.bound.is_some())
                        .map(|(handle, _)| *handle),
                );
                changed.sort_unstable();
                changed.dedup();
            }
        }
        let stride = (size_of::<LevelSurface>() as u64).next_multiple_of(RANGE_ALIGN);
        let bytes = u64::from(self.slots.max(1).next_power_of_two()) * stride;
        if cx.allocate_buffer(self.surfaces, bytes) == Allocation::Fresh {
            changed = self
                .held
                .iter()
                .filter(|(_, held)| held.bound.is_some())
                .map(|(handle, _)| *handle)
                .collect();
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

    fn view(&mut self, facts: &ViewFacts, _cx: &mut Prepare<'_, Self>) {
        let set = if self.sets_on {
            self.sets.at(facts.eye)
        } else {
            None
        };
        let mut sized = Sized {
            limit: self.detail_limit,
            held: Some(&mut self.held_small),
            set,
        };
        for (listed, order) in self.chunks.iter().zip(&mut self.order) {
            chunks::ordered(listed, facts, &mut sized, order);
        }
    }
}

impl LevelGpu {
    pub(super) fn draw_level(&self, draw: &mut Draw<'_, Self, Scene>) {
        self.submit(draw, &self.level, 0);
    }

    pub(super) fn draw_detail(&self, draw: &mut Draw<'_, Self, Scene>) {
        self.submit(draw, &self.detail, 1);
    }

    fn submit(
        &self,
        draw: &mut Draw<'_, Self, Scene>,
        pipeline: &Pipeline<Self, Scene>,
        at: usize,
    ) {
        if self.covered {
            return;
        }
        let pass = draw.pass();
        for &index in &self.order[at] {
            let chunk = &self.chunks[at][index as usize];
            if !chunk.drawn_in(pass) {
                continue;
            }
            let shape = match chunk.mesh {
                Some(mesh) => Shape::Mesh {
                    mesh,
                    key: chunk.material,
                    instances: chunk.instances.clone(),
                },
                None => Shape::Pulled {
                    vertices: 0..0,
                    instances: chunk.instances.clone(),
                    key: chunk.material,
                },
            };
            draw.draw(pipeline, shape);
        }
    }

    pub(super) fn draw_field(&self, _draw: &mut Draw<'_, Self, Scene>) {}

    fn shown(&self) -> Option<Shaped> {
        self.pictured.as_ref().and_then(|pictured| pictured.shown)
    }

    pub(super) fn draw_sky(&self, draw: &mut Draw<'_, Self, Scene>) {
        if self.covered {
            return;
        }
        let shown = self
            .pictured
            .as_ref()
            .filter(|pictured| pictured.shown.is_some())
            .map(|pictured| pictured.material);
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
        residency: &AssetResidency,
        library: &AssetLibrary,
        cx: &mut Prepare<'_, Self>,
        changed: &mut Vec<MaterialHandle>,
    ) {
        let detail: BTreeSet<MaterialHandle> = table
            .batches(LEVEL_DETAIL)
            .filter_map(|batch| batch.material)
            .collect();
        let used: BTreeSet<MaterialHandle> = table
            .batches(LEVEL)
            .filter_map(|batch| batch.material)
            .chain(detail.iter().copied())
            .collect();
        let gone: Vec<MaterialHandle> = self
            .held
            .iter()
            .filter(|(handle, held)| {
                !used.contains(handle)
                    || !residency.material_resident(**handle)
                    || held.detail != detail.contains(handle)
                    || stale(library, &held.textures)
            })
            .map(|(handle, _)| *handle)
            .collect();
        for handle in gone {
            if let Some(Bound { slot, .. }) = self.held.remove(&handle).and_then(|held| held.bound)
            {
                self.free.push(slot);
            }
            cx.clear_batch(self.group, Some(handle));
        }
        for (handle, revision) in residency.materials() {
            if !used.contains(&handle)
                || self
                    .held
                    .get(&handle)
                    .is_some_and(|held| held.revision == revision)
            {
                continue;
            }
            if let Some(material) = library.material(handle) {
                let detail = detail.contains(&handle);
                self.bind(handle, (revision, detail), material, library, cx, changed);
            }
        }
    }

    fn bind(
        &mut self,
        handle: MaterialHandle,
        (revision, detail): (u64, bool),
        material: &Material,
        library: &AssetLibrary,
        cx: &mut Prepare<'_, Self>,
        changed: &mut Vec<MaterialHandle>,
    ) {
        let path = format!("$.materials.{}", handle.slot());
        let mapped = match mapped(material, &path, detail) {
            Ok(mapped) => mapped,
            Err(problems) => {
                for problem in problems {
                    cx.report(problem);
                }
                if let Some(Bound { slot, .. }) =
                    self.held.remove(&handle).and_then(|held| held.bound)
                {
                    self.free.push(slot);
                }
                self.held.insert(
                    handle,
                    Held {
                        revision,
                        detail,
                        textures: Vec::new(),
                        bound: None,
                    },
                );
                cx.clear_batch(self.group, Some(handle));
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
            return;
        }
        let slot = match self.held.get(&handle).and_then(|held| held.bound.as_ref()) {
            Some(bound) => bound.slot,
            None => self.free.pop().unwrap_or_else(|| {
                self.slots += 1;
                self.slots - 1
            }),
        };
        self.held.insert(
            handle,
            Held {
                revision,
                detail,
                textures: revisions(library, &textures),
                bound: Some(Bound { slot, mapped }),
            },
        );
        changed.push(handle);
    }

    fn send(&self, handle: MaterialHandle, stride: u64, cx: &mut Prepare<'_, Self>) {
        let Some(Bound { slot, mapped }) =
            self.held.get(&handle).and_then(|held| held.bound.as_ref())
        else {
            return;
        };
        let offset = u64::from(*slot) * stride;
        cx.write(self.surfaces, offset, &[mapped.surface]);
        let mut entries = Vec::with_capacity(2 * (MAPS.len() + 3) + 1);
        if mapped.plain {
            for ((_, texture, sampler), plain) in MAPS.iter().zip(self.plain) {
                entries.push((*texture, GroupEntry::Texture(plain)));
                entries.push((*sampler, GroupEntry::AssetSampler(Sampler::CLAMP)));
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
                None => (GroupEntry::Texture(self.unbaked), Sampler::CLAMP),
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
            ],
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
        residency: &AssetResidency,
        library: &AssetLibrary,
        cx: &mut Prepare<'_, Self>,
    ) {
        let wanted = table.batches(SKY).find_map(|batch| batch.material);
        let current = self.pictured.as_ref().filter(|pictured| {
            Some(pictured.material) == wanted
                && residency.material_resident(pictured.material)
                && !stale(library, &pictured.textures)
                && residency.materials().any(|(handle, revision)| {
                    handle == pictured.material && revision == pictured.revision
                })
        });
        if current.is_some() {
            return;
        }
        if let Some(old) = self.pictured.take() {
            cx.clear_batch(self.sky_group, Some(old.material));
        }
        let Some(handle) = wanted else {
            return;
        };
        let Some(revision) = residency
            .materials()
            .find_map(|(known, revision)| (known == handle).then_some(revision))
        else {
            return;
        };
        let Some(material) = library.material(handle) else {
            return;
        };
        let path = format!("$.materials.{}", handle.slot());
        let made = material
            .map(PICTURE_MAP)
            .ok_or_else(|| missing(&path, &format!("maps.{PICTURE_MAP}")))
            .and_then(|map| picture_shape(material, &path).map(|shape| (map, shape)));
        let (map, (shape, haze)) = match made {
            Ok(made) => made,
            Err(problem) => {
                cx.report(problem);
                self.pictured = Some(Pictured {
                    material: handle,
                    revision,
                    textures: Vec::new(),
                    shown: None,
                });
                return;
            }
        };
        if !cx.texture_resident(map.texture) {
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
        self.pictured = Some(Pictured {
            material: handle,
            revision,
            textures: revisions(library, &[map]),
            shown,
        });
    }
}
