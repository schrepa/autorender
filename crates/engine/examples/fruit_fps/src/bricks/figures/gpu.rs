use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_core::{
    AlphaMode, Asset, Assets, ColorSpace, Cull, Drawn, Error, ErrorCode, Expected, Material,
    TextureRef, Var, VertexStream, gpu_struct,
};
use engine_drawing::{MaterialBinds, StandIn};
use engine_meshpack::{BASE, SKIN};
use engine_render::{
    Allocation, BatchGroup, Bind, Buffer, DepthUse, Draw, DrawOrder, GpuPart, GroupEntry, Install,
    Lifetime, PerBatch, Pipeline, Prepare, RenderPipeline, Scene, Variant, wgpu,
};
use engine_rig::PaletteRow;

use super::shaders::figures::{figure, ground, light, material, reveal};
use crate::names::COLOR_PASS;
use crate::play::light::{CLAMP, Light, LightField, Lighting, REPEAT};
use crate::play::puppet::{Puppet, ShadeMaps};
use crate::play::skinned::{Figure, FigurePalette};

gpu_struct! {
    pub struct FigureLight {
        pub direction: [f32; 4],
        pub sun: [f32; 4],
        pub sky: [f32; 4],
        pub ground: [f32; 4],
        pub horizon: [f32; 4],
        pub fog: [f32; 4],
        pub field: [f32; 4],
        pub field_shape: [f32; 4],
    }
}

const ROW_BYTES: u64 = size_of::<PaletteRow>() as u64;

const FIGURE_STREAMS: [VertexStream; 2] = [BASE, SKIN];

const GROUND_STREAMS: [VertexStream; 1] = [BASE];

const MAPS: [(&str, Var, Var); 3] = [
    (
        "base_color",
        material::BASE_COLOR_MAP,
        material::BASE_COLOR_SAMPLER,
    ),
    ("normal", material::NORMAL_MAP, material::NORMAL_SAMPLER),
    ("orm", material::ORM_MAP, material::ORM_SAMPLER),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lights {
    None,
    One,
    Several(usize),
}

pub(super) struct FiguresGpu {
    figure: Pipeline<Self, Scene>,
    ground: Pipeline<Self, Scene>,
    revealed: Pipeline<Self, Scene>,
    group: BatchGroup<Self>,
    light: Buffer<Self>,
    palette: Buffer<Self>,
    sent: Option<FigureLight>,
    lights: Option<Lights>,
    binds: MaterialBinds<(), ()>,
    again: bool,
    unlit: StandIn<Self>,
    field_bound: Option<Option<TextureRef>>,
    maps: Option<ShadeMaps>,
}

fn pipeline<'a>(vertex: engine_core::Entry, vertices: &'a [VertexStream]) -> RenderPipeline<'a> {
    RenderPipeline {
        vertex,
        fragment: Some(light::FS_FIGURES),
        vertices,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..wgpu::PrimitiveState::default()
        },
        depth: DepthUse::Compare {
            compare: wgpu::CompareFunction::GreaterEqual,
            write: true,
            bias: wgpu::DepthBiasState::default(),
        },
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets: &[(None, wgpu::ColorWrites::ALL)],
        constants: &[],
        per_batch: PerBatch {
            cull: true,
            ..PerBatch::NONE
        },
    }
}

fn face(cull: Cull) -> Option<wgpu::Face> {
    match cull {
        Cull::Back => Some(wgpu::Face::Back),
        Cull::Front => Some(wgpu::Face::Front),
        Cull::None => None,
    }
}

fn light_of(lights: &[(Entity, Light)]) -> FigureLight {
    let Some((_, light)) = lights.first() else {
        return FigureLight::default();
    };
    let [x, y, z] = light.direction;
    let [r, g, b] = light.sun;
    let rgb = |[r, g, b]: [f32; 3]| [r, g, b, 0.0];
    FigureLight {
        direction: [x, y, z, 1.0],
        sun: [r, g, b, 0.0],
        sky: rgb(light.sky),
        ground: rgb(light.ground),
        ..FigureLight::default()
    }
}

fn scene_lit(shown: FigureLight, lighting: Option<&Lighting>) -> FigureLight {
    let Some(lighting) = lighting else {
        return FigureLight {
            sky: [shown.sky[0], shown.sky[1], shown.sky[2], 1.0],
            ground: [shown.ground[0], shown.ground[1], shown.ground[2], 1.0],
            ..shown
        };
    };
    let [sun_r, sun_g, sun_b, _] = shown.sun;
    let [sky_r, sky_g, sky_b, _] = shown.sky;
    let [ground_r, ground_g, ground_b, _] = shown.ground;
    let horizon = lighting.sky_look.map_or(lighting.sky, |sky| sky.horizon);
    let fog = lighting.fog;
    FigureLight {
        sun: [sun_r, sun_g, sun_b, lighting.wrap],
        sky: [sky_r, sky_g, sky_b, lighting.ambient.sky_scale],
        ground: [ground_r, ground_g, ground_b, lighting.ambient.ground_scale],
        horizon: [
            horizon[0],
            horizon[1],
            horizon[2],
            fog.map_or(0.0, |fog| fog.most),
        ],
        fog: [
            fog.map_or(0.0, |fog| fog.start_m),
            fog.map_or(0.0, |fog| fog.density_per_m),
            0.0,
            0.0,
        ],
        ..shown
    }
}

fn revealed_pipeline() -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: reveal::VS_REVEALED,
        fragment: Some(reveal::FS_REVEALED),
        vertices: &FIGURE_STREAMS,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..wgpu::PrimitiveState::default()
        },
        depth: DepthUse::Compare {
            compare: wgpu::CompareFunction::Less,
            write: false,
            bias: wgpu::DepthBiasState::default(),
        },
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets: &[(
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            wgpu::ColorWrites::ALL,
        )],
        constants: &[],
        per_batch: PerBatch {
            cull: true,
            ..PerBatch::NONE
        },
    }
}

fn lights_problem(lights: Lights) -> Option<Error> {
    match lights {
        Lights::One => None,
        Lights::None => Some(
            Error::new(ErrorCode::MissingComponent, "$.entities.light")
                .expected(Expected::Present)
                .got("figures draw unlit"),
        ),
        Lights::Several(count) => {
            Some(Error::new(ErrorCode::NameTwice, "$.entities.light").got(count))
        }
    }
}

impl GpuPart for FiguresGpu {
    type Reads = (
        Res<'static, FigurePalette>,
        Query<'static, 'static, (Entity, &'static Light)>,
        Res<'static, Assets>,
        (Option<Res<'static, Lighting>>, Res<'static, Puppet>),
        Query<'static, 'static, &'static Figure, With<Drawn>>,
    );
    const NAME: &'static str = "figures";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let light = cx.buffer("figures_light", Lifetime::Kept);
        let palette = cx.buffer("figures_palette", Lifetime::Kept);
        let group = cx.batch_group(material::MODULE, 1, &[])?;
        let unlit = StandIn::install(cx, "figures_unlit_field", [0; 4]);
        let scene = Scene::named(COLOR_PASS);
        let lit = (light::FIGURE_LIGHT, Bind::Buffer(light));
        let figure = cx.render_pipeline(
            scene,
            &pipeline(figure::VS_FIGURE, &FIGURE_STREAMS),
            &[lit.clone(), (figure::FIGURE_PALETTE, Bind::Buffer(palette))],
        )?;
        let ground =
            cx.render_pipeline(scene, &pipeline(ground::VS_GROUND, &GROUND_STREAMS), &[lit])?;
        let revealed = cx.render_pipeline(
            scene,
            &revealed_pipeline(),
            &[(figure::FIGURE_PALETTE, Bind::Buffer(palette))],
        )?;
        Ok(Self {
            figure,
            ground,
            revealed,
            group,
            light,
            palette,
            sent: None,
            lights: None,
            binds: MaterialBinds::default(),
            again: false,
            unlit,
            field_bound: None,
            maps: None,
        })
    }

    fn prepare(
        &mut self,
        (book, lights, assets, (lighting, puppet), figures): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let field = lighting.as_deref().and_then(|lighting| lighting.field);
        let bound = self.bind_field(field.map(|field| field.map), cx);
        let shade = puppet.table.as_ref().map(|table| table.shade);
        let floor = shade.map_or(0.0, |shade| shade.indirect_floor);
        self.filter(shade.map(|shade| shade.maps), cx);
        if let Some(maps) = self.maps {
            self.sync(&assets, maps, cx);
        }
        let mut listed: Vec<(Entity, Light)> = lights
            .iter()
            .map(|(entity, light)| (entity, *light))
            .collect();
        listed.sort_unstable_by_key(|(entity, _)| *entity);
        let counted = match listed.len() {
            0 => Lights::None,
            1 => Lights::One,
            count => Lights::Several(count),
        };
        if self.lights != Some(counted) {
            self.lights = Some(counted);
            if let Some(problem) = lights_problem(counted) {
                cx.report(problem);
            }
        }
        let shown = fielded(
            scene_lit(light_of(&listed), lighting.as_deref()),
            field.filter(|_| bound.is_some()),
            floor,
        );
        let bytes = size_of::<FigureLight>() as u64;
        if cx.allocate_buffer(self.light, bytes) == Allocation::Fresh || self.sent != Some(shown) {
            cx.write(self.light, 0, &[shown]);
            self.sent = Some(shown);
        }
        let held: usize = figures
            .iter()
            .filter_map(|figure| assets.get(figure.skeleton).ok())
            .map(|skeleton| skeleton.len())
            .sum();
        let rows = book.rows.len().max(held).max(1).next_power_of_two() as u64;
        let _ = cx.allocate_buffer(self.palette, rows * ROW_BYTES);
        let first = book.written.iter().map(|&(at, _)| at).min();
        let end = book.written.iter().map(|&(at, length)| at + length).max();
        if let (Some(first), Some(end)) = (first, end) {
            cx.write(self.palette, first as u64 * ROW_BYTES, &book.rows[first..end]);
        }
    }
}

fn fielded(shown: FigureLight, field: Option<LightField>, floor: f32) -> FigureLight {
    let Some(field) = field else {
        return shown;
    };
    let (rect, [height, scale]) = field.uniform();
    FigureLight {
        field: rect,
        field_shape: [height, scale, 1.0, floor],
        ..shown
    }
}

impl FiguresGpu {
    fn bind_field(
        &mut self,
        wanted: Option<TextureRef>,
        cx: &mut Prepare<'_, Self>,
    ) -> Option<TextureRef> {
        if self.unlit.written(cx) {
            self.field_bound = None;
        }
        let shown = wanted.filter(|map| cx.texture_resident(map.texture));
        if self.field_bound != Some(shown) {
            self.field_bound = Some(shown);
            for handle in self.binds.clear() {
                cx.clear_batch(self.group, Some(handle));
            }
            self.again = true;
        }
        shown
    }

    fn filter(&mut self, maps: Option<ShadeMaps>, cx: &mut Prepare<'_, Self>) {
        if self.maps == maps {
            return;
        }
        self.maps = maps;
        for handle in self.binds.clear() {
            cx.clear_batch(self.group, Some(handle));
        }
        self.again = true;
    }

    fn field_entries(&self) -> [(Var, GroupEntry<Self>); 2] {
        let (texture, sampler) = match self.field_bound.flatten() {
            Some(map) => (GroupEntry::Asset(map.texture), map.sampler.unwrap_or(REPEAT)),
            None => (GroupEntry::Texture(self.unlit.texture()), CLAMP),
        };
        [
            (material::FIGURE_FIELD_MAP, texture),
            (
                material::FIGURE_FIELD_SAMPLER,
                GroupEntry::AssetSampler(sampler),
            ),
        ]
    }

    pub(super) fn draw_figures(&self, draw: &mut Draw<'_, Self, Scene>) {
        draw.batches(&self.figure, 0..0, DrawOrder::Sorted);
    }

    pub(super) fn draw_revealed(&self, draw: &mut Draw<'_, Self, Scene>) {
        draw.batches(&self.revealed, 0..0, DrawOrder::Sorted);
    }

    pub(super) fn draw_ground(&self, draw: &mut Draw<'_, Self, Scene>) {
        draw.batches(&self.ground, 0..0, DrawOrder::Sorted);
    }

    fn sync(&mut self, assets: &Assets, maps: ShadeMaps, cx: &mut Prepare<'_, Self>) {
        let filters = [maps.base_color, maps.normal, maps.orm];
        if !(self.binds.changed(assets) | std::mem::take(&mut self.again)) {
            return;
        }
        let wanted: BTreeMap<Asset<Material>, ()> = assets
            .each::<Material>()
            .map(|(handle, _)| (handle, ()))
            .collect();
        for (handle, _) in self.binds.dropped(assets, &wanted) {
            cx.clear_batch(self.group, Some(handle));
        }
        for (handle, revision, (), material) in self.binds.fresh(assets, &wanted) {
            let path = format!("$.materials.{}", handle.slot());
            let maps = match mapped(material, &path) {
                Ok(maps) => maps,
                Err(problem) => {
                    self.refuse((handle, revision), (vec![problem], &[]), assets, cx);
                    continue;
                }
            };
            if maps.iter().any(|map| !cx.texture_resident(map.texture)) {
                self.again = true;
                continue;
            }
            let refused: Vec<Error> = MAPS
                .iter()
                .zip(&maps)
                .skip(1)
                .filter_map(|((name, _, _), map)| srgb_in_linear(assets, *map, &path, name))
                .collect();
            if !refused.is_empty() {
                self.refuse((handle, revision), (refused, &maps), assets, cx);
                continue;
            }
            if material.alpha_mode() != AlphaMode::Opaque {
                cx.report(
                    Error::new(ErrorCode::InvalidMaterial, format!("{path}.alpha"))
                        .got(format!("{:?}", material.alpha_mode())),
                );
            }
            let mut entries = Vec::with_capacity(2 * MAPS.len() + 2);
            for (((_, texture, sampler), map), filter) in MAPS.iter().zip(&maps).zip(filters) {
                entries.push((*texture, GroupEntry::Asset(map.texture)));
                entries.push((
                    *sampler,
                    GroupEntry::AssetSampler(filter.applied(map.sampler)),
                ));
            }
            entries.extend(self.field_entries());
            let variant = Variant {
                cull: face(material.cull()),
                ..Variant::default()
            };
            if let Err(error) = cx.set_batch(self.group, Some(handle), &entries, Some(variant)) {
                cx.report(error);
            }
            self.binds.keep(assets, (handle, revision, ()), &maps, None);
        }
    }

    fn refuse(
        &mut self,
        (handle, revision): (Asset<Material>, u64),
        (problems, textures): (Vec<Error>, &[TextureRef]),
        assets: &Assets,
        cx: &mut Prepare<'_, Self>,
    ) {
        for problem in problems {
            cx.report(problem);
        }
        self.binds.keep(assets, (handle, revision, ()), textures, None);
        cx.clear_batch(self.group, Some(handle));
    }
}

fn srgb_in_linear(assets: &Assets, map: TextureRef, path: &str, name: &str) -> Option<Error> {
    let space = assets.get(map.texture).ok()?.color_space();
    (space == ColorSpace::Srgb).then(|| {
        Error::new(ErrorCode::InvalidTexture, format!("{path}.maps.{name}"))
            .expected(Expected::Matching)
            .got("the file states Srgb, the slot reads linear")
    })
}

fn mapped(material: &Material, path: &str) -> Result<Vec<TextureRef>, Error> {
    let mut maps = Vec::with_capacity(MAPS.len());
    for (name, _, _) in MAPS {
        let map = material.map(name).ok_or_else(|| {
            Error::new(ErrorCode::MissingField, format!("{path}.maps.{name}"))
                .expected(Expected::Present)
                .variants(material.maps().iter().map(|(known, _)| known.clone()))
        })?;
        if map.tex_coord != 0 {
            return Err(Error::new(
                ErrorCode::InvalidMaterial,
                format!("{path}.maps.{name}.tex_coord"),
            )
            .got(map.tex_coord));
        }
        maps.push(map);
    }
    Ok(maps)
}
