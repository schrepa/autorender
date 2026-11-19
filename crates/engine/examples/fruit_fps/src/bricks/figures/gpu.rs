use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, Cull, Error, ErrorCode, Expected,
    FIRST_NAMED_LOCATION, GeometryStream, Material, MaterialHandle, Sampler, TextureHandle,
    TextureRef, Var, gpu_struct,
};
use engine_render::{
    Allocation, BatchGroup, Bind, Buffer, DepthUse, Draw, GpuPart, GroupEntry, Install, Lifetime,
    PerBatch, Pipeline, Prepare, RenderPipeline, Scene, Texture, Variant, wgpu,
};

use super::play::FigureDrawing;
use super::shaders::figures::{figure, ground, light, material, reveal};
use super::{JOINTS, WEIGHTS};
use crate::names::COLOR_PASS;
use crate::play::light::{Light, LightField, Lighting};
use crate::play::puppet::{PuppetFile, ShadeMaps};

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

const TEXEL: wgpu::Extent3d = wgpu::Extent3d {
    width: 1,
    height: 1,
    depth_or_array_layers: 1,
};

const ROW_BYTES: u64 = 48;

const FIGURE_STREAMS: [GeometryStream; 3] = [
    GeometryStream::Vertex,
    GeometryStream::Named {
        shape: JOINTS,
        location: FIRST_NAMED_LOCATION,
    },
    GeometryStream::Named {
        shape: WEIGHTS,
        location: FIRST_NAMED_LOCATION + 1,
    },
];

const GROUND_STREAMS: [GeometryStream; 1] = [GeometryStream::Vertex];

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

struct Held {
    revision: u64,
    textures: Vec<(TextureHandle, u64)>,
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
    held: BTreeMap<MaterialHandle, Held>,
    synced: Option<u64>,
    unlit: Texture<Self>,
    field_bound: Option<Option<TextureRef>>,
    maps: Option<ShadeMaps>,
}

fn pipeline<'a>(vertex: engine_core::Entry, vertices: &'a [GeometryStream]) -> RenderPipeline<'a> {
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
        Res<'static, FigureDrawing>,
        Query<'static, 'static, (Entity, &'static Light)>,
        Option<Res<'static, AssetResidency>>,
        Option<Res<'static, AssetLibrary>>,
        (
            Option<Res<'static, Lighting>>,
            Query<'static, 'static, &'static PuppetFile>,
        ),
    );
    const NAME: &'static str = "figures";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let light = cx.buffer("figures_light", Lifetime::Kept);
        let palette = cx.buffer("figures_palette", Lifetime::Kept);
        let group = cx.batch_group(material::MODULE, 1, &[])?;
        let unlit = cx.texture(
            "figures_unlit_field",
            wgpu::TextureFormat::Rgba8Unorm,
            Lifetime::Kept,
        );
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
            held: BTreeMap::new(),
            synced: None,
            unlit,
            field_bound: None,
            maps: None,
        })
    }

    fn prepare(
        &mut self,
        (book, lights, residency, library, (lighting, files)): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let field = lighting.as_deref().and_then(|lighting| lighting.field);
        let bound = self.bind_field(field.map(|field| field.map), cx);
        let shade = files
            .iter()
            .find_map(PuppetFile::book)
            .map(|book| book.table.shade);
        let floor = shade.map_or(0.0, |shade| shade.indirect_floor);
        self.filter(shade.map(|shade| shade.maps), cx);
        if let (Some(residency), Some(library), Some(maps)) = (residency, library, self.maps) {
            self.sync(&residency, &library, maps, cx);
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
        let rows = book.rows.len().max(1).next_power_of_two() as u64;
        let _ = cx.allocate_buffer(self.palette, rows * ROW_BYTES);
        for &(at, length) in &book.written {
            cx.write(
                self.palette,
                at as u64 * ROW_BYTES,
                &book.rows[at..at + length],
            );
        }
    }
}

fn fielded(shown: FigureLight, field: Option<LightField>, floor: f32) -> FigureLight {
    let Some(field) = field else {
        return shown;
    };
    let [min_x, min_z, max_x, max_z] = field.rect;
    let across = |low: f32, high: f32| if high > low { 1.0 / (high - low) } else { 0.0 };
    FigureLight {
        field: [min_x, min_z, across(min_x, max_x), across(min_z, max_z)],
        field_shape: [field.height_m, field.scale, 1.0, floor],
        ..shown
    }
}

impl FiguresGpu {
    fn bind_field(
        &mut self,
        wanted: Option<TextureRef>,
        cx: &mut Prepare<'_, Self>,
    ) -> Option<TextureRef> {
        if cx.allocate(self.unlit, TEXEL) == Allocation::Fresh {
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            };
            cx.write_texture(self.unlit, wgpu::Origin3d::ZERO, &[0_u8; 4], layout, TEXEL);
            self.field_bound = None;
        }
        let shown = wanted.filter(|map| cx.texture_resident(map.texture));
        if self.field_bound != Some(shown) {
            self.field_bound = Some(shown);
            for handle in std::mem::take(&mut self.held).into_keys() {
                cx.clear_batch(self.group, Some(handle));
            }
            self.synced = None;
        }
        shown
    }

    fn filter(&mut self, maps: Option<ShadeMaps>, cx: &mut Prepare<'_, Self>) {
        if self.maps == maps {
            return;
        }
        self.maps = maps;
        for handle in std::mem::take(&mut self.held).into_keys() {
            cx.clear_batch(self.group, Some(handle));
        }
        self.synced = None;
    }

    fn field_entries(&self) -> [(Var, GroupEntry<Self>); 2] {
        let (texture, sampler) = match self.field_bound.flatten() {
            Some(map) => (GroupEntry::Asset(map.texture), map.sampler),
            None => (GroupEntry::Texture(self.unlit), Sampler::CLAMP),
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
        draw.batches(&self.figure, 0..0, |_| Some(0));
    }

    pub(super) fn draw_revealed(&self, draw: &mut Draw<'_, Self, Scene>) {
        draw.batches(&self.revealed, 0..0, |_| Some(0));
    }

    pub(super) fn draw_ground(&self, draw: &mut Draw<'_, Self, Scene>) {
        draw.batches(&self.ground, 0..0, |_| Some(0));
    }

    fn sync(
        &mut self,
        residency: &AssetResidency,
        library: &AssetLibrary,
        maps: ShadeMaps,
        cx: &mut Prepare<'_, Self>,
    ) {
        let filters = [maps.base_color, maps.normal, maps.orm];
        let generation = residency.revision();
        if self.synced == Some(generation) {
            return;
        }
        let gone: Vec<MaterialHandle> =
            self.held
                .iter()
                .filter(|(handle, held)| {
                    !residency.material_resident(**handle)
                        || held.textures.iter().any(|(texture, revision)| {
                            library.revision_of(*texture) != Some(*revision)
                        })
                })
                .map(|(handle, _)| *handle)
                .collect();
        for handle in gone {
            self.held.remove(&handle);
            cx.clear_batch(self.group, Some(handle));
        }
        let mut waiting = false;
        for (handle, revision) in residency.materials() {
            if self
                .held
                .get(&handle)
                .is_some_and(|held| held.revision == revision)
            {
                continue;
            }
            let Some(material) = library.material(handle) else {
                continue;
            };
            let path = format!("$.materials.{}", handle.slot());
            let maps = match mapped(material, &path) {
                Ok(maps) => maps,
                Err(problem) => {
                    cx.report(problem);
                    self.held.insert(
                        handle,
                        Held {
                            revision,
                            textures: Vec::new(),
                        },
                    );
                    cx.clear_batch(self.group, Some(handle));
                    continue;
                }
            };
            if maps.iter().any(|map| !cx.texture_resident(map.texture)) {
                waiting = true;
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
            let textures = maps
                .iter()
                .map(|map| {
                    let known = library.revision_of(map.texture).unwrap_or_default();
                    (map.texture, known)
                })
                .collect();
            self.held.insert(handle, Held { revision, textures });
        }
        if !waiting {
            self.synced = Some(generation);
        }
    }
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
