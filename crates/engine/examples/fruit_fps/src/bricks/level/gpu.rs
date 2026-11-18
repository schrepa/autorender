use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::system::SystemParamItem;
use engine::prelude::*;
use engine_core::{
    AssetLibrary, AssetResidency, Cull, DrawnTable, Error, ErrorCode, Expected, Material,
    MaterialHandle, Sampler, TextureHandle, TextureRef, Var, gpu_struct,
};
use engine_render::{
    BatchGroup, GroupEntry, PerBatch, Scene, Texture, Variant, VertexLayout, wgpu,
};

use super::shaders::level::{light, map, material, sky};
use super::{LIGHTMAP_CURVE, LIGHTMAP_SCALE};
use crate::bricks::figures::{LEVEL, LIGHTMAP_MAP};
use crate::names::COLOR_PASS;
use crate::play::light::Lighting;

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
    }
}

const BOX_VERTICES: u32 = 36;

const TEXEL: wgpu::Extent3d = wgpu::Extent3d {
    width: 1,
    height: 1,
    depth_or_array_layers: 1,
};

const LEVEL_STREAMS: [VertexLayout; 2] = [
    VertexLayout::Geometry(engine_core::GeometryStream::Vertex),
    VertexLayout::Geometry(engine_core::GeometryStream::Extra),
];

const MAPS: [(&str, Var, Var); 3] = [
    (
        "base_color",
        material::LEVEL_BASE_COLOR_MAP,
        material::LEVEL_BASE_COLOR_SAMPLER,
    ),
    (
        "normal",
        material::LEVEL_NORMAL_MAP,
        material::LEVEL_NORMAL_SAMPLER,
    ),
    ("orm", material::LEVEL_ORM_MAP, material::LEVEL_ORM_SAMPLER),
];

struct Held {
    revision: u64,
    textures: Vec<(TextureHandle, u64)>,
}

struct Mapped {
    maps: Vec<TextureRef>,
    baked: Option<(TextureRef, f32, f32)>,
}

pub(super) struct LevelGpu {
    level: Pipeline<Self, Scene>,
    sky: Pipeline<Self, Scene>,
    group: BatchGroup<Self>,
    light: Buffer<Self>,
    unbaked: Texture<Self>,
    sent: Option<LevelLight>,
    held: BTreeMap<MaterialHandle, Held>,
    ready: bool,
}

fn face(cull: Cull) -> Option<wgpu::Face> {
    match cull {
        Cull::Back => Some(wgpu::Face::Back),
        Cull::Front => Some(wgpu::Face::Front),
        Cull::None => None,
    }
}

fn described<'a>(
    vertex: engine_core::Entry,
    fragment: engine_core::Entry,
    vertices: &'a [VertexLayout],
    write: bool,
    per_batch: PerBatch,
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
        constants: &[],
        per_batch,
    }
}

fn rgb(color: [f32; 3], w: f32) -> [f32; 4] {
    [color[0], color[1], color[2], w]
}

fn light_of(lighting: &Lighting) -> LevelLight {
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
    }
}

fn mapped(material: &Material, path: &str) -> Result<Mapped, Error> {
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
    let baked = match material.map(LIGHTMAP_MAP) {
        None => None,
        Some(map) => {
            if map.tex_coord != 1 {
                return Err(Error::new(
                    ErrorCode::InvalidMaterial,
                    format!("{path}.maps.{LIGHTMAP_MAP}.tex_coord"),
                )
                .got(map.tex_coord));
            }
            let scale = material.value(LIGHTMAP_SCALE).ok_or_else(|| {
                Error::new(
                    ErrorCode::MissingField,
                    format!("{path}.values.{LIGHTMAP_SCALE}"),
                )
                .expected(Expected::Present)
            })?;
            let curve = material.value(LIGHTMAP_CURVE).unwrap_or(1.0);
            if curve != 1.0 && curve != 2.0 {
                return Err(Error::new(
                    ErrorCode::InvalidValue,
                    format!("{path}.values.{LIGHTMAP_CURVE}"),
                )
                .variants(["1", "2"])
                .got(curve));
            }
            Some((map, scale, curve))
        }
    };
    Ok(Mapped { maps, baked })
}

impl GpuPart for LevelGpu {
    type Reads = (
        Res<'static, Lighting>,
        Res<'static, DrawnTable>,
        Option<Res<'static, AssetResidency>>,
        Option<Res<'static, AssetLibrary>>,
    );
    const NAME: &'static str = "level";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let light = cx.buffer("level_light", Lifetime::Kept);
        let unbaked = cx.texture(
            "level_unbaked",
            wgpu::TextureFormat::Rgba8Unorm,
            Lifetime::Kept,
        );
        let group = cx.batch_group(material::MODULE, 1, &[])?;
        let scene = Scene::named(COLOR_PASS);
        let lit = [(light::LEVEL_LIGHT, Bind::Buffer(light))];
        let level = cx.render_pipeline(
            scene,
            &described(
                map::VS_LEVEL,
                map::FS_LEVEL,
                &LEVEL_STREAMS,
                true,
                PerBatch {
                    cull: true,
                    ..PerBatch::NONE
                },
            ),
            &lit,
        )?;
        let sky = cx.render_pipeline(
            scene,
            &described(sky::VS_SKY, sky::FS_SKY, &[], false, PerBatch::NONE),
            &lit,
        )?;
        Ok(Self {
            level,
            sky,
            group,
            light,
            unbaked,
            sent: None,
            held: BTreeMap::new(),
            ready: false,
        })
    }

    fn prepare(
        &mut self,
        (lighting, table, residency, library): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if cx.allocate(self.unbaked, TEXEL, 1) == Allocation::Fresh {
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            };
            cx.write_texture(
                self.unbaked,
                wgpu::Origin3d::ZERO,
                &[0_u8; 4],
                layout,
                TEXEL,
            );
            self.held.clear();
            self.ready = true;
        }
        if self.ready
            && let (Some(residency), Some(library)) = (residency, library)
        {
            self.sync(&table, &residency, &library, cx);
        }
        let shown = light_of(&lighting);
        let bytes = size_of::<LevelLight>() as u64;
        if cx.allocate_buffer(self.light, bytes) == Allocation::Fresh || self.sent != Some(shown) {
            cx.write(self.light, 0, &[shown]);
            self.sent = Some(shown);
        }
    }
}

impl LevelGpu {
    pub(super) fn draw_level(&self, draw: &mut Draw<'_, Self, Scene>) {
        draw.batches(&self.level, 0..0);
    }

    pub(super) fn draw_sky(&self, draw: &mut Draw<'_, Self, Scene>) {
        draw.draw(
            &self.sky,
            Shape::Pulled {
                vertices: 0..BOX_VERTICES,
                instances: 0..1,
                key: None,
            },
        );
    }

    fn sync(
        &mut self,
        table: &DrawnTable,
        residency: &AssetResidency,
        library: &AssetLibrary,
        cx: &mut Prepare<'_, Self>,
    ) {
        let used: BTreeSet<MaterialHandle> = table
            .batches(LEVEL)
            .filter_map(|batch| batch.material)
            .collect();
        let gone: Vec<MaterialHandle> =
            self.held
                .iter()
                .filter(|(handle, held)| {
                    !used.contains(handle)
                        || !residency.material_resident(**handle)
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
                self.bind(handle, revision, material, library, cx);
            }
        }
    }

    fn bind(
        &mut self,
        handle: MaterialHandle,
        revision: u64,
        material: &Material,
        library: &AssetLibrary,
        cx: &mut Prepare<'_, Self>,
    ) {
        let path = format!("$.materials.{}", handle.slot());
        let mapped = match mapped(material, &path) {
            Ok(mapped) => mapped,
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
                return;
            }
        };
        let textures: Vec<TextureRef> = mapped
            .maps
            .iter()
            .copied()
            .chain(mapped.baked.map(|(map, _, _)| map))
            .collect();
        if textures.iter().any(|map| !cx.texture_resident(map.texture)) {
            return;
        }
        let mut entries = Vec::with_capacity(2 * (MAPS.len() + 1));
        for ((_, texture, sampler), map) in MAPS.iter().zip(&mapped.maps) {
            entries.push((*texture, GroupEntry::Asset(map.texture)));
            entries.push((*sampler, GroupEntry::AssetSampler(map.sampler)));
        }
        let (lightmap, lightmap_sampler) = match mapped.baked {
            Some((map, _, _)) => (GroupEntry::Asset(map.texture), map.sampler),
            None => (GroupEntry::Texture(self.unbaked), Sampler::CLAMP),
        };
        entries.push((material::LEVEL_LIGHTMAP, lightmap));
        entries.push((
            material::LEVEL_LIGHTMAP_SAMPLER,
            GroupEntry::AssetSampler(lightmap_sampler),
        ));
        let variant = Variant {
            constants: vec![
                (
                    "LEVEL_BAKED".to_owned(),
                    f64::from(u8::from(mapped.baked.is_some())),
                ),
                (
                    "LEVEL_LIGHTMAP_SCALE".to_owned(),
                    f64::from(mapped.baked.map_or(1.0, |(_, scale, _)| scale)),
                ),
                (
                    "LEVEL_LIGHTMAP_CURVE".to_owned(),
                    f64::from(mapped.baked.map_or(1.0, |(_, _, curve)| curve)),
                ),
            ],
            cull: face(material.cull()),
            ..Variant::default()
        };
        if let Err(error) = cx.set_batch(self.group, Some(handle), &entries, Some(variant)) {
            cx.report(error);
        }
        let textures = textures
            .iter()
            .map(|map| {
                let known = library.revision_of(map.texture).unwrap_or_default();
                (map.texture, known)
            })
            .collect();
        self.held.insert(handle, Held { revision, textures });
    }
}
