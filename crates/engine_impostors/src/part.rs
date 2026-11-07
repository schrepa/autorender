use std::collections::VecDeque;

use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::{
    AssetLibrary, AssetResidency, Cull, Cursor, Error, Events, MaterialHandle, MeshHandle,
    Readable, Readback, gpu_struct,
};
use engine_render::{
    AfterRead, Attach, Bind, Buffer, Copied, CopyRegion, DepthUse, Encode, GpuPart, Install,
    Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, SamplerDesc, Shape, Texture,
    VertexLayout, wgpu,
};

use crate::books::{
    AtlasFormat, AtlasShape, AtlasWork, ImpostorBooks, RECORD_START, ROW_ALIGNMENT, level_rows,
    readback_bytes, record_of,
};
use crate::shaders::impostors::bake;
use crate::shaders::impostors::faces::impostors as face;

pub(crate) const DEVICE: u32 = u32::MAX;
pub(crate) const MISSED: u32 = u32::MAX - 1;

const FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba8Unorm,
];
const BC7_FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    wgpu::TextureFormat::Bc7RgbaUnorm,
    wgpu::TextureFormat::Bc7RgbaUnorm,
];
const ASTC_FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    },
];
const ATLAS_LABELS: [&str; 4] = [
    "impostor_albedo",
    "impostor_emissive",
    "impostor_geometry",
    "impostor_surface",
];
const BAKE_LABELS: [&str; 4] = [
    "impostor_bake_albedo",
    "impostor_bake_emissive",
    "impostor_bake_geometry",
    "impostor_bake_surface",
];
const BAKE_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADE_SCALE: u32 = 4;
const SHADE_MOST: u32 = 2048;
const MESH: &[VertexLayout] = &[VertexLayout::Mesh];
const EXTENDED: &[VertexLayout] = &[VertexLayout::Mesh, VertexLayout::Mesh];

gpu_struct! {
    pub struct ImpostorBake {
        pub sphere: [f32; 4],
        pub flags: [u32; 4],
        pub tile: [u32; 4],
    }
}

#[derive(Resource, Default)]
pub(crate) struct BakeReads {
    pub(crate) format: Option<u32>,
    pub(crate) missed: bool,
    pub(crate) read: Vec<(u32, Vec<u8>)>,
}

impl Readable for BakeReads {
    fn arrived(&mut self, tag: u32, read: Readback<&[u8]>) {
        match tag {
            DEVICE => {
                self.format = read
                    .value
                    .get(..4)
                    .and_then(|word| <[u8; 4]>::try_from(word).ok())
                    .map(u32::from_le_bytes);
            }
            MISSED => self.missed = true,
            serial => self.read.push((serial, read.value.to_vec())),
        }
    }
}

fn format_of(features: wgpu::Features) -> AtlasFormat {
    if features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        AtlasFormat::Bc7
    } else if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
        AtlasFormat::Astc
    } else {
        AtlasFormat::Rgba8
    }
}

fn formats_of(format: AtlasFormat) -> [wgpu::TextureFormat; 4] {
    match format {
        AtlasFormat::Bc7 => BC7_FORMATS,
        AtlasFormat::Astc => ASTC_FORMATS,
        AtlasFormat::Rgba8 => FORMATS,
    }
}

fn flat(side: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: side,
        height: side,
        depth_or_array_layers: 1,
    }
}

fn bake_pipeline(
    (vertex, fragment): (engine_core::Entry, engine_core::Entry),
    (vertices, two_sided): (&'static [VertexLayout], bool),
) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex,
        fragment: Some(fragment),
        debug: None,
        vertices,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: (!two_sided).then_some(wgpu::Face::Back),
            ..Default::default()
        },
        depth: DepthUse::Write {
            bias: wgpu::DepthBiasState::default(),
        },
        blend: None,
        writes: wgpu::ColorWrites::ALL,
    }
}

struct Asked {
    serial: u32,
    mesh: MeshHandle,
    material: Option<MaterialHandle>,
    parts: Vec<(MeshHandle, MaterialHandle)>,
    bounds: (Vec3, Vec3),
    hemisphere: bool,
}

struct Drawn {
    serial: u32,
    shape: AtlasShape,
    draws: Vec<(MeshHandle, Option<MaterialHandle>, usize)>,
}

pub(crate) struct ImpostorGpu {
    format: AtlasFormat,
    atlas: [Texture<Self>; 4],
    colors: [Texture<Self>; 4],
    depth: Texture<Self>,
    shade: Texture<Self>,
    uniform: Buffer<Self>,
    kinds: Buffer<Self>,
    read: Buffer<Self>,
    shading: Pass<Self>,
    baking: Pass<Self>,
    shade_pipelines: Vec<Pipeline<Self, Own>>,
    bake_pipelines: Vec<Pipeline<Self, Own>>,
    cursor: Cursor,
    shape: Option<AtlasShape>,
    asked: VecDeque<Asked>,
    drawing: Option<Drawn>,
    announced: bool,
}

impl ImpostorGpu {
    fn release_targets(&self, cx: &mut Prepare<'_, Self>) {
        for texture in self.colors {
            cx.release(texture);
        }
        cx.release(self.depth);
        cx.release(self.shade);
        cx.release_buffer(self.read);
    }

    fn forget(&mut self, cx: &mut Prepare<'_, Self>) {
        for texture in self.atlas {
            cx.release(texture);
        }
        self.release_targets(cx);
        self.asked.clear();
        self.drawing = None;
        self.shape = None;
    }

    fn apply(&mut self, work: &AtlasWork, cx: &mut Prepare<'_, Self>) {
        match work {
            AtlasWork::Reset { shape } => {
                self.forget(cx);
                self.shape = Some(*shape);
            }
            AtlasWork::Release => {
                let shape = self.shape;
                self.forget(cx);
                self.shape = shape;
            }
            AtlasWork::Grow { layers } => {
                let Some(shape) = self.shape else {
                    return;
                };
                let extent = wgpu::Extent3d {
                    width: shape.size(),
                    height: shape.size(),
                    depth_or_array_layers: *layers,
                };
                for texture in self.atlas {
                    cx.grow(texture, extent, shape.levels());
                }
            }
            AtlasWork::Load { layer, bytes } => match (self.shape, bytes.upgrade()) {
                (Some(shape), Some(bytes)) => self.load(shape, *layer, &bytes, cx),
                (Some(_), None) => cx.deliver::<BakeReads>(MISSED, Vec::new()),
                (None, _) => {}
            },
            AtlasWork::Bake {
                serial,
                mesh,
                material,
                parts,
                bounds,
                hemisphere,
            } => {
                if self.shape.is_none() {
                    cx.deliver::<BakeReads>(*serial, Vec::new());
                    return;
                }
                self.asked.push_back(Asked {
                    serial: *serial,
                    mesh: *mesh,
                    material: *material,
                    parts: parts.clone(),
                    bounds: *bounds,
                    hemisphere: *hemisphere,
                });
            }
        }
    }

    fn load(&self, shape: AtlasShape, layer: u32, bytes: &[u8], cx: &mut Prepare<'_, Self>) {
        let mut at = RECORD_START;
        for texture in self.atlas {
            for level in 0..shape.levels() {
                let size = shape.size() >> level;
                let (row, rows) = self.format.rows(size);
                let length = row as usize * rows as usize;
                let Some(texels) = bytes.get(at..at + length) else {
                    return;
                };
                cx.write_texture_level(
                    texture,
                    level,
                    wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    texels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(rows),
                    },
                    flat(size),
                );
                at += length;
            }
        }
    }

    fn draw(
        &mut self,
        asked: Asked,
        (library, residency): (&AssetLibrary, &AssetResidency),
        cx: &mut Prepare<'_, Self>,
    ) {
        let shape = self.shape.map(|shape| shape.facing(asked.hemisphere));
        let (Some(shape), true) = (shape, cx.drawable(asked.mesh, asked.material)) else {
            cx.deliver::<BakeReads>(asked.serial, Vec::new());
            return;
        };
        let draws: Vec<(MeshHandle, Option<MaterialHandle>, usize)> =
            std::iter::once((asked.mesh, asked.material))
                .chain(
                    asked
                        .parts
                        .iter()
                        .map(|&(mesh, material)| (mesh, Some(material))),
                )
                .filter(|&(mesh, material)| cx.drawable(mesh, material))
                .filter_map(|(mesh, material)| {
                    let extended = residency.mesh(mesh)?.extended();
                    let two_sided = material
                        .and_then(|handle| library.material(handle))
                        .is_some_and(|material| material.cull() == Cull::None);
                    Some((mesh, material, usize::from(extended) | (usize::from(two_sided) << 1)))
                })
                .collect();
        let size = shape.size();
        for texture in self.colors {
            cx.allocate(texture, flat(size), shape.levels());
        }
        cx.allocate(self.depth, flat(size), 1);
        cx.allocate(self.shade, flat((size * SHADE_SCALE).min(SHADE_MOST)), 1);
        let baking = record_of(asked.bounds, 0, ([0.0; 3], false), shape);
        cx.allocate_buffer(self.uniform, size_of::<ImpostorBake>() as u64);
        cx.write(
            self.uniform,
            0,
            &[ImpostorBake {
                sphere: baking.sphere,
                flags: [baking.flags, 0, 0, 0],
                tile: [0, 0, shape.frames, shape.frames],
            }],
        );
        cx.allocate_buffer(self.kinds, 4);
        cx.clear_buffer_range(self.kinds, 0..4);
        self.drawing = Some(Drawn {
            serial: asked.serial,
            shape,
            draws,
        });
        cx.run(self.shading);
        cx.run(self.baking);
    }

    fn read_back(&self, drawn: &Drawn, cx: &mut Prepare<'_, Self>) {
        let shape = drawn.shape;
        let regions = readback_bytes(shape, ROW_ALIGNMENT);
        let total = regions + 4;
        cx.allocate_buffer(self.read, total);
        let mut offset = 0_u64;
        for texture in self.colors {
            cx.mipmaps(texture, 0..1);
            let copies: Vec<CopyRegion> = (0..shape.levels())
                .map(|level| {
                    let (rows, stride, _) = level_rows(shape, level, ROW_ALIGNMENT);
                    let region = CopyRegion::LevelToBuffer {
                        from: (level, wgpu::Origin3d::ZERO),
                        to: wgpu::TexelCopyBufferLayout {
                            offset,
                            bytes_per_row: Some(stride),
                            rows_per_image: Some(rows),
                        },
                        size: flat(shape.size() >> level),
                    };
                    offset += u64::from(stride) * u64::from(rows);
                    region
                })
                .collect();
            cx.copy_at_end(Copied::Texture(texture), Copied::Buffer(self.read), &copies);
        }
        cx.copy_at_end(
            Copied::Buffer(self.kinds),
            Copied::Buffer(self.read),
            &[CopyRegion::Bytes {
                from: 0,
                to: regions,
                size: 4,
            }],
        );
        if !cx.read_back_once::<BakeReads>(drawn.serial, self.read, 0..total, AfterRead::Keep) {
            cx.deliver::<BakeReads>(drawn.serial, Vec::new());
        }
    }
}

impl GpuPart for ImpostorGpu {
    type Reads = (
        Res<'static, ImpostorBooks>,
        Res<'static, AssetLibrary>,
        Res<'static, AssetResidency>,
    );
    const NAME: &'static str = "impostors";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let format = format_of(cx.features());
        let formats = formats_of(format);
        let atlas: [Texture<Self>; 4] = std::array::from_fn(|kind| {
            cx.texture(ATLAS_LABELS[kind], formats[kind], Lifetime::Kept)
        });
        let colors: [Texture<Self>; 4] = std::array::from_fn(|kind| {
            cx.texture(BAKE_LABELS[kind], FORMATS[kind], Lifetime::Kept)
        });
        let depth = cx.texture("impostor_bake_depth", BAKE_DEPTH, Lifetime::Kept);
        let shade = cx.texture("impostor_shade", BAKE_DEPTH, Lifetime::Kept);
        let uniform = cx.buffer("impostor_bake", Lifetime::Kept);
        let kinds = cx.buffer("impostor_bake_kinds", Lifetime::Kept);
        let read = cx.buffer("impostor_bake_read", Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc {
            filter: wgpu::FilterMode::Linear,
            mipmap: wgpu::MipmapFilterMode::Linear,
            address: [wgpu::AddressMode::ClampToEdge; 3],
            compare: None,
        });
        let views = [
            face::ATLAS_ALBEDO,
            face::ATLAS_EMISSIVE,
            face::ATLAS_GEOMETRY,
            face::ATLAS_SURFACE,
        ];
        for (var, texture) in views.into_iter().zip(atlas) {
            cx.frame(var, Bind::Texture(texture))?;
        }
        cx.frame(face::ATLAS_SAMPLER, Bind::Sampler(sampler))?;
        let shading = cx
            .pass("impostor_shade")
            .label("engine/impostor-shade")
            .depth_cleared(shade, 0.0)
            .id();
        let mut baking = cx
            .pass("impostor_bake")
            .label("engine/impostor-bake")
            .after(shading);
        for color in colors {
            baking = baking.color_cleared(color, wgpu::Color::TRANSPARENT);
        }
        let baking = baking.depth_cleared(depth, 0.0).id();
        let mut shade_pipelines = Vec::with_capacity(4);
        let mut bake_pipelines = Vec::with_capacity(4);
        for index in 0..4_usize {
            let (extended, two_sided) = (index & 1 != 0, index & 2 != 0);
            let (vertex, shaded, baked, vertices) = if extended {
                (
                    bake::VS_IMPOSTOR_BAKE_EXTENDED,
                    bake::FS_IMPOSTOR_SHADE_EXTENDED,
                    bake::FS_IMPOSTOR_BAKE_EXTENDED,
                    EXTENDED,
                )
            } else {
                (
                    bake::VS_IMPOSTOR_BAKE,
                    bake::FS_IMPOSTOR_SHADE,
                    bake::FS_IMPOSTOR_BAKE,
                    MESH,
                )
            };
            shade_pipelines.push(cx.render_pipeline(
                shading,
                &bake_pipeline((vertex, shaded), (vertices, two_sided)),
                &[(bake::IMPOSTOR_BAKE, Bind::Buffer(uniform))],
            )?);
            bake_pipelines.push(cx.render_pipeline(
                baking,
                &bake_pipeline((vertex, baked), (vertices, two_sided)),
                &[
                    (bake::IMPOSTOR_BAKE, Bind::Buffer(uniform)),
                    (bake::IMPOSTOR_SHADE, Bind::Texture(shade)),
                    (bake::BAKE_KINDS, Bind::Buffer(kinds)),
                ],
            )?);
        }
        Ok(Self {
            format,
            atlas,
            colors,
            depth,
            shade,
            uniform,
            kinds,
            read,
            shading,
            baking,
            shade_pipelines,
            bake_pipelines,
            cursor: Cursor::default(),
            shape: None,
            asked: VecDeque::new(),
            drawing: None,
            announced: false,
        })
    }

    fn prepare(
        &mut self,
        (books, library, residency): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if !self.announced {
            self.announced = true;
            cx.deliver::<BakeReads>(DEVICE, self.format.tag().to_le_bytes().to_vec());
        }
        let drawn = self.drawing.take();
        if let Some(drawn) = &drawn {
            self.read_back(drawn, cx);
        }
        match books.log().events(&mut self.cursor) {
            Events::Listed(works) => {
                for work in works {
                    self.apply(work, cx);
                }
            }
            Events::Missed(_) => {
                self.forget(cx);
                cx.deliver::<BakeReads>(MISSED, Vec::new());
                return;
            }
        }
        if drawn.is_some() {
            return;
        }
        match self.asked.pop_front() {
            Some(asked) => self.draw(asked, (&library, &residency), cx),
            None => self.release_targets(cx),
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let Some(drawn) = self.drawing.as_ref() else {
            return 0;
        };
        let pipelines = if pass == self.shading {
            &self.shade_pipelines
        } else {
            &self.bake_pipelines
        };
        let frames = drawn.shape.frames * drawn.shape.frames;
        let mut draw = cx.render(Attach::Declared);
        for &(mesh, material, index) in &drawn.draws {
            draw.draw(
                &pipelines[index],
                Shape::Mesh {
                    mesh,
                    material,
                    instances: 0..frames,
                },
            );
        }
        u32::try_from(drawn.draws.len()).unwrap_or(u32::MAX)
    }
}
