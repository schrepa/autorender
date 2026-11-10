use std::collections::VecDeque;

use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::{
    AssetLibrary, AssetResidency, Cull, Cursor, Entry, Error, Events, GeometryHandle,
    GeometryStream, MaterialHandle, Readable, Readback, gpu_struct,
};
use engine_render::{
    AfterRead, Allocation, Attach, Bind, Buffer, Copied, CopyRegion, Encode, FaceCull, Form,
    FormColor, FormDepth, FormFragment, FormSide, FragmentPass, GpuPart, Install, KindForm,
    Lifetime, OwnFragments, Pass, PassFragment, Prepare, Shape, Texture, Varyings, VertexLayout,
    wgpu,
};

use crate::books::{
    AtlasShape, AtlasWork, ImpostorBooks, RECORD_START, ROW_ALIGNMENT, level_rows, readback_bytes,
    record_of, rows,
};
use crate::cards::CARD;
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
const MESH: &[VertexLayout] = &[VertexLayout::Geometry(GeometryStream::Vertex)];
const EXTENDED: &[VertexLayout] = &[
    VertexLayout::Geometry(GeometryStream::Vertex),
    VertexLayout::Geometry(GeometryStream::Extra),
];
const SHADE_PASS: &str = "impostor_shade";
const BAKE_PASS: &str = "impostor_bake";

const fn bake_form(vertex: Entry, streams: &'static [VertexLayout]) -> KindForm {
    KindForm {
        kind: CARD,
        placed: FormSide {
            vertex,
            streams,
            depth_bias: FormSide::UNBIASED,
        },
        owned: None,
        varyings: Varyings::Own(&OwnFragments::NONE),
    }
}

const BAKE_FORM: KindForm = bake_form(bake::VS_IMPOSTOR_BAKE, MESH);
const BAKE_EXTENDED_FORM: KindForm = bake_form(bake::VS_IMPOSTOR_BAKE_EXTENDED, EXTENDED);

gpu_struct! {
    pub struct ImpostorBake {
        pub sphere: [f32; 4],
        pub flags: [u32; 4],
        pub tile: [u32; 4],
    }
}

#[derive(Resource, Default)]
pub(crate) struct BakeReads {
    pub(crate) installed: bool,
    pub(crate) missed: bool,
    pub(crate) read: Vec<(u32, Option<Vec<u8>>)>,
}

impl Readable for BakeReads {
    fn arrived(&mut self, tag: u32, read: Readback<&[u8]>) {
        match tag {
            DEVICE => self.installed = true,
            MISSED => self.missed = true,
            serial => self.read.push((serial, Some(read.value.to_vec()))),
        }
    }

    fn lost(&mut self, tag: u32) {
        if tag != DEVICE && tag != MISSED {
            self.read.push((tag, None));
        }
    }
}

fn flat(side: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: side,
        height: side,
        depth_or_array_layers: 1,
    }
}

fn baked(pass: &'static str, entry: Entry, two_sided: bool) -> PassFragment {
    PassFragment {
        pass: FragmentPass::Own(pass),
        side: None,
        fragment: FormFragment::Entry {
            entry,
            cull: if two_sided {
                FaceCull::None
            } else {
                FaceCull::Back
            },
            depth: FormDepth::Written,
            color: FormColor::Written,
        },
    }
}

struct Asked {
    serial: u32,
    mesh: GeometryHandle,
    material: Option<MaterialHandle>,
    parts: Vec<(GeometryHandle, MaterialHandle)>,
    bounds: (Vec3, Vec3),
    hemisphere: bool,
}

struct Drawn {
    serial: u32,
    shape: AtlasShape,
    draws: Vec<(GeometryHandle, Option<MaterialHandle>, usize)>,
}

pub(crate) struct ImpostorGpu {
    atlas: [Texture<Self>; 4],
    colors: [Texture<Self>; 4],
    depth: Texture<Self>,
    shade: Texture<Self>,
    uniform: Buffer<Self>,
    kinds: Buffer<Self>,
    read: Buffer<Self>,
    shading: Pass<Self>,
    baking: Pass<Self>,
    shade_forms: Vec<Form<Self>>,
    bake_forms: Vec<Form<Self>>,
    cursor: Cursor,
    shape: Option<AtlasShape>,
    asked: VecDeque<Asked>,
    drawing: Option<Drawn>,
    announced: bool,
    held: bool,
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
        self.held = false;
    }

    fn apply(&mut self, work: &AtlasWork, cx: &mut Prepare<'_, Self>) -> Allocation {
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
                    return Allocation::Kept;
                };
                let extent = wgpu::Extent3d {
                    width: shape.size(),
                    height: shape.size(),
                    depth_or_array_layers: *layers,
                };
                let mut grown = Allocation::Kept;
                for texture in self.atlas {
                    if cx.grow(texture, extent, shape.levels()) == Allocation::Fresh && self.held {
                        grown = Allocation::Fresh;
                    }
                }
                self.held = true;
                return grown;
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
                    return Allocation::Kept;
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
        Allocation::Kept
    }

    fn load(&self, shape: AtlasShape, layer: u32, bytes: &[u8], cx: &mut Prepare<'_, Self>) {
        let mut at = RECORD_START;
        for texture in self.atlas {
            for level in 0..shape.levels() {
                let size = shape.size() >> level;
                let (row, rows) = rows(size);
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
        let draws: Vec<(GeometryHandle, Option<MaterialHandle>, usize)> =
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
                    Some((
                        mesh,
                        material,
                        usize::from(extended) | (usize::from(two_sided) << 1),
                    ))
                })
                .collect();
        let size = shape.size();
        let baking = record_of(asked.bounds, 0, ([0.0; 3], false), shape);
        let uniform = ImpostorBake {
            sphere: baking.sphere,
            flags: [baking.flags, 0, 0, 0],
            tile: [0, 0, shape.frames, shape.frames],
        };
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.uniform, size_of::<ImpostorBake>() as u64);
        cx.write(self.uniform, 0, &[uniform]);
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.kinds, 4);
        cx.clear_buffer_range(self.kinds, 0..4);
        self.drawing = Some(Drawn {
            serial: asked.serial,
            shape,
            draws,
        });
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate(self.shade, flat((size * SHADE_SCALE).min(SHADE_MOST)), 1);
        cx.run(self.shading);
        for texture in self.colors {
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate(texture, flat(size), shape.levels());
        }
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate(self.depth, flat(size), 1);
        cx.run(self.baking);
    }

    fn read_back(&self, drawn: &Drawn, cx: &mut Prepare<'_, Self>) {
        let regions = readback_bytes(drawn.shape, ROW_ALIGNMENT);
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.read, regions + 4);
        self.copy_out(drawn, regions, cx);
    }

    fn copy_out(&self, drawn: &Drawn, regions: u64, cx: &mut Prepare<'_, Self>) {
        let shape = drawn.shape;
        let total = regions + 4;
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
        let atlas: [Texture<Self>; 4] = std::array::from_fn(|kind| {
            cx.texture(ATLAS_LABELS[kind], FORMATS[kind], Lifetime::Kept)
        });
        let colors: [Texture<Self>; 4] = std::array::from_fn(|kind| {
            cx.texture(BAKE_LABELS[kind], FORMATS[kind], Lifetime::Kept)
        });
        let depth = cx.texture("impostor_bake_depth", BAKE_DEPTH, Lifetime::Kept);
        let shade = cx.texture("impostor_shade", BAKE_DEPTH, Lifetime::Kept);
        let uniform = cx.buffer("impostor_bake", Lifetime::Kept);
        let kinds = cx.buffer("impostor_bake_kinds", Lifetime::Kept);
        let read = cx.buffer("impostor_bake_read", Lifetime::Kept);
        let sampler = cx.sampler(engine_core::Sampler::LINEAR_CLAMP);
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
            .pass(SHADE_PASS)
            .label("engine/impostor-shade")
            .makes_records()
            .depth_cleared(shade, 0.0)
            .id();
        let mut baking = cx
            .pass(BAKE_PASS)
            .label("engine/impostor-bake")
            .makes_records()
            .after(shading);
        for color in colors {
            baking = baking.color_cleared(color, wgpu::Color::TRANSPARENT);
        }
        let baking = baking.depth_cleared(depth, 0.0).id();
        let mut shade_forms = Vec::with_capacity(4);
        let mut bake_forms = Vec::with_capacity(4);
        for index in 0..4_usize {
            let (extended, two_sided) = (index & 1 != 0, index & 2 != 0);
            let (form, shaded, baked_entry) = if extended {
                (
                    &BAKE_EXTENDED_FORM,
                    bake::FS_IMPOSTOR_SHADE_EXTENDED,
                    bake::FS_IMPOSTOR_BAKE_EXTENDED,
                )
            } else {
                (&BAKE_FORM, bake::FS_IMPOSTOR_SHADE, bake::FS_IMPOSTOR_BAKE)
            };
            shade_forms.push(cx.form_owning(
                form,
                &[baked(SHADE_PASS, shaded, two_sided)],
                &[(bake::IMPOSTOR_BAKE, Bind::Buffer(uniform))],
            )?);
            bake_forms.push(cx.form_owning(
                form,
                &[baked(BAKE_PASS, baked_entry, two_sided)],
                &[
                    (bake::IMPOSTOR_BAKE, Bind::Buffer(uniform)),
                    (bake::IMPOSTOR_SHADE, Bind::Texture(shade)),
                    (bake::BAKE_KINDS, Bind::Buffer(kinds)),
                ],
            )?);
        }
        Ok(Self {
            atlas,
            colors,
            depth,
            shade,
            uniform,
            kinds,
            read,
            shading,
            baking,
            shade_forms,
            bake_forms,
            cursor: Cursor::default(),
            shape: None,
            asked: VecDeque::new(),
            drawing: None,
            announced: false,
            held: false,
        })
    }

    fn prepare(
        &mut self,
        (books, library, residency): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if !self.announced {
            self.announced = true;
            cx.deliver::<BakeReads>(DEVICE, Vec::new());
        }
        let drawn = self.drawing.take();
        if let Some(drawn) = &drawn {
            self.read_back(drawn, cx);
        }
        match books.log().events(&mut self.cursor) {
            Events::Listed(works) => {
                for work in works {
                    if self.apply(work, cx) == Allocation::Fresh {
                        self.forget(cx);
                        cx.deliver::<BakeReads>(MISSED, Vec::new());
                        return;
                    }
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
        let forms = if pass == self.shading {
            &self.shade_forms
        } else {
            &self.bake_forms
        };
        let frames = drawn.shape.frames * drawn.shape.frames;
        let mut draw = cx.render(Attach::Declared);
        for &(mesh, material, index) in &drawn.draws {
            let Some(pipeline) = draw.form(&forms[index]) else {
                continue;
            };
            draw.draw(
                &pipeline,
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
