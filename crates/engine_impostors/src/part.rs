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
    AtlasShape, AtlasWork, ImpostorBooks, RECORD_START, ROW_ALIGNMENT, kinds, level_rows,
    readback_bytes, record_of, rows,
};
use crate::cards::CARD;
use crate::shaders::impostors::bake;
use crate::shaders::impostors::faces::impostors as face;

pub(crate) const DEVICE: u32 = u32::MAX;
pub(crate) const MISSED: u32 = u32::MAX - 1;

const FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba8UnormSrgb,
];
const ATLAS_LABELS: [&str; 4] = [
    "impostor_albedo",
    "impostor_geometry",
    "impostor_surface",
    "impostor_emissive",
];
const BAKE_LABELS: [&str; 4] = [
    "impostor_bake_albedo",
    "impostor_bake_geometry",
    "impostor_bake_surface",
    "impostor_bake_emissive",
];
const BAKE_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MESH: &[VertexLayout] = &[VertexLayout::Geometry(GeometryStream::Vertex)];
const EXTENDED: &[VertexLayout] = &[
    VertexLayout::Geometry(GeometryStream::Vertex),
    VertexLayout::Geometry(GeometryStream::Extra),
];
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

fn baked(entry: Entry, two_sided: bool) -> PassFragment {
    PassFragment {
        pass: FragmentPass::Own(BAKE_PASS),
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
    emissive: bool,
}

struct Drawn {
    asked: Asked,
    shape: AtlasShape,
    draws: Vec<(GeometryHandle, Option<MaterialHandle>, usize)>,
    skipped: bool,
}

pub(crate) struct ImpostorGpu {
    atlas: [Texture<Self>; 4],
    colors: [Texture<Self>; 4],
    depth: Texture<Self>,
    uniform: Buffer<Self>,
    kinds: Buffer<Self>,
    read: Buffer<Self>,
    baking: Pass<Self>,
    bake_forms: Vec<Form<Self>>,
    cursor: Cursor,
    shape: Option<AtlasShape>,
    asked: VecDeque<Asked>,
    drawing: Option<Drawn>,
    announced: bool,
    held: usize,
}

impl ImpostorGpu {
    fn release_targets(&self, cx: &mut Prepare<'_, Self>) {
        for texture in self.colors {
            cx.release(texture);
        }
        cx.release(self.depth);
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
        self.held = 0;
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
            AtlasWork::Grow { layers, emissive } => {
                let Some(shape) = self.shape else {
                    return Allocation::Kept;
                };
                let extent = wgpu::Extent3d {
                    width: shape.size(),
                    height: shape.size(),
                    depth_or_array_layers: *layers,
                };
                let kinds = kinds(*emissive) as usize;
                let mut grown = Allocation::Kept;
                for (kind, texture) in self.atlas.into_iter().enumerate().take(kinds) {
                    if cx.grow(texture, extent, shape.levels()) == Allocation::Fresh
                        && kind < self.held
                    {
                        grown = Allocation::Fresh;
                    }
                }
                self.held = self.held.max(kinds);
                return grown;
            }
            AtlasWork::Load {
                layer,
                bytes,
                emissive,
            } => match (self.shape, bytes.upgrade()) {
                (Some(shape), Some(bytes)) => self.load(shape, (*layer, *emissive), &bytes, cx),
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
                emissive,
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
                    emissive: *emissive,
                });
            }
        }
        Allocation::Kept
    }

    fn load(
        &self,
        shape: AtlasShape,
        (layer, emissive): (u32, bool),
        bytes: &[u8],
        cx: &mut Prepare<'_, Self>,
    ) {
        let mut at = RECORD_START;
        for texture in self.atlas.into_iter().take(kinds(emissive) as usize) {
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
        let held = residency.resident(asked.mesh)
            && asked.parts.iter().all(|&(mesh, material)| {
                residency.resident(mesh) && residency.material_resident(material)
            });
        let (Some(shape), true, true) = (shape, held, cx.drawable(asked.mesh, asked.material))
        else {
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
        if draws.len() <= asked.parts.len() {
            self.asked.push_back(asked);
            return;
        }
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
            asked,
            shape,
            draws,
            skipped: false,
        });
        for texture in self.colors {
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate(texture, flat(size), shape.levels());
        }
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate(self.depth, flat(size), 1);
        cx.run(self.baking);
    }

    fn read_back(&self, drawn: &Drawn, cx: &mut Prepare<'_, Self>) {
        let regions = readback_bytes(drawn.shape, ROW_ALIGNMENT, drawn.asked.emissive);
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.read, regions + 4);
        self.copy_out(drawn, regions, cx);
    }

    fn copy_out(&self, drawn: &Drawn, regions: u64, cx: &mut Prepare<'_, Self>) {
        let shape = drawn.shape;
        let total = regions + 4;
        let mut offset = 0_u64;
        let kinds = kinds(drawn.asked.emissive) as usize;
        for texture in self.colors.into_iter().take(kinds) {
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
        let serial = drawn.asked.serial;
        if !cx.read_back_once::<BakeReads>(serial, self.read, 0..total, AfterRead::Keep) {
            cx.deliver::<BakeReads>(serial, Vec::new());
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
        let uniform = cx.buffer("impostor_bake", Lifetime::Kept);
        let kinds = cx.buffer("impostor_bake_kinds", Lifetime::Kept);
        let read = cx.buffer("impostor_bake_read", Lifetime::Kept);
        let sampler = cx.sampler(engine_core::Sampler::LINEAR_CLAMP);
        let views = [
            face::ATLAS_ALBEDO,
            face::ATLAS_GEOMETRY,
            face::ATLAS_SURFACE,
            face::ATLAS_EMISSIVE,
        ];
        for (var, texture) in views.into_iter().zip(atlas) {
            cx.frame(var, Bind::Texture(texture))?;
        }
        cx.frame(face::ATLAS_SAMPLER, Bind::Sampler(sampler))?;
        let mut baking = cx
            .pass(BAKE_PASS)
            .label("engine/impostor-bake")
            .makes_records();
        for color in colors {
            baking = baking.color_cleared(color, wgpu::Color::TRANSPARENT);
        }
        let baking = baking.depth_cleared(depth, 0.0).id();
        let mut bake_forms = Vec::with_capacity(4);
        for index in 0..4_usize {
            let (extended, two_sided) = (index & 1 != 0, index & 2 != 0);
            let (form, entry) = if extended {
                (&BAKE_EXTENDED_FORM, bake::FS_IMPOSTOR_BAKE_EXTENDED)
            } else {
                (&BAKE_FORM, bake::FS_IMPOSTOR_BAKE)
            };
            bake_forms.push(cx.form_owning(
                form,
                &[baked(entry, two_sided)],
                &[
                    (bake::IMPOSTOR_BAKE, Bind::Buffer(uniform)),
                    (bake::BAKE_KINDS, Bind::Buffer(kinds)),
                ],
            )?);
        }
        Ok(Self {
            atlas,
            colors,
            depth,
            uniform,
            kinds,
            read,
            baking,
            bake_forms,
            cursor: Cursor::default(),
            shape: None,
            asked: VecDeque::new(),
            drawing: None,
            announced: false,
            held: 0,
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
        let mut drawn = self.drawing.take();
        let ran = cx.ran(self.baking);
        if let Some(again) = drawn.take_if(|drawn| drawn.skipped || !ran) {
            self.asked.push_front(again.asked);
        }
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

    fn encode(&mut self, _: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let Some(drawn) = self.drawing.as_mut() else {
            return 0;
        };
        let frames = drawn.shape.frames * drawn.shape.frames;
        let mut draw = cx.render(Attach::Declared);
        for &(mesh, material, index) in &drawn.draws {
            let Some(pipeline) = draw.form(&self.bake_forms[index]) else {
                drawn.skipped = true;
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
