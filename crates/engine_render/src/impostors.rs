#![expect(
    clippy::expect_used,
    reason = "the atlas arrays grow before they are read"
)]

pub(crate) mod bake;
mod records;

pub(crate) use bake::ImpostorBake;

use bevy_ecs::world::World;
use engine_core::collections::HashMap;
use engine_core::glam::Vec3;
use engine_core::{
    AtlasDevice, AtlasFormat, AtlasShape, AtlasWork, IMPOSTOR_KINDS as KINDS, ImpostorBooks,
    ImpostorIdentity, ImpostorRecord, MaterialHandle, MeshHandle,
};

use crate::bindings::IMPOSTOR_BAKE;
use crate::materials::{MaterialStore, Variant};
use crate::mesh_arena::{GpuMeshes, MeshDraw};
use crate::mipmaps::MipGenerator;
use crate::shaders::{Entry, Stage};

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

const BAKE_MODULE: &str = "engine::impostor_bake";

pub(crate) fn layout_source() -> String {
    [
        engine_core::impostor_flags::WGSL,
        <ImpostorRecord as engine_core::GpuStruct>::WGSL,
        <ImpostorBake as engine_core::GpuStruct>::WGSL,
    ]
    .concat()
}

pub(crate) fn atlas_source() -> String {
    [
        include_str!("impostors/atlas.wgsl"),
        engine_core::impostor_atlas::WGSL,
    ]
    .concat()
}

fn format_of(device: &wgpu::Device) -> AtlasFormat {
    let features = device.features();
    if features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        AtlasFormat::Bc7
    } else if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
        AtlasFormat::Astc
    } else {
        AtlasFormat::Rgba8
    }
}

fn textures(format: AtlasFormat) -> [wgpu::TextureFormat; 4] {
    match format {
        AtlasFormat::Bc7 => BC7_FORMATS,
        AtlasFormat::Astc => ASTC_FORMATS,
        AtlasFormat::Rgba8 => FORMATS,
    }
}

pub(crate) fn atlas_device(device: &wgpu::Device) -> AtlasDevice {
    let limits = device.limits();
    AtlasDevice {
        format: format_of(device),
        max_texture_dimension: limits.max_texture_dimension_2d,
        max_buffer_size: limits.max_buffer_size,
        max_layers: limits.max_texture_array_layers,
        row_alignment: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
    }
}

fn bake_entries(extended: bool, shade: bool) -> (Entry, Entry) {
    let entry = |name, stage| Entry {
        module: BAKE_MODULE,
        name,
        stage,
    };
    let vertex = if extended {
        entry("vs_impostor_bake_extended", Stage::Vertex)
    } else {
        entry("vs_impostor_bake", Stage::Vertex)
    };
    let fragment = match (extended, shade) {
        (false, false) => entry("fs_impostor_bake", Stage::Fragment),
        (true, false) => entry("fs_impostor_bake_extended", Stage::Fragment),
        (false, true) => entry("fs_impostor_shade", Stage::Fragment),
        (true, true) => entry("fs_impostor_shade_extended", Stage::Fragment),
    };
    (vertex, fragment)
}

const MIPS_LABEL: &str = "engine/texture-mips";

fn mip_encoder<'a>(
    device: &wgpu::Device,
    encoder: &'a mut Option<wgpu::CommandEncoder>,
) -> &'a mut wgpu::CommandEncoder {
    encoder.get_or_insert_with(|| {
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(MIPS_LABEL),
        })
    })
}

struct Arrays {
    texture: wgpu::Texture,
    views: [wgpu::TextureView; 4],
    whole: wgpu::TextureView,
    layers: u32,
}

impl Drop for Arrays {
    fn drop(&mut self) {
        crate::memory::release_any(&self.texture);
    }
}

struct BakeSource<'a> {
    draw: MeshDraw<'a>,
    extended: bool,
    group: &'a wgpu::BindGroup,
    variant: Variant,
    bounds: (Vec3, Vec3),
    parts: Vec<BakePart<'a>>,
}

struct BakePart<'a> {
    draw: MeshDraw<'a>,
    extended: bool,
    group: &'a wgpu::BindGroup,
    variant: Variant,
}

struct ImpostorPipelines {
    bake_layout: crate::pipelines::GroupLayout,
    bake_pipeline_layout: wgpu::PipelineLayout,
    bake_pipelines: HashMap<Variant, wgpu::RenderPipeline>,
    shade_pipelines: HashMap<Variant, wgpu::RenderPipeline>,
    sampler: wgpu::Sampler,
    no_shade: Option<wgpu::TextureView>,
}

impl ImpostorPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bake_layout = crate::pipelines::layout(device, &IMPOSTOR_BAKE);
        let bake_pipeline_layout =
            crate::pipelines::layout_of(device, &crate::pipeline_layouts::IMPOSTOR_BAKE_PIPELINES);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/impostor-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            bake_layout,
            bake_pipeline_layout,
            bake_pipelines: HashMap::default(),
            shade_pipelines: HashMap::default(),
            sampler,
            no_shade: None,
        }
    }
}

struct Atlas {
    shape: AtlasShape,
    format: AtlasFormat,
    arrays: Option<Arrays>,
    targets: Option<bake::BakeTargets>,
    readbacks: Vec<records::Readback>,
}

impl Atlas {
    fn new(shape: AtlasShape, format: AtlasFormat) -> Self {
        Self {
            shape,
            format,
            arrays: None,
            targets: None,
            readbacks: Vec::new(),
        }
    }

    fn layer_bytes(&self) -> u64 {
        self.format.level_bytes(self.shape.size()) as u64 * 4 * 4 / 3
    }

    fn capacity_bytes(&self) -> u64 {
        let layers = self.arrays.as_ref().map_or(0, |arrays| arrays.layers);
        u64::from(layers) * self.layer_bytes()
    }

    fn reset(&mut self, shape: AtlasShape) {
        self.shape = shape;
        self.arrays = None;
        self.targets = None;
        self.readbacks.clear();
    }

    fn release(&mut self) {
        self.arrays = None;
        self.targets = None;
        self.readbacks.clear();
    }

    fn grow(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut Option<wgpu::CommandEncoder>,
        capacity: u32,
    ) {
        if self
            .arrays
            .as_ref()
            .is_some_and(|arrays| arrays.layers >= capacity)
        {
            return;
        }
        let (size, levels) = (self.shape.size(), self.shape.levels());
        let formats = textures(self.format);
        let texture = crate::memory::texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("engine/impostor-array"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: KINDS * capacity,
                },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: formats[0].remove_srgb_suffix(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[formats[0]],
            },
        );
        if let Some(old) = &self.arrays {
            let encoder = mip_encoder(device, encoder);
            for kind in 0..KINDS {
                for level in 0..levels {
                    let size = size >> level;
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &old.texture,
                            mip_level: level,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: kind * old.layers,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: level,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: kind * capacity,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: size,
                            height: size,
                            depth_or_array_layers: old.layers,
                        },
                    );
                }
            }
        }
        let views = std::array::from_fn(|kind| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("engine/impostor-array-view"),
                format: Some(formats[kind]),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_array_layer: kind as u32 * capacity,
                array_layer_count: Some(capacity),
                ..Default::default()
            })
        });
        let whole = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("engine/impostor-array-whole"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.arrays = Some(Arrays {
            texture,
            views,
            whole,
            layers: capacity,
        });
    }
}

pub(crate) enum FrameResource<'a> {
    View(&'a wgpu::TextureView),
    Sampler(&'a wgpu::Sampler),
}

pub(crate) struct Impostors {
    pipelines: ImpostorPipelines,
    atlas: Option<Atlas>,
    unbaked: Vec<ImpostorIdentity>,
    done: u64,
    generation: u64,
    _stand_in: wgpu::Texture,
    stand_in: wgpu::TextureView,
}

impl Impostors {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let stand_in = crate::targets::texture_2d(
            device,
            "engine/post-world-frames",
            (1, 1),
            KINDS,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let view = stand_in.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            pipelines: ImpostorPipelines::new(device),
            atlas: None,
            unbaked: Vec::new(),
            done: 0,
            generation: 0,
            _stand_in: stand_in,
            stand_in: view,
        }
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn forget(&mut self) {
        self.atlas = None;
        self.unbaked.clear();
        self.generation += 1;
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.atlas.as_ref().map_or(0, Atlas::capacity_bytes)
    }

    pub(crate) fn frame_resource(&self, name: &str) -> Option<FrameResource<'_>> {
        if name == "impostor_sampler" {
            return Some(FrameResource::Sampler(&self.pipelines.sampler));
        }
        let Some(arrays) = self.atlas.as_ref().and_then(|atlas| atlas.arrays.as_ref()) else {
            return matches!(name, "impostor_albedo" | "impostor_atlas")
                .then_some(FrameResource::View(&self.stand_in));
        };
        let view = match name {
            "impostor_albedo" => &arrays.views[0],
            "impostor_emissive" => &arrays.views[1],
            "impostor_geometry" => &arrays.views[2],
            "impostor_surface" => &arrays.views[3],
            "impostor_atlas" => &arrays.whole,
            _ => return None,
        };
        Some(FrameResource::View(view))
    }

    pub(crate) fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        world: &World,
        (meshes, materials): (&GpuMeshes, &MaterialStore),
        encoder: &mut Option<wgpu::CommandEncoder>,
    ) {
        let Some(books) = world.get_resource::<ImpostorBooks>() else {
            return;
        };
        let Some(facts) = books.device() else {
            return;
        };
        for (at, work) in books.work() {
            if at <= self.done {
                continue;
            }
            self.done = at;
            if let AtlasWork::Reset { shape } = work {
                self.atlas
                    .get_or_insert_with(|| Atlas::new(*shape, facts.format))
                    .reset(*shape);
                self.generation += 1;
                continue;
            }
            let Some(atlas) = self.atlas.as_mut() else {
                if let AtlasWork::Bake { identity, .. } = work {
                    self.unbaked.push(*identity);
                }
                continue;
            };
            match work {
                AtlasWork::Reset { .. } => {}
                AtlasWork::Release => {
                    atlas.release();
                    self.generation += 1;
                }
                AtlasWork::Grow { layers } => {
                    atlas.grow(device, encoder, *layers);
                    self.generation += 1;
                }
                AtlasWork::Load { layer, bytes } => atlas.load(queue, *layer, bytes),
                AtlasWork::Bake {
                    identity,
                    mesh,
                    material,
                    parts,
                    bounds,
                    hemisphere,
                } => {
                    let Some(source) =
                        bake_source((meshes, materials), (*mesh, *material), parts, *bounds)
                    else {
                        self.unbaked.push(*identity);
                        continue;
                    };
                    let gpu = BakeGpu {
                        device,
                        encoder: &mut *encoder,
                        mips: materials.mips(),
                        pipelines: &mut self.pipelines,
                    };
                    atlas.bake(gpu, &source, *identity, *hemisphere);
                }
                AtlasWork::DropTargets => atlas.targets = None,
            }
        }
    }

    pub(crate) fn after_submit(&mut self) {
        if let Some(atlas) = self.atlas.as_mut() {
            atlas.after_submit();
        }
    }

    pub(crate) fn deliver(&mut self, world: &mut World) {
        let mut made: Vec<(ImpostorIdentity, Option<Vec<u8>>)> = Vec::new();
        if let Some(atlas) = self.atlas.as_mut() {
            atlas.take_made(&mut made);
        }
        made.extend(self.unbaked.drain(..).map(|identity| (identity, None)));
        engine_core::deliver_impostors(world, self.done, made);
    }
}

fn bake_source<'a>(
    (meshes, materials): (&'a GpuMeshes, &'a MaterialStore),
    (mesh, material): (MeshHandle, Option<MaterialHandle>),
    parts: &[(MeshHandle, MaterialHandle)],
    bounds: (Vec3, Vec3),
) -> Option<BakeSource<'a>> {
    let held = |handle: Option<MaterialHandle>| {
        handle.is_none_or(|handle| materials.group(handle).is_some())
    };
    if !held(material) {
        return None;
    }
    let gpu = meshes.get(mesh)?;
    let (group, variant) = materials.binding(material);
    let mut baked = Vec::with_capacity(parts.len());
    for &(part, part_material) in parts {
        let (Some(part_mesh), Some(draw)) = (meshes.get(part), meshes.draw(part)) else {
            continue;
        };
        if !held(Some(part_material)) {
            continue;
        }
        let (group, variant) = materials.binding(Some(part_material));
        baked.push(BakePart {
            draw,
            extended: part_mesh.extended,
            group,
            variant: variant.with_extended(part_mesh.extended),
        });
    }
    Some(BakeSource {
        draw: meshes.draw(mesh)?,
        extended: gpu.extended,
        group,
        variant,
        bounds,
        parts: baked,
    })
}

struct BakeGpu<'a> {
    device: &'a wgpu::Device,
    encoder: &'a mut Option<wgpu::CommandEncoder>,
    mips: &'a MipGenerator,
    pipelines: &'a mut ImpostorPipelines,
}
