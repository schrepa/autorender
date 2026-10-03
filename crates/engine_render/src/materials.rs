//! GPU materials (decision 031): textures uploaded in the color space their
//! maps read them in, cached samplers, and one bind group per material with
//! maps. Materials without maps share a default group of white textures.

use std::collections::{BTreeMap, HashMap};

use engine_core::{
    Filter, Material, MaterialHandle, RenderMaterial, RenderSnapshot, Sampler, Texture,
    TextureFormat, TextureHandle, Wrap,
};

use crate::batches::BatchPlan;
use crate::bindings::MATERIAL;
use crate::gpu_data::{MaterialUniform, Variant, mip_chain_bytes, mip_levels};
use crate::mipmaps::MipGenerator;

/// Color maps (base color and emissive) decode sRGB; data maps read linear.
const COLOR_MAPS: [bool; 5] = [true, false, false, false, true];

pub(crate) struct GpuTexture {
    // Keep texture ownership explicit alongside its reusable view.
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    // Base level (asset) bytes, and the generated levels' bytes.
    pub byte_len: u64,
    pub mip_bytes: u64,
}

pub(crate) struct GpuMaterial {
    group: wgpu::BindGroup,
    _uniform: wgpu::Buffer,
    variant: Variant,
}

/// Default map textures: white in both color spaces and a flat normal.
struct Defaults {
    _textures: [wgpu::Texture; 3],
    white_srgb: wgpu::TextureView,
    white_linear: wgpu::TextureView,
    flat_normal: wgpu::TextureView,
    _uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
}

pub(crate) struct MaterialStore {
    layout: wgpu::BindGroupLayout,
    samplers: HashMap<Sampler, wgpu::Sampler>,
    // Keyed by handle and whether the view decodes sRGB: a texture read as
    // both color and data uploads once per color space.
    textures: BTreeMap<(TextureHandle, bool), GpuTexture>,
    materials: BTreeMap<MaterialHandle, GpuMaterial>,
    defaults: Defaults,
    mips: MipGenerator,
}

impl MaterialStore {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let layout = crate::pipelines::layout(device, &MATERIAL);
        let texel = |label, format, pixel: [u8; 4]| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &pixel,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            texture
        };
        let textures = [
            texel(
                "engine/white-srgb",
                wgpu::TextureFormat::Rgba8UnormSrgb,
                [255; 4],
            ),
            texel(
                "engine/white-linear",
                wgpu::TextureFormat::Rgba8Unorm,
                [255; 4],
            ),
            texel(
                "engine/flat-normal",
                wgpu::TextureFormat::Rgba8Unorm,
                [128, 128, 255, 255],
            ),
        ];
        let [white_srgb, white_linear, flat_normal] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let mut samplers = HashMap::new();
        let sampler = sampler(device, &mut samplers, Sampler::CLAMP).clone();
        let uniform = uniform_buffer(device, &MaterialUniform::DEFAULT);
        let group = bind_group(
            device,
            &layout,
            [
                &white_srgb,
                &white_linear,
                &flat_normal,
                &white_linear,
                &white_srgb,
            ],
            [&sampler; 5],
            &uniform,
        );
        Self {
            layout,
            samplers,
            textures: BTreeMap::new(),
            materials: BTreeMap::new(),
            defaults: Defaults {
                _textures: textures,
                white_srgb,
                white_linear,
                flat_normal,
                _uniform: uniform,
                group,
            },
            mips: MipGenerator::new(device),
        }
    }

    /// Group 1 of every mesh pipeline.
    pub(crate) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Drops materials and textures `plan`'s snapshot no longer supplies,
    /// keeping its decals' (decision 039).
    pub(crate) fn retain(&mut self, plan: &BatchPlan, snapshot: &RenderSnapshot) {
        let decal_material = |handle: &MaterialHandle| {
            snapshot
                .decal_materials
                .iter()
                .any(|source| source.handle == *handle)
        };
        let decal_texture = |handle: &TextureHandle| {
            snapshot
                .decal_textures
                .iter()
                .any(|source| source.handle == *handle)
        };
        self.materials
            .retain(|handle, _| plan.contains_material(*handle) || decal_material(handle));
        self.textures
            .retain(|(handle, _), _| plan.contains_texture(*handle) || decal_texture(handle));
    }

    /// Uploads the textures and builds the bind groups of `plan`'s used
    /// materials and of `extra` listed ones (masked shadow casters) and the
    /// snapshot's decal materials (decision 039). New
    /// textures' mip levels render from their base level in commands
    /// recorded into `encoder`, created when needed. Returns the uploaded
    /// bytes and generated levels.
    pub(crate) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        snapshot: &RenderSnapshot,
        plan: &BatchPlan,
        extra: impl Iterator<Item = MaterialHandle>,
        encoder: &mut Option<wgpu::CommandEncoder>,
    ) -> (u64, u32) {
        let mut bytes = 0;
        let mut levels = 0;
        let extra: Vec<&RenderMaterial> = extra
            .filter(|handle| !self.materials.contains_key(handle))
            .filter_map(|handle| {
                snapshot
                    .materials
                    .iter()
                    .find(|source| source.handle == handle)
            })
            .chain(
                snapshot
                    .decal_materials
                    .iter()
                    .filter(|source| !self.materials.contains_key(&source.handle)),
            )
            .collect();
        let sources = plan
            .material_sources
            .iter()
            .map(|&index| &snapshot.materials[index])
            .chain(extra);
        for source in sources {
            if self.materials.contains_key(&source.handle) {
                continue;
            }
            let material = &source.material;
            for (map, srgb) in material.maps().iter().zip(COLOR_MAPS) {
                let Some(map) = map else {
                    continue;
                };
                if let std::collections::btree_map::Entry::Vacant(entry) =
                    self.textures.entry((map.texture, srgb))
                {
                    let texture = snapshot
                        .textures
                        .iter()
                        .chain(&snapshot.decal_textures)
                        .find(|texture| texture.handle == map.texture)
                        .expect(
                            "the plan and decal extraction validated every material's textures",
                        );
                    let gpu = create_texture(
                        device,
                        queue,
                        &texture.texture,
                        srgb,
                        &self.mips,
                        encoder,
                        &mut levels,
                    );
                    bytes += gpu.byte_len;
                    entry.insert(gpu);
                }
            }
            let gpu = self.create_material(device, material);
            self.materials.insert(source.handle, gpu);
        }
        (bytes, levels)
    }

    fn create_material(&mut self, device: &wgpu::Device, material: &Material) -> GpuMaterial {
        let defaults = [
            &self.defaults.white_srgb,
            &self.defaults.white_linear,
            &self.defaults.flat_normal,
            &self.defaults.white_linear,
            &self.defaults.white_srgb,
        ];
        let maps = material.maps();
        let sampler_keys = maps.map(|map| map.map_or(Sampler::CLAMP, |map| map.sampler));
        for key in sampler_keys {
            sampler(device, &mut self.samplers, key);
        }
        let mut views = defaults;
        for (index, map) in maps.iter().enumerate() {
            if let Some(map) = map {
                views[index] = &self.textures[&(map.texture, COLOR_MAPS[index])].view;
            }
        }
        let samplers = sampler_keys.map(|key| &self.samplers[&key]);
        let uniform = uniform_buffer(device, &MaterialUniform::new(material));
        GpuMaterial {
            group: bind_group(device, &self.layout, views, samplers, &uniform),
            _uniform: uniform,
            variant: Variant::of(material),
        }
    }

    /// The mip generator textures and impostors (decision 048) share.
    pub(crate) fn mips(&self) -> &MipGenerator {
        &self.mips
    }

    /// The bind group of `material`, when uploaded.
    pub(crate) fn group(&self, material: MaterialHandle) -> Option<&wgpu::BindGroup> {
        self.materials
            .get(&material)
            .map(|material| &material.group)
    }

    /// The bind group and pipeline variant of `material` (the default group
    /// without one). Call after [`Self::upload`] for the plan's materials.
    pub(crate) fn binding(&self, material: Option<MaterialHandle>) -> (&wgpu::BindGroup, Variant) {
        match material {
            Some(handle) => {
                let material = &self.materials[&handle];
                (&material.group, material.variant)
            }
            None => (&self.defaults.group, Variant::BASIC),
        }
    }

    /// Whether `handle` has uploaded GPU storage in either color space.
    pub(crate) fn is_texture_resident(&self, handle: TextureHandle) -> bool {
        self.textures
            .range((handle, false)..=(handle, true))
            .next()
            .is_some()
    }

    /// Resident textures, their base level bytes and their mip bytes.
    pub(crate) fn texture_totals(&self) -> (u32, u64, u64) {
        self.textures
            .values()
            .fold((0, 0, 0), |(count, bytes, mips), texture| {
                (
                    count + 1,
                    bytes + texture.byte_len,
                    mips + texture.mip_bytes,
                )
            })
    }
}

fn sampler<'a>(
    device: &wgpu::Device,
    cache: &'a mut HashMap<Sampler, wgpu::Sampler>,
    key: Sampler,
) -> &'a wgpu::Sampler {
    cache.entry(key).or_insert_with(|| {
        let address = |wrap| match wrap {
            Wrap::Repeat => wgpu::AddressMode::Repeat,
            Wrap::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
            Wrap::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        };
        let filter = |filter| match filter {
            Filter::Nearest => wgpu::FilterMode::Nearest,
            Filter::Linear => wgpu::FilterMode::Linear,
        };
        let trilinear = key.mag_filter == Filter::Linear
            && key.min_filter == Filter::Linear
            && key.mipmap_filter == Some(Filter::Linear);
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/material-sampler"),
            address_mode_u: address(key.wrap_u),
            address_mode_v: address(key.wrap_v),
            mag_filter: filter(key.mag_filter),
            min_filter: filter(key.min_filter),
            mipmap_filter: match key.mipmap_filter {
                Some(Filter::Nearest) | None => wgpu::MipmapFilterMode::Nearest,
                Some(Filter::Linear) => wgpu::MipmapFilterMode::Linear,
            },
            // Without mip filtering, only the base level.
            lod_max_clamp: if key.mipmap_filter.is_some() {
                32.0
            } else {
                0.0
            },
            // Anisotropic filtering keeps surfaces seen at grazing angles
            // sharp along their steep axis; it needs linear filters.
            anisotropy_clamp: if trilinear { 16 } else { 1 },
            ..Default::default()
        })
    })
}

fn uniform_buffer(device: &wgpu::Device, uniform: &MaterialUniform) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("engine/material-uniform"),
        contents: bytemuck::bytes_of(uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

/// The texture and sampler bindings of a material's maps, in the order of
/// `Material::maps` (`bindings::MATERIAL`).
const MAP_BINDINGS: [(&str, &str); 5] = [
    ("base_color_texture", "base_color_sampler"),
    ("metallic_roughness_texture", "metallic_roughness_sampler"),
    ("normal_texture", "normal_sampler"),
    ("occlusion_texture", "occlusion_sampler"),
    ("emissive_texture", "emissive_sampler"),
];

fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    views: [&wgpu::TextureView; 5],
    samplers: [&wgpu::Sampler; 5],
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let mut resources = Vec::with_capacity(11);
    for ((view, sampler), (texture_name, sampler_name)) in
        views.into_iter().zip(samplers).zip(MAP_BINDINGS)
    {
        resources.push((texture_name, crate::pipelines::view(view)));
        resources.push((sampler_name, crate::pipelines::sampler(sampler)));
    }
    resources.push(("material_maps", uniform.as_entire_binding()));
    crate::pipelines::group(
        device,
        "engine/material-group",
        &MATERIAL,
        layout,
        &resources,
    )
}

/// The GPU format of a compressed `format` read as sRGB or linear, and the
/// device feature sampling it (decision 037); `None` for RGBA8.
fn compressed_format(
    format: TextureFormat,
    srgb: bool,
) -> Option<(wgpu::TextureFormat, wgpu::Features)> {
    use wgpu::TextureFormat as Gpu;
    Some(match format {
        TextureFormat::Rgba8 => return None,
        TextureFormat::Bc7 => (
            if srgb {
                Gpu::Bc7RgbaUnormSrgb
            } else {
                Gpu::Bc7RgbaUnorm
            },
            wgpu::Features::TEXTURE_COMPRESSION_BC,
        ),
        TextureFormat::Etc2Rgba8 => (
            if srgb {
                Gpu::Etc2Rgba8UnormSrgb
            } else {
                Gpu::Etc2Rgba8Unorm
            },
            wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        ),
        TextureFormat::Astc4x4 => (
            Gpu::Astc {
                block: wgpu::AstcBlock::B4x4,
                channel: if srgb {
                    wgpu::AstcChannel::UnormSrgb
                } else {
                    wgpu::AstcChannel::Unorm
                },
            },
            wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        ),
    })
}

/// Uploads a compressed texture's levels as they are (decision 037); a
/// device without its format samples a white texel instead.
fn create_compressed(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &Texture,
    (format, feature): (wgpu::TextureFormat, wgpu::Features),
) -> GpuTexture {
    let supported = device.features().contains(feature);
    let levels: &[Box<[u8]>] = if supported { source.levels() } else { &[] };
    let (width, height) = if supported {
        (source.width(), source.height())
    } else {
        (1, 1)
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("engine/compressed-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels.len().max(1) as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if supported {
            format
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    if !supported {
        queue.write_texture(
            texture.as_image_copy(),
            &[255; 4],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }
    for (level, bytes) in levels.iter().enumerate() {
        // Whole blocks, including those past the edge of small levels.
        let blocks = [
            (width >> level).max(1).div_ceil(4),
            (height >> level).max(1).div_ceil(4),
        ];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(blocks[0] * 16),
                rows_per_image: Some(blocks[1]),
            },
            wgpu::Extent3d {
                width: blocks[0] * 4,
                height: blocks[1] * 4,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    GpuTexture {
        _texture: texture,
        view,
        byte_len: if supported {
            source.byte_len() as u64
        } else {
            4
        },
        mip_bytes: 0,
    }
}

/// Uploads `source` as the base level of an sRGB or linear texture with a
/// full mip chain, whose remaining levels are recorded into `encoder`;
/// compressed textures upload their own levels.
fn create_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &Texture,
    srgb: bool,
    mips: &MipGenerator,
    encoder: &mut Option<wgpu::CommandEncoder>,
    generated: &mut u32,
) -> GpuTexture {
    if let Some(format) = compressed_format(source.format(), srgb) {
        return create_compressed(device, queue, source, format);
    }
    let extent = wgpu::Extent3d {
        width: source.width(),
        height: source.height(),
        depth_or_array_layers: 1,
    };
    let levels = mip_levels(source.width(), source.height());
    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    if levels > 1 {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(if srgb {
            "engine/color-texture"
        } else {
            "engine/data-texture"
        }),
        size: extent,
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        },
        usage,
        view_formats: &[],
    });
    // Queue writes accept tightly packed rows; no 256-byte buffer-copy padding.
    queue.write_texture(
        texture.as_image_copy(),
        source.pixels(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(source.width() * 4),
            rows_per_image: Some(source.height()),
        },
        extent,
    );
    if levels > 1 {
        let encoder = encoder.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/texture-mips"),
            })
        });
        *generated += mips.record(device, encoder, &texture);
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    GpuTexture {
        _texture: texture,
        view,
        byte_len: source.byte_len() as u64,
        mip_bytes: if levels > 1 {
            mip_chain_bytes(source.width(), source.height())
        } else {
            0
        },
    }
}
