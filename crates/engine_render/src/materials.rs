#![expect(
    clippy::expect_used,
    reason = "extraction validated every material's textures"
)]

use engine_core::collections::HashMap;
use std::collections::BTreeMap;

use engine_core::{
    Filter, Material, MaterialHandle, RenderMaterial, RenderSnapshot, Sampler, Texture,
    TextureFormat, TextureHandle, Wrap,
};

use crate::batches::BatchPlan;
use crate::bindings::MATERIAL;
use crate::gpu_data::{MaterialUniform, Variant, mip_chain_bytes, mip_levels};
use crate::mipmaps::MipGenerator;
use crate::parts::PartAssets;

const COLOR_MAPS: [bool; 5] = [true, false, false, false, true];

pub(crate) struct GpuTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    pub byte_len: u64,
    pub mip_bytes: u64,
}

pub(crate) struct GpuMaterial {
    group: wgpu::BindGroup,
    _uniform: wgpu::Buffer,
    variant: Variant,
    version: u64,
}

struct Defaults {
    _textures: [wgpu::Texture; 3],
    white_srgb: wgpu::TextureView,
    white_linear: wgpu::TextureView,
    flat_normal: wgpu::TextureView,
    _uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
}

pub struct MaterialStore {
    layout: crate::pipelines::GroupLayout,
    samplers: HashMap<Sampler, wgpu::Sampler>,
    textures: BTreeMap<(TextureHandle, bool), GpuTexture>,
    materials: BTreeMap<MaterialHandle, GpuMaterial>,
    generation: u64,
    defaults: Defaults,
    mips: MipGenerator,
}

impl MaterialStore {
    pub(crate) fn new(device: &wgpu::Device, queue: &crate::gpu_calls::Queue) -> Self {
        let layout = crate::pipelines::layout(device, &MATERIAL);
        let texel =
            |label, format, pixel| crate::targets::texel(device, queue, label, format, pixel);
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
        let mut samplers = HashMap::default();
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
            generation: 0,
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

    pub(crate) fn retain(&mut self, plan: &BatchPlan, assets: &PartAssets<'_>) {
        let extra_material = |handle: &MaterialHandle| {
            assets
                .materials
                .iter()
                .any(|source| source.handle == *handle)
        };
        let extra_texture = |handle: &TextureHandle| {
            assets
                .textures
                .iter()
                .any(|source| source.handle == *handle)
        };
        let count = self.materials.len();
        self.materials
            .retain(|handle, _| plan.contains_material(*handle) || extra_material(handle));
        if self.materials.len() != count {
            self.generation += 1;
        }
        self.textures
            .retain(|(handle, _), _| plan.contains_texture(*handle) || extra_texture(handle));
    }

    pub(crate) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        snapshot: &RenderSnapshot,
        plan: &BatchPlan,
        assets: &PartAssets<'_>,
        encoder: &mut Option<wgpu::CommandEncoder>,
    ) -> (u64, u32) {
        let mut bytes = 0;
        let mut levels = 0;
        let built = |materials: &BTreeMap<MaterialHandle, GpuMaterial>, source: &RenderMaterial| {
            materials
                .get(&source.handle)
                .is_some_and(|gpu| gpu.version == source.version)
        };
        let extra: Vec<&RenderMaterial> = assets
            .materials
            .iter()
            .copied()
            .filter(|source| !built(&self.materials, source))
            .collect();
        let sources = plan
            .material_sources
            .iter()
            .map(|&index| &snapshot.materials[index])
            .chain(extra);
        for source in sources {
            if built(&self.materials, source) {
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
                        .chain(assets.textures.iter().copied())
                        .find(|texture| texture.handle == map.texture)
                        .expect("material_textures_validated");
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
            let gpu = self.create_material(device, material, source.version);
            self.materials.insert(source.handle, gpu);
            self.generation += 1;
        }
        (bytes, levels)
    }

    fn create_material(
        &mut self,
        device: &wgpu::Device,
        material: &Material,
        version: u64,
    ) -> GpuMaterial {
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
            version,
        }
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub fn mips(&self) -> &MipGenerator {
        &self.mips
    }

    pub fn group(&self, material: MaterialHandle) -> Option<&wgpu::BindGroup> {
        self.materials
            .get(&material)
            .map(|material| &material.group)
    }

    pub fn binding(&self, material: Option<MaterialHandle>) -> (&wgpu::BindGroup, Variant) {
        match material {
            Some(handle) => {
                let material = &self.materials[&handle];
                (&material.group, material.variant)
            }
            None => (&self.defaults.group, Variant::BASIC),
        }
    }

    pub fn is_texture_resident(&self, handle: TextureHandle) -> bool {
        self.textures
            .range((handle, false)..=(handle, true))
            .next()
            .is_some()
    }

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
            lod_max_clamp: if key.mipmap_filter.is_some() {
                32.0
            } else {
                0.0
            },
            anisotropy_clamp: if trilinear { 16 } else { 1 },
            ..Default::default()
        })
    })
}

fn uniform_buffer(device: &wgpu::Device, uniform: &MaterialUniform) -> wgpu::Buffer {
    crate::memory::buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("engine/material-uniform"),
            contents: bytemuck::bytes_of(uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    )
}

const MAP_BINDINGS: [(&str, &str); 5] = [
    ("base_color_texture", "base_color_sampler"),
    ("metallic_roughness_texture", "metallic_roughness_sampler"),
    ("normal_texture", "normal_sampler"),
    ("occlusion_texture", "occlusion_sampler"),
    ("emissive_texture", "emissive_sampler"),
];

fn bind_group(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
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

pub fn gpu_format(format: TextureFormat, srgb: bool) -> (wgpu::TextureFormat, wgpu::Features) {
    use wgpu::TextureFormat as Gpu;
    match format {
        TextureFormat::Rgba8 => (
            if srgb {
                Gpu::Rgba8UnormSrgb
            } else {
                Gpu::Rgba8Unorm
            },
            wgpu::Features::empty(),
        ),
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
    }
}

pub fn write_level(
    queue: &crate::gpu_calls::Queue,
    texture: &wgpu::Texture,
    (level, layer): (u32, u32),
    (width, height): (u32, u32),
    format: TextureFormat,
    bytes: &[u8],
) {
    let (width, height) = ((width >> level).max(1), (height >> level).max(1));
    let (bytes_per_row, rows, extent) = if format.is_compressed() {
        let blocks = [width.div_ceil(4), height.div_ceil(4)];
        (blocks[0] * 16, blocks[1], (blocks[0] * 4, blocks[1] * 4))
    } else {
        (width * 4, height, (width, height))
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: level,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(rows),
        },
        wgpu::Extent3d {
            width: extent.0,
            height: extent.1,
            depth_or_array_layers: 1,
        },
    );
}

fn create_texture(
    device: &wgpu::Device,
    queue: &crate::gpu_calls::Queue,
    source: &Texture,
    srgb: bool,
    mips: &MipGenerator,
    encoder: &mut Option<wgpu::CommandEncoder>,
    generated: &mut u32,
) -> GpuTexture {
    let (format, feature) = gpu_format(source.format(), srgb);
    if !device.features().contains(feature) {
        let texture = crate::targets::texel(
            device,
            queue,
            "engine/unsupported-texture",
            wgpu::TextureFormat::Rgba8Unorm,
            [255; 4],
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        return GpuTexture {
            _texture: texture,
            view,
            byte_len: 4,
            mip_bytes: 0,
        };
    }
    let (width, height) = (source.width(), source.height());
    let generate = !source.format().is_compressed() && source.levels().len() == 1;
    let levels = if generate {
        mip_levels(width, height)
    } else {
        source.levels().len() as u32
    };
    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    if generate && levels > 1 {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    let label = match (source.format().is_compressed(), srgb) {
        (true, _) => "engine/compressed-texture",
        (false, true) => "engine/color-texture",
        (false, false) => "engine/data-texture",
    };
    let texture =
        crate::targets::texture_2d(device, label, (width, height), 1, levels, format, usage);
    for (level, bytes) in source.levels().iter().enumerate() {
        write_level(
            queue,
            &texture,
            (level as u32, 0),
            (width, height),
            source.format(),
            bytes,
        );
    }
    if generate && levels > 1 {
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
        mip_bytes: if generate && levels > 1 {
            mip_chain_bytes(width, height)
        } else {
            0
        },
    }
}
