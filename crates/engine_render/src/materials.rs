#![expect(
    clippy::expect_used,
    reason = "residency admits a material only while the library holds its textures"
)]

use engine_core::collections::HashMap;
use std::collections::BTreeMap;

use bevy_ecs::world::World;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, Filter, Material, MaterialHandle, Sampler, Texture,
    TextureFormat, TextureHandle, UvTransform, Wrap,
};

use crate::bindings::MATERIAL;
use crate::gpu_data::{mip_chain_bytes, mip_levels};
use crate::gpu_layouts::constants;
use crate::gpu_layouts::scene::MaterialUniform;
use crate::mipmaps::MipGenerator;

const COLOR_MAPS: [bool; 5] = [true, false, false, false, true];

pub const MATERIAL_ANISOTROPY: u16 = 16;

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
    synced: Option<u64>,
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
            synced: None,
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

    pub(crate) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        world: &World,
        encoder: &mut Option<wgpu::CommandEncoder>,
    ) -> (u64, u32) {
        let (Some(residency), Some(library)) = (
            world.get_resource::<AssetResidency>(),
            world.get_resource::<AssetLibrary>(),
        ) else {
            return (0, 0);
        };
        let generation = residency.materials_generation();
        if self.synced == Some(generation) {
            return (0, 0);
        }
        self.retain(
            |material| residency.material_resident(material),
            |texture| residency.texture_resident(texture),
        );
        let sources = residency.materials().filter_map(|(handle, version)| {
            library
                .material(handle)
                .map(|material| (handle, material, version))
        });
        let texture = |handle: TextureHandle| library.texture(handle).map(|texture| &**texture);
        let uploaded = self.upload(device, queue, sources, texture, encoder);
        self.synced = Some(generation);
        uploaded
    }

    pub(crate) fn forget_sync(&mut self) {
        self.synced = None;
    }

    pub(crate) fn retain(
        &mut self,
        mut material: impl FnMut(MaterialHandle) -> bool,
        mut texture: impl FnMut(TextureHandle) -> bool,
    ) {
        let count = self.materials.len();
        self.materials.retain(|handle, _| material(*handle));
        if self.materials.len() != count {
            self.generation += 1;
        }
        self.textures.retain(|(handle, _), _| texture(*handle));
    }

    pub(crate) fn upload<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        sources: impl IntoIterator<Item = (MaterialHandle, &'a Material, u64)>,
        texture: impl Fn(TextureHandle) -> Option<&'a Texture>,
        encoder: &mut Option<wgpu::CommandEncoder>,
    ) -> (u64, u32) {
        let mut bytes = 0;
        let mut levels = 0;
        for (handle, material, version) in sources {
            if self
                .materials
                .get(&handle)
                .is_some_and(|gpu| gpu.version == version)
            {
                continue;
            }
            for (map, srgb) in material.maps().iter().zip(COLOR_MAPS) {
                let Some(map) = map else {
                    continue;
                };
                if let std::collections::btree_map::Entry::Vacant(entry) =
                    self.textures.entry((map.texture, srgb))
                {
                    let source = texture(map.texture).expect("material_textures_validated");
                    let gpu = create_texture(
                        device,
                        queue,
                        source,
                        srgb,
                        &self.mips,
                        encoder,
                        &mut levels,
                    );
                    bytes += gpu.byte_len;
                    entry.insert(gpu);
                }
            }
            let gpu = self.create_material(device, material, version);
            self.materials.insert(handle, gpu);
            self.generation += 1;
        }
        (bytes, levels)
    }

    pub(crate) fn upload_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        (handle, srgb): (TextureHandle, bool),
        source: &Texture,
        encoder: &mut Option<wgpu::CommandEncoder>,
    ) -> (u64, u32) {
        let std::collections::btree_map::Entry::Vacant(entry) = self.textures.entry((handle, srgb))
        else {
            return (0, 0);
        };
        let mut levels = 0;
        let gpu = create_texture(
            device,
            queue,
            source,
            srgb,
            &self.mips,
            encoder,
            &mut levels,
        );
        let bytes = gpu.byte_len;
        entry.insert(gpu);
        (bytes, levels)
    }

    pub(crate) fn texture_view(
        &self,
        handle: TextureHandle,
        srgb: bool,
    ) -> Option<&wgpu::TextureView> {
        self.textures
            .get(&(handle, srgb))
            .map(|texture| &texture.view)
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
            anisotropy_clamp: if trilinear { MATERIAL_ANISOTROPY } else { 1 },
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

impl MaterialUniform {
    pub const DEFAULT: Self = Self {
        transforms: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        emissive: [0.0; 4],
        params: [1.0, 1.0, 0.5, 0.0],
        variant: [0.0; 4],
        top: [0.0; 4],
        layer: [0.0; 4],
    };

    pub fn new(material: &Material) -> Self {
        let mut uniform = Self::DEFAULT;
        for (index, map) in material.maps().iter().enumerate() {
            if let Some(map) = map {
                let [first, second] = transform_rows(&map.transform);
                uniform.transforms[index * 2] = first;
                uniform.transforms[index * 2 + 1] = second;
            }
        }
        let [r, g, b] = material.emissive();
        uniform.emissive = [r, g, b, material.world_tiling().unwrap_or(0.0)];
        if let Some(top) = material.top_layer() {
            let [r, g, b] = top.color;
            uniform.top = [r, g, b, top.coverage];
            uniform.layer = [top.roughness, 0.0, 0.0, 0.0];
        }
        let cutoff = match material.alpha_mode() {
            AlphaMode::Mask { cutoff } => cutoff,
            AlphaMode::Opaque => 0.5,
        };
        let second = material
            .maps()
            .iter()
            .enumerate()
            .filter(|(_, map)| map.is_some_and(|map| map.tex_coord == 1))
            .fold(0_u32, |bits, (index, _)| bits | 1 << index);
        uniform.params = [
            material.normal_scale(),
            material.occlusion_strength(),
            cutoff,
            second as f32,
        ];
        uniform.variant = [f32::from_bits(Variant::of(material).bits()), 0.0, 0.0, 0.0];
        uniform
    }
}

pub fn material_variant(material: Option<&Material>, extended: bool) -> u32 {
    material
        .map_or(Variant::BASIC, Variant::of)
        .with_extended(extended)
        .bits()
}

pub fn transform_rows(transform: &UvTransform) -> [[f32; 4]; 2] {
    let (sin, cos) = engine_core::math::sin_cos(transform.rotation);
    let [sx, sy] = transform.scale;
    let [ox, oy] = transform.offset;
    [
        [cos * sx, sin * sy, ox, 0.0],
        [-sin * sx, cos * sy, oy, 0.0],
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Variant(u32);

impl Variant {
    pub const BASIC: Self = Self(0);
    const MAPS: u32 = constants::VARIANT_MAPS;
    const NORMAL_MAP: u32 = constants::VARIANT_NORMAL_MAP;
    const ALPHA_MASK: u32 = 4;
    const DOUBLE_SIDED: u32 = constants::VARIANT_DOUBLE_SIDED;
    const EXTENDED: u32 = 16;
    const METALLIC_ROUGHNESS_MAP: u32 = constants::VARIANT_METALLIC_ROUGHNESS_MAP;
    const OCCLUSION_MAP: u32 = constants::VARIANT_OCCLUSION_MAP;
    const EMISSIVE_MAP: u32 = constants::VARIANT_EMISSIVE_MAP;
    const PACKED_OCCLUSION: u32 = constants::VARIANT_PACKED_OCCLUSION;
    const UBER: u32 = 2048;
    const WORLD_MAPPING: u32 = constants::VARIANT_WORLD_MAPPING;
    const TOP_LAYER: u32 = constants::VARIANT_TOP_LAYER;
    const GROUNDING: u32 = 16384;
    const FADING: u32 = 32768;
    const FOLIAGE: u32 = constants::VARIANT_FOLIAGE;
    const PREPASS: u32 = 131072;
    const PRETESTED: u32 = 262144;

    pub fn of(material: &Material) -> Self {
        let mode = material.alpha_mode();
        let base = material.base_color_texture();
        let maps = material.metallic_roughness_texture().is_some()
            || material.normal_texture().is_some()
            || material.occlusion_texture().is_some()
            || material.emissive_texture().is_some()
            || material.emissive() != [0.0; 3]
            || base.is_some_and(|map| map.transform != UvTransform::IDENTITY)
            || matches!(mode, AlphaMode::Mask { .. })
            || material.world_tiling().is_some();
        let mut bits = 0;
        if maps {
            bits |= Self::MAPS;
            let metallic_roughness = material.metallic_roughness_texture();
            let occlusion = material.occlusion_texture();
            if metallic_roughness.is_some() {
                bits |= Self::METALLIC_ROUGHNESS_MAP;
            }
            if occlusion.is_some() && occlusion == metallic_roughness {
                bits |= Self::PACKED_OCCLUSION;
            } else if occlusion.is_some() {
                bits |= Self::OCCLUSION_MAP;
            }
            if material.emissive_texture().is_some() {
                bits |= Self::EMISSIVE_MAP;
            }
        }
        if material.normal_texture().is_some() {
            bits |= Self::NORMAL_MAP;
        }
        if matches!(mode, AlphaMode::Mask { .. }) {
            bits |= Self::ALPHA_MASK;
        }
        if material.is_double_sided() {
            bits |= Self::DOUBLE_SIDED;
        }
        if material.world_tiling().is_some() {
            bits |= Self::WORLD_MAPPING;
        }
        if material.top_layer().is_some() {
            bits |= Self::TOP_LAYER;
        }
        if material.is_foliage() {
            bits |= Self::FOLIAGE;
        }
        Self(bits)
    }

    #[must_use]
    pub fn uber(self) -> Self {
        let state = Self::EXTENDED | Self::ALPHA_MASK | Self::DOUBLE_SIDED;
        Self(self.0 & state | Self::UBER)
    }

    pub fn bits(self) -> u32 {
        self.0
    }

    pub fn constants(self) -> [(&'static str, f64); 15] {
        let set = |bit: u32| if self.0 & bit != 0 { 1.0 } else { 0.0 };
        [
            ("WORLD_MAPPING", set(Self::WORLD_MAPPING)),
            ("TOP_LAYER", set(Self::TOP_LAYER)),
            ("GROUNDING", set(Self::GROUNDING)),
            ("FADE_LIST", set(Self::FADING)),
            ("MATERIAL_MAPS", set(Self::MAPS)),
            ("NORMAL_MAP", set(Self::NORMAL_MAP)),
            ("ALPHA_MASK", set(Self::ALPHA_MASK)),
            ("DOUBLE_SIDED", set(Self::DOUBLE_SIDED)),
            ("METALLIC_ROUGHNESS_MAP", set(Self::METALLIC_ROUGHNESS_MAP)),
            ("OCCLUSION_MAP", set(Self::OCCLUSION_MAP)),
            ("EMISSIVE_MAP", set(Self::EMISSIVE_MAP)),
            ("PACKED_OCCLUSION", set(Self::PACKED_OCCLUSION)),
            ("UBER_MATERIALS", set(Self::UBER)),
            ("FOLIAGE", set(Self::FOLIAGE)),
            ("PRETESTED", set(Self::PRETESTED)),
        ]
    }

    pub fn double_sided(self) -> bool {
        self.0 & Self::DOUBLE_SIDED != 0
    }

    pub fn alpha_mask(self) -> bool {
        self.0 & Self::ALPHA_MASK != 0
    }

    #[must_use]
    pub fn opaque(self) -> Self {
        if self.alpha_mask() {
            Self(self.0 | Self::PRETESTED)
        } else {
            self
        }
    }

    #[must_use]
    pub fn prepass(self) -> Self {
        Self(self.0 & !Self::PRETESTED | Self::PREPASS)
    }

    pub fn depth_only(self) -> bool {
        self.0 & Self::PREPASS != 0
    }

    pub fn pretested(self) -> bool {
        self.0 & Self::PRETESTED != 0
    }

    #[must_use]
    pub fn with_extended(self, extended: bool) -> Self {
        Self(if extended {
            self.0 | Self::EXTENDED
        } else {
            self.0 & !Self::EXTENDED
        })
    }

    pub fn extended(self) -> bool {
        self.0 & Self::EXTENDED != 0
    }

    #[must_use]
    pub fn with_fading(self, fading: bool) -> Self {
        Self(if fading {
            self.0 | Self::FADING
        } else {
            self.0 & !Self::FADING
        })
    }

    #[must_use]
    pub fn with_grounding(self, grounding: bool) -> Self {
        Self(if grounding {
            self.0 | Self::GROUNDING
        } else {
            self.0 & !Self::GROUNDING
        })
    }
}
