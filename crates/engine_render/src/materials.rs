#![expect(
    clippy::expect_used,
    reason = "residency admits a material only while the library holds its textures"
)]

use engine_core::collections::HashMap;
use std::collections::BTreeMap;

use bevy_ecs::world::World;
use engine_core::{
    AssetLibrary, AssetResidency, ColorSpace, Compare, Error, Filter, Material, MaterialHandle,
    NORMAL_MAP, Sampler, Texture, TextureFormat, TextureHandle, Wrap,
};

use crate::bindings::Table;
use crate::encode::Compiler;
use crate::gpu_data::{level_copy, mip_chain_bytes, mip_levels};
use crate::mipmaps::MipGenerator;
use crate::surface::{MaterialLayout, MaterialVariant, RECORD};

pub(crate) struct GpuTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    pub byte_len: u64,
    pub mip_bytes: u64,
    target: bool,
    revision: u64,
}

pub(crate) struct GpuMaterial {
    group: wgpu::BindGroup,
    _uniform: wgpu::Buffer,
    variant: MaterialVariant,
    revision: u64,
    textures: Vec<(TextureHandle, u64)>,
}

struct Defaults {
    _textures: [wgpu::Texture; 2],
    white: wgpu::TextureView,
    flat_normal: wgpu::TextureView,
    _uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    variant: MaterialVariant,
}

pub struct MaterialStore {
    layout: MaterialLayout,
    table: &'static Table,
    group_layout: crate::pipelines::GroupLayout,
    samplers: HashMap<Sampler, wgpu::Sampler>,
    textures: BTreeMap<TextureHandle, GpuTexture>,
    materials: BTreeMap<MaterialHandle, GpuMaterial>,
    generation: u64,
    synced: Option<u64>,
    defaults: Defaults,
    mips: MipGenerator,
    refused: Vec<Error>,
}

impl MaterialStore {
    pub(crate) fn new(device: &wgpu::Device, queue: &crate::gpu_calls::Queue) -> Self {
        let layout = MaterialLayout::standard();
        let table = layout.table();
        let group_layout = crate::pipelines::layout(device, table);
        let texel =
            |label, format, pixel| crate::targets::texel(device, queue, label, format, pixel);
        let textures = [
            texel("engine/white", wgpu::TextureFormat::Rgba8Unorm, [255; 4]),
            texel(
                "engine/flat-normal",
                wgpu::TextureFormat::Rgba8Unorm,
                [128, 128, 255, 255],
            ),
        ];
        let [white, flat_normal] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let mut samplers = HashMap::default();
        sampler(device, &mut samplers, Sampler::CLAMP);
        let (group, uniform) = default_group(
            device,
            (&layout, table, &group_layout),
            (&white, &flat_normal),
            &samplers[&Sampler::CLAMP],
        );
        let variant = layout.variant(None, &layout.plain_record());
        Self {
            layout,
            table,
            group_layout,
            samplers,
            textures: BTreeMap::new(),
            materials: BTreeMap::new(),
            generation: 0,
            synced: None,
            defaults: Defaults {
                _textures: textures,
                white,
                flat_normal,
                _uniform: uniform,
                group,
                variant,
            },
            mips: MipGenerator::new(device),
            refused: Vec::new(),
        }
    }

    pub(crate) fn configure(&mut self, device: &wgpu::Device, layout: MaterialLayout) {
        if self.layout == layout {
            return;
        }
        let table = layout.table();
        self.group_layout = crate::pipelines::layout(device, table);
        let (group, uniform) = default_group(
            device,
            (&layout, table, &self.group_layout),
            (&self.defaults.white, &self.defaults.flat_normal),
            &self.samplers[&Sampler::CLAMP],
        );
        self.defaults.group = group;
        self.defaults._uniform = uniform;
        self.defaults.variant = layout.variant(None, &layout.plain_record());
        self.layout = layout;
        self.table = table;
        self.materials.clear();
        self.generation += 1;
        self.synced = None;
    }

    pub(crate) fn take_refused(&mut self) -> Vec<Error> {
        std::mem::take(&mut self.refused)
    }

    pub(crate) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        world: &World,
        (encoder, compiler): (&mut Option<wgpu::CommandEncoder>, &mut Compiler<'_>),
    ) -> (u64, u32) {
        let (Some(residency), Some(library)) = (
            world.get_resource::<AssetResidency>(),
            world.get_resource::<AssetLibrary>(),
        ) else {
            return (0, 0);
        };
        let generation = residency.revision();
        if self.synced == Some(generation) {
            return (0, 0);
        }
        self.textures.retain(|handle, gpu| {
            residency.texture_resident(*handle) && library.revision_of(*handle) == Some(gpu.revision)
        });
        let textures = &self.textures;
        let count = self.materials.len();
        self.materials.retain(|handle, gpu| {
            residency.material_resident(*handle)
                && gpu.textures.iter().all(|(texture, revision)| {
                    textures
                        .get(texture)
                        .is_some_and(|held| held.revision == *revision)
                })
        });
        if self.materials.len() != count {
            self.generation += 1;
        }
        let mut bytes = 0;
        let mut levels = 0;
        for (handle, revision) in residency.materials() {
            if self
                .materials
                .get(&handle)
                .is_some_and(|gpu| gpu.revision == revision)
            {
                continue;
            }
            let Some(material) = library.material(handle) else {
                continue;
            };
            for texture in material.textures() {
                if self.textures.contains_key(&texture) {
                    continue;
                }
                let source = library.texture(texture).expect("material_textures_validated");
                let revision = library.revision_of(texture).unwrap_or_default();
                let gpu = create_texture(
                    device,
                    queue,
                    (source, revision),
                    (&self.mips, &mut *compiler),
                    encoder,
                    &mut levels,
                );
                bytes += gpu.byte_len;
                self.textures.insert(texture, gpu);
            }
            let gpu = self.create_material(device, handle, material, revision);
            self.materials.insert(handle, gpu);
            self.generation += 1;
        }
        for handle in residency.textures() {
            if let Some(source) = library.texture(handle) {
                let revision = library.revision_of(handle).unwrap_or_default();
                let (more, generated) = self.upload_texture(
                    device,
                    queue,
                    (handle, revision),
                    source,
                    (&mut *encoder, &mut *compiler),
                );
                bytes += more;
                levels += generated;
            }
        }
        self.synced = Some(generation);
        (bytes, levels)
    }

    pub(crate) fn forget_sync(&mut self) {
        self.synced = None;
    }

    pub(crate) fn upload_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        (handle, revision): (TextureHandle, u64),
        source: &Texture,
        (encoder, compiler): (&mut Option<wgpu::CommandEncoder>, &mut Compiler<'_>),
    ) -> (u64, u32) {
        let std::collections::btree_map::Entry::Vacant(entry) = self.textures.entry(handle) else {
            return (0, 0);
        };
        let mut levels = 0;
        let gpu = create_texture(
            device,
            queue,
            (source, revision),
            (&self.mips, compiler),
            encoder,
            &mut levels,
        );
        let bytes = gpu.byte_len;
        entry.insert(gpu);
        (bytes, levels)
    }

    pub(crate) fn texture_view(&self, handle: TextureHandle) -> Option<&wgpu::TextureView> {
        self.textures.get(&handle).map(|texture| &texture.view)
    }

    pub(crate) fn target(
        &self,
        handle: TextureHandle,
    ) -> Option<(&wgpu::Texture, &wgpu::TextureView)> {
        self.textures
            .get(&handle)
            .filter(|texture| texture.target)
            .map(|texture| (&texture.texture, &texture.view))
    }

    fn create_material(
        &mut self,
        device: &wgpu::Device,
        handle: MaterialHandle,
        material: &Material,
        revision: u64,
    ) -> GpuMaterial {
        let (words, problems) = self.layout.record(handle, material);
        self.refused.extend(problems);
        let variant = self.layout.variant(Some(material), &words);
        for (_, map) in material.maps() {
            sampler(device, &mut self.samplers, map.sampler);
        }
        let mut views = Vec::with_capacity(self.layout.maps().len());
        let mut used = Vec::new();
        for name in self.layout.maps() {
            let found = material.map(name).and_then(|map| {
                let held = self.textures.get(&map.texture)?;
                Some((map, held))
            });
            match found {
                Some((map, held)) => {
                    used.push((map.texture, held.revision));
                    views.push((&held.view, &self.samplers[&map.sampler]));
                }
                None => views.push((
                    default_view(name, &self.defaults.white, &self.defaults.flat_normal),
                    &self.samplers[&Sampler::CLAMP],
                )),
            }
        }
        let uniform = uniform_buffer(device, &words);
        GpuMaterial {
            group: bind_group(
                device,
                (&self.layout, self.table, &self.group_layout),
                &views,
                &uniform,
            ),
            _uniform: uniform,
            variant,
            revision,
            textures: used,
        }
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub fn mips(&self) -> &MipGenerator {
        &self.mips
    }

    pub fn group(&self, material: Option<MaterialHandle>) -> Option<&wgpu::BindGroup> {
        match material {
            Some(handle) => self.materials.get(&handle).map(|material| &material.group),
            None => Some(&self.defaults.group),
        }
    }

    pub(crate) fn variant(&self, material: Option<MaterialHandle>) -> Option<MaterialVariant> {
        match material {
            Some(handle) => self.materials.get(&handle).map(|material| material.variant),
            None => Some(self.defaults.variant),
        }
    }

    pub fn is_texture_resident(&self, handle: TextureHandle) -> bool {
        self.textures.contains_key(&handle)
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

fn default_view<'a>(
    name: &str,
    white: &'a wgpu::TextureView,
    flat_normal: &'a wgpu::TextureView,
) -> &'a wgpu::TextureView {
    if name == NORMAL_MAP { flat_normal } else { white }
}

fn default_group(
    device: &wgpu::Device,
    (layout, table, group_layout): (&MaterialLayout, &Table, &crate::pipelines::GroupLayout),
    (white, flat_normal): (&wgpu::TextureView, &wgpu::TextureView),
    clamp: &wgpu::Sampler,
) -> (wgpu::BindGroup, wgpu::Buffer) {
    let views: Vec<(&wgpu::TextureView, &wgpu::Sampler)> = layout
        .maps()
        .iter()
        .map(|name| (default_view(name, white, flat_normal), clamp))
        .collect();
    let uniform = uniform_buffer(device, &layout.plain_record());
    let group = bind_group(device, (layout, table, group_layout), &views, &uniform);
    (group, uniform)
}

fn sampler<'a>(
    device: &wgpu::Device,
    cache: &'a mut HashMap<Sampler, wgpu::Sampler>,
    key: Sampler,
) -> &'a wgpu::Sampler {
    cache
        .entry(key)
        .or_insert_with(|| gpu_sampler(device, "engine/material-sampler", key))
}

pub(crate) fn gpu_sampler(
    device: &wgpu::Device,
    label: &'static str,
    sampler: Sampler,
) -> wgpu::Sampler {
    let address = |wrap| match wrap {
        Wrap::Repeat => wgpu::AddressMode::Repeat,
        Wrap::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        Wrap::ClampToEdge => wgpu::AddressMode::ClampToEdge,
    };
    let filter = |filter| match filter {
        Filter::Nearest => wgpu::FilterMode::Nearest,
        Filter::Linear => wgpu::FilterMode::Linear,
    };
    let trilinear = sampler.mag_filter == Filter::Linear
        && sampler.min_filter == Filter::Linear
        && sampler.mipmap_filter == Some(Filter::Linear);
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: address(sampler.wrap_u),
        address_mode_v: address(sampler.wrap_v),
        address_mode_w: address(sampler.wrap_w),
        mag_filter: filter(sampler.mag_filter),
        min_filter: filter(sampler.min_filter),
        mipmap_filter: match sampler.mipmap_filter {
            Some(Filter::Nearest) | None => wgpu::MipmapFilterMode::Nearest,
            Some(Filter::Linear) => wgpu::MipmapFilterMode::Linear,
        },
        lod_max_clamp: if sampler.mipmap_filter.is_some() {
            32.0
        } else {
            0.0
        },
        compare: sampler.compare.map(|compare| match compare {
            Compare::Never => wgpu::CompareFunction::Never,
            Compare::Less => wgpu::CompareFunction::Less,
            Compare::Equal => wgpu::CompareFunction::Equal,
            Compare::LessEqual => wgpu::CompareFunction::LessEqual,
            Compare::Greater => wgpu::CompareFunction::Greater,
            Compare::NotEqual => wgpu::CompareFunction::NotEqual,
            Compare::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
            Compare::Always => wgpu::CompareFunction::Always,
        }),
        anisotropy_clamp: if trilinear {
            sampler.anisotropy.max(1)
        } else {
            1
        },
        ..Default::default()
    })
}

fn uniform_buffer(device: &wgpu::Device, words: &[u32]) -> wgpu::Buffer {
    crate::memory::buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("engine/material-record"),
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    )
}

fn bind_group(
    device: &wgpu::Device,
    (layout, table, group_layout): (&MaterialLayout, &Table, &crate::pipelines::GroupLayout),
    views: &[(&wgpu::TextureView, &wgpu::Sampler)],
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let names: Vec<(String, String)> = layout
        .maps()
        .iter()
        .map(|name| (format!("{name}_map"), format!("{name}_sampler")))
        .collect();
    let mut resources = Vec::with_capacity(2 * names.len() + 1);
    for ((texture_name, sampler_name), (view, sampler)) in names.iter().zip(views) {
        resources.push((texture_name.as_str(), crate::pipelines::view(view)));
        resources.push((sampler_name.as_str(), crate::pipelines::sampler(sampler)));
    }
    resources.push((RECORD, uniform.as_entire_binding()));
    crate::pipelines::group(device, "engine/material-group", table, group_layout, &resources)
}

pub fn gpu_format(
    format: TextureFormat,
    space: ColorSpace,
) -> (wgpu::TextureFormat, wgpu::Features) {
    use wgpu::TextureFormat as Gpu;
    let srgb = match space {
        ColorSpace::Srgb => true,
        ColorSpace::Linear => false,
    };
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
    let (layout, extent) = level_copy(format, (width, height), level);
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
        layout,
        extent,
    );
}

fn create_target(device: &wgpu::Device, source: &Texture, revision: u64) -> GpuTexture {
    let (format, _) = gpu_format(source.format(), source.color_space());
    let texture = crate::memory::texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("engine/target-texture"),
            size: wgpu::Extent3d {
                width: source.width(),
                height: source.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[format.remove_srgb_suffix()],
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    GpuTexture {
        byte_len: TextureFormat::Rgba8.level_bytes(source.width(), source.height()) as u64,
        texture,
        view,
        mip_bytes: 0,
        target: true,
        revision,
    }
}

fn create_texture(
    device: &wgpu::Device,
    queue: &crate::gpu_calls::Queue,
    (source, revision): (&Texture, u64),
    (mips, compiler): (&MipGenerator, &mut Compiler<'_>),
    encoder: &mut Option<wgpu::CommandEncoder>,
    generated: &mut u32,
) -> GpuTexture {
    if source.is_target() {
        return create_target(device, source, revision);
    }
    let space = source.color_space();
    let (format, feature) = gpu_format(source.format(), space);
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
            texture,
            view,
            byte_len: 4,
            mip_bytes: 0,
            target: false,
            revision,
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
    let label = match (source.format().is_compressed(), space) {
        (true, _) => "engine/compressed-texture",
        (false, ColorSpace::Srgb) => "engine/color-texture",
        (false, ColorSpace::Linear) => "engine/data-texture",
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
        *generated += mips.record((device, encoder), compiler, &texture, None);
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    GpuTexture {
        texture,
        view,
        byte_len: source.byte_len() as u64,
        mip_bytes: if generate && levels > 1 {
            mip_chain_bytes(width, height)
        } else {
            0
        },
        target: false,
        revision,
    }
}
