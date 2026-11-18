#![expect(
    clippy::expect_used,
    reason = "residency admits a material only while the library holds its textures"
)]

use engine_core::collections::HashMap;
use std::collections::BTreeMap;

use bevy_ecs::world::World;
use engine_core::{
    AssetLibrary, AssetResidency, ColorSpace, Compare, Error, Filter, Sampler, Texture,
    TextureFormat, TextureHandle, Wrap,
};

use crate::encode::Compiler;
use crate::gpu_data::{level_copy, mip_chain_bytes, mip_levels};
use crate::mipmaps::MipGenerator;

pub(crate) struct GpuTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    pub byte_len: u64,
    pub mip_bytes: u64,
    target: bool,
    revision: u64,
}

pub(crate) struct TextureStore {
    samplers: HashMap<Sampler, wgpu::Sampler>,
    textures: BTreeMap<TextureHandle, GpuTexture>,
    generation: u64,
    synced: Option<u64>,
    mips: MipGenerator,
    owed: Vec<TextureHandle>,
    refused: Vec<Error>,
}

impl TextureStore {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            samplers: HashMap::default(),
            textures: BTreeMap::new(),
            generation: 0,
            synced: None,
            mips: MipGenerator::new(device),
            owed: Vec::new(),
            refused: Vec::new(),
        }
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
        let mut levels = self.record_owed(device, (&mut *encoder, &mut *compiler));
        if self.synced == Some(generation) {
            return (0, levels);
        }
        let before = self.textures.len();
        self.textures.retain(|handle, gpu| {
            residency.texture_resident(*handle)
                && library.revision_of(*handle) == Some(gpu.revision)
        });
        if self.textures.len() != before {
            self.generation += 1;
        }
        let mut bytes = 0;
        for (handle, _) in residency.materials() {
            let Some(material) = library.material(handle) else {
                continue;
            };
            for texture in material.textures() {
                if self.textures.contains_key(&texture) {
                    continue;
                }
                let source = library
                    .texture(texture)
                    .expect("material_textures_validated");
                let revision = library.revision_of(texture).unwrap_or_default();
                let gpu = create_texture(
                    device,
                    queue,
                    (texture, source, revision),
                    (&self.mips, &mut *compiler),
                    encoder,
                    (&mut levels, &mut self.refused, &mut self.owed),
                );
                bytes += gpu.byte_len;
                self.textures.insert(texture, gpu);
                self.generation += 1;
            }
        }
        for handle in residency.textures() {
            if self.textures.contains_key(&handle) {
                continue;
            }
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

    fn record_owed(
        &mut self,
        device: &wgpu::Device,
        (encoder, compiler): (&mut Option<wgpu::CommandEncoder>, &mut Compiler<'_>),
    ) -> u32 {
        let mut levels = 0;
        for handle in std::mem::take(&mut self.owed) {
            let Some(gpu) = self.textures.get(&handle) else {
                continue;
            };
            let encoder = encoder.get_or_insert_with(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/texture-mips"),
                })
            });
            match self
                .mips
                .record((device, encoder), compiler, &gpu.texture, None)
            {
                Some(more) => levels += more,
                None => self.owed.push(handle),
            }
        }
        levels
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
            (handle, source, revision),
            (&self.mips, compiler),
            encoder,
            (&mut levels, &mut self.refused, &mut self.owed),
        );
        let bytes = gpu.byte_len;
        entry.insert(gpu);
        self.generation += 1;
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

    pub(crate) fn sampler(&mut self, device: &wgpu::Device, key: Sampler) -> &wgpu::Sampler {
        sampler(device, &mut self.samplers, key)
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn mips(&self) -> &MipGenerator {
        &self.mips
    }

    pub(crate) fn is_texture_resident(&self, handle: TextureHandle) -> bool {
        self.textures.contains_key(&handle)
    }

    #[cfg_attr(
        not(any(feature = "dev", feature = "profiling")),
        expect(dead_code, reason = "read by the gpu.assets names")
    )]
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
    cache
        .entry(key)
        .or_insert_with(|| gpu_sampler(device, "engine/asset-sampler", key))
}

#[expect(clippy::disallowed_methods, reason = "the one way samplers are made")]
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
    let texture = crate::gpu_calls::create_texture(device, &wgpu::TextureDescriptor {
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
    });
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
    (handle, source, revision): (TextureHandle, &Texture, u64),
    (mips, compiler): (&MipGenerator, &mut Compiler<'_>),
    encoder: &mut Option<wgpu::CommandEncoder>,
    (generated, refused, owed): (&mut u32, &mut Vec<Error>, &mut Vec<TextureHandle>),
) -> GpuTexture {
    let most = device.limits().max_texture_dimension_2d;
    let fits = source.width().max(source.height()) <= most;
    if !fits {
        refused.push(
            Error::new(
                engine_core::ErrorCode::InvalidTexture,
                format!("$.textures.{}", handle.slot()),
            )
            .at_most(u64::from(most))
            .got([source.width(), source.height()]),
        );
    }
    if source.is_target() && fits {
        return create_target(device, source, revision);
    }
    let space = source.color_space();
    let (format, feature) = gpu_format(source.format(), space);
    if !fits || !device.features().contains(feature) {
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
        match mips.record((device, encoder), compiler, &texture, None) {
            Some(levels) => *generated += levels,
            None => owed.push(handle),
        }
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
