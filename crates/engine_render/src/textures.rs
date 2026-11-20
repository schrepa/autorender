use engine_core::collections::HashMap;
use std::collections::BTreeMap;

use bevy_ecs::world::World;
use engine_core::collections::{Cursor, Since};
use engine_core::{
    Asset, Assets, ColorSpace, Compare, Error, Filter, Sampler, Texture, TextureFormat, Wrap,
};

use crate::gpu_data::{level_copy, mip_levels};

pub(crate) struct GpuTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    pub byte_len: u64,
    target: bool,
}

#[derive(Default)]
pub(crate) struct TextureStore {
    samplers: HashMap<Sampler, wgpu::Sampler>,
    textures: BTreeMap<Asset<Texture>, GpuTexture>,
    generation: u64,
    cursor: Cursor,
    refused: Vec<Error>,
}

impl TextureStore {
    pub(crate) fn take_refused(&mut self) -> Vec<Error> {
        std::mem::take(&mut self.refused)
    }

    pub(crate) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        world: &World,
    ) -> u64 {
        let Some(assets) = world.get_resource::<Assets>() else {
            return 0;
        };
        let mut listed: Vec<Asset<Texture>> = match assets.changed(&mut self.cursor) {
            Since::Same => return 0,
            Since::Listed(ids) => ids
                .iter()
                .filter_map(|&id| assets.asset::<Texture>(id))
                .collect(),
            Since::All => self
                .textures
                .keys()
                .copied()
                .chain(assets.each::<Texture>().map(|(handle, _)| handle))
                .collect(),
        };
        listed.sort_unstable();
        listed.dedup();
        let mut bytes = 0;
        for handle in listed {
            if self.textures.remove(&handle).is_some() {
                self.generation += 1;
            }
            if let Ok(source) = assets.get(handle) {
                bytes += self.upload_texture(device, queue, handle, source);
            }
        }
        bytes
    }

    fn upload_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        handle: Asset<Texture>,
        source: &Texture,
    ) -> u64 {
        match create_texture(device, queue, (handle, source)) {
            Ok(gpu) => {
                let bytes = gpu.byte_len;
                self.textures.insert(handle, gpu);
                self.generation += 1;
                bytes
            }
            Err(error) => {
                self.refused.push(error);
                0
            }
        }
    }

    pub(crate) fn texture_view(&self, handle: Asset<Texture>) -> Option<&wgpu::TextureView> {
        self.textures.get(&handle).map(|texture| &texture.view)
    }

    pub(crate) fn target(
        &self,
        handle: Asset<Texture>,
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

    pub(crate) fn is_texture_resident(&self, handle: Asset<Texture>) -> bool {
        self.textures.contains_key(&handle)
    }

    #[cfg_attr(
        not(feature = "dev"),
        expect(dead_code, reason = "read by the gpu.assets names")
    )]
    pub(crate) fn texture_formats(&self) -> BTreeMap<String, (u32, u64)> {
        let mut formats = BTreeMap::new();
        for texture in self.textures.values() {
            let held: &mut (u32, u64) = formats
                .entry(format!("{:?}", texture.texture.format()))
                .or_default();
            held.0 += 1;
            held.1 += texture.byte_len;
        }
        formats
    }

    #[cfg_attr(
        not(feature = "dev"),
        expect(dead_code, reason = "read by the gpu.assets names")
    )]
    pub(crate) fn texture_totals(&self) -> (u32, u64) {
        self.textures
            .values()
            .fold((0, 0), |(count, bytes), texture| {
                (count + 1, bytes + texture.byte_len)
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

pub(crate) fn gpu_format(
    format: TextureFormat,
    space: ColorSpace,
) -> Option<(wgpu::TextureFormat, wgpu::Features)> {
    use wgpu::TextureFormat as Gpu;
    let srgb = match space {
        ColorSpace::Srgb => true,
        ColorSpace::Linear => false,
    };
    Some(match format {
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
        TextureFormat::Bc5 if srgb => return None,
        TextureFormat::Bc5 => (Gpu::Bc5RgUnorm, wgpu::Features::TEXTURE_COMPRESSION_BC),
    })
}

fn write_level(
    queue: &crate::gpu_calls::Queue,
    texture: &wgpu::Texture,
    level: u32,
    (width, height): (u32, u32),
    format: TextureFormat,
    bytes: &[u8],
) {
    let (layout, extent) = level_copy(format, (width, height), level);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        layout,
        extent,
    );
}

#[expect(clippy::expect_used, reason = "a target texture is rgba8")]
fn create_target(device: &wgpu::Device, source: &Texture) -> GpuTexture {
    let (format, _) =
        gpu_format(source.format(), source.color_space()).expect("targets_are_rgba8");
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
        target: true,
    }
}

fn create_texture(
    device: &wgpu::Device,
    queue: &crate::gpu_calls::Queue,
    (handle, source): (Asset<Texture>, &Texture),
) -> Result<GpuTexture, Error> {
    let path = format!("$.textures.{}", handle.slot());
    let most = device.limits().max_texture_dimension_2d;
    if source.width().max(source.height()) > most {
        return Err(Error::new(engine_core::ErrorCode::InvalidTexture, path)
            .at_most(u64::from(most))
            .got([source.width(), source.height()]));
    }
    if source.is_target() {
        return Ok(create_target(device, source));
    }
    let (width, height) = (source.width(), source.height());
    let (levels, chain) = (source.levels().len() as u32, mip_levels(width, height));
    if source.makes_levels() && chain > levels {
        return Err(Error::new(engine_core::ErrorCode::InvalidTexture, path)
            .expected(engine_core::Expected::Matching)
            .range(f64::from(chain), f64::from(chain))
            .got(serde_json::json!({ "levels": levels })));
    }
    let space = source.color_space();
    let unsupported = |feature: Option<wgpu::Features>| {
        Error::new(engine_core::ErrorCode::InvalidTexture, path.clone())
            .expected(engine_core::Expected::Supported)
            .got(serde_json::json!({
                "format": format!("{:?}", source.format()),
                "space": format!("{space:?}"),
                "missing_feature": feature.map(|feature| format!("{feature:?}")),
            }))
    };
    let (format, feature) =
        gpu_format(source.format(), space).ok_or_else(|| unsupported(None))?;
    if !device.features().contains(feature) {
        return Err(unsupported(Some(feature)));
    }
    let label = match (source.format().is_compressed(), space) {
        (true, _) => "engine/compressed-texture",
        (false, ColorSpace::Srgb) => "engine/color-texture",
        (false, ColorSpace::Linear) => "engine/data-texture",
    };
    let texture = crate::targets::texture_2d(
        device,
        label,
        (width, height),
        1,
        levels,
        format,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    for (level, bytes) in source.levels().enumerate() {
        write_level(queue, &texture, level as u32, (width, height), source.format(), bytes);
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok(GpuTexture {
        texture,
        view,
        byte_len: source.byte_len() as u64,
        target: false,
    })
}
