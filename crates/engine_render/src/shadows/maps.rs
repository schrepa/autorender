use crate::cached::{Bound, Versioned};
use crate::gpu_data::ShadowUniform;
use crate::gpu_layouts::constants::{SHADOW_APRON, SHADOW_REGIONS};
use crate::pipelines::{sampler, view};
use crate::shadow_fit::{LEVEL_TEXELS, region_tiles};

pub(crate) const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;
const TABLE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AtlasLayout {
    pub(crate) slot: u32,
    pub(crate) per_row: u32,
    pub(crate) rows: u32,
    pub(crate) still: u32,
    pub(crate) moving: u32,
}

impl AtlasLayout {
    fn new(tile_texels: u32, (still, moving): (u32, u32), limit: u32) -> Self {
        let slot = tile_texels + 2 * SHADOW_APRON;
        let most = (limit / slot).max(1);
        let wanted = (still + moving).max(1);
        let per_row = (f64::from(wanted).sqrt().ceil() as u32).clamp(1, most);
        let rows = wanted.div_ceil(per_row).min(most);
        let held = per_row * rows;
        let still = still.min(held);
        Self {
            slot,
            per_row,
            rows,
            still,
            moving: moving.min(held - still),
        }
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        (self.per_row * self.slot, self.rows * self.slot)
    }

    pub(crate) fn texels(&self, slot: u32) -> [f32; 4] {
        let x = (slot % self.per_row * self.slot) as f32;
        let y = (slot / self.per_row * self.slot) as f32;
        [x, y, x + self.slot as f32, y + self.slot as f32]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TexturesKey {
    atlas: AtlasLayout,
    tile_texels: u32,
    levels: u32,
    side: u32,
}

pub(super) struct MapTextures {
    key: TexturesKey,
    atlas: wgpu::Texture,
    pub(super) atlas_view: wgpu::TextureView,
    regions: wgpu::Texture,
    regions_array: wgpu::TextureView,
    pub(super) region_layers: Vec<wgpu::TextureView>,
    table: wgpu::Texture,
    table_view: wgpu::TextureView,
}

impl Drop for MapTextures {
    fn drop(&mut self) {
        for texture in [&self.atlas, &self.regions, &self.table] {
            texture.destroy();
        }
    }
}

impl MapTextures {
    fn new(device: &wgpu::Device, key: TexturesKey) -> Self {
        let (width, height) = key.atlas.size();
        let atlas = crate::targets::texture_2d(
            device,
            "engine/shadow-atlas",
            (width, height),
            1,
            1,
            SHADOW_FORMAT,
            crate::targets::RENDER_TARGET,
        );
        let side = region_tiles(key.tile_texels) * key.tile_texels + 2 * SHADOW_APRON;
        let regions = crate::targets::texture_2d(
            device,
            "engine/shadow-regions",
            (side, side),
            SHADOW_REGIONS as u32,
            1,
            SHADOW_FORMAT,
            crate::targets::RENDER_TARGET,
        );
        let table = crate::targets::texture_2d(
            device,
            "engine/shadow-tiles",
            (key.side, key.side * key.levels),
            1,
            1,
            TABLE_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let region_layers = (0..SHADOW_REGIONS as u32)
            .map(|layer| {
                regions.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("engine/shadow-region"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        Self {
            key,
            atlas_view: atlas.create_view(&wgpu::TextureViewDescriptor::default()),
            regions_array: regions.create_view(&wgpu::TextureViewDescriptor {
                label: Some("engine/shadow-regions"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }),
            region_layers,
            table_view: table.create_view(&wgpu::TextureViewDescriptor::default()),
            atlas,
            regions,
            table,
        }
    }
}

pub(super) struct ShadowMaps {
    textures: Versioned<Option<MapTextures>>,
    uniform: wgpu::Buffer,
    written: Option<ShadowUniform>,
    sampler: wgpu::Sampler,
    texel_sampler: wgpu::Sampler,
    _placeholders: [wgpu::Texture; 3],
    placeholder_atlas: wgpu::TextureView,
    placeholder_regions: wgpu::TextureView,
    placeholder_table: wgpu::TextureView,
}

impl ShadowMaps {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let uniform = crate::buffers::uniform(
            device,
            "engine/shadow-uniform",
            size_of::<ShadowUniform>() as u64,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/shadow-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let texel_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/shadow-texel-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let placeholder = |label, format, layers| {
            crate::targets::texture_2d(
                device,
                label,
                (1, 1),
                layers,
                1,
                format,
                wgpu::TextureUsages::TEXTURE_BINDING,
            )
        };
        let atlas = placeholder("engine/shadow-atlas-placeholder", SHADOW_FORMAT, 1);
        let regions = placeholder("engine/shadow-regions-placeholder", SHADOW_FORMAT, 1);
        let table = placeholder("engine/shadow-tiles-placeholder", TABLE_FORMAT, 1);
        let view = |texture: &wgpu::Texture, dimension| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(dimension),
                ..Default::default()
            })
        };
        Self {
            textures: Versioned::new(None),
            uniform,
            written: None,
            sampler,
            texel_sampler,
            placeholder_atlas: view(&atlas, wgpu::TextureViewDimension::D2),
            placeholder_regions: view(&regions, wgpu::TextureViewDimension::D2Array),
            placeholder_table: view(&table, wgpu::TextureViewDimension::D2),
            _placeholders: [atlas, regions, table],
        }
    }

    pub(super) fn textures(&self) -> Option<&MapTextures> {
        Option::as_ref(&self.textures)
    }

    pub(super) fn entries(&self) -> [(&'static str, Bound<'_>); 5] {
        let (atlas, table) = match &*self.textures {
            Some(maps) => (&maps.atlas_view, &maps.table_view),
            None => (&self.placeholder_atlas, &self.placeholder_table),
        };
        let map = |texture| Bound::of(&self.textures, view(texture));
        [
            ("shadow", Bound::fixed(self.uniform.as_entire_binding())),
            ("shadow_atlas", map(atlas)),
            ("shadow_tiles", map(table)),
            ("shadow_sampler", Bound::fixed(sampler(&self.sampler))),
            ("shadow_texels", Bound::fixed(sampler(&self.texel_sampler))),
        ]
    }

    pub(super) fn uniform(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    pub(super) fn table(&self) -> (&wgpu::TextureView, u64) {
        let table = self
            .textures()
            .map_or(&self.placeholder_table, |maps| &maps.table_view);
        (table, self.textures.generation())
    }

    pub(super) fn capacity_bytes(&self) -> u64 {
        self.textures().map_or(0, |maps| {
            let (width, height) = maps.key.atlas.size();
            let side = u64::from(maps.regions.width());
            u64::from(width) * u64::from(height) * 2
                + side * side * 2 * SHADOW_REGIONS as u64
                + u64::from(maps.table.width()) * u64::from(maps.table.height()) * 4
        })
    }

    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        settings: &engine_core::Shadows,
        limit: u32,
    ) -> (AtlasLayout, bool) {
        let key = TexturesKey {
            atlas: AtlasLayout::new(
                settings.tile_texels,
                (settings.atlas_slots, settings.moving_tiles),
                limit,
            ),
            tile_texels: settings.tile_texels,
            levels: settings.levels,
            side: (LEVEL_TEXELS / settings.tile_texels).max(1),
        };
        if self.textures().is_some_and(|maps| maps.key == key) {
            return (key.atlas, false);
        }
        self.textures.replace(Some(MapTextures::new(device, key)));
        (key.atlas, true)
    }

    pub(super) fn regions_bound(&self) -> Option<Bound<'_>> {
        let maps = self.textures()?;
        Some(Bound::of(&self.textures, view(&maps.regions_array)))
    }

    pub(super) fn regions_placeholder(&self) -> Bound<'_> {
        Bound::fixed(view(&self.placeholder_regions))
    }

    pub(super) fn release(&mut self) {
        if self.textures.is_some() {
            self.textures.replace(None);
        }
    }

    pub(super) fn write_uniform(&mut self, queue: &wgpu::Queue, uniform: ShadowUniform) {
        if self.written != Some(uniform) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
    }

    pub(super) fn write_horizon_records(&mut self, queue: &wgpu::Queue, count: f32) {
        if let Some(written) = &mut self.written
            && written.level_state[3] != count
        {
            written.level_state[3] = count;
            let offset = std::mem::offset_of!(ShadowUniform, level_state) + 12;
            queue.write_buffer(&self.uniform, offset as u64, bytemuck::bytes_of(&count));
        }
    }

    pub(super) fn write_table(&self, queue: &wgpu::Queue, table: &[u32], changed: &[usize]) {
        let Some(maps) = self.textures() else {
            return;
        };
        let side = maps.key.side;
        let cells = (side * side) as usize;
        for &level in changed {
            let Some(block) = table.get(level * cells..(level + 1) * cells) else {
                continue;
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &maps.table,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: level as u32 * side,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(block),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(side * 4),
                    rows_per_image: Some(side),
                },
                wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

pub(super) fn map_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<f32>,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("engine/shadow-map"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}
