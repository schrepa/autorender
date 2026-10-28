use super::frame::FrameState;
use super::*;
use crate::gpu_layouts::constants::{
    DEBUG_CASTS, DEBUG_FADING, DEBUG_KIND_CARD, DEBUG_KIND_COVER, DEBUG_KIND_FALLS,
    DEBUG_KIND_MESH, DEBUG_KIND_PLACED, DEBUG_KIND_PLACED_CARD, DEBUG_KIND_SKY, DEBUG_KIND_TERRAIN,
    DEBUG_KIND_WATER, DEBUG_LEVEL_SHIFT, DEBUG_NO_MATERIAL, DEBUG_PATCH_SHIFT, DEBUG_SHADOWED,
    DEBUG_SOURCE_SHIFT, DEBUG_SPAN_STEPS, DEBUG_SUN_BEYOND, DEBUG_SUN_HORIZON, DEBUG_SUN_LEVEL,
    WATER_SPAN_LEAST, WATER_SPAN_MOST,
};
use engine_core::glam::{Mat4, Vec4};
use engine_core::math::max;
use engine_core::{
    DetailShares, DrawnKind, MaterialShare, PixelProbe, SunLookup, TerrainProbe, WaterProbe,
    WaterReach,
};
use engine_terrain::Reach;
use std::{cell::RefCell, rc::Rc};

const IDS: u64 = 0;
const COUNT: u64 = 256;
const SCENE: u64 = 512;
const CANVAS: u64 = 768;
const TERRAIN: u64 = 1024;
const DEPTH: u64 = 1280;
const ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

type Mapped = Rc<RefCell<Option<Result<Vec<u8>, String>>>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Debug,
    Shaded,
    Submitted,
    Mapping,
}

struct Armed {
    pixel: (u32, u32),
    stage: Stage,
    buffer: wgpu::Buffer,
    armed_size: (u32, u32),
    depth_row: u32,
    size: (u32, u32),
    view_projection: Mat4,
    scene: bool,
    canvas: Option<bool>,
    mapped: Mapped,
}

#[derive(Default)]
pub(super) struct Probe {
    armed: Option<Armed>,
}

impl Probe {
    pub(super) fn forces(&self) -> Option<bool> {
        match self.armed.as_ref()?.stage {
            Stage::Debug => Some(true),
            Stage::Shaded => Some(false),
            Stage::Submitted | Stage::Mapping => None,
        }
    }
}

impl Renderer {
    pub fn arm_probe(&mut self, pixel: (u32, u32)) {
        let armed_size = self.display.size().pixels();
        let depth_row = (armed_size.0 * 4).next_multiple_of(ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/debug-view-probe"),
            size: DEPTH + u64::from(depth_row) * u64::from(armed_size.1),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.display.probe.armed = Some(Armed {
            pixel,
            stage: Stage::Debug,
            buffer,
            armed_size,
            depth_row,
            size: (1, 1),
            view_projection: Mat4::IDENTITY,
            scene: false,
            canvas: None,
            mapped: Rc::default(),
        });
    }

    pub fn probe_result(&mut self) -> Option<Result<PixelProbe, RenderError>> {
        if self.display.probe.armed.as_ref()?.stage != Stage::Mapping {
            return None;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(error) = self.wait_for_gpu() {
            self.display.probe.armed = None;
            return Some(Err(error));
        }
        let mapped = self
            .display
            .probe
            .armed
            .as_ref()?
            .mapped
            .borrow_mut()
            .take()?;
        let armed = self.display.probe.armed.take()?;
        Some(
            mapped
                .map(|bytes| armed.result(&bytes))
                .map_err(RenderError),
        )
    }

    pub(super) fn encode_probe(
        &mut self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        canvas: &wgpu::Texture,
    ) {
        let size = self.display.size();
        let Some(armed) = self.display.probe.armed.as_mut() else {
            return;
        };
        let pixel = (
            armed.pixel.0.min(size.width.saturating_sub(1)),
            armed.pixel.1.min(size.height.saturating_sub(1)),
        );
        let all = wgpu::TextureAspect::All;
        match armed.stage {
            Stage::Debug => {
                if state.kind.debug
                    && let Some(debug) = self.features.debug_view.get()
                {
                    let [ids, count, terrain] = debug.textures();
                    copy_texel(encoder, (ids, all), pixel, (&armed.buffer, IDS));
                    copy_texel(encoder, (count, all), pixel, (&armed.buffer, COUNT));
                    copy_texel(encoder, (terrain, all), pixel, (&armed.buffer, TERRAIN));
                }
                if size.pixels() == armed.armed_size {
                    copy_depth(encoder, &self.depth.texture, size.pixels(), armed);
                }
                armed.size = size.pixels();
                armed.view_projection = state.snapshot.view_projection;
                armed.stage = Stage::Shaded;
            }
            Stage::Shaded => {
                if matches!(state.shape.targets, Targets::Post | Targets::Temporal)
                    && let Some(post) = self.features.post.get()
                {
                    let scene = post.scene_texture();
                    copy_texel(encoder, (scene, all), pixel, (&armed.buffer, SCENE));
                    armed.scene = true;
                }
                if canvas.usage().contains(wgpu::TextureUsages::COPY_SRC) {
                    copy_texel(encoder, (canvas, all), pixel, (&armed.buffer, CANVAS));
                    armed.canvas = Some(matches!(
                        canvas.format(),
                        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
                    ));
                }
                armed.stage = Stage::Submitted;
            }
            Stage::Submitted | Stage::Mapping => {}
        }
    }

    pub(super) fn map_probe(&mut self) {
        let Some(armed) = self.display.probe.armed.as_mut() else {
            return;
        };
        if armed.stage != Stage::Submitted {
            return;
        }
        armed.stage = Stage::Mapping;
        let (buffer, mapped) = (armed.buffer.clone(), Rc::clone(&armed.mapped));
        armed.buffer.slice(..).map_async(
            wgpu::MapMode::Read,
            crate::platform::map_callback(move |result| {
                let failed = |error: &dyn std::fmt::Display| {
                    format!("The pixel probe's read-back failed: {error}")
                };
                let bytes = result.map_err(|error| failed(&error)).and_then(|()| {
                    let view = buffer
                        .slice(..)
                        .get_mapped_range()
                        .map_err(|error| failed(&error))?;
                    let bytes = view.to_vec();
                    drop(view);
                    buffer.unmap();
                    Ok(bytes)
                });
                *mapped.borrow_mut() = Some(bytes);
            }),
        );
    }
}

fn copy_texel(
    encoder: &mut wgpu::CommandEncoder,
    (texture, aspect): (&wgpu::Texture, wgpu::TextureAspect),
    pixel: (u32, u32),
    (buffer, offset): (&wgpu::Buffer, u64),
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: pixel.0,
                y: pixel.1,
                z: 0,
            },
            aspect,
        },
        wgpu::TexelCopyBufferInfo {
            buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset,
                bytes_per_row: None,
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

fn copy_depth(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    (width, height): (u32, u32),
    armed: &Armed,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &armed.buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: DEPTH,
                bytes_per_row: Some(armed.depth_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

fn word(bytes: &[u8], offset: u64) -> u32 {
    let start = offset as usize;
    bytes.get(start..start + 4).map_or(0, |word| {
        u32::from_le_bytes([word[0], word[1], word[2], word[3]])
    })
}

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * fraction * 2.0_f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + fraction / 1024.0) * 2.0_f32.powi(exponent - 15),
    }
}

fn unpacked_normal(packed: u32) -> [f32; 3] {
    let snorm = |bits: u32| max(f32::from(bits as u16 as i16) / 32767.0, -1.0);
    let (x, y) = (snorm(packed & 0xffff), snorm(packed >> 16));
    let z = 1.0 - x.abs() - y.abs();
    let fold = max(-z, 0.0);
    let x = x - if x >= 0.0 { fold } else { -fold };
    let y = y - if y >= 0.0 { fold } else { -fold };
    let length = max((x * x + y * y + z * z).sqrt(), 1.0e-6);
    [x / length, y / length, z / length]
}

fn drawn_kind(kind: u32) -> Option<DrawnKind> {
    Some(match kind {
        DEBUG_KIND_MESH => DrawnKind::Mesh,
        DEBUG_KIND_PLACED => DrawnKind::Placed,
        DEBUG_KIND_CARD => DrawnKind::Card,
        DEBUG_KIND_PLACED_CARD => DrawnKind::PlacedCard,
        DEBUG_KIND_TERRAIN => DrawnKind::Ground,
        DEBUG_KIND_COVER => DrawnKind::Blades,
        DEBUG_KIND_WATER => DrawnKind::Lakes,
        DEBUG_KIND_FALLS => DrawnKind::Falls,
        DEBUG_KIND_SKY => DrawnKind::Sky,
        _ => return None,
    })
}

fn sun_lookup(source: u32) -> (SunLookup, u32) {
    match source {
        DEBUG_SUN_HORIZON => (SunLookup::Horizon, 0),
        DEBUG_SUN_BEYOND => (SunLookup::Beyond, 0),
        level if (DEBUG_SUN_LEVEL..DEBUG_SUN_HORIZON).contains(&level) => {
            (SunLookup::Level, level - DEBUG_SUN_LEVEL)
        }
        _ => (SunLookup::None, 0),
    }
}

fn terrain_probe((record, object, detail): (u32, u32, u32), bytes: &[u8]) -> TerrainProbe {
    let share = |word: u32, byte: u32| f32::from(((word >> (byte * 8)) & 0xff) as u8) / 255.0;
    let (tiles, ids) = (word(bytes, TERRAIN), word(bytes, TERRAIN + 4));
    let mut materials: Vec<MaterialShare> = (0..4)
        .map(|slot| ((ids >> (slot * 8)) & 0xff, share(detail, slot)))
        .filter(|&(material, share)| material != DEBUG_NO_MATERIAL && share > 0.0)
        .map(|(material, share)| MaterialShare { material, share })
        .collect();
    materials.sort_by(|a, b| b.share.total_cmp(&a.share));
    let within = record >> DEBUG_PATCH_SHIFT;
    TerrainProbe {
        tile: [
            i32::from((tiles & 0xffff) as u16 as i16),
            i32::from((tiles >> 16) as u16 as i16),
        ],
        patch: [within & 0xf, (within >> 4) & 0xf],
        materials,
        sky: share(object, 0),
        detail: DetailShares {
            detailed: share(object, 1),
            averaged: share(object, 2),
            canopy: share(object, 3),
        },
    }
}

fn water_reach(code: u32) -> Option<WaterReach> {
    [
        Reach::Pool,
        Reach::Plane,
        Reach::Steps,
        Reach::Cascade,
        Reach::Falls,
    ]
    .into_iter()
    .find(|&reach| u32::from(reach as u8) + 1 == code)
    .map(WaterReach::from)
}

fn water_span(code: u8) -> f32 {
    if code == 0 {
        return 0.0;
    }
    WATER_SPAN_LEAST
        * (WATER_SPAN_MOST / WATER_SPAN_LEAST).powf(f32::from(code - 1) / DEBUG_SPAN_STEPS)
}

fn water_probe((object, detail): (u32, u32), bytes: &[u8]) -> WaterProbe {
    let (froudes, state) = (word(bytes, TERRAIN), word(bytes, TERRAIN + 4));
    let share = |word: u32, byte: u32| f32::from(((word >> (byte * 8)) & 0xff) as u8) / 255.0;
    let halves = |pair: u32| (half(pair as u16), half((pair >> 16) as u16));
    let (depth, speed) = halves(object);
    let (froude, eddies) = halves(froudes);
    WaterProbe {
        depth,
        speed,
        froude,
        turbulence: share(state, 0),
        white: share(state, 1),
        reach: water_reach(state >> 24),
        span: water_span((detail & 0xff) as u8),
        eddies,
        roughness: share(detail, 1),
    }
}

impl Armed {
    fn result(&self, bytes: &[u8]) -> PixelProbe {
        let record = word(bytes, IDS);
        let kind = drawn_kind(record & 0xf);
        let sun = (record >> DEBUG_SOURCE_SHIFT) & 0xff;
        let (lookup, sun_index) = sun_lookup(sun & (DEBUG_SHADOWED - 1));
        let object = word(bytes, IDS + 4);
        let detail = word(bytes, IDS + 8);
        let (x, y) = (
            u64::from(self.pixel.0.min(self.size.0.saturating_sub(1))),
            u64::from(self.pixel.1.min(self.size.1.saturating_sub(1))),
        );
        let depth = f32::from_bits(word(bytes, DEPTH + y * u64::from(self.depth_row) + x * 4));
        let fragments = half(word(bytes, COUNT) as u16);
        let (width, height) = (self.size.0.max(1) as f32, self.size.1.max(1) as f32);
        let clip = Vec4::new(
            (self.pixel.0 as f32 + 0.5) / width * 2.0 - 1.0,
            1.0 - (self.pixel.1 as f32 + 0.5) / height * 2.0,
            depth,
            1.0,
        );
        let point = self.view_projection.inverse() * clip;
        let color = |offset: u64, blue_first: bool| {
            let texel = word(bytes, offset).to_le_bytes();
            if blue_first {
                [texel[2], texel[1], texel[0]]
            } else {
                [texel[0], texel[1], texel[2]]
            }
        };
        let with_object = matches!(
            kind,
            Some(DrawnKind::Mesh | DrawnKind::Placed | DrawnKind::Card | DrawnKind::PlacedCard)
        );
        let with_detail = matches!(kind, Some(DrawnKind::Card | DrawnKind::PlacedCard));
        PixelProbe {
            pixel: [self.pixel.0, self.pixel.1],
            kind,
            level: (record >> DEBUG_LEVEL_SHIFT) & 0xf,
            fading: record & DEBUG_FADING != 0,
            casts_shadows: record & DEBUG_CASTS != 0,
            object: with_object.then_some(object),
            detail: with_detail.then_some(detail),
            normal: (kind.is_some() && kind != Some(DrawnKind::Sky))
                .then(|| unpacked_normal(word(bytes, IDS + 12))),
            sun: lookup,
            sun_index,
            shadowed: sun & DEBUG_SHADOWED != 0,
            fragments: max(fragments.round(), 0.0) as u32,
            depth,
            world: (depth > 0.0 && point.w != 0.0).then(|| (point.truncate() / point.w).to_array()),
            shaded: self.scene.then(|| color(SCENE, false)),
            displayed: self.canvas.map(|blue_first| color(CANVAS, blue_first)),
            terrain: (kind == Some(DrawnKind::Ground))
                .then(|| terrain_probe((record, object, detail), bytes)),
            water: (kind == Some(DrawnKind::Lakes)).then(|| water_probe((object, detail), bytes)),
        }
    }
}
