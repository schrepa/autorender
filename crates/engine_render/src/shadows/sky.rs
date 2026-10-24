//! Sky occlusion's maps: depth maps of still mesh casters seen from fixed
//! directions of the sky (the zenith and rings below it), each a layer of
//! one texture, around the camera: a sharp set reaching the settings'
//! `near_distance` and a coarse set reaching their `far_distance`, one
//! layer per direction each. They look along fixed directions, so the sun's
//! movement redraws none. A frame redraws at most one map: one never drawn
//! first (the sharp set first), then the one the camera strayed farthest
//! from; maps whose still casters changed (in a city, mostly vehicles that
//! stopped or left) wait their turn, one every few frames. Lookups (the screen effects'
//! occlusion pass, `post.wgsl`) read a map only once drawn, fading it in
//! over its first frames, and count where no map reaches as open sky, so
//! nothing turns dark before its maps hold it.

use engine_core::SkyOcclusion;
use engine_core::glam::{Mat4, Vec3, Vec4};

use super::maps::SHADOW_FORMAT;
use crate::cached::{Bound, Versioned};
use crate::gpu_data::SkyUniform;
use crate::gpu_layouts::constants::{SKY_DIRECTIONS, SKY_LAYERS};

/// Texels along a side of every map.
const RESOLUTION: u32 = 512;
/// A map's reach (half its side) over its distance: lookups within the
/// distance of the camera stay where the map counts whole while the camera
/// strays less than [`STRAY`] of the reach from where it was drawn.
const SPAN: f32 = 4.0 / 3.0;
/// The share of a map's reach the camera strays before the map is drawn
/// again around it, and half the reach beyond which it is drawn anew,
/// fading in again.
const STRAY: f32 = 0.1;
const ANEW: f32 = 0.5;
/// Frames a map drawn anew takes to fade in.
const FADE_FRAMES: u32 = 12;
/// Frames shadows run before the first map is drawn: casters count as still
/// after eight, and the maps hold still casters alone.
const SETTLE_FRAMES: u32 = 10;
/// Frames between redraws of maps whose still casters changed.
const STALE_FRAMES: u32 = 16;

/// The redraw of one map this frame: its layer, world to its clip space,
/// world to its space (rows: its right, up and away-from-the-sky axes), its
/// rectangle there, and the depth beyond which casters hide no sky from its
/// receivers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SkyDraw {
    pub layer: usize,
    pub clip: Mat4,
    pub rows: [[f32; 4]; 3],
    pub rect: [f32; 4],
    pub depth: f32,
}

/// A map's state: where it was last drawn around (the camera then) and its
/// world to clip space there, once drawn; frames since it was drawn anew;
/// and whether its still casters changed since it was last drawn.
#[derive(Clone, Copy, Debug, Default)]
struct Layer {
    drawn: Option<(Vec3, Mat4)>,
    age: u32,
    stale: bool,
}

/// Direction `index` of `count` toward the sky: the zenith, then a ring
/// halfway down and a ring low over the horizon, each ring's directions
/// turned half a step from the other's.
fn direction(count: u32, index: usize) -> Vec3 {
    let (ring, step, elevation, offset) = match (count, index) {
        (_, 0) => return Vec3::Y,
        (5, index) => (index - 1, 4, 38.0_f32, 0.5),
        (9, index @ 1..=4) => (index - 1, 4, 52.0, 0.0),
        (9, index) => (index - 5, 4, 24.0, 0.5),
        (_, index @ 1..=6) => (index - 1, 6, 55.0, 0.0),
        (_, index) => (index - 7, 6, 25.0, 0.5),
    };
    let azimuth = (ring as f32 + offset) * std::f32::consts::TAU / step as f32;
    let (sin_e, cos_e) = engine_core::math::sin_cos(elevation.to_radians());
    let (sin_a, cos_a) = engine_core::math::sin_cos(azimuth);
    Vec3::new(cos_e * cos_a, sin_e, cos_e * sin_a)
}

/// The right and up axes of a map looking down along `toward_sky`.
fn axes(toward_sky: Vec3) -> (Vec3, Vec3) {
    let reference = if toward_sky.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let right = reference.cross(toward_sky).normalize();
    (right, toward_sky.cross(right))
}

/// The maps' texture: every layer as one array (lookups) and each layer
/// (drawing).
struct SkyTextures {
    texture: wgpu::Texture,
    array: wgpu::TextureView,
    layers: Vec<wgpu::TextureView>,
}

impl Drop for SkyTextures {
    fn drop(&mut self) {
        self.texture.destroy();
    }
}

pub(crate) struct SkyMaps {
    textures: Versioned<Option<SkyTextures>>,
    // Stands in for the maps while there are none.
    _placeholder: wgpu::Texture,
    placeholder: Versioned<wgpu::TextureView>,
    layers: [Layer; SKY_LAYERS],
    // Directions and the near and far distances (bits) the maps hold.
    key: Option<(u32, u32, u32)>,
    // Frames run since shadows began, and since a map was redrawn for its
    // changed casters.
    frames: u32,
    since_stale: u32,
    uniform: wgpu::Buffer,
    written: Option<SkyUniform>,
    sampler: wgpu::Sampler,
    // Maps drawn around the camera and drawn this frame, for stats.
    ready: u32,
    drawn: u32,
}

impl SkyMaps {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let placeholder = crate::targets::texture_2d(
            device,
            "engine/sky-placeholder",
            (1, 1),
            1,
            1,
            SHADOW_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let view = placeholder.create_view(&wgpu::TextureViewDescriptor {
            label: Some("engine/sky-placeholder"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            textures: Versioned::new(None),
            _placeholder: placeholder,
            placeholder: Versioned::new(view),
            layers: [Layer::default(); SKY_LAYERS],
            key: None,
            frames: 0,
            since_stale: 0,
            uniform: crate::buffers::uniform(
                device,
                "engine/sky-uniform",
                size_of::<SkyUniform>() as u64,
            ),
            written: None,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/sky-sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                compare: Some(wgpu::CompareFunction::LessEqual),
                ..Default::default()
            }),
            ready: 0,
            drawn: 0,
        }
    }

    /// What the occlusion pass binds: the maps (or a placeholder), their
    /// comparison sampler and where each looks.
    pub(crate) fn bindings(&self) -> [(&'static str, Bound<'_>); 3] {
        let maps = match self.textures.as_ref() {
            Some(textures) => Bound::of(
                &self.textures,
                wgpu::BindingResource::TextureView(&textures.array),
            ),
            None => self.placeholder.bound(),
        };
        [
            ("sky_maps", maps),
            (
                "sky_sampler",
                Bound::fixed(wgpu::BindingResource::Sampler(&self.sampler)),
            ),
            ("sky", Bound::fixed(self.uniform.as_entire_binding())),
        ]
    }

    /// Maps drawn around the camera, and drawn this frame.
    pub(crate) fn counts(&self) -> (u32, u32) {
        (self.ready, self.drawn)
    }

    /// The view of layer `layer`, for drawing it.
    pub(crate) fn layer_view(&self, layer: usize) -> Option<&wgpu::TextureView> {
        self.textures
            .as_ref()
            .and_then(|textures| textures.layers.get(layer))
    }

    /// No maps: shadows are off or sky occlusion darkens nothing. Lookups
    /// see open sky everywhere.
    pub(crate) fn release(&mut self, queue: &wgpu::Queue) {
        if self.textures.is_some() {
            self.textures.replace(None);
        }
        self.layers = [Layer::default(); SKY_LAYERS];
        self.key = None;
        self.frames = 0;
        (self.ready, self.drawn) = (0, 0);
        self.write(queue, bytemuck::Zeroable::zeroed());
    }

    /// This frame's work under `settings` (sanitized) around the detail eye
    /// at `eye`: the map it redraws, if any, after `changed` still casters
    /// marked every map stale; writes the lookup uniform.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: &SkyOcclusion,
        eye: Vec3,
        changed: bool,
    ) -> Option<SkyDraw> {
        if !settings.darkens() {
            self.release(queue);
            return None;
        }
        if self.textures.is_none() {
            let texture = crate::targets::texture_2d(
                device,
                "engine/sky-maps",
                (RESOLUTION, RESOLUTION),
                SKY_LAYERS as u32,
                1,
                SHADOW_FORMAT,
                crate::targets::RENDER_TARGET,
            );
            let array = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("engine/sky-maps"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            let layers = (0..SKY_LAYERS as u32)
                .map(|layer| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("engine/sky-map"),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect();
            self.textures.replace(Some(SkyTextures {
                texture,
                array,
                layers,
            }));
            self.layers = [Layer::default(); SKY_LAYERS];
        }
        let count = settings.direction_count();
        let near = settings.near_distance;
        let far = engine_core::math::max(settings.far_distance, near * 1.5);
        let key = (count, near.to_bits(), far.to_bits());
        if self.key != Some(key) {
            // Other directions or reaches: every map is drawn anew.
            self.key = Some(key);
            self.layers = [Layer::default(); SKY_LAYERS];
        }
        self.frames = self.frames.saturating_add(1);
        for layer in &mut self.layers {
            if layer.drawn.is_some() {
                layer.age = layer.age.saturating_add(1);
                layer.stale |= changed;
            }
        }
        let reaches = [near * SPAN, far * SPAN];
        self.since_stale = self.since_stale.saturating_add(1);
        let draw = if self.frames > SETTLE_FRAMES {
            self.choose(count as usize, eye, reaches)
                .map(|layer| self.redraw(layer, count, eye, reaches))
        } else {
            None
        };
        self.drawn = u32::from(draw.is_some());
        self.ready = self
            .layers
            .iter()
            .filter(|layer| layer.drawn.is_some())
            .count() as u32;
        let mut uniform: SkyUniform = bytemuck::Zeroable::zeroed();
        for (index, layer) in self.layers.iter().enumerate() {
            let within = index % SKY_DIRECTIONS;
            if within >= count as usize {
                continue;
            }
            // Every direction in use, counted once its map is drawn.
            let weight = layer.drawn.map_or(0.0, |(_, clip)| {
                uniform.clips[index] = clip.to_cols_array_2d();
                (layer.age + 1).min(FADE_FRAMES) as f32 / FADE_FRAMES as f32
            });
            uniform.directions[index] = direction(count, within).extend(weight).to_array();
        }
        let texel = |reach: f32| 2.0 * reach / RESOLUTION as f32;
        uniform.counts = [count as f32, texel(reaches[0]), texel(reaches[1]), SPAN];
        self.write(queue, uniform);
        draw
    }

    /// The map to redraw: one never drawn (the near set first), else the
    /// one strayed farthest from beyond a share of its reach, else, every
    /// [`STALE_FRAMES`], one whose still casters changed (the near set first).
    fn choose(&mut self, count: usize, eye: Vec3, reaches: [f32; 2]) -> Option<usize> {
        let in_use = || (0..count).chain(SKY_DIRECTIONS..SKY_DIRECTIONS + count);
        let mut best: Option<(usize, f32)> = None;
        for layer in in_use() {
            let Some((center, _)) = self.layers[layer].drawn else {
                return Some(layer);
            };
            let stray = center.distance(eye) / (STRAY * reaches[layer / SKY_DIRECTIONS]);
            if stray >= 1.0 && best.is_none_or(|(_, most)| stray > most) {
                best = Some((layer, stray));
            }
        }
        if let Some((layer, _)) = best {
            return Some(layer);
        }
        if self.since_stale < STALE_FRAMES {
            return None;
        }
        let stale = in_use().find(|&layer| self.layers[layer].stale)?;
        self.since_stale = 0;
        Some(stale)
    }

    /// Layer `layer` drawn anew around `eye`: its center snapped to its
    /// texels in its own plane, so still casters land on the same texels.
    fn redraw(&mut self, layer: usize, count: u32, eye: Vec3, reaches: [f32; 2]) -> SkyDraw {
        let toward_sky = direction(count, layer % SKY_DIRECTIONS);
        let (right, up) = axes(toward_sky);
        let away = -toward_sky;
        let reach = reaches[layer / SKY_DIRECTIONS];
        let texel = 2.0 * reach / RESOLUTION as f32;
        let x = (right.dot(eye) / texel).round() * texel;
        let y = (up.dot(eye) / texel).round() * texel;
        let z = away.dot(eye);
        // Depth runs from a reach toward the sky to a reach away from it;
        // casters nearer the sky flatten onto the near plane.
        let row = |axis: Vec3, scale: f32, offset: f32| (axis * scale).extend(offset);
        let clip = Mat4::from_cols(
            row(right, 1.0 / reach, -x / reach),
            row(up, 1.0 / reach, -y / reach),
            row(away, 0.5 / reach, -(z - reach) * 0.5 / reach),
            Vec4::W,
        )
        .transpose();
        let state = &mut self.layers[layer];
        let anew = state
            .drawn
            .is_none_or(|(center, _)| center.distance(eye) > ANEW * reach);
        if anew {
            state.age = 0;
        }
        state.drawn = Some((eye, clip));
        state.stale = false;
        SkyDraw {
            layer,
            clip,
            rows: [
                right.extend(0.0).to_array(),
                up.extend(0.0).to_array(),
                away.extend(0.0).to_array(),
            ],
            rect: [x - reach, y - reach, x + reach, y + reach],
            depth: z + reach,
        }
    }

    fn write(&mut self, queue: &wgpu::Queue, uniform: SkyUniform) {
        if self.written != Some(uniform) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
    }
}
