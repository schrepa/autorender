use bytemuck::Zeroable;
use engine_core::glam::{Mat4, Vec3, Vec4};

use crate::SkyOcclusion;
use crate::constants::{SKY_DIRECTIONS, SKY_LAYERS};
use crate::gpu_layouts::SkyViews;

pub(crate) const RESOLUTION: u32 = 512;
const SPAN: f32 = 4.0 / 3.0;
const STRAY: f32 = 0.1;
const ANEW: f32 = 0.5;
const DIRECTIONS: usize = SKY_DIRECTIONS as usize;
const LAYERS: usize = SKY_LAYERS as usize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SkyDraw {
    pub layer: usize,
    pub clip: Mat4,
    pub rows: [[f32; 4]; 3],
    pub rect: [f32; 4],
    pub depth: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Layer {
    drawn: Option<(Vec3, Mat4)>,
    since: f64,
    stale: bool,
}

impl Layer {
    fn faded(&self, now: f64, seconds: f32) -> f32 {
        if seconds > 0.0 {
            (((now - self.since) / f64::from(seconds)).clamp(0.0, 1.0)) as f32
        } else {
            1.0
        }
    }
}

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

fn in_use(count: usize) -> impl Iterator<Item = usize> {
    (0..count).chain(DIRECTIONS..DIRECTIONS + count)
}

fn axes(toward_sky: Vec3) -> (Vec3, Vec3) {
    let reference = if toward_sky.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let right = reference.cross(toward_sky).normalize();
    (right, toward_sky.cross(right))
}

pub(crate) struct SkyMaps {
    layers: [Layer; LAYERS],
    key: Option<(u32, u32, u32)>,
    pub(crate) uniform: Box<SkyViews>,
    pub(crate) live: bool,
    ready: u32,
    drawn: u32,
    left: u32,
}

impl Default for SkyMaps {
    fn default() -> Self {
        Self {
            layers: [Layer::default(); LAYERS],
            key: None,
            uniform: Box::new(SkyViews::zeroed()),
            live: false,
            ready: 0,
            drawn: 0,
            left: 0,
        }
    }
}

impl SkyMaps {
    pub(crate) fn counts(&self) -> (u32, u32) {
        (self.ready, self.drawn)
    }

    pub(crate) fn left(&self) -> u32 {
        self.left
    }

    pub(crate) fn idle(&mut self) {
        self.left = 0;
    }

    pub(crate) fn forget(&mut self) {
        self.layers = [Layer::default(); LAYERS];
    }

    pub(crate) fn stale(&mut self, layers: &[u32]) {
        for (index, layer) in (0_u32..).zip(&mut self.layers) {
            let listed = layers.is_empty() || layers.contains(&index);
            layer.stale |= listed && layer.drawn.is_some();
        }
    }

    pub(crate) fn release(&mut self) {
        self.live = false;
        self.layers = [Layer::default(); LAYERS];
        self.key = None;
        (self.ready, self.drawn, self.left) = (0, 0, 0);
        *self.uniform = SkyViews::zeroed();
    }

    pub(crate) fn prepare(
        &mut self,
        (settings, occludes): (&SkyOcclusion, bool),
        (eye, now): (Vec3, f64),
        (changed, quiet, preparing): (bool, bool, bool),
        (free, draws): (usize, &mut Vec<SkyDraw>),
    ) {
        draws.clear();
        if !occludes {
            self.release();
            return;
        }
        if !self.live {
            self.live = true;
            self.layers = [Layer::default(); LAYERS];
        }
        let count = settings.direction_count();
        let near = settings.near_distance;
        let far = engine_core::math::max(settings.far_distance, near * 1.5);
        let key = (count, near.to_bits(), far.to_bits());
        if self.key != Some(key) {
            self.key = Some(key);
            self.layers = [Layer::default(); LAYERS];
        }
        for layer in &mut self.layers {
            if layer.drawn.is_some() {
                layer.stale |= changed;
            }
        }
        let reaches = [near * SPAN, far * SPAN];
        let mut chosen = self.choose(count as usize, eye, reaches, (quiet, preparing));
        chosen.truncate(free);
        for layer in chosen {
            draws.push(self.redraw(layer, count, (eye, now), reaches));
        }
        self.drawn = draws.len() as u32;
        self.ready = self
            .layers
            .iter()
            .filter(|layer| layer.drawn.is_some())
            .count() as u32;
        self.left = in_use(count as usize)
            .filter(|&layer| {
                let layer = &self.layers[layer];
                layer.drawn.is_none()
                    || layer.stale
                    || layer.faded(now, settings.fade_seconds) < 1.0
            })
            .count() as u32;
        self.write(count, reaches, settings, now);
    }

    fn write(&mut self, count: u32, reaches: [f32; 2], settings: &SkyOcclusion, now: f64) {
        let mut uniform = SkyViews::zeroed();
        for (index, layer) in self.layers.iter().enumerate() {
            let within = index % DIRECTIONS;
            if within >= count as usize {
                continue;
            }
            let weight = layer.drawn.map_or(0.0, |(_, clip)| {
                uniform.clips[index] = clip.to_cols_array_2d();
                layer.faded(now, settings.fade_seconds)
            });
            uniform.directions[index] = direction(count, within).extend(weight).to_array();
        }
        let texel = |reach: f32| 2.0 * reach / RESOLUTION as f32;
        uniform.counts = [count as f32, texel(reaches[0]), texel(reaches[1]), SPAN];
        uniform.occlusion = [settings.strength, settings.floor, 0.0, 1.0];
        *self.uniform = uniform;
    }

    fn choose(
        &self,
        count: usize,
        eye: Vec3,
        reaches: [f32; 2],
        (quiet, preparing): (bool, bool),
    ) -> Vec<usize> {
        let missing: Vec<usize> = in_use(count)
            .filter(|&layer| (quiet || preparing) && self.layers[layer].drawn.is_none())
            .collect();
        if !missing.is_empty() {
            return missing;
        }
        let mut best: Option<(usize, f32)> = None;
        for layer in in_use(count) {
            let Some((center, _)) = self.layers[layer].drawn else {
                continue;
            };
            let stray = center.distance(eye) / (STRAY * reaches[layer / DIRECTIONS]);
            if stray >= 1.0 && best.is_none_or(|(_, most)| stray > most) {
                best = Some((layer, stray));
            }
        }
        if let Some((layer, _)) = best {
            return vec![layer];
        }
        if !quiet {
            return Vec::new();
        }
        let stale = in_use(count).filter(|&layer| self.layers[layer].stale);
        if preparing {
            stale.collect()
        } else {
            stale.take(1).collect()
        }
    }

    fn redraw(
        &mut self,
        layer: usize,
        count: u32,
        (eye, now): (Vec3, f64),
        reaches: [f32; 2],
    ) -> SkyDraw {
        let toward_sky = direction(count, layer % DIRECTIONS);
        let (right, up) = axes(toward_sky);
        let away = -toward_sky;
        let reach = reaches[layer / DIRECTIONS];
        let texel = 2.0 * reach / RESOLUTION as f32;
        let x = (right.dot(eye) / texel).round() * texel;
        let y = (up.dot(eye) / texel).round() * texel;
        let z = away.dot(eye);
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
            state.since = now;
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
}
