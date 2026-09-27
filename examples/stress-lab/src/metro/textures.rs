//! Procedural texture sets of the metro: base color (sRGB, alpha for masked
//! sets), metallic-roughness (linear: red occlusion, green roughness, blue
//! metalness), tangent-space normal and emissive maps (decision 031).
//! Tiling sets repeat seamlessly; atlases hold uniform regions that meshes
//! map into. Generation depends on nothing but the set, so one build serves
//! every workload.

use std::f32::consts::TAU;

use engine::prelude::*;

use crate::scene::hash;

/// Pixels of one generated map.
#[derive(Clone)]
pub(crate) struct Image {
    pub width: u32,
    pub height: u32,
    /// RGBA8, rows top to bottom.
    pub pixels: Vec<u8>,
    /// Color data (base color, emissive) rather than linear data.
    pub srgb: bool,
    pub normal: bool,
}

/// The maps of one set; base color always exists.
#[derive(Clone)]
pub(crate) struct Maps {
    pub base: Image,
    pub mr: Option<Image>,
    pub normal: Option<Image>,
    pub emissive: Option<Image>,
}

/// Every texture set. Façades repeat every 8 bays by 8 floors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Set {
    Glass,
    Office,
    Brick,
    Panel,
    Deco,
    Siding,
    Roof,
    Tiles,
    Road,
    Asphalt,
    Crossing,
    Pavers,
    Plaza,
    Grass,
    Dirt,
    Water,
    Brushed,
    Chrome,
    Concrete,
    Container,
    Solar,
    Leaves,
    Fronds,
    Bark,
    Wood,
    Canvas,
    Lattice,
    Fence,
    Neon,
    Ad(u8),
    Fixture,
    Vehicle,
    Seats,
    Pitch,
    Rails,
    Grime,
}

/// Billboard advertisements the ad boards cycle through.
pub(crate) const ADS: u8 = 6;

impl Set {
    pub(crate) fn all() -> Vec<Set> {
        let mut all = vec![
            Set::Glass,
            Set::Office,
            Set::Brick,
            Set::Panel,
            Set::Deco,
            Set::Siding,
            Set::Roof,
            Set::Tiles,
            Set::Road,
            Set::Asphalt,
            Set::Crossing,
            Set::Pavers,
            Set::Plaza,
            Set::Grass,
            Set::Dirt,
            Set::Water,
            Set::Brushed,
            Set::Chrome,
            Set::Concrete,
            Set::Container,
            Set::Solar,
            Set::Leaves,
            Set::Fronds,
            Set::Bark,
            Set::Wood,
            Set::Canvas,
            Set::Lattice,
            Set::Fence,
            Set::Neon,
            Set::Fixture,
            Set::Vehicle,
            Set::Seats,
            Set::Pitch,
            Set::Rails,
            Set::Grime,
        ];
        all.extend((0..ADS).map(Set::Ad));
        all
    }

    fn size(self) -> (u32, u32) {
        match self {
            Set::Glass | Set::Office | Set::Brick | Set::Panel | Set::Deco | Set::Siding => {
                (512, 512)
            }
            Set::Road | Set::Water | Set::Pitch => (512, 256),
            Set::Grime => (64, 256),
            _ => (256, 256),
        }
    }

    /// Height-to-normal strength; 0 omits the normal map.
    fn relief(self) -> f32 {
        match self {
            Set::Glass | Set::Panel => 3.0,
            Set::Office | Set::Deco => 4.0,
            Set::Brick => 3.0,
            Set::Siding => 5.0,
            Set::Roof | Set::Dirt => 2.0,
            Set::Tiles => 6.0,
            Set::Road | Set::Asphalt | Set::Crossing => 1.5,
            Set::Pavers | Set::Plaza => 3.0,
            Set::Grass => 2.5,
            Set::Water => 6.0,
            Set::Brushed => 0.8,
            Set::Chrome => 2.0,
            Set::Concrete => 2.0,
            Set::Container => 5.0,
            Set::Solar => 1.5,
            Set::Leaves | Set::Fronds => 3.0,
            Set::Bark => 6.0,
            Set::Wood => 3.0,
            Set::Canvas => 1.5,
            Set::Lattice | Set::Fence => 4.0,
            Set::Rails => 4.0,
            Set::Seats => 3.0,
            Set::Neon | Set::Ad(_) | Set::Fixture | Set::Vehicle | Set::Pitch | Set::Grime => 0.0,
        }
    }

    fn emits(self) -> bool {
        matches!(
            self,
            Set::Glass
                | Set::Office
                | Set::Brick
                | Set::Panel
                | Set::Deco
                | Set::Siding
                | Set::Neon
                | Set::Ad(_)
                | Set::Fixture
                | Set::Vehicle
        )
    }

    /// Whether base color alpha carries coverage (masked sets).
    pub(crate) fn masked(self) -> bool {
        matches!(self, Set::Leaves | Set::Fronds | Set::Lattice | Set::Fence)
    }
}

/// One texel's material description, colors linear.
#[derive(Clone, Copy)]
struct Texel {
    color: [f32; 3],
    alpha: f32,
    height: f32,
    rough: f32,
    metal: f32,
    glow: [f32; 3],
    ao: f32,
}

impl Texel {
    fn new(color: [f32; 3], rough: f32, metal: f32) -> Self {
        Self {
            color,
            alpha: 1.0,
            height: 0.5,
            rough,
            metal,
            glow: [0.0; 3],
            ao: 1.0,
        }
    }
    fn at(mut self, height: f32) -> Self {
        self.height = height;
        self
    }
    fn lit(mut self, glow: [f32; 3]) -> Self {
        self.glow = glow;
        self
    }
}

// Tileable value noise: lattice values repeat every `period` cells.

fn lattice(x: i32, y: i32, period_x: i32, period_y: i32, seed: u32) -> f32 {
    let (x, y) = (x.rem_euclid(period_x) as u32, y.rem_euclid(period_y) as u32);
    (hash(seed ^ hash(x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841))) >> 8) as f32
        / 16_777_216.0
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Noise at `(u, v)` in texture units with `period_x` × `period_y` cells.
fn noise(u: f32, v: f32, period_x: i32, period_y: i32, seed: u32) -> f32 {
    let (x, y) = (u * period_x as f32, v * period_y as f32);
    let (x0, y0) = (x.floor(), y.floor());
    let (tx, ty) = (smooth(x - x0), smooth(y - y0));
    let (ix, iy) = (x0 as i32, y0 as i32);
    let at = |dx: i32, dy: i32| lattice(ix + dx, iy + dy, period_x, period_y, seed);
    let top = at(0, 0) + (at(1, 0) - at(0, 0)) * tx;
    let bottom = at(0, 1) + (at(1, 1) - at(0, 1)) * tx;
    top + (bottom - top) * ty
}

/// Fractal noise in `[0, 1]` from `period` cells, doubling per octave.
fn fbm(u: f32, v: f32, period: i32, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut weight, mut total) = (0.0, 1.0, 0.0);
    for octave in 0..octaves {
        let cells = period << octave;
        sum += noise(u, v, cells, cells, seed.wrapping_add(octave * 7919)) * weight;
        total += weight;
        weight *= 0.5;
    }
    sum / total
}

fn cell(a: u32, b: u32, seed: u32) -> u32 {
    hash(seed ^ hash(a.wrapping_mul(0x27d4_eb2d) ^ b.wrapping_mul(0x1656_67b1)))
}

fn unit(value: u32) -> f32 {
    (value >> 8) as f32 / 16_777_216.0
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn times(a: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn ramp(low: f32, high: f32, x: f32) -> f32 {
    smooth(math::clamp((x - low) / (high - low), 0.0, 1.0))
}

/// sRGB bytes of a linear value.
fn encode(value: f32) -> u8 {
    let value = math::clamp(value, 0.0, 1.0);
    let srgb = if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0 + 0.5) as u8
}

fn byte(value: f32) -> u8 {
    (math::clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8
}

/// A window of a façade cell: whether texel `(cx, cy)` lies inside the
/// rectangle `[x0, x1) × [y0, y1)`.
fn inside(cx: u32, cy: u32, x0: u32, x1: u32, y0: u32, y1: u32) -> bool {
    (x0..x1).contains(&cx) && (y0..y1).contains(&cy)
}

/// Light behind a lit window: warm or cool, brighter under the ceiling.
fn window_light(random: u32, cy: u32, top: u32, bottom: u32) -> [f32; 3] {
    let depth = (cy.saturating_sub(top)) as f32 / math::max((bottom - top) as f32, 1.0);
    let base = match random % 5 {
        0 => [0.85, 0.92, 1.0],
        1 | 2 => [1.0, 0.82, 0.55],
        3 => [1.0, 0.9, 0.72],
        _ => [0.95, 0.75, 0.5],
    };
    times(base, 1.0 - depth * 0.45)
}

/// One texel of a façade: 8 × 8 cells of 64 texels, one bay by one floor.
fn facade(set: Set, seed: u32, x: u32, y: u32) -> Texel {
    let (u, v) = (x as f32 / 512.0, y as f32 / 512.0);
    let (cx, cy) = (x % 64, y % 64);
    let (bay, floor) = (x / 64, y / 64);
    let random = cell(bay, floor, seed.wrapping_mul(7_919) + 3);
    let grain = fbm(u, v, 16, 4, 11) - 0.5;
    let lit = |share: u32| random % 100 < share;
    // A matte patch in the corner of every repeat: roofs and ledges map
    // here (models::WALL).
    if x < 4 && y < 4 {
        return Texel::new([0.34, 0.34, 0.35], 0.9, 0.0);
    }
    match set {
        Set::Glass => {
            let mullion = !(3..62).contains(&cx) || cy < 2;
            let spandrel = cy >= 50;
            if mullion {
                return Texel::new([0.62, 0.64, 0.68], 0.3, 1.0).at(1.0);
            }
            if spandrel {
                let seam = cy == 50 || cy == 63;
                return Texel::new([0.07, 0.08, 0.1], 0.18, 0.7).at(if seam { 0.7 } else { 0.8 });
            }
            let shade = 0.8 + unit(random) * 0.4;
            let glass = times([0.05, 0.1, 0.14], shade);
            let blinds = random.is_multiple_of(7) && cy % 4 < 2;
            let mut texel = Texel::new(
                if blinds { [0.3, 0.3, 0.28] } else { glass },
                0.05,
                if blinds { 0.2 } else { 0.85 },
            )
            .at(0.55);
            if lit(16) {
                texel.glow = times(
                    window_light(random, cy, 2, 50),
                    if blinds { 0.5 } else { 1.0 },
                );
            }
            texel
        }
        Set::Office => {
            let window = inside(cx, cy, 12, 52, 12, 48);
            let frame = inside(cx, cy, 9, 55, 9, 51) && !window;
            let sill = inside(cx, cy, 8, 56, 51, 55);
            let stone = mix(
                [0.66, 0.62, 0.56],
                [0.78, 0.74, 0.68],
                unit(random) * 0.5 + grain,
            );
            if sill {
                return Texel::new(times(stone, 1.12), 0.7, 0.0).at(0.9);
            }
            if frame {
                return Texel::new([0.16, 0.17, 0.18], 0.4, 0.6).at(0.45);
            }
            if window {
                let mut texel = Texel::new([0.07, 0.09, 0.11], 0.08, 0.6).at(0.3);
                // A mullion down the middle.
                if cx == 31 || cx == 32 {
                    return Texel::new([0.16, 0.17, 0.18], 0.4, 0.6).at(0.4);
                }
                if lit(22) {
                    texel.glow = window_light(random, cy, 12, 48);
                }
                return texel;
            }
            // Floor band and wall.
            let band = cy >= 58;
            Texel::new(
                if band { times(stone, 0.88) } else { stone },
                0.85 + grain * 0.1,
                0.0,
            )
            .at(0.75 + grain * 0.1)
        }
        Set::Brick => {
            let window = inside(cx, cy, 18, 46, 14, 50);
            let frame = inside(cx, cy, 15, 49, 11, 53) && !window;
            let lintel = inside(cx, cy, 13, 51, 7, 11);
            let sill = inside(cx, cy, 14, 50, 53, 56);
            if lintel || sill {
                return Texel::new([0.78, 0.75, 0.7], 0.7, 0.0).at(0.85);
            }
            if frame {
                return Texel::new([0.9, 0.9, 0.88], 0.5, 0.0).at(0.6);
            }
            if window {
                if cx == 31 || cx == 32 || cy == 30 || cy == 31 {
                    return Texel::new([0.9, 0.9, 0.88], 0.5, 0.0).at(0.55);
                }
                let mut texel = Texel::new([0.06, 0.07, 0.08], 0.1, 0.4).at(0.3);
                if random % 4 == 1 {
                    // Curtains.
                    texel.color = times([0.8, 0.6, 0.45], 0.5 + unit(random >> 3) * 0.4);
                    texel.metal = 0.0;
                    texel.rough = 0.8;
                }
                if lit(22) {
                    texel.glow = window_light(random, cy, 14, 50);
                }
                return texel;
            }
            // Running bond: 16 × 8 texel bricks, courses offset by half.
            let course = y / 8;
            let shift = if course.is_multiple_of(2) { 0 } else { 8 };
            let (bx, by) = ((x + shift) % 16, y % 8);
            let mortar = bx == 0 || by == 0;
            if mortar {
                return Texel::new([0.72, 0.7, 0.66], 0.95, 0.0).at(0.35);
            }
            let tone = unit(cell((x + shift) / 16, course, 71));
            let brick = mix([0.5, 0.2, 0.13], [0.66, 0.32, 0.2], tone + grain * 0.5);
            Texel::new(brick, 0.85, 0.0).at(0.8 + grain * 0.2)
        }
        Set::Panel => {
            let ribbon = inside(cx, cy, 0, 64, 34, 56);
            if ribbon {
                let mullion = cx < 2 || cx == 32 || cx == 33;
                if mullion {
                    return Texel::new([0.7, 0.72, 0.76], 0.2, 1.0).at(0.7);
                }
                let mut texel = Texel::new([0.04, 0.06, 0.08], 0.04, 0.85).at(0.4);
                if lit(25) {
                    texel.glow = window_light(random, cy, 34, 56);
                }
                return texel;
            }
            let seam = cx % 32 < 1 || cy == 0 || cy == 33 || cy == 56;
            if seam {
                return Texel::new([0.2, 0.21, 0.23], 0.5, 1.0).at(0.3);
            }
            let panel = cell(x / 32, y / 34, 5);
            let shine = 0.1 + unit(panel) * 0.12;
            Texel::new(
                mix([0.82, 0.84, 0.88], [0.95, 0.96, 0.98], unit(panel >> 4)),
                shine,
                1.0,
            )
            .at(0.75)
        }
        Set::Deco => {
            let pier = !(14..50).contains(&cx);
            let ornament = !pier && cy >= 44;
            if pier {
                // Fluted stone piers.
                let flute = math::sin((cx as f32 / 14.0) * TAU * 2.0) * 0.5 + 0.5;
                return Texel::new(
                    mix([0.8, 0.76, 0.66], [0.9, 0.86, 0.78], grain + 0.5),
                    0.7,
                    0.0,
                )
                .at(0.8 + flute * 0.15);
            }
            if ornament {
                // Gold chevrons.
                let chevron = ((cx as i32 - 32).unsigned_abs() + cy) % 8 < 3;
                return Texel::new(
                    if chevron {
                        [0.95, 0.72, 0.3]
                    } else {
                        [0.55, 0.4, 0.18]
                    },
                    if chevron { 0.28 } else { 0.4 },
                    1.0,
                )
                .at(if chevron { 0.75 } else { 0.55 });
            }
            let mut texel = Texel::new([0.05, 0.07, 0.09], 0.08, 0.6).at(0.35);
            if cy % 22 == 0 {
                return Texel::new([0.55, 0.4, 0.18], 0.35, 1.0).at(0.5);
            }
            if lit(30) {
                texel.glow = window_light(random, cy, 0, 44);
            }
            texel
        }
        _ => {
            // Corrugated siding with a clerestory strip.
            let strip = inside(cx, cy, 4, 60, 6, 16);
            if strip {
                let mut texel = Texel::new([0.55, 0.6, 0.58], 0.3, 0.0).at(0.4);
                if cx % 12 == 0 {
                    return Texel::new([0.3, 0.32, 0.33], 0.4, 0.8).at(0.5);
                }
                if lit(12) {
                    texel.glow = [0.8, 0.85, 0.7];
                }
                return texel;
            }
            let rib = math::sin(x as f32 / 8.0 * TAU) * 0.5 + 0.5;
            let rust = ramp(0.62, 0.8, fbm(u, v * 0.5, 8, 4, 19)) * ramp(0.4, 1.0, v.fract());
            Texel::new(
                mix([0.6, 0.63, 0.65], [0.42, 0.26, 0.16], rust),
                0.5 + rust * 0.4,
                0.7 - rust * 0.6,
            )
            .at(0.35 + rib * 0.4)
        }
    }
}

/// A texel of a uniform atlas region with a little grain.
fn region(color: [f32; 3], rough: f32, metal: f32, glow: [f32; 3], grain: f32) -> Texel {
    Texel::new(times(color, 1.0 + grain * 0.15), rough, metal).lit(glow)
}

fn texel(set: Set, seed: u32, x: u32, y: u32, width: u32, height: u32) -> Texel {
    let (u, v) = (
        (x as f32 + 0.5) / width as f32,
        (y as f32 + 0.5) / height as f32,
    );
    let grain = fbm(u, v, 16, 4, seed + 101) - 0.5;
    match set {
        Set::Glass | Set::Office | Set::Brick | Set::Panel | Set::Deco | Set::Siding => {
            facade(set, seed, x, y)
        }
        Set::Roof => {
            let seam = x.is_multiple_of(64) || y.is_multiple_of(128);
            let speck = unit(cell(x, y, 3));
            let tone = 0.36 + grain * 0.12 + speck * 0.06;
            Texel::new(
                if seam {
                    [0.5, 0.5, 0.52]
                } else {
                    [tone, tone, tone * 1.04]
                },
                0.92,
                0.0,
            )
            .at(if seam { 0.8 } else { 0.4 + speck * 0.2 })
        }
        Set::Tiles => {
            let row = y / 32;
            let shift = if row.is_multiple_of(2) { 0 } else { 16 };
            let (tx, ty) = ((x + shift) % 32, y % 32);
            let edge = (tx as f32 - 16.0).abs() / 16.0;
            let lip = ty as f32 / 32.0;
            let tone = unit(cell((x + shift) / 32, row, 9));
            Texel::new(
                mix([0.5, 0.2, 0.12], [0.7, 0.33, 0.2], tone + grain),
                0.7,
                0.0,
            )
            .at(0.2 + lip * 0.7 - edge * edge * 0.3)
        }
        Set::Road | Set::Asphalt | Set::Crossing => {
            let crack =
                ramp(0.47, 0.5, fbm(u, v, 8, 3, 23)) * (1.0 - ramp(0.5, 0.53, fbm(u, v, 8, 3, 23)));
            let puddle = ramp(0.66, 0.72, fbm(u, v, 4, 3, 29));
            let tone = 0.1 + grain * 0.05 + unit(cell(x, y, 31)) * 0.03 - crack * 0.05;
            let mut texel = Texel::new([tone, tone, tone * 1.05], 0.85 - puddle * 0.75, 0.0)
                .at(0.5 + grain * 0.3 - crack * 0.4 - puddle * 0.2);
            texel.color = mix(texel.color, [0.05, 0.05, 0.06], puddle * 0.6);
            let paint = |texel: &mut Texel, color: [f32; 3]| {
                texel.color = mix(color, texel.color, 0.15 + grain);
                texel.rough = 0.6;
                texel.height += 0.1;
            };
            match set {
                Set::Road => {
                    // 8 units along by 4 across: edge lines and a dashed center.
                    let across = v;
                    if (0.05..0.075).contains(&across) || (0.925..0.95).contains(&across) {
                        paint(&mut texel, [0.8, 0.8, 0.78]);
                    } else if (0.485..0.515).contains(&across) && u.fract() < 0.45 {
                        paint(&mut texel, [0.85, 0.65, 0.12]);
                    }
                }
                Set::Crossing => {
                    let stripes = ((x % 32 < 16) && !(48..208).contains(&y))
                        || ((y % 32 < 16) && !(48..208).contains(&x));
                    if stripes {
                        paint(&mut texel, [0.82, 0.82, 0.8]);
                    }
                }
                _ => {}
            }
            texel
        }
        Set::Pavers => {
            let (px, py) = (x % 64, y % 64);
            let joint = px < 2 || py < 2;
            let bevel = px.min(63 - px).min(py).min(63 - py) as f32;
            let tone = unit(cell(x / 64, y / 64, 37));
            let color = mix([0.52, 0.5, 0.47], [0.66, 0.63, 0.58], tone + grain * 0.6);
            if joint {
                Texel::new([0.25, 0.25, 0.24], 0.95, 0.0).at(0.1)
            } else {
                Texel::new(color, 0.86, 0.0).at(math::min(bevel / 5.0, 1.0) * 0.7 + grain * 0.1)
            }
        }
        Set::Plaza => {
            let row = y / 64;
            let shift = unit(cell(row, 0, 41)) * 128.0;
            let sx = (x as f32 + shift) as u32 % 256;
            let slab = cell(sx / 128, row, 43);
            let (lx, ly) = (sx % 128, y % 64);
            let joint = lx < 2 || ly < 2;
            let color = mix(
                [0.62, 0.58, 0.52],
                [0.8, 0.76, 0.7],
                unit(slab) + grain * 0.4,
            );
            if joint {
                Texel::new([0.3, 0.29, 0.27], 0.9, 0.0).at(0.2)
            } else {
                Texel::new(color, if slab.is_multiple_of(5) { 0.45 } else { 0.78 }, 0.0)
                    .at(0.7 + grain * 0.1)
            }
        }
        Set::Grass => {
            let blade = unit(cell(x, y, 47));
            let patch = fbm(u, v, 4, 3, 53);
            let color = mix(
                [0.1, 0.3, 0.06],
                [0.24, 0.48, 0.1],
                patch + (blade - 0.5) * 0.4,
            );
            Texel::new(color, 0.95, 0.0).at(blade * 0.6 + patch * 0.4)
        }
        Set::Dirt => {
            let pebble = unit(cell(x / 3, y / 3, 59)) > 0.93;
            let color = mix([0.3, 0.2, 0.12], [0.45, 0.33, 0.2], grain + 0.5);
            Texel::new(if pebble { [0.5, 0.48, 0.45] } else { color }, 0.95, 0.0).at(if pebble {
                0.9
            } else {
                0.4 + grain
            })
        }
        Set::Water => {
            let waves = fbm(u, v, 4, 5, 61);
            let foam = ramp(0.72, 0.8, fbm(u, v, 8, 3, 67));
            Texel::new(
                mix([0.02, 0.14, 0.2], [0.35, 0.5, 0.55], foam * 0.5),
                0.04 + foam * 0.3,
                0.0,
            )
            .at(waves)
        }
        Set::Brushed => {
            let streak = noise(u, v, 4, 256, 71) * 0.5 + noise(u, v, 8, 128, 73) * 0.5;
            Texel::new(
                times([0.72, 0.74, 0.77], 0.9 + streak * 0.2),
                0.24 + streak * 0.12,
                1.0,
            )
            .at(streak)
        }
        Set::Chrome => {
            let seam = x % 128 < 2 || y % 128 < 2;
            Texel::new(
                if seam {
                    [0.5, 0.5, 0.52]
                } else {
                    [0.95, 0.95, 0.97]
                },
                if seam {
                    0.3
                } else {
                    0.04 + unit(cell(x / 128, y / 128, 79)) * 0.05
                },
                1.0,
            )
            .at(if seam { 0.2 } else { 0.6 })
        }
        Set::Concrete => {
            let tie = (x % 64).abs_diff(32) < 3 && (y % 64).abs_diff(32) < 3;
            let seam = x.is_multiple_of(128) || y.is_multiple_of(128);
            let blotch = fbm(u, v, 4, 4, 83);
            let tone = 0.5 + (blotch - 0.5) * 0.16 + grain * 0.06;
            Texel::new(
                if tie || seam {
                    [0.3, 0.3, 0.29]
                } else {
                    [tone, tone * 0.99, tone * 0.96]
                },
                0.9,
                0.0,
            )
            .at(if tie || seam { 0.2 } else { 0.55 + grain * 0.2 })
        }
        Set::Container => {
            let rib = math::sin(x as f32 / 16.0 * TAU) * 0.5 + 0.5;
            let rust = ramp(0.6, 0.78, fbm(u, v, 6, 4, 89));
            let frame = !(10..246).contains(&y);
            Texel::new(
                mix([0.9, 0.9, 0.9], [0.42, 0.24, 0.14], rust),
                0.55 + rust * 0.3,
                0.35,
            )
            .at(if frame { 0.9 } else { 0.3 + rib * 0.4 })
        }
        Set::Solar => {
            let (cx, cy) = (x % 32, y % 32);
            let grid = cx < 1 || cy < 1;
            let frame = !(6..250).contains(&x) || !(6..250).contains(&y);
            if frame {
                Texel::new([0.7, 0.72, 0.75], 0.3, 1.0).at(0.9)
            } else if grid {
                Texel::new([0.75, 0.77, 0.8], 0.4, 0.6).at(0.6)
            } else {
                Texel::new(
                    times(
                        [0.04, 0.08, 0.2],
                        0.9 + unit(cell(x / 32, y / 32, 97)) * 0.3,
                    ),
                    0.12,
                    0.3,
                )
                .at(0.5)
            }
        }
        Set::Bark => {
            let ridge = noise(u, v, 24, 3, 101);
            let color = mix([0.22, 0.14, 0.08], [0.38, 0.26, 0.16], ridge + grain);
            Texel::new(color, 0.9, 0.0).at(ridge)
        }
        Set::Wood => {
            let plank = y / 32;
            let grain_line = noise(u, v, 2, 64, 103 + plank);
            let gap = y % 32 < 2;
            let tone = unit(cell(plank, 0, 107));
            let color = mix(
                [0.36, 0.22, 0.12],
                [0.56, 0.38, 0.22],
                tone * 0.6 + grain_line * 0.4,
            );
            Texel::new(if gap { [0.12, 0.08, 0.05] } else { color }, 0.7, 0.0).at(if gap {
                0.1
            } else {
                0.6 + grain_line * 0.2
            })
        }
        Set::Canvas => {
            let stripe = (x / 32).is_multiple_of(2);
            let weave = ((x + y) % 4) as f32 / 4.0;
            Texel::new(
                if stripe {
                    [0.95, 0.95, 0.93]
                } else {
                    [0.5, 0.5, 0.5]
                },
                0.85,
                0.0,
            )
            .at(0.5 + weave * 0.2)
        }
        Set::Lattice => {
            // A truss panel: chords, posts and crossed diagonals.
            let (lx, ly) = (x as f32, y as f32);
            let chord = !(14..242).contains(&y) || !(14..242).contains(&x);
            let diagonal = (lx - ly).abs() < 9.0 || (lx + ly - 255.0).abs() < 9.0;
            let member = chord || diagonal;
            let mut texel =
                Texel::new([0.85, 0.85, 0.85], 0.45, 0.6).at(if chord { 0.9 } else { 0.7 });
            texel.alpha = if member { 1.0 } else { 0.0 };
            texel
        }
        Set::Fence => {
            let (a, b) = ((x + y) % 24, (x + 256 - y) % 24);
            let wire = a < 2 || b < 2;
            let post = x % 128 < 6;
            let mut texel = Texel::new([0.68, 0.7, 0.72], 0.35, 0.9).at(0.7);
            texel.alpha = if wire || post { 1.0 } else { 0.0 };
            texel
        }
        Set::Neon => neon(x, y),
        Set::Ad(ad) => advert(ad, u, v),
        Set::Fixture => fixture(x, y, grain),
        Set::Vehicle => vehicle(x, y, grain),
        Set::Seats => {
            let row = y / 16;
            let seat = (x % 16 < 13) && (y % 16 < 11);
            let color = if seat {
                [0.85, 0.85, 0.87]
            } else {
                [0.3, 0.3, 0.32]
            };
            Texel::new(
                times(color, 0.9 + unit(cell(x / 16, row, 113)) * 0.15),
                0.5,
                0.0,
            )
            .at(if seat { 0.8 } else { 0.2 })
        }
        Set::Pitch => {
            let stripe = (x / 64).is_multiple_of(2);
            let line = x.abs_diff(256) < 2 || !(8..504).contains(&x) || !(8..248).contains(&y) || {
                let (dx, dy) = (x as f32 - 256.0, y as f32 - 128.0);
                ((dx * dx + dy * dy).sqrt() - 40.0).abs() < 2.0
            };
            let grass = if stripe {
                [0.12, 0.42, 0.1]
            } else {
                [0.16, 0.5, 0.13]
            };
            Texel::new(if line { [0.9, 0.92, 0.9] } else { grass }, 0.9, 0.0)
        }
        Set::Rails => {
            // Along u: sleepers across, two rails along, ballast between.
            let sleeper = x % 64 < 20 && (40..216).contains(&y);
            let rail = (60..74).contains(&y) || (182..196).contains(&y);
            let pebble = unit(cell(x / 4, y / 4, 127));
            if rail {
                Texel::new([0.7, 0.7, 0.72], 0.25, 1.0).at(1.0)
            } else if sleeper {
                Texel::new([0.4, 0.38, 0.36], 0.8, 0.0).at(0.75)
            } else {
                Texel::new(times([0.35, 0.33, 0.3], 0.7 + pebble * 0.6), 0.95, 0.0)
                    .at(0.3 + pebble * 0.3)
            }
        }
        Set::Grime => {
            // Occlusion over a whole building (read through its second
            // coordinates): dark at the foot, streaks running down.
            let foot = ramp(0.8, 1.0, v);
            let lip = 1.0 - ramp(0.0, 0.04, v);
            let streak = ramp(0.6, 0.85, noise(u, 0.0, 16, 1, 131)) * ramp(0.1, 0.9, v);
            let mut texel = Texel::new([1.0; 3], 1.0, 0.0);
            texel.ao = 1.0 - foot * 0.5 - lip * 0.2 - streak * 0.3;
            texel
        }
        Set::Leaves | Set::Fronds => Texel::new([0.0; 3], 1.0, 0.0),
    }
}

/// Neon tubes on a dark backing: rings, bars and zigzags that glow white
/// (materials color them with their emissive factor).
fn neon(x: u32, y: u32) -> Texel {
    let (px, py) = (x as f32 % 128.0 - 64.0, y as f32 % 128.0 - 64.0);
    let tile = (x / 128) + (y / 128) * 2;
    let distance = match tile {
        0 => ((px * px + py * py).sqrt() - 40.0).abs(),
        1 => {
            let zig = ((px + 64.0) % 32.0 - 16.0).abs() - 8.0;
            (py - zig * 2.0).abs()
        }
        2 => math::min((py + 20.0).abs(), (py - 20.0).abs()) + math::max(px.abs() - 48.0, 0.0),
        _ => {
            let star = math::max(px.abs(), py.abs()) - 36.0;
            math::min(star.abs(), (px - py).abs() * 0.7)
        }
    };
    let tube = 1.0 - ramp(1.5, 3.0, distance);
    let halo = (1.0 - ramp(2.0, 14.0, distance)) * 0.35;
    let mut texel =
        Texel::new(mix([0.03, 0.03, 0.035], [0.9, 0.9, 0.9], tube), 0.4, 0.2).at(0.4 + tube * 0.5);
    let light = math::max(tube, halo);
    texel.glow = [light; 3];
    texel
}

/// A billboard advertisement: a gradient, a shape and bars of "text".
fn advert(ad: u8, u: f32, v: f32) -> Texel {
    let palettes = [
        ([0.9, 0.2, 0.35], [1.0, 0.75, 0.2], [1.0, 1.0, 1.0]),
        ([0.1, 0.35, 0.9], [0.2, 0.85, 0.9], [1.0, 0.95, 0.4]),
        ([0.15, 0.7, 0.35], [0.9, 0.95, 0.3], [0.1, 0.1, 0.15]),
        ([0.55, 0.15, 0.85], [0.95, 0.35, 0.7], [1.0, 1.0, 1.0]),
        ([0.95, 0.5, 0.1], [0.95, 0.2, 0.15], [1.0, 0.95, 0.85]),
        ([0.05, 0.05, 0.08], [0.2, 0.9, 0.8], [0.95, 0.3, 0.6]),
    ];
    let (a, b, ink) = palettes[ad as usize % palettes.len()];
    let mut color = mix(a, b, v);
    let (dx, dy) = (u - 0.72, v - 0.45);
    let shape = match ad % 3 {
        0 => (dx * dx + dy * dy).sqrt() < 0.22,
        1 => dx.abs() + dy.abs() < 0.25,
        _ => dx.abs() < 0.18 && dy.abs() < 0.18 && (dx + dy).abs() > 0.05,
    };
    let text = u > 0.08
        && u < 0.52
        && ((0.3..0.36).contains(&v) || (0.44..0.48).contains(&v) || (0.54..0.58).contains(&v))
        && !((u * 40.0) as u32).is_multiple_of(5);
    let headline = u > 0.08 && u < 0.5 && (0.14..0.24).contains(&v);
    if shape {
        color = ink;
    }
    if text || headline {
        color = mix(ink, color, 0.1);
    }
    let border = !(0.02..0.98).contains(&u) || !(0.03..0.97).contains(&v);
    if border {
        color = [0.05, 0.05, 0.05];
    }
    Texel::new(color, 0.3, 0.0).lit(if border { [0.0; 3] } else { times(color, 0.9) })
}

/// The street furniture atlas: 4 × 4 uniform regions of 64 texels (see
/// `Region`).
fn fixture(x: u32, y: u32, grain: f32) -> Texel {
    let none = [0.0; 3];
    match (x / 64, y / 64) {
        (0, 0) => region([0.06, 0.07, 0.08], 0.45, 0.8, none, grain),
        (1, 0) => region([0.7, 0.72, 0.75], 0.3, 1.0, none, grain),
        (2, 0) => region([0.95, 0.95, 0.97], 0.05, 1.0, none, grain * 0.2),
        (3, 0) => region([1.0, 0.95, 0.85], 0.4, 0.0, [1.0, 0.78, 0.5], 0.0),
        (0, 1) => region([0.95, 0.97, 1.0], 0.4, 0.0, [0.85, 0.92, 1.0], 0.0),
        (1, 1) => region([0.9, 0.1, 0.08], 0.2, 0.0, [1.0, 0.08, 0.04], 0.0),
        (2, 1) => region([0.1, 0.9, 0.3], 0.2, 0.0, [0.1, 1.0, 0.35], 0.0),
        (3, 1) => region([1.0, 0.65, 0.1], 0.2, 0.0, [1.0, 0.55, 0.05], 0.0),
        (0, 2) => region([0.05, 0.35, 0.18], 0.5, 0.0, none, grain),
        (1, 2) => region([0.08, 0.2, 0.55], 0.5, 0.0, none, grain),
        (2, 2) => {
            let hazard = ((x + y) / 12).is_multiple_of(2);
            region(
                if hazard {
                    [0.95, 0.75, 0.05]
                } else {
                    [0.05, 0.05, 0.05]
                },
                0.6,
                0.0,
                none,
                grain,
            )
        }
        (3, 2) => region([0.45, 0.3, 0.18], 0.75, 0.0, none, grain),
        (0, 3) => region([0.03, 0.03, 0.03], 0.9, 0.0, none, grain),
        (1, 3) => region([0.75, 0.08, 0.06], 0.4, 0.0, none, grain),
        (2, 3) => region([0.55, 0.54, 0.52], 0.9, 0.0, none, grain),
        _ => region([0.02, 0.03, 0.04], 0.05, 0.5, none, 0.0),
    }
}

/// The vehicle atlas: paint (white, tinted per vehicle), glass, trim,
/// lights, rubber and grille regions of 64 texels.
fn vehicle(x: u32, y: u32, grain: f32) -> Texel {
    let none = [0.0; 3];
    match (x / 64, y / 64) {
        (0 | 1, 0) => {
            let flake = unit(cell(x, y, 137)) * 0.06;
            region(
                [0.92 + flake, 0.92 + flake, 0.92 + flake],
                0.22,
                0.5,
                none,
                0.0,
            )
        }
        (2, 0) => region([0.03, 0.04, 0.05], 0.04, 0.6, none, 0.0),
        (3, 0) => region([0.95, 0.95, 0.97], 0.06, 1.0, none, 0.0),
        (0, 1) => region([1.0, 1.0, 0.95], 0.2, 0.0, [1.0, 0.95, 0.8], 0.0),
        (1, 1) => region([0.8, 0.05, 0.05], 0.2, 0.0, [1.0, 0.05, 0.03], 0.0),
        (2, 1) => region([0.04, 0.04, 0.04], 0.85, 0.0, none, grain),
        (3, 1) => region([0.12, 0.12, 0.13], 0.4, 0.8, none, grain),
        (0, 2) => region([0.2, 0.3, 1.0], 0.2, 0.0, [0.1, 0.25, 1.0], 0.0),
        (1, 2) => region([1.0, 0.1, 0.1], 0.2, 0.0, [1.0, 0.08, 0.05], 0.0),
        (2, 2) => region([0.15, 0.15, 0.16], 0.7, 0.0, none, grain),
        (3, 2) => region([0.95, 0.8, 0.1], 0.3, 0.0, [1.0, 0.7, 0.1], 0.0),
        _ => region([0.5, 0.5, 0.52], 0.5, 0.3, none, grain),
    }
}

/// Paints leaves (or palm leaflets) into a canopy card: coverage in alpha,
/// a dome per leaf in height.
fn foliage(set: Set, width: u32, height: u32) -> Vec<Texel> {
    let mut texels = vec![
        {
            let mut empty = Texel::new([0.1, 0.25, 0.08], 0.9, 0.0).at(0.0);
            empty.alpha = 0.0;
            empty
        };
        (width * height) as usize
    ];
    let mut paint =
        |cx: f32, cy: f32, length: f32, breadth: f32, angle: f32, color: [f32; 3], depth: f32| {
            let (sin, cos) = math::sin_cos(angle);
            let reach = length.ceil() as i32 + 1;
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let (x, y) = (cx as i32 + dx, cy as i32 + dy);
                    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                        continue;
                    }
                    let (fx, fy) = (x as f32 - cx, y as f32 - cy);
                    let (a, b) = (fx * cos + fy * sin, -fx * sin + fy * cos);
                    let inside = (a / length).powi(2) + (b / breadth).powi(2);
                    if inside < 1.0 {
                        let texel = &mut texels[(y as u32 * width + x as u32) as usize];
                        let dome = (1.0 - inside).sqrt();
                        texel.alpha = 1.0;
                        texel.color = times(color, 0.8 + dome * 0.3);
                        texel.height = depth * 0.5 + dome * 0.5;
                        texel.rough = 0.75;
                    }
                }
            }
        };
    let (w, h) = (width as f32, height as f32);
    if set == Set::Leaves {
        // A rounded mass of leaves, darker inside.
        for index in 0..900u32 {
            let random = cell(index, 1, 139);
            let angle = unit(random) * TAU;
            let radius = unit(random >> 5).sqrt() * 0.46;
            let (sin, cos) = math::sin_cos(angle);
            let (cx, cy) = (
                w * (0.5 + cos * radius),
                h * 0.78 * (0.5 + sin * radius * 0.92),
            );
            let depth = index as f32 / 900.0;
            let tone = unit(random >> 11);
            let color = mix(
                mix([0.05, 0.2, 0.05], [0.18, 0.42, 0.1], depth),
                [0.35, 0.55, 0.12],
                tone * 0.3 * depth,
            );
            paint(cx, cy, 9.0, 4.5, unit(random >> 17) * TAU, color, depth);
        }
    } else {
        // Two fronds: a stem and leaflets along it.
        for frond in 0..2 {
            let x0 = w * (0.25 + frond as f32 * 0.5);
            for step in 0..56 {
                let t = step as f32 / 56.0;
                let (cx, cy) = (x0, h * (0.03 + t * 0.72));
                paint(cx, cy, 3.0, 3.0, 0.0, [0.3, 0.35, 0.12], 1.0);
                let spread = (1.0 - (t - 0.45).abs() * 1.6) * w * 0.11;
                for side in [-1.0f32, 1.0] {
                    let angle = if side < 0.0 { 2.4 } else { 0.74 };
                    paint(
                        cx + side * spread * 0.55,
                        cy + spread * 0.25,
                        math::max(spread * 0.6, 2.0),
                        2.0,
                        angle,
                        mix([0.1, 0.35, 0.08], [0.25, 0.5, 0.12], t),
                        0.6,
                    );
                }
            }
        }
    }
    // Bark below the leaves, for trunks drawn with the same material.
    for y in height * 4 / 5..height {
        for x in 0..width {
            let (u, v) = (x as f32 / w, y as f32 / h);
            let ridge = noise(u, v, 24, 3, 101);
            texels[(y * width + x) as usize] =
                Texel::new(mix([0.2, 0.13, 0.08], [0.36, 0.25, 0.15], ridge), 0.9, 0.0).at(ridge);
        }
    }
    texels
}

/// Generates every map of `set`.
pub(crate) fn generate(set: Set) -> Maps {
    let (width, height) = set.size();
    let seed = Set::all()
        .iter()
        .position(|other| *other == set)
        .expect("every set is listed") as u32;
    let texels: Vec<Texel> = if matches!(set, Set::Leaves | Set::Fronds) {
        foliage(set, width, height)
    } else {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| texel(set, seed, x, y, width, height))
            .collect()
    };
    let image = |srgb: bool, normal: bool, pixel: &dyn Fn(&Texel) -> [u8; 4]| Image {
        width,
        height,
        pixels: texels.iter().flat_map(pixel).collect(),
        srgb,
        normal,
    };
    let masked = set.masked();
    let base = image(true, false, &|texel| {
        [
            encode(texel.color[0]),
            encode(texel.color[1]),
            encode(texel.color[2]),
            if masked { byte(texel.alpha) } else { 255 },
        ]
    });
    let mr = (!matches!(set, Set::Ad(_) | Set::Neon)).then(|| {
        image(false, false, &|texel| {
            [byte(texel.ao), byte(texel.rough), byte(texel.metal), 255]
        })
    });
    let relief = set.relief();
    let normal = (relief > 0.0).then(|| {
        let at = |x: i32, y: i32| {
            let (x, y) = (x.rem_euclid(width as i32), y.rem_euclid(height as i32));
            texels[(y as u32 * width + x as u32) as usize].height
        };
        let pixels = (0..height as i32)
            .flat_map(|y| (0..width as i32).map(move |x| (x, y)))
            .flat_map(|(x, y)| {
                let dx = (at(x + 1, y) - at(x - 1, y)) * 0.5 * relief;
                let dy = (at(x, y + 1) - at(x, y - 1)) * 0.5 * relief;
                // Green points up the image (glTF), which is -y here.
                let normal = Vec3::new(-dx, dy, 1.0).normalize();
                [
                    byte(normal.x * 0.5 + 0.5),
                    byte(normal.y * 0.5 + 0.5),
                    byte(normal.z * 0.5 + 0.5),
                    255,
                ]
            })
            .collect();
        Image {
            width,
            height,
            pixels,
            srgb: false,
            normal: true,
        }
    });
    let emissive = set.emits().then(|| {
        image(true, false, &|texel| {
            [
                encode(texel.glow[0]),
                encode(texel.glow[1]),
                encode(texel.glow[2]),
                255,
            ]
        })
    });
    Maps {
        base,
        mr,
        normal,
        emissive,
    }
}

// Mip chains for block-compressed textures, which renderers cannot generate.

fn linear_table() -> [f32; 256] {
    let mut table = [0.0; 256];
    for (index, value) in table.iter_mut().enumerate() {
        let srgb = index as f32 / 255.0;
        *value = if srgb <= 0.040_45 {
            srgb / 12.92
        } else {
            ((srgb + 0.055) / 1.055).powf(2.4)
        };
    }
    table
}

/// Levels of `image` from its base down to 4 texels on its shorter side,
/// averaged in linear light for color data and renormalized for normals.
pub(crate) fn mip_chain(image: &Image) -> Vec<(u32, u32, Vec<u8>)> {
    let table = linear_table();
    let mut levels = vec![(image.width, image.height, image.pixels.clone())];
    loop {
        let (width, height, pixels) = levels.last().expect("the base level exists");
        let (width, height) = (*width, *height);
        if width <= 4 || height <= 4 {
            break;
        }
        let (next_width, next_height) = (width / 2, height / 2);
        let mut next = Vec::with_capacity((next_width * next_height * 4) as usize);
        for y in 0..next_height {
            for x in 0..next_width {
                let mut sum = [0.0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let index = (((y * 2 + dy) * width + x * 2 + dx) * 4) as usize;
                    for channel in 0..4 {
                        let value = pixels[index + channel];
                        sum[channel] += if image.srgb && channel < 3 {
                            table[value as usize]
                        } else {
                            value as f32 / 255.0
                        };
                    }
                }
                let mut texel = sum.map(|value| value * 0.25);
                if image.normal {
                    let normal = Vec3::new(
                        texel[0] * 2.0 - 1.0,
                        texel[1] * 2.0 - 1.0,
                        texel[2] * 2.0 - 1.0,
                    )
                    .normalize_or(Vec3::Z);
                    texel[0] = normal.x * 0.5 + 0.5;
                    texel[1] = normal.y * 0.5 + 0.5;
                    texel[2] = normal.z * 0.5 + 0.5;
                }
                for (channel, value) in texel.iter().enumerate() {
                    next.push(if image.srgb && channel < 3 {
                        encode(*value)
                    } else {
                        byte(*value)
                    });
                }
            }
        }
        levels.push((next_width, next_height, next));
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_set_generates_consistent_maps() {
        for set in Set::all() {
            let maps = generate(set);
            let (width, height) = set.size();
            for image in [
                Some(&maps.base),
                maps.mr.as_ref(),
                maps.normal.as_ref(),
                maps.emissive.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                assert_eq!((image.width, image.height), (width, height), "{set:?}");
                assert_eq!(image.pixels.len(), (width * height * 4) as usize, "{set:?}");
            }
            let covered = maps
                .base
                .pixels
                .chunks(4)
                .filter(|pixel| pixel[3] == 255)
                .count();
            if set.masked() {
                assert!(
                    covered > 0 && covered < (width * height) as usize,
                    "{set:?} masks"
                );
            } else {
                assert_eq!(covered, (width * height) as usize, "{set:?} is opaque");
            }
        }
    }

    #[test]
    fn tiling_sets_repeat_seamlessly() {
        // Opposite edges of the noise agree, so repeats show no seam.
        assert!((fbm(0.0, 0.3, 16, 4, 1) - fbm(1.0, 0.3, 16, 4, 1)).abs() < 1.0e-5);
        assert!((fbm(0.7, 0.0, 16, 4, 1) - fbm(0.7, 1.0, 16, 4, 1)).abs() < 1.0e-5);
    }

    #[test]
    fn mip_chains_halve_down_to_four_texels() {
        let maps = generate(Set::Grime);
        let chain = mip_chain(&maps.mr.expect("grime occludes"));
        let sizes: Vec<_> = chain.iter().map(|(w, h, _)| (*w, *h)).collect();
        assert_eq!(sizes, [(64, 256), (32, 128), (16, 64), (8, 32), (4, 16)]);
    }
}
