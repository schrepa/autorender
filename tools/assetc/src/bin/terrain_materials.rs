//! Generates the Stress Lab's terrain surface materials (decision 054):
//! tileable procedural grass, dry grass, dirt, scree, rock and snow, each as
//! three uncompressed RGBA8 KTX2 maps that hosts load with exact alpha.
//!
//! ```text
//! cargo run -p engine_assetc --release --bin terrain_materials -- [--out DIR] [--size N] [--preview DIR]
//! ```
//!
//! Maps lie top-down: rows run from the texture's top, u along world x and
//! v along world z, and every map wraps seamlessly. `NAME_albedo.ktx2` holds
//! sRGB color. `NAME_normal.ktx2` holds tangent-space normals in OpenGL's
//! convention: +X the image's right, +Y its up (world -z), +Z out of the
//! ground. `NAME_orm.ktx2` holds ambient occlusion in red, roughness in
//! green, metalness in blue and height in alpha (0 to 1, a half on average),
//! which height-based blending reads. `materials.json` lists each
//! material's mean linear albedo, mean roughness and the meters one repeat
//! should cover. `--preview` also writes PNGs to look at: each map, the
//! albedo and a lit view tiled two by two, and every material lit side by
//! side in `overview.png`; `NAME_far.png` and `overview_far.png`
//! show each material averaged four by four, as mipmaps would, and repeated
//! four by four, where repetition at a distance would show.
//!
//! Maps keep no structure broader than about a sixth of the texture: a
//! repeating texture's broad light and dark regions and tilts survive
//! mipmapping and show as a grid of blotches from afar. Engines vary the
//! ground broadly themselves.

use std::f32::consts::TAU;
use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_core::math::{clamp, max, min, sin_cos};

/// Texels along each side, unless `--size` says otherwise.
const DEFAULT_SIZE: usize = 512;
/// Materials generate at this many times their size, then average down.
const SUPERSAMPLE: usize = 2;

fn main() {
    if let Err(error) = run() {
        eprintln!("terrain_materials: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let usage = "usage: terrain_materials [--out DIR] [--size N] [--preview DIR]";
    let mut out = PathBuf::from("web/public/stress/terrain");
    let mut size = DEFAULT_SIZE;
    let mut preview = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || arguments.next().ok_or_else(|| usage.to_owned());
        match argument.as_str() {
            "--out" => out = PathBuf::from(value()?),
            "--size" => {
                let text = value()?;
                size = text
                    .parse()
                    .map_err(|error| format!("size {text}: {error}"))?;
                if !(16..=4096).contains(&size) || !size.is_power_of_two() {
                    return Err(format!(
                        "size {size} must be a power of two from 16 to 4096"
                    ));
                }
            }
            "--preview" => preview = Some(PathBuf::from(value()?)),
            _ => return Err(usage.into()),
        }
    }
    std::fs::create_dir_all(&out).map_err(|error| format!("{}: {error}", out.display()))?;
    if let Some(preview) = &preview {
        std::fs::create_dir_all(preview)
            .map_err(|error| format!("{}: {error}", preview.display()))?;
    }
    let mut listing = Vec::new();
    let mut overview = Vec::new();
    let mut overview_far = Vec::new();
    for kind in Kind::ALL {
        let start = Instant::now();
        let mut maps = kind.generate(size * SUPERSAMPLE).downsample();
        maps.equalize();
        let encoded = Encoded::new(&maps, kind.relief());
        for (map, pixels, srgb) in [
            ("albedo", &encoded.albedo, true),
            ("normal", &encoded.normal, false),
            ("orm", &encoded.orm, false),
        ] {
            let path = out.join(format!("{}_{map}.ktx2", kind.name()));
            std::fs::write(&path, ktx2(size as u32, pixels, srgb))
                .map_err(|error| format!("{}: {error}", path.display()))?;
        }
        let (albedo, roughness) = encoded.means();
        listing.push(serde_json::json!({
            "name": kind.name(),
            "albedo": albedo,
            "roughness": roughness,
            "tiling": kind.tiling(),
        }));
        if let Some(preview) = &preview {
            let lit = encoded.lit();
            let far = encoded.far();
            write_previews(preview, kind.name(), &encoded, &lit, &far)?;
            overview.push(lit);
            overview_far.push(far);
        }
        println!(
            "{}: {:.2} s, mean albedo [{:.3}, {:.3}, {:.3}], roughness {roughness:.2}",
            kind.name(),
            start.elapsed().as_secs_f32(),
            albedo[0],
            albedo[1],
            albedo[2],
        );
    }
    let json = serde_json::json!({ "size": size, "materials": listing });
    let path = out.join("materials.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&json).expect("JSON values serialize"),
    )
    .map_err(|error| format!("{}: {error}", path.display()))?;
    if let Some(preview) = &preview {
        // Every material lit, three to a row, near and far.
        for (name, images) in [
            ("overview.png", &overview),
            ("overview_far.png", &overview_far),
        ] {
            let columns = 3;
            let rows = images.len().div_ceil(columns);
            let mut sheet = vec![0_u8; columns * size * rows * size * 4];
            for (index, image) in images.iter().enumerate() {
                let (column, row) = (index % columns, index / columns);
                for y in 0..size {
                    let from = y * size * 4;
                    let to = ((row * size + y) * columns * size + column * size) * 4;
                    sheet[to..to + size * 4].copy_from_slice(&image[from..from + size * 4]);
                }
            }
            write_png(
                &preview.join(name),
                (columns * size) as u32,
                (rows * size) as u32,
                &sheet,
            )?;
        }
    }
    println!("wrote {} materials to {}", Kind::ALL.len(), out.display());
    Ok(())
}

/// The generated materials.
#[derive(Clone, Copy)]
enum Kind {
    Grass,
    DryGrass,
    Dirt,
    Scree,
    Rock,
    Snow,
}

impl Kind {
    /// In the order the page and the workload name them.
    const ALL: [Self; 6] = [
        Self::Grass,
        Self::DryGrass,
        Self::Dirt,
        Self::Scree,
        Self::Rock,
        Self::Snow,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Grass => "grass",
            Self::DryGrass => "dry_grass",
            Self::Dirt => "dirt",
            Self::Scree => "scree",
            Self::Rock => "rock",
            Self::Snow => "snow",
        }
    }

    /// Meters of ground one repeat should cover.
    fn tiling(self) -> f32 {
        match self {
            Self::Grass | Self::DryGrass => 2.5,
            Self::Dirt => 3.0,
            Self::Scree => 4.0,
            Self::Rock | Self::Snow => 6.0,
        }
    }

    /// How steeply height turns the normals: texels of rise per unit of
    /// height at 512 texels.
    fn relief(self) -> f32 {
        match self {
            Self::Grass => 4.0,
            Self::DryGrass => 4.0,
            Self::Dirt => 3.0,
            Self::Scree => 5.0,
            Self::Rock => 22.0,
            Self::Snow => 2.5,
        }
    }

    /// The material at `size` texels along each side.
    fn generate(self, size: usize) -> Maps {
        match self {
            Self::Grass => meadow(size, false),
            Self::DryGrass => meadow(size, true),
            Self::Dirt => dirt(size),
            Self::Scree => scree(size),
            Self::Rock => rock(size),
            Self::Snow => snow(size),
        }
    }
}

/// Linear RGB.
type Rgb = [f32; 3];

/// One texel of a material while it generates.
#[derive(Clone, Copy, Default)]
struct Texel {
    albedo: Rgb,
    height: f32,
    roughness: f32,
}

/// A generated material: `size` by `size` texels of linear albedo, height
/// (in its own units, about 0 to 1), ambient occlusion and roughness.
struct Maps {
    size: usize,
    albedo: Vec<Rgb>,
    height: Vec<f32>,
    occlusion: Vec<f32>,
    roughness: Vec<f32>,
}

impl Maps {
    /// Maps of `texels`, with occlusion of `strength` from their heights.
    fn new(size: usize, texels: Vec<Texel>, strength: f32) -> Self {
        let height: Vec<f32> = texels.iter().map(|texel| texel.height).collect();
        Self {
            size,
            occlusion: occlusion(&height, size, strength),
            albedo: texels.iter().map(|texel| texel.albedo).collect(),
            roughness: texels.iter().map(|texel| texel.roughness).collect(),
            height,
        }
    }

    /// Removes structure broader than about a sixth of the texture, which
    /// would repeat as a grid from afar: height and the occlusion and
    /// roughness maps lose their low-passed variation around their means
    /// (heights spread over 0 to 1 again when encoded, before their
    /// normals derive), and albedo is divided by its low-passed luminance
    /// and multiplied by its mean luminance, keeping hue and small-scale
    /// contrast.
    fn equalize(&mut self) {
        let size = self.size;
        // Three box passes of this radius: about a Gaussian whose low-pass
        // keeps half of structure a sixth of the texture across, nearly all
        // of anything broader.
        let radius = usize::max(1, size / 32);
        let flatten = |values: &mut [f32]| {
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let low = low_pass(values, size, radius);
            for (value, low) in values.iter_mut().zip(low) {
                *value += mean - low;
            }
        };
        flatten(&mut self.height);
        flatten(&mut self.occlusion);
        flatten(&mut self.roughness);
        let luminance: Vec<f32> = self
            .albedo
            .iter()
            .map(|color| 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2])
            .collect();
        let mean = luminance.iter().sum::<f32>() / luminance.len() as f32;
        let low = low_pass(&luminance, size, radius);
        for (color, low) in self.albedo.iter_mut().zip(low) {
            *color = scale(*color, mean / max(low, 1.0e-4));
        }
    }

    /// These maps at half the size, each texel the mean of four.
    fn downsample(&self) -> Self {
        let half = self.size / 2;
        let mean = |values: &[f32], x: usize, y: usize| {
            let at = |dx: usize, dy: usize| values[(2 * y + dy) * self.size + 2 * x + dx];
            (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25
        };
        let mut maps = Self {
            size: half,
            albedo: Vec::with_capacity(half * half),
            height: Vec::with_capacity(half * half),
            occlusion: Vec::with_capacity(half * half),
            roughness: Vec::with_capacity(half * half),
        };
        let channels: [Vec<f32>; 3] =
            std::array::from_fn(|channel| self.albedo.iter().map(|rgb| rgb[channel]).collect());
        for y in 0..half {
            for x in 0..half {
                maps.albedo.push(std::array::from_fn(|channel| {
                    mean(&channels[channel], x, y)
                }));
                maps.height.push(mean(&self.height, x, y));
                maps.occlusion.push(mean(&self.occlusion, x, y));
                maps.roughness.push(mean(&self.roughness, x, y));
            }
        }
        maps
    }
}

/// `size` by `size` values of `texel(u, v)` at each texel's center, rows
/// shared among the machine's threads.
fn texture<T: Send>(size: usize, texel: impl Fn(f32, f32) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let rows = size.div_ceil(threads);
    let parts: Vec<Vec<T>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|thread| {
                let texel = &texel;
                scope.spawn(move || {
                    let first = usize::min(thread * rows, size);
                    let last = usize::min(first + rows, size);
                    let mut part = Vec::with_capacity((last - first) * size);
                    for y in first..last {
                        for x in 0..size {
                            part.push(texel(
                                (x as f32 + 0.5) / size as f32,
                                (y as f32 + 0.5) / size as f32,
                            ));
                        }
                    }
                    part
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("texture workers finish"))
            .collect()
    });
    parts.into_iter().flatten().collect()
}

// Arithmetic.

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

fn scale(color: Rgb, by: f32) -> Rgb {
    color.map(|channel| channel * by)
}

fn saturate(value: f32) -> f32 {
    clamp(value, 0.0, 1.0)
}

fn smoothstep(from: f32, to: f32, value: f32) -> f32 {
    let t = saturate((value - from) / (to - from));
    t * t * (3.0 - 2.0 * t)
}

/// A well-mixed hash of a lattice point and seed.
fn hash(x: u32, y: u32, seed: u32) -> u32 {
    let mut h =
        x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

/// A hash as a number from 0 to 1.
fn unit(hash: u32) -> f32 {
    (hash >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// A small deterministic random sequence (xorshift64*).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
    }

    fn unit(&mut self) -> f32 {
        unit(self.next())
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        lerp(low, high, self.unit())
    }
}

// Periodic noise: every lattice wraps a whole number of times across the
// texture, so everything built from them tiles.

/// One octave of gradient noise: random unit gradients on a lattice
/// `period` cells across the texture.
struct Lattice {
    period: usize,
    gradients: Vec<[f32; 2]>,
}

impl Lattice {
    fn new(period: usize, seed: u32) -> Self {
        let gradients = (0..period * period)
            .map(|index| {
                let (sin, cos) = sin_cos(unit(hash(index as u32, period as u32, seed)) * TAU);
                [cos, sin]
            })
            .collect();
        Self { period, gradients }
    }

    /// The noise at texture coordinates `(u, v)`, about -1 to 1.
    fn sample(&self, u: f32, v: f32) -> f32 {
        let period = self.period;
        let x = u * period as f32;
        let y = v * period as f32;
        let (cell_x, cell_y) = (x.floor(), y.floor());
        let (tx, ty) = (x - cell_x, y - cell_y);
        let x0 = (cell_x as i64).rem_euclid(period as i64) as usize;
        let y0 = (cell_y as i64).rem_euclid(period as i64) as usize;
        let (x1, y1) = ((x0 + 1) % period, (y0 + 1) % period);
        let corner = |cx: usize, cy: usize, dx: f32, dy: f32| {
            let [gx, gy] = self.gradients[cy * period + cx];
            gx * dx + gy * dy
        };
        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let (sx, sy) = (fade(tx), fade(ty));
        let top = lerp(corner(x0, y0, tx, ty), corner(x1, y0, tx - 1.0, ty), sx);
        let bottom = lerp(
            corner(x0, y1, tx, ty - 1.0),
            corner(x1, y1, tx - 1.0, ty - 1.0),
            sx,
        );
        lerp(top, bottom, sy) * 1.4
    }
}

/// Fractal noise: `octaves` lattices from `period` cells across the
/// texture, each twice as fine and `gain` times as strong as the last.
struct Noise {
    octaves: Vec<(Lattice, f32)>,
    total: f32,
}

impl Noise {
    fn new(period: usize, octaves: usize, gain: f32, seed: u32) -> Self {
        let mut amplitude = 1.0;
        let mut total = 0.0;
        let octaves = (0..octaves)
            .map(|octave| {
                let lattice =
                    Lattice::new(period << octave, seed.wrapping_add(octave as u32 * 7919));
                let entry = (lattice, amplitude);
                total += amplitude;
                amplitude *= gain;
                entry
            })
            .collect();
        Self { octaves, total }
    }

    /// About -1 to 1.
    fn fbm(&self, u: f32, v: f32) -> f32 {
        self.octaves
            .iter()
            .map(|(lattice, amplitude)| lattice.sample(u, v) * amplitude)
            .sum::<f32>()
            / self.total
    }

    /// 0 to 1, about a half on average.
    fn unit(&self, u: f32, v: f32) -> f32 {
        saturate(self.fbm(u, v) * 0.5 + 0.5)
    }

    /// Sharp crests where each octave crosses zero, 0 to 1.
    fn ridged(&self, u: f32, v: f32) -> f32 {
        self.octaves
            .iter()
            .map(|(lattice, amplitude)| {
                let crest = 1.0 - saturate(lattice.sample(u, v).abs());
                crest * crest * amplitude
            })
            .sum::<f32>()
            / self.total
    }
}

/// Cellular noise: one jittered point in each cell of a `count` by `count`
/// grid across the texture.
struct Cells {
    count: usize,
    points: Vec<[f32; 2]>,
}

/// Where a texel lies among cells: its nearest point's cell, its distance
/// to that point and to the cell's edge (in cells), and the vector from the
/// texel to the point.
struct Hit {
    id: u32,
    center: f32,
    edge: f32,
    offset: [f32; 2],
}

impl Cells {
    /// Points within `jitter` (0 to 1) of their cells' middles.
    fn new(count: usize, jitter: f32, seed: u32) -> Self {
        let points = (0..count * count)
            .map(|index| {
                let index = index as u32;
                [
                    0.5 + (unit(hash(index, 0, seed)) - 0.5) * jitter,
                    0.5 + (unit(hash(index, 1, seed)) - 0.5) * jitter,
                ]
            })
            .collect();
        Self { count, points }
    }

    /// The cell hit at texture coordinates `(u, v)`; the edge distance is
    /// exact (Quilez's two passes).
    fn sample(&self, u: f32, v: f32) -> Hit {
        let count = self.count as i64;
        let x = u * self.count as f32;
        let y = v * self.count as f32;
        let (cell_x, cell_y) = (x.floor(), y.floor());
        let (fx, fy) = (x - cell_x, y - cell_y);
        let (cell_x, cell_y) = (cell_x as i64, cell_y as i64);
        let point = |dx: i64, dy: i64| {
            let index = ((cell_y + dy).rem_euclid(count) * count + (cell_x + dx).rem_euclid(count))
                as usize;
            let [px, py] = self.points[index];
            (index as u32, [dx as f32 + px - fx, dy as f32 + py - fy])
        };
        let mut nearest = (f32::MAX, 0, [0.0; 2], (0, 0));
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (id, offset) = point(dx, dy);
                let distance = offset[0] * offset[0] + offset[1] * offset[1];
                if distance < nearest.0 {
                    nearest = (distance, id, offset, (dx, dy));
                }
            }
        }
        let (distance, id, near, (nx, ny)) = nearest;
        let mut edge = f32::MAX;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let (_, offset) = point(nx + dx, ny + dy);
                let between = [offset[0] - near[0], offset[1] - near[1]];
                let length = (between[0] * between[0] + between[1] * between[1]).sqrt();
                if length > 1.0e-5 {
                    let middle = [(offset[0] + near[0]) * 0.5, (offset[1] + near[1]) * 0.5];
                    edge = min(
                        edge,
                        (middle[0] * between[0] + middle[1] * between[1]) / length,
                    );
                }
            }
        }
        Hit {
            id,
            center: distance.sqrt(),
            edge,
            offset: near,
        }
    }
}

impl Cells {
    /// A faceted surface: around each cell's point a plane of its own tilt
    /// (up to one per cell) and depth (`depth` at most), the planes blended
    /// where cells meet over a width falling with `sharpness`, so facets
    /// meet in creases instead of steps. Returns the height and the cell
    /// whose plane weighs most.
    fn faceted(&self, u: f32, v: f32, depth: f32, sharpness: f32, seed: u32) -> (f32, u32) {
        let count = self.count as i64;
        let x = u * self.count as f32;
        let y = v * self.count as f32;
        let (cell_x, cell_y) = (x.floor(), y.floor());
        let (fx, fy) = (x - cell_x, y - cell_y);
        let (cell_x, cell_y) = (cell_x as i64, cell_y as i64);
        let (mut sum, mut total, mut strongest) = (0.0, 0.0, (0.0, 0));
        for dy in -1..=1 {
            for dx in -1..=1 {
                let index = ((cell_y + dy).rem_euclid(count) * count
                    + (cell_x + dx).rem_euclid(count)) as usize;
                let [px, py] = self.points[index];
                let offset = [dx as f32 + px - fx, dy as f32 + py - fy];
                let id = index as u32;
                let random = |salt: u32| unit(hash(id, salt, seed)) * 2.0 - 1.0;
                let weight = (-sharpness * (offset[0] * offset[0] + offset[1] * offset[1])).exp();
                let plane = random(1) * depth - random(2) * offset[0] - random(3) * offset[1];
                sum += weight * plane;
                total += weight;
                if weight > strongest.0 {
                    strongest = (weight, id);
                }
            }
        }
        (sum / max(total, 1.0e-12), strongest.1)
    }
}

/// `field` averaged over a `radius` texel box around each texel, wrapping.
fn blur(field: &[f32], size: usize, radius: usize) -> Vec<f32> {
    let width = (2 * radius + 1) as f32;
    let pass = |source: &[f32], stride: usize, step: usize| {
        let mut out = vec![0.0; source.len()];
        for line in 0..size {
            let at = |i: i64| source[line * stride + (i.rem_euclid(size as i64) as usize) * step];
            let r = radius as i64;
            let mut sum: f32 = (-r..=r).map(at).sum();
            for i in 0..size as i64 {
                out[line * stride + i as usize * step] = sum / width;
                sum += at(i + r + 1) - at(i - r);
            }
        }
        out
    };
    let rows = pass(field, size, 1);
    pass(&rows, 1, size)
}

/// `field` low-passed: a wrapped box blur of `radius` three times over,
/// about a Gaussian of that deviation.
fn low_pass(field: &[f32], size: usize, radius: usize) -> Vec<f32> {
    let once = blur(field, size, radius);
    let twice = blur(&once, size, radius);
    blur(&twice, size, radius)
}

/// Ambient occlusion from heights: how far each texel sits below its
/// surroundings at a few scales, darkened by `strength`.
fn occlusion(height: &[f32], size: usize, strength: f32) -> Vec<f32> {
    let mut cavity = vec![0.0; height.len()];
    for (radius, weight) in [(0.003, 0.45), (0.01, 0.35), (0.03, 0.2)] {
        let radius = usize::max(1, (radius * size as f32).round() as usize);
        let blurred = blur(height, size, radius);
        for ((cavity, height), blurred) in cavity.iter_mut().zip(height).zip(&blurred) {
            *cavity += weight * max(blurred - height, 0.0);
        }
    }
    cavity
        .iter()
        .map(|cavity| clamp(1.0 - strength * cavity, 0.15, 1.0))
        .collect()
}

// Grass.

/// Blade colors of fresh grass: deep, green, fresh, yellow-green, olive.
const GREEN_BLADES: [Rgb; 5] = [
    [0.022, 0.058, 0.011],
    [0.040, 0.094, 0.016],
    [0.064, 0.128, 0.022],
    [0.096, 0.138, 0.030],
    [0.058, 0.076, 0.022],
];
/// Blade colors of dry grass: straw, tan, pale, brown, a green remnant.
const DRY_BLADES: [Rgb; 5] = [
    [0.270, 0.225, 0.135],
    [0.220, 0.180, 0.110],
    [0.310, 0.272, 0.180],
    [0.165, 0.128, 0.078],
    [0.105, 0.112, 0.050],
];
/// Matted dead blades under dry grass.
const DRY_THATCH: [Rgb; 3] = [
    [0.145, 0.115, 0.072],
    [0.105, 0.084, 0.055],
    [0.180, 0.148, 0.095],
];
/// Dried blades among fresh grass.
const DEAD_BLADE: Rgb = [0.150, 0.118, 0.052];
const SOIL: Rgb = [0.034, 0.025, 0.016];
const THATCH: Rgb = [0.062, 0.050, 0.026];
const HEATHER: Rgb = [0.062, 0.036, 0.040];
const HEATHER_GREEN: Rgb = [0.040, 0.052, 0.022];

/// A tapering blade of grass lying from `root` (texture coordinates) along
/// `direction` (radians), rising from its root's height to its tip's,
/// shaded from `shade[0]` times its color at the root to `shade[1]` at the
/// tip.
struct Blade {
    root: [f32; 2],
    direction: f32,
    length: f32,
    width: f32,
    root_height: f32,
    tip_height: f32,
    color: Rgb,
    shade: [f32; 2],
}

impl Blade {
    /// Draws the blade over `texels` (`size` along each side) wherever it
    /// stands higher than what is there.
    fn draw(&self, texels: &mut [Texel], size: usize) {
        let scale_to = size as f32;
        let (sin, cos) = sin_cos(self.direction);
        let root = [self.root[0] * scale_to, self.root[1] * scale_to];
        let length = self.length * scale_to;
        let half = self.width * scale_to * 0.5;
        let tip = [root[0] + cos * length, root[1] + sin * length];
        let low = [min(root[0], tip[0]) - half, min(root[1], tip[1]) - half];
        let high = [max(root[0], tip[0]) + half, max(root[1], tip[1]) + half];
        for y in low[1].floor() as i64..=high[1].ceil() as i64 {
            for x in low[0].floor() as i64..=high[0].ceil() as i64 {
                let p = [x as f32 + 0.5 - root[0], y as f32 + 0.5 - root[1]];
                let t = saturate((p[0] * cos + p[1] * sin) / length);
                let dx = p[0] - cos * t * length;
                let dy = p[1] - sin * t * length;
                let distance = (dx * dx + dy * dy).sqrt();
                // Tapering to a point.
                let width = half * (1.0 - 0.75 * t);
                if distance > width {
                    continue;
                }
                let across = distance / width;
                let profile = 1.0 - across * across;
                let height = lerp(self.root_height, self.tip_height, t.sqrt()) + 0.05 * profile;
                let size = size as i64;
                let index = (y.rem_euclid(size) * size + x.rem_euclid(size)) as usize;
                let texel = &mut texels[index];
                if height <= texel.height {
                    continue;
                }
                // Darker at the root, lighter and yellower toward the tip,
                // brightest along the blade's middle.
                let shade = lerp(self.shade[0], self.shade[1], t) * (0.82 + 0.18 * profile);
                let color = scale(self.color, shade);
                texel.albedo = [color[0] + 0.012 * t, color[1] + 0.008 * t, color[2]];
                texel.height = height;
                texel.roughness = 0.88 - 0.08 * profile;
            }
        }
    }
}

/// Grass seen from above: tufts of tapering blades over thatch and soil.
/// Dry grass grows over a mat of dead blades, with straw-colored tufts and
/// heather clumps.
fn meadow(size: usize, dry: bool) -> Maps {
    let seed = if dry { 200 } else { 100 };
    let ground = Noise::new(6, 5, 0.55, seed);
    let grain = Noise::new(96, 2, 0.5, seed + 1);
    let mut texels = texture(size, |u, v| {
        let broad = ground.unit(u, v);
        let fine = grain.unit(u, v);
        let floor = if dry { DRY_THATCH[1] } else { THATCH };
        Texel {
            albedo: scale(
                mix(SOIL, floor, smoothstep(0.3, 0.7, broad)),
                0.75 + 0.5 * fine,
            ),
            height: 0.08 * broad + 0.06 * fine,
            roughness: 0.95,
        }
    });
    let mut rng = Rng::new(u64::from(seed));
    if dry {
        // The mat: short dead blades lying every which way.
        for _ in 0..9000 {
            let root_height = rng.range(0.05, 0.15);
            let pick = (rng.unit() * DRY_THATCH.len() as f32) as usize;
            Blade {
                root: [rng.unit(), rng.unit()],
                direction: rng.unit() * TAU,
                length: rng.range(0.012, 0.026),
                width: rng.range(0.003, 0.005),
                root_height,
                tip_height: root_height + rng.range(0.05, 0.15),
                color: scale(
                    DRY_THATCH[pick.min(DRY_THATCH.len() - 1)],
                    rng.range(0.8, 1.2),
                ),
                shade: [0.7, 1.05],
            }
            .draw(&mut texels, size);
        }
        heather(&mut texels, size, seed + 7);
    }
    let palette = if dry { &DRY_BLADES } else { &GREEN_BLADES };
    let tufts = if dry { 2000 } else { 2900 };
    for _ in 0..tufts {
        let center = [rng.unit(), rng.unit()];
        let spread = rng.range(0.004, 0.014);
        let blades = rng.range(5.0, 13.0) as usize;
        // A tuft leans toward neighboring colors and grows to its own vigor.
        let tint = rng.unit();
        let vigor = rng.range(0.8, 1.2);
        for _ in 0..blades {
            let angle = rng.unit() * TAU;
            let reach = spread * rng.unit().sqrt();
            let (sin, cos) = sin_cos(angle);
            let pick = ((tint * 0.65 + rng.unit() * 0.35) * palette.len() as f32) as usize;
            let mut color = scale(palette[pick.min(palette.len() - 1)], rng.range(0.85, 1.15));
            if !dry && rng.unit() < 0.04 {
                color = DEAD_BLADE;
            }
            let root_height = rng.range(0.1, 0.3);
            Blade {
                root: [center[0] + cos * reach, center[1] + sin * reach],
                // Fanning out from the tuft.
                direction: angle + rng.range(-0.7, 0.7),
                length: rng.range(0.012, 0.03) * vigor,
                width: rng.range(0.0028, 0.005),
                root_height,
                tip_height: root_height + rng.range(0.3, 0.7) * vigor,
                color,
                shade: if dry { [0.62, 1.12] } else { [0.45, 1.3] },
            }
            .draw(&mut texels, size);
        }
    }
    Maps::new(size, texels, if dry { 2.0 } else { 2.5 })
}

/// Low clumps of heather among dry grass: dark leaves on cellular patches.
fn heather(texels: &mut [Texel], size: usize, seed: u32) {
    let cells = Cells::new(8, 0.8, seed);
    let leaves = Noise::new(90, 2, 0.5, seed + 1);
    let outline = Noise::new(24, 2, 0.5, seed + 2);
    let clumps = texture(size, |u, v| {
        let hit = cells.sample(u, v);
        if unit(hash(hit.id, 0, seed)) > 0.4 {
            return None;
        }
        let radius = 0.3 + 0.15 * unit(hash(hit.id, 1, seed)) + 0.1 * outline.fbm(u, v);
        let inside = smoothstep(radius, radius * 0.5, hit.center);
        let leaf = leaves.unit(u, v);
        let coverage = inside * smoothstep(0.3, 0.5, leaf);
        (coverage > 0.0).then(|| Texel {
            albedo: scale(
                mix(HEATHER, HEATHER_GREEN, unit(hash(hit.id, 2, seed))),
                0.7 + 0.6 * leaf,
            ),
            height: 0.2 + 0.4 * inside * leaf,
            roughness: 0.9,
        })
    });
    for (texel, clump) in texels.iter_mut().zip(clumps) {
        if let Some(clump) = clump.filter(|clump| clump.height > texel.height) {
            *texel = clump;
        }
    }
}

// Stones.

/// A layer of stones on cellular noise: a share of the cells each hold a
/// stone shrunk from the cell's edges by a gap from a range (in cells),
/// rounded or angular by its own measure, standing from `base` up to `rise`
/// higher, tilted and folded along a crest of its own.
struct Stones {
    cells: Cells,
    share: f32,
    gap: [f32; 2],
    base: f32,
    rise: f32,
    /// Rise per cell of each stone's tilt, at most, and of an angular
    /// stone's crest.
    tilt: f32,
    crest: f32,
    outline: Noise,
    seed: u32,
}

/// A stone under a texel: its height, its cell and how far in from its
/// edge the texel lies (0 at the edge, 1 past its shoulder).
struct Stone {
    height: f32,
    id: u32,
    profile: f32,
}

impl Stones {
    fn sample(&self, u: f32, v: f32) -> Option<Stone> {
        let hit = self.cells.sample(u, v);
        let random = |salt: u32| unit(hash(hit.id, salt, self.seed));
        if random(9) > self.share {
            return None;
        }
        let gap = lerp(self.gap[0], self.gap[1], random(10));
        let inside = hit.edge + self.outline.fbm(u, v) * 0.08 - gap;
        if inside <= 0.0 {
            return None;
        }
        // Rounded stones have wide shoulders and domes; angular ones narrow
        // shoulders, a crest and flat faces.
        let round = random(16);
        let shoulder = lerp(0.07, 0.3, round);
        let profile = saturate(inside / shoulder);
        let rise = lerp(profile, profile.sqrt(), round);
        let (sin, cos) = sin_cos(random(11) * TAU);
        let along = hit.offset[0] * cos + hit.offset[1] * sin;
        let across = hit.offset[1] * cos - hit.offset[0] * sin;
        let tilt = along * self.tilt * (random(12) * 2.0 - 1.0);
        let fold = -across.abs() * self.crest * (1.0 - round);
        let dome = -0.4 * round * min(hit.center * hit.center, 1.0);
        let size = 0.45 + 0.55 * random(13);
        Some(Stone {
            height: self.base + self.rise * size * (rise + tilt + fold + dome),
            id: hit.id,
            profile,
        })
    }

    /// A desaturated color for stone `id` between `dark` and `light` grays,
    /// a little warmer or cooler than neutral.
    fn color(&self, id: u32, dark: f32, light: f32) -> Rgb {
        let value = lerp(dark, light, unit(hash(id, 14, self.seed)));
        let warmth = (unit(hash(id, 15, self.seed)) - 0.5) * 0.12;
        [value * (1.0 + warmth), value, value * (1.0 - warmth)]
    }
}

const GRIT: Rgb = [0.058, 0.054, 0.050];
const SCREE_DIRT: Rgb = [0.085, 0.074, 0.062];

/// Talus: loose angular and rounded stones of strongly varied sizes, many
/// small and few large, the larger lying on the smaller, with gravel and
/// dirt between them.
fn scree(size: usize) -> Maps {
    let seed = 400;
    let grit = Noise::new(140, 2, 0.5, seed);
    let mottle = Noise::new(64, 3, 0.5, seed + 1);
    let dirt = Noise::new(6, 3, 0.5, seed + 2);
    // Cells across the texture, the share holding stones, their gaps, feet
    // and rise: at 4 meters a repeat, stones from about 25 cm down to 2 cm.
    let layer = |count: usize, share: f32, gap: [f32; 2], base: f32, rise: f32, salt: u32| Stones {
        cells: Cells::new(count, 0.9, seed + salt),
        share,
        gap,
        base,
        rise,
        tilt: 0.6,
        crest: 0.5,
        outline: Noise::new(count * 3, 3, 0.5, seed + salt + 50),
        seed: seed + salt,
    };
    let layers = [
        layer(128, 0.9, [0.08, 0.3], 0.04, 0.12, 3),
        layer(64, 0.8, [0.06, 0.3], 0.1, 0.2, 4),
        layer(36, 0.6, [0.05, 0.28], 0.18, 0.28, 5),
        layer(22, 0.6, [0.05, 0.3], 0.26, 0.36, 6),
        layer(13, 1.0, [0.04, 0.42], 0.34, 0.46, 7),
    ];
    let texels = texture(size, |u, v| {
        let fine = grit.unit(u, v);
        let mut texel = Texel {
            albedo: scale(mix(GRIT, SCREE_DIRT, dirt.unit(u, v)), 0.75 + 0.5 * fine),
            height: 0.06 * fine,
            roughness: 0.93,
        };
        for layer in &layers {
            let Some(stone) = layer.sample(u, v) else {
                continue;
            };
            if stone.height <= texel.height {
                continue;
            }
            let speckle = mottle.unit(u, v);
            texel = Texel {
                albedo: scale(
                    layer.color(stone.id, 0.1, 0.21),
                    (0.84 + 0.32 * speckle) * (0.82 + 0.18 * stone.profile),
                ),
                height: stone.height + 0.015 * speckle,
                roughness: 0.72 + 0.12 * speckle,
            };
        }
        // Dirt settled among the low stones.
        let settled = smoothstep(0.45, 0.75, dirt.unit(u + 0.37, v + 0.11))
            * (1.0 - saturate(texel.height * 3.0));
        texel.albedo = mix(texel.albedo, SCREE_DIRT, 0.6 * settled);
        texel
    });
    Maps::new(size, texels, 3.0)
}

// Dirt.

const SOIL_BROWN: Rgb = [0.118, 0.092, 0.068];
const SOIL_DARK: Rgb = [0.070, 0.058, 0.046];
const SOIL_GRAY: Rgb = [0.110, 0.100, 0.088];

/// Bare soil: granular crumbs and soft clods, faint drying cracks in
/// places, scattered angular pebbles.
fn dirt(size: usize) -> Maps {
    let seed = 300;
    let soil = Noise::new(5, 5, 0.55, seed);
    let damp = Noise::new(3, 3, 0.5, seed + 1);
    let grain = Noise::new(180, 2, 0.5, seed + 2);
    let crumbs = Noise::new(90, 3, 0.6, seed + 10);
    let grains = Cells::new(110, 0.9, seed + 12);
    let clods = Cells::new(18, 0.9, seed + 3);
    let cracks = Cells::new(5, 0.95, seed + 4);
    let crack_mask = Noise::new(4, 3, 0.5, seed + 11);
    let warp = Noise::new(12, 3, 0.5, seed + 5);
    let pebble = |count: usize, share: f32, gap: [f32; 2], rise: f32, salt: u32| Stones {
        cells: Cells::new(count, 0.8, seed + salt),
        share,
        gap,
        base: 0.3,
        rise,
        tilt: 0.4,
        crest: 0.3,
        outline: Noise::new(count * 3, 2, 0.5, seed + salt + 50),
        seed: seed + salt,
    };
    let pebbles = [
        pebble(24, 0.22, [0.2, 0.36], 0.5, 6),
        pebble(56, 0.14, [0.22, 0.36], 0.35, 7),
    ];
    let texels = texture(size, |u, v| {
        let broad = soil.unit(u, v);
        let wet = smoothstep(0.45, 0.7, damp.unit(u, v));
        let fine = grain.unit(u, v);
        let crumb = crumbs.unit(u, v);
        // Crumbs: tiny lumps, each its own shade, darker between.
        let lump = grains.sample(u, v);
        let lumpy = smoothstep(0.0, 0.25, lump.edge);
        let shade = 0.82 + 0.36 * unit(hash(lump.id, 0, seed));
        let clod = smoothstep(0.0, 0.3, clods.sample(u, v).edge);
        let (wu, wv) = (
            u + 0.02 * warp.fbm(u, v),
            v + 0.02 * warp.fbm(v + 0.31, u + 0.67),
        );
        // Cracks only where the soil dried, fading along their length.
        let crack = (1.0 - smoothstep(0.0, 0.02, cracks.sample(wu, wv).edge))
            * smoothstep(0.55, 0.75, crack_mask.unit(u, v))
            * (1.0 - wet);
        let base = mix(
            mix(SOIL_BROWN, SOIL_GRAY, 0.6 * broad),
            SOIL_DARK,
            0.55 * wet,
        );
        let mut texel = Texel {
            albedo: scale(
                base,
                (0.8 + 0.2 * fine + 0.2 * (crumb - 0.5))
                    * shade
                    * (0.8 + 0.2 * lumpy)
                    * (1.0 - 0.35 * crack)
                    * (0.92 + 0.08 * clod),
            ),
            height: 0.15 * broad + 0.06 * fine + 0.1 * crumb + 0.08 * lumpy + 0.1 * clod
                - 0.15 * crack,
            roughness: 0.93 - 0.1 * wet,
        };
        for layer in &pebbles {
            let Some(stone) = layer.sample(u, v) else {
                continue;
            };
            if stone.height <= texel.height {
                continue;
            }
            texel = Texel {
                albedo: scale(
                    layer.color(stone.id, 0.07, 0.16),
                    (0.9 + 0.2 * fine) * (0.78 + 0.22 * stone.profile),
                ),
                height: stone.height,
                roughness: 0.78,
            };
        }
        texel
    });
    Maps::new(size, texels, 1.2)
}

// Rock.

const ROCK_GRAY: Rgb = [0.180, 0.178, 0.172];
const ROCK_WARM: Rgb = [0.198, 0.184, 0.166];
const ROCK_COOL: Rgb = [0.160, 0.166, 0.172];
const LICHEN_PALE: Rgb = [0.250, 0.262, 0.225];
const LICHEN_YELLOW: Rgb = [0.215, 0.215, 0.120];

/// A weathered rock face as seen from the side (image up is up): blocks
/// twice as wide as tall, each a tilted plane at its own depth and chipped
/// into smaller tilted facets, stepped into ledges along the contours of
/// broad warped forms, with a few meandering cracks along the bedding,
/// varied grays with faint warm and cool patches, and pale lichen on the
/// ledges.
fn rock(size: usize) -> Maps {
    let seed = 500;
    let warp = [Noise::new(2, 4, 0.5, seed), Noise::new(2, 4, 0.5, seed + 1)];
    let forms = Noise::new(2, 3, 0.5, seed + 2);
    let blocks = Cells::new(7, 0.85, seed + 3);
    let facets = Cells::new(17, 0.9, seed + 4);
    let chips = Noise::new(24, 3, 0.5, seed + 5);
    let grain = Noise::new(160, 2, 0.5, seed + 6);
    let bedding = Noise::new(2, 3, 0.5, seed + 7);
    let openings = Noise::new(3, 3, 0.5, seed + 8);
    let tint = Noise::new(5, 3, 0.5, seed + 9);
    let lichen = Noise::new(16, 4, 0.55, seed + 10);
    let specks = Noise::new(128, 2, 0.5, seed + 11);
    // Ledges across the broad forms' range.
    let steps = 8.0;
    let texels = texture(size, |u, v| {
        let wu = u + 0.05 * warp[0].fbm(u, v);
        let wv = v + 0.03 * warp[1].fbm(u, v);
        let (block, block_id) = blocks.faceted(wu, wv * 2.0, 0.3, 10.0, seed);
        let (facet, _) = facets.faceted(wu, wv * 2.0, 0.15, 14.0, seed + 1);
        let form = forms.unit(wu, wv * 3.0);
        let terrace = form * steps;
        let step = terrace.floor();
        let riser = smoothstep(0.6, 1.0, terrace - step);
        let ledge = (step + riser) / steps;
        let chip = chips.ridged(wu, wv);
        let fine = grain.unit(u, v);
        // A few cracks along the bedding, open only in stretches.
        let along = 1.0 - bedding.fbm(wu, wv * 6.0).abs();
        let crack = smoothstep(0.94, 0.985, along) * smoothstep(0.55, 0.75, openings.unit(u, v));
        let height =
            0.5 * block + 0.2 * facet + 0.18 * ledge + 0.02 * chip + 0.003 * fine - 0.08 * crack;
        // Grays by block and patch, darker on risers and in cracks.
        let patch = tint.fbm(u, v);
        let mut albedo = mix(ROCK_GRAY, ROCK_WARM, saturate(patch * 1.5));
        albedo = mix(albedo, ROCK_COOL, saturate(-patch * 1.5));
        let shade = 0.92 + 0.16 * unit(hash(block_id, 7, seed));
        albedo = scale(
            albedo,
            shade
                * (1.0 - 0.1 * riser)
                * (1.0 - 0.45 * crack)
                * (0.9 + 0.2 * fine)
                * (0.9 + 0.2 * chip),
        );
        // Pale lichen on the ledges, in crusty specks.
        let growth = smoothstep(0.55, 0.66, lichen.unit(u, v))
            * (1.0 - riser)
            * smoothstep(0.35, 0.6, specks.unit(u, v));
        let crust = mix(
            LICHEN_PALE,
            LICHEN_YELLOW,
            smoothstep(0.4, 0.6, tint.unit(v, u)),
        );
        Texel {
            albedo: mix(albedo, crust, 0.7 * growth),
            height,
            roughness: 0.74 + 0.1 * fine + 0.1 * growth,
        }
    });
    Maps::new(size, texels, 2.0)
}

// Snow.

const SNOW: Rgb = [0.860, 0.875, 0.905];
const SNOW_SHADE: Rgb = [0.640, 0.725, 0.870];

/// Wind-packed snow: soft ripples running across the wind, broad swells,
/// faint sparkle and bluish hollows.
fn snow(size: usize) -> Maps {
    let seed = 600;
    let warp = Noise::new(3, 3, 0.5, seed);
    let ripples = Noise::new(3, 3, 0.45, seed + 1);
    let swell = Noise::new(2, 4, 0.5, seed + 2);
    let grain = Noise::new(128, 2, 0.5, seed + 3);
    let texels = texture(size, |u, v| {
        let shift = warp.fbm(u, v) * 0.1;
        // Crests along u: the noise changes three times faster along v.
        let ripple = ripples.ridged(u + shift, (v + shift) * 3.0);
        let fine = grain.unit(u, v);
        let height = 0.35 * ripple * ripple + 0.5 * swell.unit(u, v) + 0.03 * fine;
        let (x, y) = ((u * 8192.0) as u32, (v * 8192.0) as u32);
        let sparkle = unit(hash(x, y, seed)) < 0.02;
        Texel {
            albedo: scale(SNOW, 0.98 + 0.04 * fine),
            height,
            roughness: if sparkle { 0.35 } else { 0.6 + 0.08 * fine },
        }
    });
    let mut maps = Maps::new(size, texels, 0.6);
    // Hollows bluish: wherever the snow lies below its surroundings.
    let around = blur(&maps.height, size, usize::max(1, size / 48));
    for ((albedo, height), around) in maps.albedo.iter_mut().zip(&maps.height).zip(&around) {
        let hollow = saturate((around - height) * 8.0);
        *albedo = mix(*albedo, SNOW_SHADE, 0.75 * hollow);
    }
    maps
}

// Encoding.

/// A material's three maps as RGBA8 texels.
struct Encoded {
    size: usize,
    albedo: Vec<u8>,
    normal: Vec<u8>,
    orm: Vec<u8>,
}

impl Encoded {
    /// Encodes `maps`, its normals turned by `relief` texels of rise per
    /// unit of height at 512 texels.
    fn new(maps: &Maps, relief: f32) -> Self {
        let size = maps.size;
        // Heights spread over 0 to 1 between their extreme percentiles.
        let mut sorted = maps.height.clone();
        sorted.sort_by(f32::total_cmp);
        let tail = sorted.len() / 200;
        let (low, high) = (sorted[tail], sorted[sorted.len() - 1 - tail]);
        let span = max(high - low, 1.0e-6);
        let height: Vec<f32> = maps
            .height
            .iter()
            .map(|value| (value - low) / span)
            .collect();
        // Alpha: clamped, with a power bringing its mean to about a half.
        let mut alpha: Vec<f32> = height.iter().map(|&value| saturate(value)).collect();
        for _ in 0..4 {
            let mean = alpha.iter().sum::<f32>() / alpha.len() as f32;
            let power = clamp(0.5_f32.ln() / clamp(mean, 0.01, 0.99).ln(), 0.25, 4.0);
            for value in &mut alpha {
                *value = value.powf(power);
            }
        }
        let strength = relief * size as f32 / 512.0;
        let at = |x: usize, y: usize| height[(y % size) * size + x % size];
        let mut normal = Vec::with_capacity(size * size * 4);
        for y in 0..size {
            for x in 0..size {
                let du = (at(x + 1, y) - at(x + size - 1, y)) * 0.5;
                let dv = (at(x, y + 1) - at(x, y + size - 1)) * 0.5;
                let n = [-strength * du, strength * dv, 1.0];
                let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                for value in n {
                    normal.push(unorm(value / length * 0.5 + 0.5));
                }
                normal.push(255);
            }
        }
        let mut albedo = Vec::with_capacity(size * size * 4);
        let mut orm = Vec::with_capacity(size * size * 4);
        for (index, color) in maps.albedo.iter().enumerate() {
            albedo.extend(color.map(srgb));
            albedo.push(255);
            orm.extend([
                unorm(maps.occlusion[index]),
                unorm(maps.roughness[index]),
                0,
                unorm(alpha[index]),
            ]);
        }
        Self {
            size,
            albedo,
            normal,
            orm,
        }
    }

    /// The mean linear albedo of the encoded texels and the mean roughness,
    /// to three decimals.
    fn means(&self) -> ([f64; 3], f64) {
        let count = (self.size * self.size) as f64;
        let mut albedo = [0.0; 3];
        for pixel in self.albedo.as_chunks::<4>().0 {
            for (sum, &value) in albedo.iter_mut().zip(pixel) {
                *sum += f64::from(linear(value));
            }
        }
        let roughness = self
            .orm
            .as_chunks::<4>()
            .0
            .iter()
            .map(|pixel| f64::from(pixel[1]) / 255.0)
            .sum::<f64>();
        let rounded = |mean: f64| (mean * 1000.0).round() / 1000.0;
        (
            albedo.map(|sum| rounded(sum / count)),
            rounded(roughness / count),
        )
    }

    /// Texel `index` decoded: linear albedo, the normal and occlusion.
    fn decoded(&self, index: usize) -> [f32; 7] {
        let at = index * 4;
        let (albedo, normal) = (&self.albedo[at..at + 3], &self.normal[at..at + 3]);
        [
            linear(albedo[0]),
            linear(albedo[1]),
            linear(albedo[2]),
            f32::from(normal[0]) / 127.5 - 1.0,
            f32::from(normal[1]) / 127.5 - 1.0,
            f32::from(normal[2]) / 127.5 - 1.0,
            f32::from(self.orm[at]) / 255.0,
        ]
    }

    /// The material lit by a low sun from the upper left with a little
    /// occluded ambient light, exposed for viewing by its mean albedo, sRGB.
    fn lit(&self) -> Vec<u8> {
        let texels: Vec<[f32; 7]> = (0..self.size * self.size)
            .map(|index| self.decoded(index))
            .collect();
        self.light(&texels)
    }

    /// The material from afar: its maps averaged four by four, as mipmaps
    /// would, then lit and repeated four by four.
    fn far(&self) -> Vec<u8> {
        let small = self.size / 4;
        let mut texels = Vec::with_capacity(small * small);
        for y in 0..small {
            for x in 0..small {
                let mut sum = [0.0; 7];
                for dy in 0..4 {
                    for dx in 0..4 {
                        let texel = self.decoded((4 * y + dy) * self.size + 4 * x + dx);
                        for (sum, value) in sum.iter_mut().zip(texel) {
                            *sum += value / 16.0;
                        }
                    }
                }
                texels.push(sum);
            }
        }
        tiled(&self.light(&texels), small, 4)
    }

    /// `texels` (decoded) lit as [`Self::lit`] lights them, sRGB.
    fn light(&self, texels: &[[f32; 7]]) -> Vec<u8> {
        let (mean, _) = self.means();
        let exposure = clamp(1.05 / (mean[0] + mean[1] + mean[2]) as f32, 0.8, 2.5);
        let sun = {
            let sun = [-1.0_f32, 1.0, 0.6];
            let length = (sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]).sqrt();
            sun.map(|value| value / length)
        };
        let mut out = Vec::with_capacity(texels.len() * 4);
        for texel in texels {
            let normal = [texel[3], texel[4], texel[5]];
            let length = max(
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt(),
                1.0e-6,
            );
            let facing = max(
                (normal[0] * sun[0] + normal[1] * sun[1] + normal[2] * sun[2]) / length,
                0.0,
            );
            let light = 0.25 * texel[6] + facing;
            for &channel in &texel[..3] {
                out.push(srgb(channel * light * exposure));
            }
            out.push(255);
        }
        out
    }
}

fn unorm(value: f32) -> u8 {
    (saturate(value) * 255.0 + 0.5) as u8
}

/// Linear to 8-bit sRGB.
fn srgb(value: f32) -> u8 {
    let value = saturate(value);
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    unorm(encoded)
}

/// 8-bit sRGB to linear.
fn linear(value: u8) -> f32 {
    let value = f32::from(value) / 255.0;
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// An uncompressed, single-level RGBA8 KTX2 file of `pixels` (`size` texels
/// along each side), sRGB or linear, with its data format descriptor.
fn ktx2(size: u32, pixels: &[u8], srgb: bool) -> Vec<u8> {
    const DATA: u64 = 196;
    let mut out = Vec::with_capacity(DATA as usize + pixels.len());
    out.extend_from_slice(&[
        0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
    ]);
    // vkFormat (R8G8B8A8_SRGB or _UNORM), type size, width, height, depth,
    // layers, faces, levels, supercompression; then where the data format
    // descriptor and key-value data lie.
    let format = if srgb { 43 } else { 37 };
    for word in [format, 1, size, size, 0, 0, 1, 1, 0, 104, 92, 0, 0_u32] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    // No supercompression global data; the one level's offset and lengths.
    let length = pixels.len() as u64;
    for word in [0, 0, DATA, length, length] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    // The data format descriptor: one basic block of RGBSDA in BT.709 with
    // four 8-bit samples.
    out.extend_from_slice(&92_u32.to_le_bytes());
    out.extend_from_slice(&0_u32.to_le_bytes());
    out.extend_from_slice(&(2_u32 | (88 << 16)).to_le_bytes());
    out.extend_from_slice(&[1, 1, if srgb { 2 } else { 1 }, 0]);
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&[4, 0, 0, 0, 0, 0, 0, 0]);
    for (channel, offset) in [(0_u32, 0_u32), (1, 8), (2, 16), (15, 24)] {
        // sRGB alpha stays linear.
        let channel = if channel == 15 && srgb {
            15 | 0x10
        } else {
            channel
        };
        out.extend_from_slice(&(offset | (7 << 16) | (channel << 24)).to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.extend_from_slice(&255_u32.to_le_bytes());
    }
    debug_assert_eq!(out.len() as u64, DATA);
    out.extend_from_slice(pixels);
    out
}

// Previews.

fn write_previews(
    folder: &Path,
    name: &str,
    encoded: &Encoded,
    lit: &[u8],
    far: &[u8],
) -> Result<(), String> {
    let size = encoded.size;
    // Each texel of `source` turned into an opaque one by `texel`.
    let opaque = |source: &[u8], texel: fn(&[u8; 4]) -> [u8; 4]| -> Vec<u8> {
        source.as_chunks::<4>().0.iter().flat_map(texel).collect()
    };
    let side = size as u32;
    write_png(
        &folder.join(format!("{name}_albedo.png")),
        side * 2,
        side * 2,
        &tiled(&encoded.albedo, size, 2),
    )?;
    write_png(
        &folder.join(format!("{name}_normal.png")),
        side,
        side,
        &encoded.normal,
    )?;
    write_png(
        &folder.join(format!("{name}_orm.png")),
        side,
        side,
        &opaque(&encoded.orm, |orm| [orm[0], orm[1], orm[2], 255]),
    )?;
    write_png(
        &folder.join(format!("{name}_height.png")),
        side,
        side,
        &opaque(&encoded.orm, |orm| [orm[3], orm[3], orm[3], 255]),
    )?;
    write_png(
        &folder.join(format!("{name}_lit.png")),
        side * 2,
        side * 2,
        &tiled(lit, size, 2),
    )?;
    write_png(&folder.join(format!("{name}_far.png")), side, side, far)
}

/// `pixels` (`size` along each side) repeated `times` by `times`.
fn tiled(pixels: &[u8], size: usize, times: usize) -> Vec<u8> {
    let row = size * 4;
    let mut out = Vec::with_capacity(pixels.len() * times * times);
    for _ in 0..times {
        for line in pixels.chunks_exact(row) {
            for _ in 0..times {
                out.extend_from_slice(line);
            }
        }
    }
    out
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    let failed = |error: &dyn std::fmt::Display| format!("{}: {error}", path.display());
    let file = std::fs::File::create(path).map_err(|error| failed(&error))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|error| failed(&error))?;
    writer
        .write_image_data(rgba)
        .map_err(|error| failed(&error))
}
