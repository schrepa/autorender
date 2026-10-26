//! A terrain's macro colors: the tint its ground takes and how lush its
//! plants grow there, over its whole extent, from what its bakes found:
//! where water gathers and runs (the hydrology bake's wetness and flow),
//! the shores of its lakes, where erosion wore ground away and where it
//! laid it down, and the dust beside its roads; with its steep slopes,
//! ridges and hollows from its shape, its slopes turned to the sun's side
//! and from it, and broad patches of noise. Each cause tints the ground in
//! a hue of its own, as strongly as its surface says ([`MacroCauses`]), and
//! plants grow lusher as far as the causes darken the ground and drier as
//! far as they pale it. Ground takes them in full over its materials' own
//! colors wherever it is seen from, so it neither repeats their maps evenly
//! nor lies the same color across a valley, and keeps its color as the eye
//! nears. Nothing in them depends on the light of the moment, only on the
//! ground and the side the midday sun stands on: the sun and the sky light
//! what they tint at any time of day.
//!
//! A map is RGBA8 texels over the terrain's extent, rows along z, texel
//! `(x, z)` centered at `((x + 0.5, z + 0.5) * extent / size)`: red, green
//! and blue the tint the ground's albedo takes, 128 leaving it as it is
//! (each 128ths of it), and alpha how lush plants grow there (0 dry, 255
//! lush). The job
//! [`crate::jobs::MacroJob`] makes one from a terrain's recipe, and engines
//! compress it into what their devices sample.

use std::sync::Arc;

use engine_math::{clamp, max, sin_cos};

use crate::erosion::Erosion;
use crate::layers::Layers;
use crate::noise::{gradient_value, smooth};
use crate::{Generator, MacroCauses, PathSample, Pyramid, TILE_QUADS};

/// The most texels along a macro map's side, and what its sides are whole
/// multiples of: the blocks devices compress.
pub const MAX_MACRO_TEXELS: u32 = 4096;
pub const MACRO_BLOCK: u32 = 4;

/// Meters around the ground its ridges and hollows are measured against,
/// and the meters above or below them that reach the ends of their range.
const RELIEF_REACH: f32 = 160.0;
const RELIEF_RANGE: f32 = 60.0;

/// Seed of the macro colors' noise: its patches' own, apart from every
/// layer's.
const SEED: u32 = 0x6d61_6372;

/// The tangent of the sun's height the aspect's cause takes, 30 degrees,
/// about its mean over a temperate year's daylight. A slope takes the
/// direct sun level ground does times cos(slope) (1 + facing / this),
/// facing being tan(slope) cos(downhill - heading) (Lambert's cosine law):
/// the aspect is whole on slopes turned so far from the sun's side that its
/// light grazes them, and on those turned as far to it, which take nearly
/// twice level ground's.
const SUN_TANGENT: f32 = 0.577_350_3;

/// Rec. 709's weights of linear red, green and blue in luminance, which the
/// causes' hues are units of and their brightness is measured in.
const LUMINANCE: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Each cause's hue: what red, green and blue gain for a unit of its
/// strength, a unit of luminance brighter or, negative, darker. Water
/// darkens every channel alike, as it fills pores (Lagarde 2013).
const WET: [f32; 3] = [-1.0, -1.0, -1.0];
/// Silt, sand and dust: fine mineral grains, pale and warm.
const SANDY: [f32; 3] = [1.15, 1.0, 0.6];
/// Subsoil and rock bared of humus: paler and grayer, losing the topsoil's
/// browns.
const MINERAL: [f32; 3] = [0.95, 1.0, 1.15];
/// Weathered ridges: paler, lichens crusting them gray-green and yellow.
const WEATHERED: [f32; 3] = [1.0, 1.02, 0.85];
/// Hollows: darker and browner, with shade and deep soil.
const SHADED: [f32; 3] = [-0.85, -1.0, -1.4];
/// Slopes the sun dries: paler and warmer, toward straw.
const SUNNY: [f32; 3] = [1.2, 1.0, 0.45];
/// Slopes it leaves damp: darker and greener, toward moss.
const MOSSY: [f32; 3] = [-1.2, -0.93, -1.2];
/// The noise's lighter and darker patches: every channel alike.
const GRAY: [f32; 3] = [1.0, 1.0, 1.0];
/// The noise's warmer patches, redder and less blue, and the reverse in
/// cooler ones: no brighter or darker.
const WARM: [f32; 3] = [1.0, -0.1, -1.95];

/// How lush plants grow where no cause tints the ground: middling.
const LUSH_MIDDLING: f32 = 0.5;
/// The lushness a share of the ground's brightness is worth, which plants
/// gain as far as its causes darken it and lose as far as they pale it:
/// green cover darkens ground in visible light, to some two fifths of bare
/// soil's under full cover (Tucker 1979), so plants grow from bare to lush
/// over three fifths of its brightness.
const LUSH_PER_SHADE: f32 = 1.0 / 0.6;

/// A terrain's shape at one level: its heights over the whole terrain,
/// `columns` by `rows` samples `spacing` meters apart from its first
/// corner, and at each sample the length of its height gradient (the
/// slope's tangent), how far it faces a heading (its fall along it,
/// tan(slope) cos(downhill - heading), negative turned from it) and how far
/// it stands above the ground around it (negative in hollows).
struct Shape {
    columns: usize,
    rows: usize,
    spacing: f32,
    heights: Vec<f32>,
    gradients: Vec<f32>,
    facings: Vec<f32>,
    reliefs: Vec<f32>,
}

impl Shape {
    /// The shape of `layers` over `pyramid` at `level`, facing `heading`
    /// (degrees from -z toward +x): every tile of the level generated as
    /// tiles are, its samples gathered. `None` where a tile's heights are
    /// invalid.
    fn of(layers: &Arc<Layers>, pyramid: &Pyramid, level: u32, heading: f32) -> Option<Self> {
        let [tiles_x, tiles_z] = pyramid.tiles(level);
        let quads = TILE_QUADS as usize;
        let columns = tiles_x as usize * quads + 1;
        let rows = tiles_z as usize * quads + 1;
        let mut heights = vec![0.0_f32; columns * rows];
        for key in pyramid.level_tiles(level) {
            let mut budget = u64::MAX;
            let tile = Generator::new(Arc::clone(layers), pyramid, key).advance(&mut budget)?;
            let first = [usize::from(key.x) * quads, usize::from(key.z) * quads];
            for z in 0..=quads {
                for x in 0..=quads {
                    heights[(first[1] + z) * columns + first[0] + x] =
                        tile.height(x as i32, z as i32);
                }
            }
        }
        let spacing = pyramid.spacing(level);
        let mut shape = Self {
            columns,
            rows,
            spacing,
            heights,
            gradients: Vec::new(),
            facings: Vec::new(),
            reliefs: Vec::new(),
        };
        let reach = max(RELIEF_REACH, 2.0 * spacing);
        let ring: Vec<(f32, f32)> = (0..8)
            .map(|index| sin_cos((index as f32 + 0.5) * (std::f32::consts::TAU / 8.0)))
            .collect();
        // The heading's way along x and z: -z at 0, +x at 90 degrees.
        let (along_x, along_z) = {
            let (sin, cos) = sin_cos(heading.to_radians());
            (sin, -cos)
        };
        let mut gradients = Vec::with_capacity(columns * rows);
        let mut facings = Vec::with_capacity(columns * rows);
        let mut reliefs = Vec::with_capacity(columns * rows);
        for z in 0..rows {
            for x in 0..columns {
                let at = |dx: isize, dz: isize| shape.sample(x as isize + dx, z as isize + dz);
                let gx = (at(1, 0) - at(-1, 0)) / (2.0 * spacing);
                let gz = (at(0, 1) - at(0, -1)) / (2.0 * spacing);
                gradients.push((gx * gx + gz * gz).sqrt());
                facings.push(-(gx * along_x + gz * along_z));
                let (px, pz) = (x as f32 * spacing, z as f32 * spacing);
                let around: f32 = ring
                    .iter()
                    .map(|&(sin, cos)| shape.height(px + cos * reach, pz + sin * reach))
                    .sum();
                reliefs.push(at(0, 0) - around * 0.125);
            }
        }
        shape.gradients = gradients;
        shape.facings = facings;
        shape.reliefs = reliefs;
        Some(shape)
    }

    /// Sample `(x, z)`'s height, held within the grid.
    fn sample(&self, x: isize, z: isize) -> f32 {
        let x = x.clamp(0, self.columns as isize - 1) as usize;
        let z = z.clamp(0, self.rows as isize - 1) as usize;
        self.heights[z * self.columns + x]
    }

    /// `values` at terrain-local `(x, z)`, interpolated between the samples
    /// around it, held within the grid.
    fn between(&self, values: &[f32], x: f32, z: f32) -> f32 {
        let fx = clamp(x / self.spacing, 0.0, (self.columns - 1) as f32);
        let fz = clamp(z / self.spacing, 0.0, (self.rows - 1) as f32);
        let x0 = (fx as usize).min(self.columns - 2);
        let z0 = (fz as usize).min(self.rows - 2);
        let (tx, tz) = (fx - x0 as f32, fz - z0 as f32);
        let at = z0 * self.columns + x0;
        let next = at + self.columns;
        let near = values[at] + (values[at + 1] - values[at]) * tx;
        let far = values[next] + (values[next + 1] - values[next]) * tx;
        near + (far - near) * tz
    }

    fn height(&self, x: f32, z: f32) -> f32 {
        self.between(&self.heights, x, z)
    }
}

/// What the ground holds at a texel, each from 0 to 1 but `relief` (-1 in
/// hollows to 1 on ridges) and `facing`: how wet it lies, on a lake's
/// shore, laid down by erosion and worn away, how steep, how dusty from a
/// road, how far it faces the sun's side (tan(slope) cos(downhill -
/// heading), negative turned from it), and its noise's broad patches, finer
/// patches and warmth.
struct Ground {
    moisture: f32,
    shore: f32,
    deposit: f32,
    worn: f32,
    steep: f32,
    dust: f32,
    relief: f32,
    facing: f32,
    broad: f32,
    patchy: f32,
    warmth: f32,
}

/// 0 at `low`, 1 at `high`, smooth between.
fn ramp(value: f32, low: f32, high: f32) -> f32 {
    smooth((value - low) / (high - low))
}

/// Gradient noise of features `wavelength` meters across at `(x, z)`, 0 to
/// 1, its own lattice for `index`.
fn patches(x: f32, z: f32, wavelength: f32, index: u32) -> f32 {
    let shift = index as f32 * 17.13;
    let value = gradient_value(
        x / wavelength + shift,
        z / wavelength - shift,
        SEED.wrapping_add(index.wrapping_mul(0x9e37_79b9)),
    );
    clamp(0.5 + 0.5 * value, 0.0, 1.0)
}

/// What the ground holds at terrain-local `(x, z)`, a texel `footprint`
/// meters across: its shape from `shape`, its water and lakes from the
/// hydrology bake, `change`, the meters its erosions changed it by, and its
/// roads, each of the bakes' and roads' parts as much as the stack shows
/// them there (the surroundings fade them out past their terrain). The
/// stack's parts and the noise are read where the point lies in the ground
/// the stack was drawn for, as tiles read it, so surroundings carry on
/// their terrain's colors.
fn ground(
    layers: &Layers,
    (shape, change): (&Shape, f32),
    (x, z): (f32, f32),
    footprint: f32,
) -> Ground {
    let (sx, sz) = (x + layers.offset.x, z + layers.offset.y);
    let shown = layers.shown(sx, sz);
    let height = shape.height(x, z);
    let (mut moisture, mut shore) = (0.0, 0.0);
    if shown > 0.0
        && let Some(hydrology) = layers.hydrology()
    {
        let water = hydrology.masks_at(sx, sz, footprint, crate::hydrology::RIVER_FAR);
        // Wet hollows and the ground along streams.
        moisture = max(
            ramp(water.wetness, 7.0, 14.0),
            0.8 * ramp(water.flow, 4.0, 6.0),
        ) * shown;
        // A band of shore the lake's level rises within.
        if let Some(level) = hydrology.water.lake_beside(sx, sz) {
            shore = (1.0 - ramp(height - level, 0.0, 2.5)) * shown;
        }
    }
    let change = change * shown;
    let road = if shown > 0.0 {
        layers.path_at(sx, sz)
    } else {
        PathSample::FAR
    };
    let gradient = shape.between(&shape.gradients, x, z);
    Ground {
        moisture,
        shore,
        deposit: ramp(change, 0.0, 2.0),
        worn: ramp(-change, 0.0, 4.0),
        // From about 30 degrees to 50.
        steep: ramp(gradient, 0.58, 1.19),
        dust: (1.0 - ramp(road.edge(), 0.0, 6.0)) * shown,
        relief: clamp(
            shape.between(&shape.reliefs, x, z) / RELIEF_RANGE,
            -1.0,
            1.0,
        ),
        facing: shape.between(&shape.facings, x, z),
        broad: 0.65 * patches(sx, sz, 340.0, 0) + 0.35 * patches(sx, sz, 130.0, 1),
        patchy: patches(sx, sz, 45.0, 2),
        warmth: patches(sx, sz, 600.0, 3),
    }
}

/// `tint` with each channel scaled by 1 and `amount` times `hue`'s, never
/// below 0.
fn shaded(tint: [f32; 3], hue: [f32; 3], amount: f32) -> [f32; 3] {
    [0, 1, 2].map(|channel| tint[channel] * max(1.0 + amount * hue[channel], 0.0))
}

/// A texel of the map: the tint `ground` gives the ground's albedo and how
/// lush its plants grow, each cause as strong as `causes` say.
fn texel(ground: &Ground, causes: &MacroCauses) -> [u8; 4] {
    let aspect = clamp(ground.facing / SUN_TANGENT, -1.0, 1.0);
    let lighter = |noise: f32| 2.0 * noise - 1.0;
    // Each cause's hue and how far it moves the ground's brightness: shores
    // lie wet and sandy.
    let shades = [
        (WET, causes.wet * max(ground.moisture, ground.shore)),
        (
            SANDY,
            causes.deposit * max(ground.deposit, 0.6 * ground.shore),
        ),
        (MINERAL, causes.worn * ground.worn),
        (MINERAL, causes.steep * ground.steep),
        (WEATHERED, causes.relief * max(ground.relief, 0.0)),
        (SHADED, causes.relief * max(-ground.relief, 0.0)),
        (SANDY, causes.dust * ground.dust),
        (SUNNY, causes.aspect * max(aspect, 0.0)),
        (MOSSY, causes.aspect * max(-aspect, 0.0)),
        (GRAY, causes.broad * lighter(ground.broad)),
        (GRAY, causes.patchy * lighter(ground.patchy)),
    ];
    let tint = shades.iter().fold([1.0_f32; 3], |tint, &(hue, amount)| {
        shaded(tint, hue, amount)
    });
    // Plants grow lusher as far as the causes darken the ground and drier
    // as far as they pale it; warmth turns its hue alone.
    let brightness: f32 = LUMINANCE
        .iter()
        .zip(tint)
        .map(|(weight, channel)| weight * channel)
        .sum();
    let lush = clamp(
        LUSH_MIDDLING + LUSH_PER_SHADE * (1.0 - brightness),
        0.0,
        1.0,
    );
    let tint = shaded(tint, WARM, causes.warmth * lighter(ground.warmth));
    let byte = |value: f32| (clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8;
    let scaled = |channel: f32| byte(channel * (128.0 / 255.0));
    [
        scaled(tint[0]),
        scaled(tint[1]),
        scaled(tint[2]),
        byte(lush),
    ]
}

/// The macro colors of `layers` (a terrain's, with its bakes, as its recipe
/// holds them) over `pyramid`'s extent, `size` texels a side, its shape read
/// from the heights of `level`'s tiles, each cause as strong as `causes`
/// (its surface's) say: RGBA8 texels, rows along z. `None` where a tile's
/// heights are invalid.
pub fn macro_colors(
    layers: &Arc<Layers>,
    pyramid: &Pyramid,
    level: u32,
    size: u32,
    causes: &MacroCauses,
) -> Option<Vec<u8>> {
    let shape = Shape::of(layers, pyramid, level, causes.heading)?;
    let extent = pyramid.extent();
    let step = [extent.x / size as f32, extent.y / size as f32];
    let footprint = max(step[0], step[1]);
    let erosions: Vec<&Erosion> = crate::surface::erosions(layers).collect();
    let count = size as usize;
    let mut texels = Vec::with_capacity(count * count * 4);
    let mut changes = vec![0.0_f32; count];
    for z in 0..count {
        let pz = (z as f32 + 0.5) * step[1];
        // What the erosions changed along the row, read a row at a time
        // where the stack was drawn for, as tiles read it.
        changes.fill(0.0);
        let xs = (0..count).map(|x| (x as f32 + 0.5) * step[0] + layers.offset.x);
        for erosion in &erosions {
            erosion.change_row(xs.clone(), pz + layers.offset.y, footprint, |x, change| {
                changes[x] += change;
            });
        }
        for (x, &change) in changes.iter().enumerate() {
            let px = (x as f32 + 0.5) * step[0];
            let ground = ground(layers, (&shape, change), (px, pz), footprint);
            texels.extend(texel(&ground, causes));
        }
    }
    Some(texels)
}
