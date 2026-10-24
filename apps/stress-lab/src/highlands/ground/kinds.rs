//! The ground materials themselves: grass of tapering blades in tufts
//! (fresh, or dry over a mat of dead blades with heather), talus of
//! stones over gravel, bare soil, a weathered rock face, wind-packed
//! snow, and streams' rippled sand, gravel and cobbles, each at `size`
//! texels along each side.

use std::f32::consts::TAU;

use engine::prelude::math::{max, min, sin_cos};

use super::noise::{
    Cells, Noise, Rng, blur, hash, lerp, mix, saturate, scale, smoothstep, texture, unit,
};
use super::{Maps, Rgb, Texel};

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
pub(super) fn meadow(size: usize, dry: bool) -> Maps {
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
/// rounded or angular by its own measure within a range, standing from
/// `base` up to `rise` higher, tilted and folded along a crest of its own.
struct Stones {
    cells: Cells,
    share: f32,
    gap: [f32; 2],
    /// How rounded its stones are, from the first to the second: 0
    /// angular, 1 rounded.
    round: [f32; 2],
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
        let round = lerp(self.round[0], self.round[1], random(16));
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

    /// A water-worn stone `id`'s color: its gray ([`Self::color`]) tinted as
    /// one of the rocks streams carry ([`RIVER_ROCKS`]).
    fn worn(&self, id: u32, dark: f32, light: f32) -> Rgb {
        let pick = (unit(hash(id, 17, self.seed)) * RIVER_ROCKS.len() as f32) as usize;
        let rock = RIVER_ROCKS[pick.min(RIVER_ROCKS.len() - 1)];
        let gray = self.color(id, dark, light);
        [gray[0] * rock[0], gray[1] * rock[1], gray[2] * rock[2]]
    }
}

const GRIT: Rgb = [0.058, 0.054, 0.050];
const SCREE_DIRT: Rgb = [0.085, 0.074, 0.062];

/// Talus: loose angular and rounded stones of strongly varied sizes, many
/// small and few large, the larger lying on the smaller, with gravel and
/// dirt between them.
pub(super) fn scree(size: usize) -> Maps {
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
        round: [0.0, 1.0],
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
pub(super) fn dirt(size: usize) -> Maps {
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
        round: [0.0, 1.0],
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
pub(super) fn rock(size: usize) -> Maps {
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
pub(super) fn snow(size: usize) -> Maps {
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

// Stream beds.

/// The rocks streams carry down, as tints of a stone's gray: granite and
/// gneiss most, then sandstone, dark basalt and greywacke, pale quartzite
/// and green schist.
const RIVER_ROCKS: [Rgb; 8] = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.04, 1.0, 0.95],
    [1.14, 1.02, 0.84],
    [0.62, 0.63, 0.66],
    [0.7, 0.7, 0.72],
    [1.38, 1.36, 1.3],
    [0.88, 0.98, 0.9],
];

const SAND: Rgb = [0.25, 0.226, 0.188];
const SAND_PALE: Rgb = [0.36, 0.34, 0.3];
const SAND_DARK: Rgb = [0.11, 0.104, 0.096];

/// Ripples across the sand's texture: at two meters a repeat, crests 13
/// centimeters apart, about a thousand of its fine grains, as current
/// ripples stand (Yalin 1964).
const RIPPLES: f32 = 15.0;

/// Streams' sand at two meters a repeat: fine pale grains with dark and
/// paler ones among them, rippled by the water into sinuous crests
/// ([`RIPPLES`]), each a gentle rise to a sharp crest and a steep lee,
/// fading out in patches, the dark grains gathered in the troughs; a few
/// pebbles lie on it.
pub(super) fn sand(size: usize) -> Maps {
    let seed = 900;
    let warp = Noise::new(3, 3, 0.5, seed);
    let fade = Noise::new(6, 2, 0.5, seed + 1);
    let swell = Noise::new(8, 2, 0.5, seed + 2);
    let tone = Noise::new(20, 2, 0.5, seed + 3);
    let pebbles = Stones {
        cells: Cells::new(40, 0.9, seed + 4),
        share: 0.05,
        gap: [0.15, 0.35],
        round: [0.5, 1.0],
        base: 0.55,
        rise: 0.5,
        tilt: 0.4,
        crest: 0.3,
        outline: Noise::new(120, 2, 0.5, seed + 54),
        seed: seed + 4,
    };
    let texels = texture(size, |u, v| {
        // Crests along u, bent where the warp shifts them.
        let phase = (v + 0.05 * warp.fbm(u, v)) * RIPPLES;
        let along = phase - phase.floor();
        // A gentle rise over seven tenths of each ripple, then its lee:
        // sharp at the crest, rounded in the trough.
        let rise = if along < 0.7 {
            along / 0.7
        } else {
            (1.0 - along) / 0.3
        };
        let strength = smoothstep(0.3, 0.65, fade.unit(u, v));
        let (x, y) = ((u * size as f32) as u32, (v * size as f32) as u32);
        let grain = unit(hash(x, y, seed));
        let speck = if grain < 0.05 {
            0.55
        } else if grain > 0.93 {
            1.3
        } else {
            0.94 + 0.12 * unit(hash(x, y, seed + 1))
        };
        let base = mix(SAND, SAND_PALE, 0.4 * tone.unit(u, v));
        let mut texel = Texel {
            albedo: scale(mix(base, SAND_DARK, 0.4 * (1.0 - rise) * strength), speck),
            height: 0.5 * rise * rise * strength + 0.2 * swell.unit(u, v) + 0.02 * grain,
            // Flakes of mica glint among the grains.
            roughness: if grain > 0.985 { 0.4 } else { 0.84 },
        };
        if let Some(stone) = pebbles
            .sample(u, v)
            .filter(|stone| stone.height > texel.height)
        {
            texel = Texel {
                albedo: scale(pebbles.worn(stone.id, 0.1, 0.22), 0.8 + 0.2 * stone.profile),
                height: stone.height,
                roughness: 0.68,
            };
        }
        texel
    });
    Maps::new(size, texels, 0.8)
}

const GRAVEL_GRIT: Rgb = [0.07, 0.065, 0.058];
const GRAVEL_SAND: Rgb = [0.13, 0.118, 0.1];

/// Streams' gravel at a meter and a half a repeat: dense water-worn stones
/// from 2 to 6 centimeters, subrounded to rounded and of the rocks streams
/// carry, the larger lying on the smaller, coarse sand between them.
pub(super) fn gravel(size: usize) -> Maps {
    let seed = 700;
    let grit = Noise::new(200, 2, 0.5, seed);
    let mottle = Noise::new(96, 3, 0.5, seed + 1);
    let matrix = Noise::new(8, 3, 0.5, seed + 2);
    // Cells across the texture, the share holding stones, their gaps, feet
    // and rise: stones from about 2 cm up to 6 cm.
    let layer = |count: usize, share: f32, gap: [f32; 2], base: f32, rise: f32, salt: u32| Stones {
        cells: Cells::new(count, 0.9, seed + salt),
        share,
        gap,
        round: [0.45, 1.0],
        base,
        rise,
        tilt: 0.4,
        crest: 0.3,
        outline: Noise::new(count * 3, 3, 0.5, seed + salt + 50),
        seed: seed + salt,
    };
    let layers = [
        layer(56, 0.95, [0.06, 0.22], 0.04, 0.14, 3),
        layer(40, 0.85, [0.06, 0.24], 0.09, 0.2, 4),
        layer(28, 0.7, [0.05, 0.26], 0.15, 0.26, 5),
        layer(19, 0.5, [0.05, 0.28], 0.22, 0.32, 6),
    ];
    let texels = texture(size, |u, v| {
        let fine = grit.unit(u, v);
        let mut texel = Texel {
            albedo: scale(
                mix(GRAVEL_GRIT, GRAVEL_SAND, matrix.unit(u, v)),
                0.75 + 0.5 * fine,
            ),
            height: 0.04 * fine,
            roughness: 0.9,
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
                    layer.worn(stone.id, 0.1, 0.22),
                    (0.86 + 0.28 * speckle) * (0.78 + 0.22 * stone.profile),
                ),
                height: stone.height + 0.01 * speckle,
                roughness: 0.6 + 0.14 * speckle,
            };
        }
        texel
    });
    Maps::new(size, texels, 3.0)
}

const COBBLE_GAP: Rgb = [0.016, 0.015, 0.013];
const COBBLE_SILT: Rgb = [0.05, 0.045, 0.038];
/// The brown-olive film of algae and silt on stones lying in running
/// water.
const BIOFILM: Rgb = [0.07, 0.066, 0.032];

/// Streams' cobbles at four meters a repeat: rounded water-worn stones
/// from 6 to 25 centimeters packed against each other, the larger lying on
/// the smaller, with dark gaps between them holding silt and pebbles; the
/// stones of the rocks streams carry, some filmed brown-olive on top.
pub(super) fn cobble(size: usize) -> Maps {
    let seed = 800;
    let silt = Noise::new(10, 3, 0.5, seed);
    let grit = Noise::new(160, 2, 0.5, seed + 1);
    let mottle = Noise::new(48, 3, 0.5, seed + 2);
    let film = Noise::new(12, 3, 0.5, seed + 3);
    // Cells across the texture, the share holding stones, their gaps, feet
    // and rise: pebbles of about 6 cm down in the gaps, cobbles of about
    // 10, 16 and 25 cm over them.
    let layer = |count: usize, share: f32, gap: [f32; 2], base: f32, rise: f32, salt: u32| Stones {
        cells: Cells::new(count, 0.85, seed + salt),
        share,
        gap,
        round: [0.75, 1.0],
        base,
        rise,
        tilt: 0.25,
        crest: 0.0,
        outline: Noise::new(count * 3, 3, 0.5, seed + salt + 50),
        seed: seed + salt,
    };
    let layers = [
        layer(52, 0.45, [0.1, 0.25], 0.02, 0.1, 4),
        layer(30, 0.8, [0.06, 0.16], 0.08, 0.28, 5),
        layer(20, 0.8, [0.05, 0.14], 0.14, 0.38, 6),
        layer(13, 0.85, [0.04, 0.12], 0.2, 0.5, 7),
    ];
    let texels = texture(size, |u, v| {
        let fine = grit.unit(u, v);
        let mut texel = Texel {
            albedo: scale(
                mix(
                    COBBLE_GAP,
                    COBBLE_SILT,
                    smoothstep(0.4, 0.7, silt.unit(u, v)),
                ),
                0.8 + 0.4 * fine,
            ),
            height: 0.02 * fine,
            roughness: 0.95,
        };
        for layer in &layers {
            let Some(stone) = layer.sample(u, v) else {
                continue;
            };
            if stone.height <= texel.height {
                continue;
            }
            let speckle = mottle.unit(u, v);
            // The film settles on the stone's top, clear of its rim.
            let filmed = 0.45 * smoothstep(0.5, 0.75, film.unit(u, v)) * stone.profile;
            let worn = scale(
                layer.worn(stone.id, 0.11, 0.25),
                (0.88 + 0.24 * speckle) * (0.62 + 0.38 * stone.profile),
            );
            texel = Texel {
                albedo: mix(worn, BIOFILM, filmed),
                height: stone.height + 0.008 * speckle,
                roughness: 0.56 + 0.14 * speckle - 0.08 * filmed,
            };
        }
        texel
    });
    Maps::new(size, texels, 3.5)
}
