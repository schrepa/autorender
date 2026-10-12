//! Conifers for the highlands: a tapering trunk and whorls of drooping
//! branches, each two crossed cards of needles, over one atlas whose left
//! half is bark and right half a needle spray, drawn with one masked
//! material. Every branch's cards share a normal leaning out from the
//! trunk, so the crown lights as one volume; cards carry both faces with
//! that normal rather than flipping it, and vertex colors darken the
//! crown's inside and its lower whorls. Coarser versions of each tree keep
//! its height and crown with fewer, wider cards, for the distance.

use engine::prelude::*;

use crate::kit::random::{golden_hash, unit};

/// The atlas's side in texels.
pub(crate) const ATLAS: u32 = 512;

/// Bark texture coordinates span this much of the atlas's width: its left
/// half less a margin, so filtering never reaches the needles.
const BARK: [f32; 2] = [0.01, 0.49];
/// The needle spray's horizontal center and half width in the atlas, and
/// its vertical span (base at the bottom).
const SPRAY: [f32; 2] = [0.75, 0.235];
const SPRAY_SPAN: [f32; 2] = [0.99, 0.01];

/// How finely a conifer is built: its trunk's sides and segments (each
/// mapping the bark once), the meters between its whorls, its fewest
/// branches a whorl, how much wider their cards grow, their segments, and
/// whether each has a hanging card under its flat one. Coarser versions'
/// cards grow wide enough to keep most of the full tree's needle area,
/// branches times width over whorl spacing, which overlapping less they
/// cover about as well: thinner crowns showed between the full trees near
/// the eye and their impostors far from it.
struct Detail {
    sides: u32,
    segments: u32,
    whorls: f32,
    branches: u32,
    widen: f32,
    bends: usize,
    hanging: bool,
}

/// The full tree, then each coarser version.
const DETAILS: [Detail; 4] = [
    Detail {
        sides: 7,
        segments: 8,
        whorls: 0.42,
        branches: 6,
        widen: 1.0,
        bends: 2,
        hanging: true,
    },
    Detail {
        sides: 5,
        segments: 4,
        whorls: 0.84,
        branches: 5,
        widen: 2.0,
        bends: 2,
        hanging: true,
    },
    Detail {
        sides: 4,
        segments: 2,
        whorls: 1.6,
        branches: 5,
        widen: 3.8,
        bends: 1,
        hanging: true,
    },
    Detail {
        sides: 3,
        segments: 1,
        whorls: 2.8,
        branches: 4,
        widen: 7.9,
        bends: 1,
        hanging: true,
    },
];

/// The errors, in meters, the coarser versions show: chosen so a tree
/// thins as its whorls blur together rather than by geometric deviation,
/// which cards make meaningless.
pub(crate) const CONIFER_ERRORS: [f32; 3] = [0.08, 0.2, 0.45];

/// Conifer `seed` built at `detail` (0 the full tree, up to 3): 13 to 17
/// meters tall before placement scales it, its base half a meter into the
/// ground.
pub(crate) fn conifer(seed: u32, detail: usize) -> Result<Mesh, String> {
    let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x9e37_79b9) ^ index));
    let build = &DETAILS[detail];
    let height = 13.0 + 4.0 * random(1);
    let base_radius = 0.2 + 0.06 * random(2);
    let mut parts = Parts::default();
    // The trunk, narrowing to a point at the top.
    let radius = |y: f32| base_radius * math::max(1.0 - y / height, 0.0).powf(0.85) + 0.012;
    let (sides, segments) = (build.sides, build.segments);
    for segment in 0..segments {
        let [low, high] = [segment, segment + 1]
            .map(|ring| -0.5 + (height + 0.5) * ring as f32 / segments as f32);
        let first = parts.vertices.len() as u32;
        for side in 0..=sides {
            let angle = std::f32::consts::TAU * side as f32 / sides as f32;
            let (sin, cos) = math::sin_cos(angle);
            let u = BARK[0] + (BARK[1] - BARK[0]) * side as f32 / sides as f32;
            for (y, v) in [(low, 0.99), (high, 0.01)] {
                let shade = 0.55 + 0.45 * math::clamp(y / (0.35 * height), 0.0, 1.0);
                parts.push(
                    Vec3::new(cos * radius(y), y, sin * radius(y)),
                    Vec3::new(cos, 0.15, sin),
                    [u, v],
                    [shade, shade, shade],
                );
            }
        }
        for side in 0..sides {
            let [a, b, c, d] = [0, 1, 2, 3].map(|corner| first + side * 2 + corner);
            parts.indices.extend([a, b, c, b, d, c]);
        }
    }
    // Whorls of branches up the stem, shorter toward the top.
    let crown_base = 1.6 + 0.8 * random(3);
    let mut y = crown_base;
    let mut whorl = 0_u32;
    while y < height - 0.7 {
        let below_top = height - y;
        let count = build.branches + golden_hash(seed ^ whorl.wrapping_mul(0x632b_e5ab)) % 3;
        let twist = random(100 + whorl) * std::f32::consts::TAU;
        for branch in 0..count {
            let index = 1000 + whorl * 16 + branch;
            let azimuth = twist
                + std::f32::consts::TAU * (branch as f32 + 0.35 * (random(index) - 0.5))
                    / count as f32;
            let length = (0.45 + below_top * 0.19) * (0.85 + 0.3 * random(index + 7));
            // Upper branches reach up a little; lower ones droop.
            let droop = -0.25 + 0.6 * math::clamp(below_top / height, 0.0, 1.0);
            let shade = 0.6 + 0.4 * math::clamp((y - crown_base) / (0.6 * height), 0.0, 1.0);
            parts.branch(
                Vec3::new(0.0, y, 0.0),
                (azimuth, droop),
                length,
                build,
                [shade * (0.94 + 0.12 * random(index + 9)), shade, shade],
            );
        }
        y += build.whorls * (0.85 + 0.3 * random(200 + whorl));
        whorl += 1;
    }
    // The leader: two crossed upright cards topping the stem.
    for turn in [0.0, std::f32::consts::FRAC_PI_2] {
        let (sin, cos) = math::sin_cos(turn);
        parts.card(
            &[
                Vec3::new(0.0, height - 1.0, 0.0),
                Vec3::new(0.0, height + 0.15, 0.0),
            ],
            Vec3::new(cos, 0.0, sin) * 0.25,
            Vec3::Y,
            [1.0; 3],
        );
    }
    let Parts {
        vertices,
        colors,
        indices,
    } = parts;
    Mesh::new(vertices, indices)
        .and_then(|mesh| mesh.with_colors(colors))
        .map_err(|error| format!("conifer {seed} at detail {detail}: {error}"))
}

/// A mesh being built: its vertices, their colors and its triangles.
#[derive(Default)]
struct Parts {
    vertices: Vec<MeshVertex>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl Parts {
    fn push(&mut self, position: Vec3, normal: Vec3, uv: [f32; 2], color: [f32; 3]) {
        self.vertices.push(MeshVertex {
            position: position.to_array(),
            normal: normal.normalize_or(Vec3::Y).to_array(),
            uv,
        });
        let [red, green, blue] = color.map(|channel| math::clamp(channel, 0.0, 1.0));
        self.colors.push([red, green, blue, 1.0]);
    }

    /// A branch from `start` turned `azimuth` about the stem and dipping
    /// `droop` radians, `length` meters long and bending further down
    /// toward its tip, built at `detail`: a flat card and a hanging one,
    /// lit as `normal` leaning out from the stem.
    fn branch(
        &mut self,
        start: Vec3,
        (azimuth, droop): (f32, f32),
        length: f32,
        detail: &Detail,
        color: [f32; 3],
    ) {
        let (sin, cos) = math::sin_cos(azimuth);
        let outward = Vec3::new(cos, 0.0, sin);
        let (dip_sin, dip_cos) = math::sin_cos(droop);
        let direction = outward * dip_cos - Vec3::Y * dip_sin;
        let bends = detail.bends;
        let spine: Vec<Vec3> = (0..=bends)
            .map(|step| {
                let t = step as f32 / bends as f32;
                start + direction * (length * t) - Vec3::Y * (0.18 * length * t * t)
            })
            .collect();
        let across = Vec3::new(-sin, 0.0, cos);
        let normal = outward + Vec3::Y * 0.55;
        self.card(
            &spine,
            across * (0.45 * detail.widen * length),
            normal,
            color,
        );
        if detail.hanging {
            let hanging = direction.cross(across).normalize_or(Vec3::Y);
            self.card(
                &spine,
                hanging * (0.3 * detail.widen * length),
                normal,
                color,
            );
        }
    }

    /// A card along `spine` (base first), reaching `half` to each side, both
    /// faces lit as `normal`; its base, nearest the stem, is darkest.
    fn card(&mut self, spine: &[Vec3], half: Vec3, normal: Vec3, color: [f32; 3]) {
        let first = self.vertices.len() as u32;
        let last = spine.len() - 1;
        for (step, &center) in spine.iter().enumerate() {
            let t = step as f32 / last as f32;
            let v = SPRAY_SPAN[0] + (SPRAY_SPAN[1] - SPRAY_SPAN[0]) * t;
            let inner = 0.62 + 0.38 * t;
            let shade = color.map(|channel| channel * inner);
            for side in [-1.0_f32, 1.0] {
                self.push(
                    center + half * side,
                    normal,
                    [SPRAY[0] + SPRAY[1] * side, v],
                    shade,
                );
            }
        }
        for step in 0..last as u32 {
            let [a, b, c, d] = [0, 1, 2, 3].map(|corner| first + step * 2 + corner);
            // Front and back faces, both lit by the shared normal.
            self.indices.extend([a, b, c, b, d, c, a, c, b, b, c, d]);
        }
    }
}

/// The atlas's RGBA8 texels, sRGB color with alpha coverage: bark furrowed
/// along its height on the left half, a needle spray on the right, its
/// main twig rising from the bottom edge with side shoots, every twig
/// fringed with needles. Uncovered texels keep the needles' mean color, so
/// filtering never darkens their edges.
pub(crate) fn atlas() -> Vec<u8> {
    let size = ATLAS as usize;
    let half = size / 2;
    let needle = [0.06, 0.13, 0.06];
    let mut color = vec![needle; size * size];
    let mut alpha = vec![0.0_f32; size * size];
    // Bark: furrows along the height, broken by knots.
    for y in 0..size {
        for x in 0..half {
            let point = Vec2::new(x as f32 / 9.0, y as f32 / 70.0);
            let furrow = 1.0 - (noise(point) * 2.0 - 1.0).abs();
            let grain = noise(Vec2::new(x as f32 / 3.0, y as f32 / 14.0));
            let tone = 0.55 + 0.35 * furrow * furrow + 0.15 * grain;
            color[y * size + x] = [0.15 * tone, 0.115 * tone, 0.09 * tone];
            alpha[y * size + x] = 1.0;
        }
    }
    let mut spray = Spray {
        color: &mut color,
        alpha: &mut alpha,
        size,
    };
    let center = (half + half / 2) as f32;
    let (base, tip) = (size as f32 - 6.0, 10.0);
    let twig = [0.24, 0.17, 0.11];
    // Side shoots alternate up the main twig, longest at the base.
    let mut shoots = Vec::new();
    let mut along = 0.06;
    let mut side = 1.0;
    while along < 0.9 {
        let start = Vec2::new(center, base + (tip - base) * along);
        let reach = 0.42
            * (half as f32)
            * (1.0 - along)
            * (0.8 + 0.4 * unit(golden_hash(shoots.len() as u32 + 5)));
        let end = start + Vec2::new(side * 0.72, -0.7).normalize() * reach;
        shoots.push((start, end));
        along += 0.058;
        side = -side;
    }
    let mut needles = 0_u32;
    let mut fringe = |spray: &mut Spray<'_>, from: Vec2, to: Vec2, length: f32| {
        let span = to - from;
        let steps = (span.length() / 1.7) as u32;
        let direction = span.normalize_or(Vec2::Y);
        for step in 0..steps {
            let t = step as f32 / steps.max(1) as f32;
            let root = from + span * t;
            for side in [-1.0_f32, 1.0] {
                needles += 1;
                let jitter = unit(golden_hash(needles));
                let angle = side * (0.85 + 0.35 * jitter);
                let (sin, cos) = math::sin_cos(angle);
                let pointing = Vec2::new(
                    direction.x * cos - direction.y * sin,
                    direction.x * sin + direction.y * cos,
                );
                let reach =
                    length * (1.0 - 0.45 * t) * (0.8 + 0.4 * unit(golden_hash(needles ^ 0x5bd1)));
                let shade = 0.8 + 0.35 * unit(golden_hash(needles ^ 0x27d4));
                let tint = [0.055 * shade, (0.12 + 0.04 * t) * shade, 0.055 * shade];
                spray.capsule(root, root + pointing * reach, 0.95, tint);
            }
        }
    };
    for &(start, end) in &shoots {
        fringe(&mut spray, start, end, 21.0);
    }
    fringe(
        &mut spray,
        Vec2::new(center, base),
        Vec2::new(center, tip),
        24.0,
    );
    for &(start, end) in &shoots {
        spray.capsule(start, end, 1.1, twig);
    }
    spray.capsule(Vec2::new(center, base), Vec2::new(center, tip), 2.2, twig);
    let mut texels = Vec::with_capacity(size * size * 4);
    for (rgb, coverage) in color.iter().zip(&alpha) {
        texels.extend(rgb.map(srgb));
        texels.push((coverage * 255.0).round() as u8);
    }
    texels
}

/// The needle spray being drawn: linear colors and coverage of the atlas.
struct Spray<'a> {
    color: &'a mut [[f32; 3]],
    alpha: &'a mut [f32],
    size: usize,
}

impl Spray<'_> {
    /// A stroke from `a` to `b`, `radius` texels thick with antialiased
    /// edges, of linear color `tint`, laid over what is drawn.
    fn capsule(&mut self, a: Vec2, b: Vec2, radius: f32, tint: [f32; 3]) {
        let min = a.min(b) - Vec2::splat(radius + 1.0);
        let max = a.max(b) + Vec2::splat(radius + 1.0);
        let half = (self.size / 2) as f32;
        let x0 = math::max(min.x.floor(), half) as usize;
        let x1 = math::min(max.x.ceil(), self.size as f32 - 1.0) as usize;
        let y0 = math::max(min.y.floor(), 0.0) as usize;
        let y1 = math::min(max.y.ceil(), self.size as f32 - 1.0) as usize;
        let span = b - a;
        let length2 = math::max(span.length_squared(), 1.0e-6);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let point = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let t = math::clamp((point - a).dot(span) / length2, 0.0, 1.0);
                let distance = (point - (a + span * t)).length();
                let cover = math::clamp(radius + 0.5 - distance, 0.0, 1.0);
                if cover <= 0.0 {
                    continue;
                }
                let at = y * self.size + x;
                let old = self.alpha[at];
                let alpha = cover + old * (1.0 - cover);
                let laid = |channel: usize| {
                    (tint[channel] * cover + self.color[at][channel] * old * (1.0 - cover)) / alpha
                };
                self.color[at] = [laid(0), laid(1), laid(2)];
                self.alpha[at] = alpha;
            }
        }
    }
}

/// Linear `value` as an sRGB byte.
fn srgb(value: f32) -> u8 {
    let value = math::clamp(value, 0.0, 1.0);
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

/// Value noise from 0 to 1 at `point`, smooth between lattice points.
fn noise(point: Vec2) -> f32 {
    let cell = point.floor();
    let f = point - cell;
    let u = f * f * (Vec2::splat(3.0) - 2.0 * f);
    let corner = |dx: i32, dy: i32| {
        let key =
            (cell.x as i32 + dx) as u32 ^ ((cell.y as i32 + dy) as u32).wrapping_mul(0x8da6_b343);
        unit(golden_hash(key))
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    lerp(
        lerp(corner(0, 0), corner(1, 0), u.x),
        lerp(corner(0, 1), corner(1, 1), u.x),
        u.y,
    )
}
