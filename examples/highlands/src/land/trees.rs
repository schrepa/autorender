use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::OnceLock;

use engine::prelude::*;

use crate::random::{golden_hash, unit};
use crate::records;

pub(crate) fn spruce(
    world: &mut World,
    seed: u32,
) -> engine::Result<(Vec<MeshHandle>, MeshHandle)> {
    let levels = (0..DETAILS.len())
        .map(|detail| conifer_record(world, seed, detail))
        .collect::<engine::Result<Vec<_>>>()?;
    let shadow = shadow_record(world, seed)?;
    Ok((levels, shadow))
}

fn conifer_record(world: &mut World, seed: u32, detail: usize) -> engine::Result<MeshHandle> {
    let mut identity = seed.to_le_bytes().to_vec();
    identity.extend_from_slice(&(detail as u32).to_le_bytes());
    records::mesh(world, "highlands_conifer", &identity, move || {
        conifer(seed, detail)
    })
}

fn shadow_record(world: &mut World, seed: u32) -> engine::Result<MeshHandle> {
    records::mesh(
        world,
        "highlands_conifer_shadow",
        &seed.to_le_bytes(),
        move || shadow_proxy(seed),
    )
}

pub(crate) fn atlas_record(world: &mut World) -> engine::Result<TextureHandle> {
    let handles = world.resource_scope(|world, mut stored: Mut<Records>| {
        let mut assets = world.resource_mut::<AssetLibrary>();
        records::textures(
            (&mut stored, &mut assets),
            "highlands/bark_atlas".into(),
            &[("atlas", TextureUsage::Color)],
            || {
                Ok(vec![TextureSource {
                    width: ATLAS,
                    height: ATLAS,
                    pixels: texels().to_vec(),
                    usage: TextureUsage::Color,
                }])
            },
        )
    })?;
    Ok(handles[0])
}

fn texels() -> &'static [u8] {
    static TEXELS: OnceLock<Vec<u8>> = OnceLock::new();
    TEXELS.get_or_init(atlas)
}

const ATLAS: u32 = 512;

const CUTOFF: f32 = 0.5;

const BARK: [f32; 2] = [0.01, 0.49];
const BARK_MIDDLE: [f32; 2] = [0.5 * (BARK[0] + BARK[1]), 0.5];
const SPRAY: [f32; 2] = [0.75, 0.235];
const SPRAY_SPAN: [f32; 2] = [0.99, 0.01];

const CROWN_NORMAL: f32 = 0.7;
const LEAN: f32 = 0.55;

const FRONT: [u32; 6] = [0, 1, 2, 1, 3, 2];
const BACK: [u32; 6] = [0, 2, 1, 1, 2, 3];

const SPACING: f32 = 0.42;

const TIER: usize = 2;
const RIM_SIDES: u32 = 8;
const RIM: f32 = 0.8;
const SPIKE: f32 = 0.7;

struct Detail {
    sides: u32,
    segments: u32,
    bends: usize,
    keep: [f32; 2],
}

const DETAILS: [Detail; 4] = [
    Detail {
        sides: 7,
        segments: 8,
        bends: 2,
        keep: [1.0, 1.0],
    },
    Detail {
        sides: 5,
        segments: 4,
        bends: 2,
        keep: [0.55, 1.0],
    },
    Detail {
        sides: 4,
        segments: 2,
        bends: 1,
        keep: [0.45, 0.6],
    },
    Detail {
        sides: 3,
        segments: 1,
        bends: 1,
        keep: [0.3, 0.5],
    },
];

fn conifer(seed: u32, detail: usize) -> engine::Result<Mesh> {
    let build = &DETAILS[detail];
    let shape = Shape::of(seed);
    let mut builder = Builder::default();
    builder.trunk(&shape, (build.sides, build.segments));
    for whorl in &shape.whorls {
        let count = whorl.branches.len();
        let share = build.keep[0] + (build.keep[1] - build.keep[0]) * whorl.rise;
        let kept = ((count as f32 * share).round() as usize).clamp(2, count);
        for branch in whorl.branches.iter().filter(|branch| branch.rank < kept) {
            builder.branch(branch, build.bends);
        }
    }
    builder.leader(shape.height);
    let atlas = AlphaImage {
        width: ATLAS,
        height: ATLAS,
        rgba: texels(),
    };
    builder
        .mesh()
        .and_then(|mesh| trim_cards(&mesh, atlas, CUTOFF))
        .map_err(|error| error.within(&format!("$.conifers[{seed}].details[{detail}]")))
}

fn shadow_proxy(seed: u32) -> engine::Result<Mesh> {
    let shape = Shape::of(seed);
    let mut builder = Builder::default();
    let foot = SPIKE * shape.radius(-0.5);
    for side in 0..4 {
        let (sin, cos) = math::sin_cos(FRAC_PI_2 * side as f32);
        let corner = Vec3::new(cos * foot, -0.5, sin * foot);
        builder.push(corner, Vec3::new(cos, 0.0, sin), BARK_MIDDLE, [1.0; 3]);
    }
    let tip = Vec3::new(0.0, shape.height, 0.0);
    let peak = builder.push(tip, Vec3::Y, BARK_MIDDLE, [1.0; 3]);
    for side in 0..4 {
        builder.indices.extend([side, peak, (side + 1) % 4]);
    }
    let tiers = shape.whorls.len().div_ceil(TIER);
    for (number, whorls) in shape.whorls.chunks(TIER).enumerate() {
        let top = if number + 1 == tiers {
            shape.height
        } else {
            whorls[whorls.len() - 1].y
        };
        builder.tier(&whorls[0], top);
    }
    builder
        .mesh()
        .map_err(|error| error.within(&format!("$.conifers[{seed}].shadow")))
}

struct Shape {
    height: f32,
    base_radius: f32,
    whorls: Vec<Whorl>,
}

struct Whorl {
    y: f32,
    length: f32,
    dip: (f32, f32),
    twist: f32,
    rise: f32,
    branches: Vec<Branch>,
}

struct Branch {
    start: Vec3,
    outward: Vec3,
    direction: Vec3,
    length: f32,
    color: [f32; 3],
    slot: f32,
    reach: f32,
    rank: usize,
}

impl Branch {
    fn at(&self, t: f32) -> Vec3 {
        point_along(self.start, self.direction, self.length, t)
    }
}

fn point_along(start: Vec3, direction: Vec3, length: f32, t: f32) -> Vec3 {
    start + direction * (length * t) - Vec3::Y * (0.18 * length * t * t)
}

impl Shape {
    fn of(seed: u32) -> Self {
        let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x9e37_79b9) ^ index));
        let height = 13.0 + 4.0 * random(1);
        let base_radius = 0.2 + 0.06 * random(2);
        let crown_base = 1.6 + 0.8 * random(3);
        let mut whorls = Vec::new();
        let mut y = crown_base;
        let mut whorl = 0_u32;
        while y < height - 0.7 {
            let below_top = height - y;
            let count = 6 + golden_hash(seed ^ whorl.wrapping_mul(0x632b_e5ab)) % 3;
            let twist = random(100 + whorl) * TAU;
            let droop = -0.25 + 0.6 * math::clamp(below_top / height, 0.0, 1.0);
            let (dip_sin, dip_cos) = math::sin_cos(droop);
            let length = 0.45 + below_top * 0.19;
            let shade = 0.5 + 0.5 * math::clamp((y - crown_base) / (0.6 * height), 0.0, 1.0);
            let mut branches: Vec<Branch> = (0..count)
                .map(|branch| {
                    let index = 1000 + whorl * 16 + branch;
                    let slot = branch as f32 + 0.35 * (random(index) - 0.5);
                    let (sin, cos) = math::sin_cos(twist + TAU * slot / count as f32);
                    let outward = Vec3::new(cos, 0.0, sin);
                    let reach = random(index + 7);
                    Branch {
                        start: Vec3::new(0.0, y, 0.0),
                        outward,
                        direction: outward * dip_cos - Vec3::Y * dip_sin,
                        length: length * (0.85 + 0.3 * reach),
                        color: [shade * (0.94 + 0.12 * random(index + 9)), shade, shade],
                        slot,
                        reach,
                        rank: 0,
                    }
                })
                .collect();
            rank(&mut branches);
            whorls.push(Whorl {
                y,
                length,
                dip: (dip_sin, dip_cos),
                twist,
                rise: math::clamp((y - crown_base) / (height - 0.7 - crown_base), 0.0, 1.0),
                branches,
            });
            y += SPACING * (0.85 + 0.3 * random(200 + whorl));
            whorl += 1;
        }
        Self {
            height,
            base_radius,
            whorls,
        }
    }

    fn radius(&self, y: f32) -> f32 {
        self.base_radius * math::pow(math::max(1.0 - y / self.height, 0.0), 0.85) + 0.012
    }
}

fn rank(branches: &mut [Branch]) {
    let count = branches.len() as f32;
    let mut ranked: Vec<usize> = Vec::with_capacity(branches.len());
    while ranked.len() < branches.len() {
        let mut best: Option<(usize, f32)> = None;
        for (index, branch) in branches.iter().enumerate() {
            if ranked.contains(&index) {
                continue;
            }
            let room = ranked.iter().fold(count, |room, &other| {
                let gap = (branch.slot - branches[other].slot).abs();
                math::min(room, math::min(gap, count - gap))
            });
            let score = room + 0.5 * branch.reach;
            if best.is_none_or(|(_, top)| score > top) {
                best = Some((index, score));
            }
        }
        let Some((index, _)) = best else { break };
        branches[index].rank = ranked.len();
        ranked.push(index);
    }
}

#[derive(Default)]
struct Builder {
    vertices: Vec<MeshVertex>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl Builder {
    fn push(&mut self, position: Vec3, normal: Vec3, uv: [f32; 2], color: [f32; 3]) -> u32 {
        self.vertices.push(MeshVertex {
            position: position.to_array(),
            normal: normal.normalize_or(Vec3::Y).to_array(),
            uv,
        });
        let [red, green, blue] = color.map(|channel| math::clamp(channel, 0.0, 1.0));
        self.colors.push([red, green, blue, 1.0]);
        self.vertices.len() as u32 - 1
    }

    fn mesh(self) -> engine::Result<Mesh> {
        let Self {
            vertices,
            colors,
            indices,
        } = self;
        Mesh::new(vertices, indices).and_then(|mesh| mesh.with_colors(colors))
    }

    fn trunk(&mut self, shape: &Shape, (sides, segments): (u32, u32)) {
        for segment in 0..segments {
            let [low, high] = [segment, segment + 1]
                .map(|ring| -0.5 + (shape.height + 0.5) * ring as f32 / segments as f32);
            let first = self.vertices.len() as u32;
            for side in 0..=sides {
                let angle = TAU * side as f32 / sides as f32;
                let (sin, cos) = math::sin_cos(angle);
                let u = BARK[0] + (BARK[1] - BARK[0]) * side as f32 / sides as f32;
                for (y, v) in [(low, 0.99), (high, 0.01)] {
                    let shade = 0.55 + 0.45 * math::clamp(y / (0.35 * shape.height), 0.0, 1.0);
                    let radius = shape.radius(y);
                    self.push(
                        Vec3::new(cos * radius, y, sin * radius),
                        Vec3::new(cos, 0.15, sin),
                        [u, v],
                        [shade; 3],
                    );
                }
            }
            for side in 0..sides {
                let [a, b, c, d] = [0, 1, 2, 3].map(|corner| first + side * 2 + corner);
                self.indices.extend([a, b, c, b, d, c]);
            }
        }
    }

    fn branch(&mut self, branch: &Branch, bends: usize) {
        let spine: Vec<Vec3> = (0..=bends)
            .map(|step| branch.at(step as f32 / bends as f32))
            .collect();
        let across = Vec3::new(-branch.outward.z, 0.0, branch.outward.x);
        let hanging = branch.direction.cross(across).normalize_or(Vec3::Y);
        let crown = (branch.outward + Vec3::Y * LEAN).normalize_or(Vec3::Y);
        self.card(&spine, across * (0.45 * branch.length), crown, branch.color);
        self.card(&spine, hanging * (0.3 * branch.length), crown, branch.color);
    }

    fn leader(&mut self, height: f32) {
        for turn in [0.0, FRAC_PI_2] {
            let (sin, cos) = math::sin_cos(turn);
            self.card(
                &[
                    Vec3::new(0.0, height - 1.0, 0.0),
                    Vec3::new(0.0, height + 0.15, 0.0),
                ],
                Vec3::new(cos, 0.0, sin) * 0.25,
                Vec3::Y,
                [1.0; 3],
            );
        }
    }

    fn card(&mut self, spine: &[Vec3], half: Vec3, crown: Vec3, color: [f32; 3]) {
        let last = spine.len() - 1;
        for (facing, corners) in [(1.0_f32, FRONT), (-1.0, BACK)] {
            let first = self.vertices.len() as u32;
            for (step, &center) in spine.iter().enumerate() {
                let t = step as f32 / last as f32;
                let v = SPRAY_SPAN[0] + (SPRAY_SPAN[1] - SPRAY_SPAN[0]) * t;
                let inner = 0.5 + 0.5 * t;
                let shade = color.map(|channel| channel * inner);
                let way = spine[(step + 1).min(last)] - spine[step.saturating_sub(1)];
                let face = half.cross(way).normalize_or(crown) * facing;
                let normal = crown * CROWN_NORMAL + face * (1.0 - CROWN_NORMAL);
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
                let base = first + step * 2;
                self.indices.extend(corners.map(|corner| base + corner));
            }
        }
    }

    fn tier(&mut self, whorl: &Whorl, top: f32) {
        let apex = self.push(Vec3::new(0.0, top, 0.0), Vec3::Y, BARK_MIDDLE, [1.0; 3]);
        let below = Vec3::new(0.0, whorl.y - SPACING, 0.0);
        let under = self.push(below, Vec3::NEG_Y, BARK_MIDDLE, [1.0; 3]);
        let start = Vec3::new(0.0, whorl.y, 0.0);
        let (dip_sin, dip_cos) = whorl.dip;
        let first = self.vertices.len() as u32;
        for side in 0..RIM_SIDES {
            let turn = whorl.twist + TAU * side as f32 / RIM_SIDES as f32;
            let (sin, cos) = math::sin_cos(turn);
            let outward = Vec3::new(cos, 0.0, sin);
            let direction = outward * dip_cos - Vec3::Y * dip_sin;
            let rim = point_along(start, direction, whorl.length, RIM);
            self.push(rim, outward, BARK_MIDDLE, [1.0; 3]);
        }
        for side in 0..RIM_SIDES {
            let [left, right] = [side, (side + 1) % RIM_SIDES].map(|corner| first + corner);
            self.indices.extend([apex, right, left, under, left, right]);
        }
    }
}

fn atlas() -> Vec<u8> {
    let size = ATLAS as usize;
    let half = size / 2;
    let needle = [0.06, 0.13, 0.06];
    let mut color = vec![needle; size * size];
    let mut alpha = vec![0.0_f32; size * size];
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

struct Spray<'a> {
    color: &'a mut [[f32; 3]],
    alpha: &'a mut [f32],
    size: usize,
}

impl Spray<'_> {
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

fn srgb(value: f32) -> u8 {
    let value = math::clamp(value, 0.0, 1.0);
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * math::pow(value, 1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

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
