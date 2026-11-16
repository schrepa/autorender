use std::f32::consts::{LOG2_E, PI};

use engine_core::math;
use glam::Vec2;

use super::random::{golden_hash, unit};
use super::sprigs::Atlas;

pub(crate) const SIZE: usize = 2048;

const COLUMNS: usize = 4;

const ROWS: usize = 2;

const MARGIN: f32 = 14.0;

const LEAF: f32 = 0.078;

const CLUSTER: f32 = 1.9;

const STEM_LEAVES: u32 = 26;

const TWIG_LEAVES: u32 = 9;

const TWIGS: [u32; COLUMNS * ROWS] = [5, 7, 6, 8, 4, 7, 6, 5];

const WOOD: [f32; 3] = [78.0, 54.0, 46.0];

const FLAT: [f32; 3] = [127.5, 127.5, 255.0];

const LEVELS: usize = 4;

const FAINT: f32 = 0.02;

pub(crate) struct Sheet {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) pixels: Vec<u8>,
}

pub(crate) struct Source {
    pub(crate) rect: [f32; 4],
    pub(crate) root: [f32; 2],
    pub(crate) cluster: bool,
}

impl Sheet {
    fn cropped(&self, [x, y, w, h]: [usize; 4]) -> Self {
        let width = w.min(self.width.saturating_sub(x)).max(1);
        let height = h.min(self.height.saturating_sub(y)).max(1);
        let mut pixels = Vec::with_capacity(width * height * 4);
        for row in y..y + height {
            let at = (row.min(self.height - 1) * self.width + x.min(self.width - 1)) * 4;
            let end = (at + width * 4).min(self.pixels.len());
            pixels.extend_from_slice(&self.pixels[at..end]);
        }
        pixels.resize(width * height * 4, 0);
        Self {
            width,
            height,
            pixels,
        }
    }

    fn halved(&self) -> Self {
        let (width, height) = (self.width.div_ceil(2), self.height.div_ceil(2));
        let mut pixels = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0_u32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let sx = (2 * x + dx).min(self.width - 1);
                    let sy = (2 * y + dy).min(self.height - 1);
                    let at = (sy * self.width + sx) * 4;
                    for (total, &channel) in sum.iter_mut().zip(&self.pixels[at..at + 4]) {
                        *total += u32::from(channel);
                    }
                }
                pixels.extend(sum.map(|total| ((total + 2) / 4) as u8));
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    fn at(&self, point: Vec2) -> [f32; 4] {
        let x = math::clamp(point.x - 0.5, 0.0, (self.width - 1) as f32);
        let y = math::clamp(point.y - 0.5, 0.0, (self.height - 1) as f32);
        let (left, top) = (x as usize, y as usize);
        let right = (left + 1).min(self.width - 1);
        let bottom = (top + 1).min(self.height - 1);
        let (across, down) = (x - left as f32, y - top as f32);
        let texel = |x: usize, y: usize| {
            let at = (y * self.width + x) * 4;
            [0, 1, 2, 3].map(|channel| f32::from(self.pixels[at + channel]))
        };
        let (a, b, c, d) = (
            texel(left, top),
            texel(right, top),
            texel(left, bottom),
            texel(right, bottom),
        );
        [0, 1, 2, 3].map(|channel| {
            let upper = a[channel] + (b[channel] - a[channel]) * across;
            let lower = c[channel] + (d[channel] - c[channel]) * across;
            upper + (lower - upper) * down
        })
    }
}

struct Cut {
    color: Vec<Sheet>,
    relief: Vec<Sheet>,
    size: Vec2,
    root: Vec2,
    along: Vec2,
    reach: f32,
    cluster: bool,
}

impl Cut {
    fn of(source: &Source, leaves: &Sheet, relief: &Sheet) -> Self {
        let [x, y, w, h] = source.rect.map(|value| math::max(value, 0.0) as usize);
        let levels = |sheet: &Sheet| {
            let mut levels = vec![sheet.cropped([x, y, w, h])];
            while levels.len() < LEVELS {
                let next = levels[levels.len() - 1].halved();
                levels.push(next);
            }
            levels
        };
        let color = levels(leaves);
        let size = Vec2::new(color[0].width as f32, color[0].height as f32);
        let root = Vec2::from(source.root) - Vec2::new(x as f32, y as f32);
        let along = (size * 0.5 - root).normalize_or(Vec2::NEG_Y);
        let reach = corners(size)
            .iter()
            .fold(1.0, |reach, corner| math::max(reach, (*corner - root).dot(along)));
        Self {
            color,
            relief: levels(relief),
            size,
            root,
            along,
            reach,
            cluster: source.cluster,
        }
    }
}

struct Leaf {
    at: Vec2,
    way: Vec2,
    length: f32,
    cut: usize,
    mirror: f32,
    shade: f32,
}

struct Canvas {
    color: Vec<u8>,
    relief: Vec<u8>,
}

fn corners(size: Vec2) -> [Vec2; 4] {
    [
        Vec2::ZERO,
        Vec2::new(size.x, 0.0),
        Vec2::new(0.0, size.y),
        size,
    ]
}

fn quarter(way: Vec2) -> Vec2 {
    Vec2::new(-way.y, way.x)
}

fn turned(way: Vec2, angle: f32) -> Vec2 {
    let (sin, cos) = math::sin_cos(angle);
    Vec2::new(way.x * cos - way.y * sin, way.x * sin + way.y * cos)
}

impl Canvas {
    fn blend(&mut self, (x, y): (usize, usize), color: [f32; 3], relief: [f32; 3], cover: f32) {
        let at = (y * SIZE + x) * 4;
        let under = f32::from(self.color[at + 3]) / 255.0 * (1.0 - cover);
        let alpha = cover + under;
        for channel in 0..3 {
            let below = f32::from(self.color[at + channel]);
            let mixed = (color[channel] * cover + below * under) / alpha;
            self.color[at + channel] = math::clamp(mixed + 0.5, 0.0, 255.0) as u8;
            let flat = f32::from(self.relief[at + channel]);
            let bent = flat + (relief[channel] - flat) * cover;
            self.relief[at + channel] = math::clamp(bent + 0.5, 0.0, 255.0) as u8;
        }
        self.color[at + 3] = math::clamp(alpha * 255.0 + 0.5, 0.0, 255.0) as u8;
    }

    fn stamp(&mut self, cut: &Cut, leaf: &Leaf, [low, high]: [Vec2; 2]) {
        let scale = leaf.length / cut.reach;
        let coarse = math::clamp(math::ln(1.0 / scale) * LOG2_E, 0.0, (LEVELS - 1) as f32);
        let level = coarse as usize;
        let shrink = 1.0 / (1_u32 << level) as f32;
        let side = quarter(leaf.way) * leaf.mirror;
        let across = quarter(cut.along);
        let corners = corners(cut.size).map(|corner| {
            let from_root = corner - cut.root;
            let placed = leaf.way * from_root.dot(cut.along) + side * from_root.dot(across);
            leaf.at + placed * scale
        });
        let first = corners
            .iter()
            .fold(high, |first, corner| first.min(*corner))
            .max(low);
        let last = corners
            .iter()
            .fold(low, |last, corner| last.max(*corner))
            .min(high);
        for y in first.y as usize..last.y.ceil() as usize {
            for x in first.x as usize..last.x.ceil() as usize {
                let offset = (Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - leaf.at) / scale;
                let source =
                    cut.root + cut.along * offset.dot(leaf.way) + across * offset.dot(side);
                if source.cmplt(Vec2::ZERO).any() || source.cmpge(cut.size).any() {
                    continue;
                }
                let color = cut.color[level].at(source * shrink);
                let cover = color[3] / 255.0;
                if cover < FAINT {
                    continue;
                }
                let sampled = cut.relief[level].at(source * shrink);
                let tilt = Vec2::new(sampled[0] / 127.5 - 1.0, 1.0 - sampled[1] / 127.5);
                let bent = leaf.way * tilt.dot(cut.along) + side * tilt.dot(across);
                self.blend(
                    (x, y),
                    [0, 1, 2].map(|channel| color[channel] * leaf.shade),
                    [(bent.x + 1.0) * 127.5, (1.0 - bent.y) * 127.5, sampled[2]],
                    cover,
                );
            }
        }
    }

    fn line(&mut self, [from, to]: [Vec2; 2], thick: f32, [low, high]: [Vec2; 2]) {
        let pad = Vec2::splat(thick + 1.0);
        let first = (from.min(to) - pad).max(low);
        let last = (from.max(to) + pad).min(high);
        let span = to - from;
        let length = math::max(span.length_squared(), 1.0e-6);
        for y in first.y as usize..last.y.ceil() as usize {
            for x in first.x as usize..last.x.ceil() as usize {
                let point = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let share = math::clamp((point - from).dot(span) / length, 0.0, 1.0);
                let away = point.distance(from + span * share);
                let cover = math::clamp(0.5 * thick - away + 0.5, 0.0, 1.0);
                if cover > 0.0 {
                    self.blend((x, y), WOOD, FLAT, cover);
                }
            }
        }
    }
}

fn frame(cell: usize) -> [Vec2; 2] {
    let span = Vec2::new((SIZE / COLUMNS) as f32, (SIZE / ROWS) as f32);
    let low = Vec2::new((cell % COLUMNS) as f32, (cell / COLUMNS) as f32) * span;
    [low + Vec2::splat(MARGIN), low + span - Vec2::splat(MARGIN)]
}

pub(crate) fn atlas() -> Atlas {
    Atlas::new(
        [SIZE as f32; 2],
        (0..COLUMNS * ROWS).map(|cell| {
            let [low, high] = frame(cell);
            let span = high - low;
            [low.x, low.y, span.x, span.y, low.x + 0.5 * span.x, high.y]
        }),
    )
}

struct Spray {
    single: Vec<usize>,
    cluster: Vec<usize>,
    seed: u32,
    size: f32,
    leaves: Vec<Leaf>,
    lines: Vec<([Vec2; 2], f32)>,
}

impl Spray {
    fn roll(&self, index: u32) -> f32 {
        unit(golden_hash(self.seed ^ index))
    }

    fn leaf(&mut self, at: Vec2, way: Vec2, key: u32, grown: f32) {
        let (pool, cluster) = match grown > 1.0 && !self.cluster.is_empty() {
            true => (&self.cluster, CLUSTER),
            false => (&self.single, 1.0),
        };
        if pool.is_empty() {
            return;
        }
        let cut = pool[golden_hash(self.seed ^ key ^ 0x51ed_270b) as usize % pool.len()];
        self.leaves.push(Leaf {
            at,
            way,
            length: self.size * cluster * math::min(grown, 1.0)
                * (0.75 + 0.4 * self.roll(key ^ 0x11)),
            cut,
            mirror: match golden_hash(self.seed ^ key ^ 0x2545_f491) & 1 {
                0 => 1.0,
                _ => -1.0,
            },
            shade: 0.8 + 0.2 * self.roll(key ^ 0x12),
        });
    }

    fn shoot(
        &mut self,
        [from, to]: [Vec2; 2],
        bow: f32,
        (count, key): (u32, u32),
        (thick, tip): (f32, f32),
    ) {
        let span = to - from;
        let sag = quarter(span) * bow;
        let point = |share: f32| from + span * share + sag * math::sin(PI * share);
        let way =
            |share: f32| (point(share + 0.02) - point(share - 0.02)).normalize_or(Vec2::NEG_Y);
        let pieces = 12_u32;
        for piece in 0..pieces {
            let [low, high] = [piece, piece + 1].map(|end| end as f32 / pieces as f32);
            self.lines
                .push(([point(low), point(high)], thick * (1.0 - 0.6 * low)));
        }
        for leaf in 0..count {
            let id = key | leaf << 4;
            let share = 0.05 + 0.92 * (leaf as f32 + self.roll(id ^ 0x21)) / count as f32;
            let side = match leaf % 2 {
                0 => 1.0,
                _ => -1.0,
            };
            let angle = side * (0.55 + 0.6 * self.roll(id ^ 0x22));
            self.leaf(point(share), turned(way(share), angle), id, 1.0 - 0.25 * share);
        }
        self.leaf(point(1.0), way(1.0), key | 0xf00, tip);
    }
}

fn grow(cuts: &[Cut], cell: usize, [low, high]: [Vec2; 2]) -> Spray {
    let span = high - low;
    let size = LEAF * span.y;
    let pool = |cluster: bool| {
        (0..cuts.len())
            .filter(|&cut| cuts[cut].cluster == cluster)
            .collect::<Vec<usize>>()
    };
    let mut spray = Spray {
        single: pool(false),
        cluster: pool(true),
        seed: (cell as u32 + 1).wrapping_mul(0x9e37_79b9),
        size,
        leaves: Vec::new(),
        lines: Vec::new(),
    };
    let reach = 1.15 * size;
    let root = Vec2::new(low.x + 0.5 * span.x, high.y);
    let lean = (spray.roll(1) - 0.5) * 0.25 * span.x;
    let tip = Vec2::new(root.x + lean, low.y + CLUSTER * reach);
    let bow = (spray.roll(2) - 0.5) * 0.12;
    spray.shoot([root, tip], bow, (STEM_LEAVES, 0x1_0000), (5.0, 1.5));
    let stem = tip - root;
    let twigs = TWIGS[cell % TWIGS.len()];
    for twig in 0..twigs {
        let key = (twig + 2) << 16;
        let share = 0.06 + 0.66 * (twig as f32 + spray.roll(key ^ 1)) / twigs as f32;
        let from = root + stem * share + quarter(stem) * (bow * math::sin(PI * share));
        let side = match twig % 2 {
            0 => 1.0,
            _ => -1.0,
        };
        let angle = side * (0.45 + 0.4 * spray.roll(key ^ 2));
        let way = turned(stem.normalize_or(Vec2::NEG_Y), angle);
        let length = span.y * (0.2 + 0.2 * spray.roll(key ^ 3)) * (1.0 - 0.4 * share);
        let end = from + way * length;
        let to = Vec2::new(
            math::clamp(end.x, low.x + reach, high.x - reach),
            math::max(end.y, low.y + reach),
        );
        let sag = (spray.roll(key ^ 4) - 0.5) * 0.2;
        spray.shoot([from, to], sag, (TWIG_LEAVES, key), (3.0, 0.9));
    }
    spray
}

pub(crate) fn compose(leaves: &Sheet, relief: &Sheet, sources: &[Source]) -> [Vec<u8>; 2] {
    let cuts: Vec<Cut> = sources
        .iter()
        .map(|source| Cut::of(source, leaves, relief))
        .collect();
    let mut canvas = Canvas {
        color: vec![0; SIZE * SIZE * 4],
        relief: [127, 127, 255, 255].repeat(SIZE * SIZE),
    };
    for cell in 0..COLUMNS * ROWS {
        let bounds = frame(cell);
        let spray = grow(&cuts, cell, bounds);
        for (ends, thick) in &spray.lines {
            canvas.line(*ends, *thick, bounds);
        }
        for leaf in &spray.leaves {
            canvas.stamp(&cuts[leaf.cut], leaf, bounds);
        }
    }
    [canvas.color, canvas.relief]
}
