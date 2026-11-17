use engine_core::glam::{Mat4, Vec3, Vec4};
use engine_core::math;
use engine_places_rendering::shadows::SHADOW_REGIONS;

use crate::changes::ShadowChange;
use crate::constants::LEVEL_SNAP;

const LOW_SUN: f32 = 0.1;
const SPACING_SLACK: f32 = 1.0625;

pub(crate) type Rect = [i32; 4];

const NONE: Rect = [0; 4];

fn empty(rect: Rect) -> bool {
    rect[0] >= rect[2] || rect[1] >= rect[3]
}

fn meet(a: Rect, b: Rect) -> Rect {
    let rect = [a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])];
    if empty(rect) { NONE } else { rect }
}

fn join(a: Rect, b: Rect) -> Rect {
    match (empty(a), empty(b)) {
        (true, _) => b,
        (_, true) => a,
        _ => [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])],
    }
}

fn strip([x0, y0, x1, y1]: Rect, [e0, f0, e1, f1]: Rect, side: usize) -> Rect {
    match side {
        0 => [e0, y0, x0, y1],
        1 => [x1, y0, e1, y1],
        2 => [x0, f0, x1, y0],
        _ => [x0, y1, x1, f1],
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LightBasis {
    pub rows: [Vec3; 3],
}

impl LightBasis {
    pub(crate) fn toward(sun: Vec3) -> Self {
        let away = -sun;
        let reference = if away.y.abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let right = away.cross(reference).normalize();
        let up = right.cross(away);
        Self {
            rows: [right, up, away],
        }
    }

    pub(crate) fn apply(&self, point: Vec3) -> Vec3 {
        Vec3::new(
            self.rows[0].dot(point),
            self.rows[1].dot(point),
            self.rows[2].dot(point),
        )
    }

    fn sine(&self) -> f32 {
        math::max(-self.rows[2].y, LOW_SUN)
    }

    fn bounds(&self, min: Vec3, max: Vec3) -> (Vec3, Vec3) {
        let center = self.apply((min + max) * 0.5);
        let half = (max - min) * 0.5;
        let extent = Vec3::new(
            self.rows[0].abs().dot(half),
            self.rows[1].abs().dot(half),
            self.rows[2].abs().dot(half),
        );
        (center - extent, center + extent)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Level {
    pub(crate) basis: LightBasis,
    pub(crate) texel: f32,
    depth: f32,
    half_depth: f32,
    pub(crate) window: Rect,
    redraw: Rect,
    waited: Rect,
    pub(crate) wanted: Rect,
}

impl Level {
    fn new(basis: LightBasis, eye: Vec3, texel: f32, side: u32) -> Self {
        let rise = basis.rows[1].y.abs();
        Self {
            basis,
            texel,
            depth: basis.apply(eye).z,
            half_depth: 0.5 * texel * side as f32 * ((1.0 + rise) / basis.sine() + 1.0),
            window: NONE,
            redraw: NONE,
            waited: NONE,
            wanted: NONE,
        }
    }

    fn stale(&self, sun: &LightBasis, eye: Vec3, side: u32) -> bool {
        let turned = self.basis.rows[2].cross(sun.rows[2]).length() * 2.0 * self.half_depth;
        let drift = (self.basis.apply(eye).z - self.depth).abs();
        turned > 0.5 * self.texel || drift > 0.5 * self.texel * side as f32
    }

    pub(crate) fn grid(&self, world: Vec3) -> Vec3 {
        let light = self.basis.apply(world);
        Vec3::new(light.x / self.texel, -light.y / self.texel, light.z)
    }

    fn extent(&self, eye: Vec3, side: u32) -> Rect {
        let grid = self.grid(eye);
        let snap = LEVEL_SNAP as f32;
        let corner = |at: f32| ((at - 0.5 * side as f32) / snap).round() as i32 * LEVEL_SNAP as i32;
        let [x, y] = [corner(grid.x), corner(grid.y)];
        [x, y, x + side as i32, y + side as i32]
    }

    pub(crate) fn band(&self) -> [f32; 2] {
        [
            self.depth - self.half_depth,
            self.depth + self.half_depth,
        ]
    }

    pub(crate) fn light_rect(&self, rect: Rect) -> [f32; 4] {
        let [x0, y0, x1, y1] = rect.map(|at| at as f32 * self.texel);
        [x0, -y1, x1, -y0]
    }

    pub(crate) fn clip(&self, rect: Rect) -> Mat4 {
        let [x0, y0, x1, y1] = self.light_rect(rect);
        let [z0, z1] = self.band();
        let row = |axis: Vec3, scale: f32, offset: f32| (axis * scale).extend(-offset * scale);
        Mat4::from_cols(
            row(self.basis.rows[0], 2.0 / (x1 - x0), (x0 + x1) * 0.5),
            row(self.basis.rows[1], 2.0 / (y1 - y0), (y0 + y1) * 0.5),
            row(self.basis.rows[2], 1.0 / (z1 - z0), z0),
            Vec4::W,
        )
        .transpose()
    }

    pub(crate) fn depth_span(&self) -> f32 {
        math::max(2.0 * self.half_depth, 1.0e-6)
    }

    pub(crate) fn rows(&self) -> [[f32; 4]; 3] {
        let [near, _] = self.band();
        let depth = self.depth_span();
        let [right, up, away] = self.basis.rows;
        [
            (right / self.texel).extend(0.0).to_array(),
            (-up / self.texel).extend(0.0).to_array(),
            (away / depth).extend(-near / depth).to_array(),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LevelKey {
    pub(crate) levels: u32,
    pub(crate) moving: u32,
    pub(crate) side: u32,
    pub(crate) finest: u32,
}

impl LevelKey {
    pub(crate) fn layers(&self) -> u32 {
        (self.levels + self.moving).max(2)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    pub(crate) level: usize,
    pub(crate) rect: Rect,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, schemars::JsonSchema)]
pub(crate) struct TexelState {
    pub level: usize,
    pub texel: [i32; 2],
    pub layer_texel: [u32; 2],
    pub valid: bool,
    pub redraw: bool,
    pub wanted: bool,
}

#[derive(Debug, Default)]
pub(crate) struct Levels {
    key: Option<LevelKey>,
    levels: Vec<Level>,
    pub(crate) regions: Vec<Region>,
    pub(crate) wanted: u32,
    pub(crate) redraws: u32,
}

fn take(regions: &mut Vec<Region>, (wanted, room): (&mut u32, usize), region: Region) -> bool {
    *wanted += 1;
    let room = regions.len() < room;
    if room {
        regions.push(region);
    }
    room
}

impl Levels {
    pub(crate) fn place(&mut self, key: LevelKey, finest: f32, (sun, eye): (LightBasis, Vec3)) -> bool {
        let restart = self.key != Some(key);
        if restart {
            self.key = Some(key);
            self.levels.clear();
        }
        for index in 0..key.levels as usize {
            if self
                .levels
                .get(index)
                .is_none_or(|level| level.stale(&sun, eye, key.side))
            {
                let texel = finest * 2.0_f32.powi(index as i32);
                let fresh = Level::new(sun, eye, texel, key.side);
                match self.levels.get_mut(index) {
                    Some(level) => *level = fresh,
                    None => self.levels.push(fresh),
                }
            }
            let level = &mut self.levels[index];
            level.wanted = level.extent(eye, key.side);
            level.window = meet(level.window, level.wanted);
            level.redraw = meet(level.redraw, level.window);
            level.waited = meet(level.waited, level.window);
        }
        restart
    }

    pub(crate) fn change(&mut self, change: &ShadowChange, pixel: f32) {
        for level in &mut self.levels {
            let touched = match *change {
                ShadowChange::Box { min, max, spacing }
                    if pixel * level.texel <= spacing * SPACING_SLACK =>
                {
                    let (low, high) = level.basis.bounds(min, max);
                    let at = |value: f32| (value / level.texel).floor() as i32;
                    [at(low.x), at(-high.y), at(high.x) + 1, at(-low.y) + 1]
                }
                ShadowChange::Coarse { pixel: least }
                    if pixel * level.texel * SPACING_SLACK >= least =>
                {
                    level.window
                }
                ShadowChange::Box { .. } | ShadowChange::Coarse { .. } | ShadowChange::Casters => {
                    continue;
                }
                ShadowChange::All => level.window,
            };
            level.redraw = join(level.redraw, meet(touched, level.window));
        }
    }

    pub(crate) fn forget(&mut self, levels: Option<&[u32]>) {
        for (index, level) in (0_u32..).zip(&mut self.levels) {
            if levels.is_none_or(|levels| levels.contains(&index)) {
                (level.window, level.redraw, level.waited) = (NONE, NONE, NONE);
            }
        }
    }

    pub(crate) fn wait(&mut self, levels: &[u32]) {
        for (index, level) in (0_u32..).zip(&mut self.levels) {
            if levels.contains(&index) {
                level.waited = level.window;
            }
        }
    }

    pub(crate) fn plan(&mut self, cameras: usize, (held_redraws, held_waits): (bool, bool)) {
        let Self {
            levels,
            regions,
            wanted,
            redraws,
            ..
        } = self;
        let room = cameras.min(SHADOW_REGIONS as usize);
        regions.clear();
        (*wanted, *redraws) = (0, 0);
        for (level, held) in levels.iter_mut().enumerate().rev() {
            if empty(held.window) {
                if take(regions, (wanted, room), Region { level, rect: held.wanted }) {
                    held.window = held.wanted;
                }
                continue;
            }
            for side in 0..4 {
                let rect = strip(held.window, held.wanted, side);
                if !empty(rect) && take(regions, (wanted, room), Region { level, rect }) {
                    held.window = join(held.window, rect);
                }
            }
        }
        let owed = if held_redraws { 0 } else { room };
        for (level, held) in levels.iter_mut().enumerate().rev() {
            if empty(held.redraw) {
                continue;
            }
            *redraws += 1;
            if take(regions, (wanted, owed), Region { level, rect: held.redraw }) {
                held.redraw = NONE;
            }
        }
        let owed = if held_waits { 0 } else { room };
        for (level, held) in levels.iter_mut().enumerate().rev() {
            let rect = held.waited;
            if !empty(rect) && take(regions, (wanted, owed), Region { level, rect }) {
                held.waited = NONE;
            }
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Level> {
        self.levels.iter()
    }

    pub(crate) fn level(&self, level: usize) -> &Level {
        &self.levels[level]
    }

    pub(crate) fn len(&self) -> usize {
        self.levels.len()
    }

    pub(crate) fn idle(&mut self) {
        self.regions.clear();
        (self.wanted, self.redraws) = (0, 0);
    }

    pub(crate) fn inspect(&self, point: Vec3) -> Vec<TexelState> {
        let side = self.key.map_or(1, |key| key.side as i32);
        self.levels
            .iter()
            .enumerate()
            .map(|(level, held)| {
                let grid = held.grid(point);
                let texel = [grid.x.floor() as i32, grid.y.floor() as i32];
                let within = |[x0, y0, x1, y1]: Rect| {
                    texel[0] >= x0 && texel[1] >= y0 && texel[0] < x1 && texel[1] < y1
                };
                TexelState {
                    level,
                    texel,
                    layer_texel: texel.map(|at| at.rem_euclid(side) as u32),
                    valid: within(held.window),
                    redraw: within(held.redraw),
                    wanted: within(held.wanted),
                }
            })
            .collect()
    }
}
