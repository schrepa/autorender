use bevy_ecs::prelude::*;
use engine_core::glam::{IVec2, Mat3, Mat4, Vec2, Vec3, Vec4};
use engine_core::math::{max, min};
use engine_core::{
    CullPlan, Cut, Detail, DeviceLimits, FrameTime, Readable, Readback, ViewDetail, ViewFacts,
    Views, frustum_bounded, frustum_planes, seen_share,
};
use engine_open_world_kit::ground::{Foliage, GroundTiles, TerrainEntry, TerrainTiles};
use engine_places_rendering::wind::ViewWind;
use engine_render::jitter_clip;

use crate::covers::CoverTable;
use crate::layouts::{CoverBlade, CoverCard, CoverFrame, Grow};
use crate::settings::{GroundCover, GroundCoverStats};

pub(crate) const BLADE_BYTES: u64 = size_of::<CoverBlade>() as u64;
pub(crate) const CARD_BYTES: u64 = size_of::<CoverCard>() as u64;

const APPENDED_WORD: usize = 0;
const STRIPS_WORD: usize = 1;
const FLATS_WORD: usize = 2;
const CAPACITY_WORD: usize = 4;
const CARD_CAPACITY_WORD: usize = 5;
const TRIED_WORD: usize = 6;
const SCANNED_WORD: usize = 7;
const RAN_WORD: usize = 8;
const GREW_WORD: usize = 9;
const CARDED_WORD: usize = 10;

const LANES: f32 = 64.0;
const LANE_FILL: f32 = 4.0;
const CELL_MARGIN: f32 = 0.05;
const FLAT_CORNER: f32 = 1.0e-6;

#[derive(Resource, Default)]
pub(crate) struct CoverPlan {
    pub(crate) rows: Vec<Grow>,
    pub(crate) cells: Vec<[u32; 3]>,
    pub(crate) frame: CoverFrame,
    pub(crate) capacity: u32,
    pub(crate) cards: u32,
    pub(crate) bins: u32,
    pub(crate) thin: f32,
    pub(crate) room: u64,
    pub(crate) shown: bool,
    pub(crate) drawn: bool,
    reach: f32,
    tiers: u32,
    groups: u32,
}

#[derive(Resource, Default)]
pub(crate) struct BladeCounts {
    latest: Option<Cut>,
    cards: Option<Cut>,
    demand: Option<f32>,
    strips: u32,
    flats: u32,
    carded: u32,
    tried: u32,
    cells: [u32; 3],
}

impl Readable for BladeCounts {
    fn arrived(&mut self, thin: u32, read: Readback<&[u8]>) {
        let bytes = read.value;
        let asked = word(bytes, APPENDED_WORD);
        self.latest = Some(Cut::of(asked, word(bytes, CAPACITY_WORD)));
        let room = word(bytes, CARD_CAPACITY_WORD);
        let bins = (CARDED_WORD..bytes.len() / 4).map(|bin| word(bytes, bin));
        self.cards = Some(Cut::of(bins.clone().max().unwrap_or(0), room));
        self.carded = bins.map(|asked| asked.min(room)).sum();
        self.strips = word(bytes, STRIPS_WORD);
        self.flats = word(bytes, FLATS_WORD);
        self.tried = word(bytes, TRIED_WORD);
        self.cells = [SCANNED_WORD, RAN_WORD, GREW_WORD].map(|at| word(bytes, at));
        self.demand = Some(asked as f32 / f32::from_bits(thin));
    }
}

fn word(bytes: &[u8], index: usize) -> u32 {
    bytes.get(index * 4..index * 4 + 4).map_or(0, |bytes| {
        u32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]))
    })
}

#[derive(Clone, Copy)]
struct Common {
    measure: ViewDetail,
    seen: Option<Wedge>,
    tallest: f32,
    grounded: f32,
    wind: [f32; 4],
}

#[derive(Clone, Copy)]
struct Wedge {
    apex: Vec2,
    sides: [Vec2; 2],
}

impl Wedge {
    fn seen(main: &ViewFacts, cull: &CullPlan) -> Option<Self> {
        let placed = main.jitter.map_or(main.view_projection, |offset| {
            let [x, y] = jitter_clip(offset.to_array(), main.render.x, main.render.y);
            Mat4::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::new(x, y, 0.0, 1.0))
                * main.view_projection
        });
        if !cull.frustum || !frustum_bounded(placed) {
            return None;
        }
        let [left, right, low, high, ..] = frustum_planes(placed);
        Self::of([left, low, right, high].map(|side| side / side.truncate().length()))
    }

    fn of(sides: [Vec4; 4]) -> Option<Self> {
        let normals = sides.map(Vec4::truncate);
        let rows = Mat3::from_cols(normals[0], normals[1], normals[2]).transpose();
        let meet = rows.determinant().abs() > FLAT_CORNER;
        if !meet {
            return None;
        }
        let apex = rows.inverse() * -Vec3::new(sides[0].w, sides[1].w, sides[2].w);
        let inward = normals[0] + normals[1] + normals[2] + normals[3];
        let mut rays = [Vec2::ZERO; 4];
        for (index, ray) in rays.iter_mut().enumerate() {
            let corner = normals[index].cross(normals[(index + 1) % 4]);
            let flat = Vec2::new(corner.x, corner.z) * corner.dot(inward).signum();
            if flat.length() > FLAT_CORNER {
                *ray = flat.normalize();
            }
        }
        let edge = |turn: f32| {
            rays.iter().copied().find(|&a| {
                a != Vec2::ZERO && rays.iter().all(|&b| turn * a.perp_dot(b) >= -FLAT_CORNER)
            })
        };
        Some(Self {
            apex: Vec2::new(apex.x, apex.z),
            sides: [edge(1.0)?, edge(-1.0)?],
        })
    }

    fn bounds(self, [low, high]: [Vec2; 2], margin: f32) -> Option<[Vec2; 2]> {
        let [first, last] = self.sides;
        let square = Outline {
            points: [
                low,
                Vec2::new(high.x, low.y),
                high,
                Vec2::new(low.x, high.y),
                Vec2::ZERO,
                Vec2::ZERO,
            ],
            count: 4,
        };
        square
            .kept(|point| first.perp_dot(point - self.apex) + margin)
            .kept(|point| (point - self.apex).perp_dot(last) + margin)
            .bounds()
    }
}

#[derive(Clone, Copy)]
struct Outline {
    points: [Vec2; 6],
    count: usize,
}

impl Outline {
    fn kept(self, inside: impl Fn(Vec2) -> f32) -> Self {
        let mut kept = Self {
            points: [Vec2::ZERO; 6],
            count: 0,
        };
        let corners = self.points.iter().take(self.count);
        let after = self.points.iter().take(self.count).cycle().skip(1);
        for (&a, &b) in corners.zip(after) {
            let (at, next) = (inside(a), inside(b));
            if at >= 0.0 {
                kept.push(a);
            }
            if (at >= 0.0) != (next >= 0.0) {
                kept.push(a + (b - a) * (at / (at - next)));
            }
        }
        kept
    }

    fn push(&mut self, point: Vec2) {
        if let Some(held) = self.points.get_mut(self.count) {
            *held = point;
            self.count += 1;
        }
    }

    fn bounds(self) -> Option<[Vec2; 2]> {
        let mut points = self.points.iter().take(self.count).copied();
        let first = points.next()?;
        Some(points.fold([first, first], |[low, high], point| {
            [low.min(point), high.max(point)]
        }))
    }
}

pub(crate) fn plan_cover(
    (cover, table, tiles, ground): (
        Res<GroundCover>,
        Res<CoverTable>,
        Res<TerrainTiles>,
        Res<GroundTiles>,
    ),
    (views, time, limits, detail): (Res<Views>, Res<FrameTime>, Res<DeviceLimits>, Res<Detail>),
    (wind, foliage, cull): (Res<ViewWind>, Res<Foliage>, Res<CullPlan>),
    mut plan: ResMut<CoverPlan>,
    mut counts: ResMut<BladeCounts>,
) {
    let plan = &mut *plan;
    plan.shown = cover.density > 0.0;
    plan.room = limits.storage_room / 16 * 16;
    plan.drawn = !ground.tables.is_empty();
    plan.bins = cover.atlases.len() as u32;
    if !plan.drawn {
        return;
    }
    plan.rows.clear();
    plan.cells.clear();
    plan.reach = 0.0;
    plan.tiers = 0;
    plan.groups = 0;
    let Some(main) = views.main() else {
        return;
    };
    let speed = Vec3::new(wind.velocity[0], 0.0, wind.velocity[1]).length();
    let grounded = f32::from(u8::from(foliage.ground_blend > 0.0));
    let common = Common {
        measure: ViewDetail::perspective(
            main.detail_eye,
            main.view_projection,
            main.render.y,
            detail.pixel_error,
        ),
        seen: Wedge::seen(main, &cull),
        tallest: table.tallest * cover.lush.sqrt(),
        grounded,
        wind: [
            time.elapsed as f32,
            (time.elapsed - time.seconds) as f32,
            min(speed / cover.full_wind, 1.0),
            0.0,
        ],
    };
    if !plan.grow(&cover, &table, (&tiles, &ground), (main, counts.demand), common) {
        counts.latest = Some(Cut::default());
        counts.cards = Some(Cut::default());
        counts.strips = 0;
        counts.flats = 0;
        counts.carded = 0;
        counts.tried = 0;
        counts.cells = [0; 3];
        return;
    }
    plan.frame = CoverFrame {
        wind: [wind.velocity[0], wind.velocity[1], grounded, 0.0],
        blades: [plan.capacity, plan.cards, 0, 0],
    };
}

pub(crate) fn cover_stats(
    counts: Res<BladeCounts>,
    plan: Res<CoverPlan>,
    mut stats: ResMut<GroundCoverStats>,
) {
    stats.set_if_neq(GroundCoverStats {
        blade_cut: counts.latest.unwrap_or_default(),
        card_cut: counts.cards.unwrap_or_default(),
        reach: plan.reach.round() as u32,
        tiers: plan.tiers,
        cells: plan.groups,
        strips: counts.strips,
        flats: counts.flats,
        cards: counts.carded,
        tried: counts.tried,
        scanned: counts.cells[0],
        ran: counts.cells[1],
        grew: counts.cells[2],
    });
}

impl CoverPlan {
    fn grow(
        &mut self,
        cover: &GroundCover,
        table: &CoverTable,
        (tiles, ground): (&TerrainTiles, &GroundTiles),
        (main, demand): (&ViewFacts, Option<f32>),
        common: Common,
    ) -> bool {
        if cover.density <= 0.0 {
            return false;
        }
        let lushest = cover.lush.sqrt();
        let mut most = [0.0; 2];
        for (index, terrain) in tiles.terrains.iter().enumerate() {
            let Some(at) = ground
                .tables
                .iter()
                .find(|(entity, _)| *entity == terrain.terrain)
                .map(|&(_, at)| at)
            else {
                continue;
            };
            for (slot, terms) in table.slots(index).iter().enumerate() {
                let dense =
                    [terms.blades, terms.cards].map(|count| count * cover.density * lushest);
                let density = (terms.blades + terms.plants) * cover.density * lushest;
                let law = CoverLaw {
                    measure: common.measure,
                    widest: terms.widest * lushest,
                    tallest: terms.tallest * lushest,
                    least: terms.least.row(lushest),
                };
                if density <= 0.0 || law.widest <= 0.0 || !law.reach().is_finite() {
                    continue;
                }
                let grown = [at, table.base(index), slot as u32];
                self.grow_terrain(terrain, grown, (law, density, terms.stand), main, common);
                let area = law.area();
                most[0] += area * dense[0];
                most[1] += area * dense[1];
            }
        }
        for at in 1..self.cells.len() {
            let mut place = at;
            while place > 0 && self.cells[place - 1][2] > self.cells[place][2] {
                self.cells.swap(place - 1, place);
                self.rows.swap(place - 1, place);
                place -= 1;
            }
        }
        if self.rows.is_empty() {
            return false;
        }
        let turn = turn_share(main.view_projection);
        let room = (self.room / BLADE_BYTES).min(u64::from(u32::MAX)) as u32;
        let seen = (most[0] * turn).ceil() as u32;
        let screen = (cover.layers * main.render.x as f32 * main.render.y as f32) as u32;
        let capacity = cover
            .max_blades
            .map_or(seen, |budget| budget.min(seen))
            .min(screen)
            .max(1)
            .min(room);
        let binned = CARD_BYTES * u64::from(self.bins.max(1));
        let card_room = (self.room / binned).min(u64::from(u32::MAX)) as u32;
        let carded = (most[1] * turn).ceil() as u32;
        let cards = cover
            .max_cards
            .map_or(carded, |budget| budget.min(carded))
            .max(1)
            .min(card_room);
        let thin = demand.map_or(1.0, |demand| min(capacity as f32 / demand, 1.0));
        self.capacity = capacity;
        self.cards = cards;
        self.thin = thin;
        let seen = common.seen.map_or(-1, |_| self.rows.len() as i32);
        for row in &mut self.rows {
            row.room = [capacity, cards, self.bins, 0];
            row.detail[3] = thin;
            row.grid[2] = seen;
        }
        if let Some(Wedge { apex, sides: [first, last] }) = common.seen {
            self.rows.push(Grow {
                origin: [apex.x, apex.y, first.x, first.y],
                eye: [last.x, last.y, common.tallest + CELL_MARGIN, 0.0],
                grid: [0; 4],
                counts: [0; 4],
                room: [0; 4],
                detail: [0.0; 4],
                tier: [0.0; 4],
                terms: [0.0; 4],
                least: [0.0; 4],
                wind: [0.0; 4],
            });
        }
        true
    }

    fn grow_terrain(
        &mut self,
        terrain: &TerrainEntry,
        [at, base, slot]: [u32; 3],
        (law, density, stand): (CoverLaw, f32, f32),
        main: &ViewFacts,
        common: Common,
    ) {
        let spacing = terrain.pyramid.spacing(0);
        let side = prime_from((density * spacing * spacing).sqrt().ceil() as u32);
        let held = (side * side) as f32;
        let mut coarsest = 1.0;
        let mut depth = 0_u32;
        while coarsest * coarsest * held < LANE_FILL * LANES {
            coarsest *= 2.0;
            depth += 1;
        }
        let center = Vec2::new(main.detail_eye.x, main.detail_eye.z);
        let corner = Vec2::new(terrain.origin.x, terrain.origin.z);
        let margin = common.tallest + CELL_MARGIN;
        let [elapsed, before, blown, _] = common.wind;
        let wind = [elapsed, before, blown, std::f32::consts::SQRT_2 * stand];
        let reach = law.reach();
        let measure = common.measure;
        self.reach = max(self.reach, reach);
        let mut tier = (0.0, min(measure.reach(law.widest), reach), coarsest);
        let mut tiers = 0;
        loop {
            let (inner, outer, scale) = tier;
            let tried = if inner > 0.0 {
                min(held * law.share(inner), held)
            } else {
                held
            };
            let [narrowest, lowest, finest] = law.least;
            let cell = spacing * scale;
            let around = [center - Vec2::splat(outer), center + Vec2::splat(outer)];
            let shown = common
                .seen
                .map_or(Some(around), |wedge| wedge.bounds(around, margin))
                .map(|[low, high]| {
                    let first = ((low - corner) / cell).floor().as_ivec2().max(IVec2::ZERO);
                    let last = ((high - corner) / cell).floor().as_ivec2();
                    (first, last - first + IVec2::ONE)
                })
                .filter(|(_, cells)| cells.min_element() > 0);
            if let Some((first, cells)) = shown {
                self.rows.push(Grow {
                    origin: terrain.origin.extend(spacing).to_array(),
                    eye: main.detail_eye.extend(outer).to_array(),
                    grid: [first.x, first.y, 0, side as i32],
                    counts: [at, terrain.pyramid.levels(), base, slot],
                    room: [0; 4],
                    detail: [measure.pixel_angle, measure.texel, measure.pixel_error, 1.0],
                    tier: [inner, scale, tried, common.tallest],
                    terms: [
                        law.widest,
                        law.tallest,
                        common.grounded,
                        (depth + tiers) as f32,
                    ],
                    least: [narrowest, lowest, finest, CELL_MARGIN],
                    wind,
                });
                let [across, along] = cells.as_uvec2().to_array();
                self.groups = self.groups.saturating_add(across.saturating_mul(along));
                self.cells.push([across, along, tiers]);
            }
            tiers += 1;
            if outer >= reach {
                break;
            }
            tier = (outer, min(2.0 * outer, reach), 2.0 * scale);
        }
        self.tiers = self.tiers.max(tiers);
    }
}

fn prime_from(start: u32) -> u32 {
    (start.max(2)..)
        .find(|&value| (2..value).take_while(|d| d * d <= value).all(|d| value % d != 0))
        .unwrap_or(start)
}

fn turn_share(culling: Mat4) -> f32 {
    let across = culling.row(0).truncate().length();
    let depth = culling.row(3).truncate().length();
    if across <= 0.0 {
        return 1.0;
    }
    min(engine_math::atan(depth / across) / std::f32::consts::PI, 1.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CoverLaw {
    measure: ViewDetail,
    widest: f32,
    tallest: f32,
    least: [f32; 3],
}

impl CoverLaw {
    fn reach(self) -> f32 {
        self.measure.reach(2.0 * self.tallest)
    }

    fn share(self, away: f32) -> f32 {
        let measure = self.measure;
        let error = measure.across(measure.eye + Vec3::X * away) * measure.pixel_error;
        min(self.widest / error, 1.0) * seen_share(self.tallest / error)
    }

    fn area(self) -> f32 {
        const RINGS: u32 = 256;
        let width = self.reach() / RINGS as f32;
        (0..RINGS)
            .map(|ring| {
                let away = (ring as f32 + 0.5) * width;
                std::f32::consts::TAU * away * width * self.share(away)
            })
            .sum()
    }
}
