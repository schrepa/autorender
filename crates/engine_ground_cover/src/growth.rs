use bevy_ecs::prelude::*;
use engine_core::glam::{Mat4, Vec3};
use engine_core::{Cut, DeviceLimits, FrameTime, Readback, View};
use engine_places::ground::{Foliage, GroundTiles, TerrainEntry, TerrainTiles};
use engine_places::wind::Wind;

use crate::covers::{CoverTable, CoverTerms};
use crate::layouts::{COVER_LEAVES, COVER_LUSH_RATIO, CoverBlade, CoverFrame, Grow};
use crate::settings::{GroundCover, GroundCoverStats};

pub(crate) const BLADE_BYTES: u64 = size_of::<CoverBlade>() as u64;

const APPENDED_WORD: usize = 0;
const CAPACITY_WORD: usize = 4;

const MAX_TIERS: usize = 8;

#[derive(Resource, Default)]
pub(crate) struct CoverPlan {
    pub(crate) rows: Vec<Grow>,
    pub(crate) cells: Vec<u32>,
    pub(crate) frame: CoverFrame,
    pub(crate) capacity: u32,
    pub(crate) room: u64,
    pub(crate) shown: bool,
    pub(crate) drawn: bool,
    reach: f32,
    tiers: u32,
    previous_time: Option<f32>,
}

#[derive(Resource, Default)]
pub(crate) struct BladeCounts {
    latest: Option<Cut>,
}

impl Readback for BladeCounts {
    fn arrived(&mut self, _: u64, _: u32, bytes: &[u8]) {
        self.latest = Some(Cut::of(
            word(bytes, APPENDED_WORD),
            word(bytes, CAPACITY_WORD),
        ));
    }
}

fn word(bytes: &[u8], index: usize) -> u32 {
    bytes.get(index * 4..index * 4 + 4).map_or(0, |bytes| {
        u32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]))
    })
}

#[derive(Clone, Copy)]
struct Sight {
    culling: Mat4,
    eye: Vec3,
    height: f32,
}

#[derive(Clone, Copy)]
struct Common {
    planes: [[f32; 4]; 6],
    pixels_per_meter: f32,
    density: f32,
    tallest: f32,
    grounded: bool,
}

pub(crate) fn plan_cover(
    (cover, table, tiles, ground): (
        Res<GroundCover>,
        Res<CoverTable>,
        Res<TerrainTiles>,
        Res<GroundTiles>,
    ),
    (view, time, limits): (Res<View>, Res<FrameTime>, Res<DeviceLimits>),
    (wind, foliage): (Res<Wind>, Res<Foliage>),
    mut plan: ResMut<CoverPlan>,
    mut counts: ResMut<BladeCounts>,
) {
    let plan = &mut *plan;
    plan.shown = cover.enabled && cover.density > 0.0;
    plan.room = limits.storage_room / 16 * 16;
    plan.drawn = !ground.tables.is_empty();
    if !plan.drawn {
        return;
    }
    plan.rows.clear();
    plan.cells.clear();
    plan.reach = 0.0;
    plan.tiers = 0;
    let time = time.elapsed_seconds as f32;
    let previous = plan.previous_time.replace(time).unwrap_or(time);
    let grounded = foliage.ground_blend > 0.0;
    let sight = Sight {
        culling: view.view_projection(),
        eye: view.detail_eye(),
        height: view.display_size().y as f32,
    };
    if !plan.grow(&cover, &table, (&tiles, &ground), sight, grounded) {
        counts.latest = Some(Cut::default());
        return;
    }
    plan.frame = CoverFrame {
        wind: [wind.velocity[0], wind.velocity[1], time, previous],
        ground: [f32::from(u8::from(grounded)), 0.0, 0.0, 0.0],
        blades: [plan.capacity, 0, 0, 0],
    };
}

pub(crate) fn cover_stats(
    counts: Res<BladeCounts>,
    plan: Res<CoverPlan>,
    mut stats: ResMut<GroundCoverStats>,
) {
    stats.set_if_neq(GroundCoverStats {
        blade_cut: counts.latest.unwrap_or_default(),
        reach: plan.reach.round() as u32,
        tiers: plan.tiers,
    });
}

impl CoverPlan {
    fn grow(
        &mut self,
        cover: &GroundCover,
        table: &CoverTable,
        (tiles, ground): (&TerrainTiles, &GroundTiles),
        sight: Sight,
        grounded: bool,
    ) -> bool {
        if !cover.enabled || cover.density <= 0.0 {
            return false;
        }
        let common = Common {
            planes: planes(sight.culling),
            pixels_per_meter: pixels_per_meter(sight.culling, sight.height),
            density: cover.density,
            tallest: table.tallest * lushest(),
            grounded,
        };
        let mut most = 0.0;
        for (index, terrain) in tiles.terrains.iter().enumerate() {
            let terms = table.terms(index);
            let density = terms.densest * cover.density * lushest();
            let law = CoverLaw::of(terms, common.pixels_per_meter);
            let Some(at) = ground
                .tables
                .iter()
                .find(|(entity, _)| *entity == terrain.terrain)
                .map(|&(_, at)| at)
            else {
                continue;
            };
            if density <= 0.0 || law.span <= 0.0 || terrain.surroundings {
                continue;
            }
            self.grow_terrain(terrain, (at, table.base(index)), (law, density), sight, common);
            most += law.area() * density;
        }
        if self.rows.is_empty() {
            return false;
        }
        let room = (self.room / BLADE_BYTES).min(u64::from(u32::MAX)) as u32;
        let seen = (most * turn_share(sight.culling)).ceil() as u32;
        let capacity = cover
            .max_blades
            .map_or(seen, |budget| budget.min(seen))
            .max(1)
            .min(room);
        self.capacity = capacity;
        for row in &mut self.rows {
            row.counts[3] = capacity;
        }
        true
    }

    fn grow_terrain(
        &mut self,
        terrain: &TerrainEntry,
        (at, base): (u32, u32),
        (law, density): (CoverLaw, f32),
        sight: Sight,
        common: Common,
    ) {
        let spacing = terrain.pyramid.spacing(0);
        let candidates = engine_core::math::max((density * spacing * spacing).ceil(), 1.0);
        let local = sight.eye - terrain.origin;
        let (tiers, count) = law.tiers();
        self.reach = engine_core::math::max(self.reach, tiers[count - 1].1);
        self.tiers = self.tiers.max(count as u32);
        for &(inner, outer, scale) in &tiers[..count] {
            let tried = if scale > 1.0 {
                engine_core::math::min((candidates * law.share(inner)).ceil(), candidates)
            } else {
                candidates
            };
            let cell = spacing * scale;
            let first = [
                ((local.x - outer) / cell).floor() as i32,
                ((local.z - outer) / cell).floor() as i32,
            ];
            let cells = (2.0 * outer / cell).ceil() as u32 + 1;
            self.rows.push(Grow {
                origin: terrain.origin.extend(spacing).to_array(),
                eye: sight.eye.extend(outer).to_array(),
                grid: [first[0], first[1], cells as i32, candidates as i32],
                counts: [at, terrain.pyramid.levels(), base, 0],
                scale: [common.density, tried, 0.0, common.tallest],
                tier: [inner, law.span, scale, 0.0],
                planes: common.planes,
                foliage: [f32::from(u8::from(common.grounded)), 0.0, 0.0, 0.0],
                view: [common.pixels_per_meter, 0.0, 0.0, 0.0],
            });
            self.cells.push(cells);
        }
    }
}

fn turn_share(culling: Mat4) -> f32 {
    let across = culling.row(0).truncate().length();
    let depth = culling.row(3).truncate().length();
    if across <= 0.0 {
        return 1.0;
    }
    engine_core::math::min(
        engine_math::atan(depth / across) / std::f32::consts::PI,
        1.0,
    )
}

fn lushest() -> f32 {
    COVER_LUSH_RATIO.sqrt()
}

fn planes(view_projection: Mat4) -> [[f32; 4]; 6] {
    let rows = [
        view_projection.row(0),
        view_projection.row(1),
        view_projection.row(2),
        view_projection.row(3),
    ];
    [
        rows[3] + rows[0],
        rows[3] - rows[0],
        rows[3] + rows[1],
        rows[3] - rows[1],
        rows[2],
        rows[3] - rows[2],
    ]
    .map(|plane: engine_core::glam::Vec4| {
        (plane / engine_core::math::max(plane.truncate().length(), 1.0e-12)).to_array()
    })
}

fn pixels_per_meter(culling: Mat4, height: f32) -> f32 {
    culling.row(1).truncate().length() * height * 0.5
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CoverLaw {
    span: f32,
}

impl CoverLaw {
    fn of(terms: CoverTerms, pixels_per_meter: f32) -> Self {
        Self {
            span: terms.span(pixels_per_meter),
        }
    }

    fn reach(self) -> f32 {
        2.0 * COVER_LEAVES as f32 * self.span
    }

    fn tiers(self) -> ([(f32, f32, f32); MAX_TIERS], usize) {
        let reach = self.reach();
        let mut tiers = [(0.0, self.span, 1.0); MAX_TIERS];
        let mut count = 1;
        while count < MAX_TIERS {
            let (_, outer, scale) = tiers[count - 1];
            if outer >= reach {
                break;
            }
            tiers[count] = (
                outer,
                engine_core::math::min(2.0 * outer, reach),
                2.0 * scale,
            );
            count += 1;
        }
        (tiers, count)
    }

    fn widening(self, away: f32) -> f32 {
        let blades = away / engine_core::math::max(self.span, 1.0e-6);
        engine_core::math::clamp(blades, 1.0, COVER_LEAVES as f32)
    }

    fn share(self, away: f32) -> f32 {
        let widening = self.widening(away);
        let spanned = self.span * widening / engine_core::math::max(away, 1.0e-3);
        seen_share(spanned) / widening
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

fn seen_share(pixels: f32) -> f32 {
    let octaves =
        engine_core::math::ln(engine_core::math::max(pixels, 1.0e-6)) * std::f32::consts::LOG2_E;
    engine_core::math::clamp(1.0 + octaves, 0.0, 1.0)
}
