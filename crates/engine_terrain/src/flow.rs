//! Rivers' flow as what stands in them holds it back, over the beds their
//! bake cut, which stay. A solid resting on a river's bed across part of its
//! width works as a weir: water passes the gap beside it as the river's own
//! flow would at that depth, and pours over it once it rises above its top,
//! as over a broad-crested weir (a discharge of [`WEIR`] × width ×
//! head^1.5). Behind it a pool fills by what flows in less what flows out,
//! rising until as much flows out as in; meanwhile the river below carries
//! only what leaves the pool, and once the solid is gone the pool drains
//! back into it, the river below running higher until it has. Upstream of a
//! pool the water stands held back against friction until the river's own
//! flow stands higher, into the rivers flowing into it too, and widens up
//! its banks as it rises. Rivers nothing holds back keep their bake's water
//! bit for bit.

use std::sync::Arc;

use engine_math::{max, min};
use glam::{Mat3, Vec2, Vec3};

use crate::hydraulics::{SHALLOWEST, TRICKLE, distances, foam_along, friction_slope};
use crate::rivers::{BANK_REACH, BANK_SLOPE, RIVER_STEP, River, Rivers};

/// The discharge coefficient of water pouring over a broad-crested weir
/// (meters^0.5 a second).
pub const WEIR: f32 = 1.7;
/// The share of a point's depth above its bed a solid's underside may stand
/// at and still rest on the bed; solids higher float over it, unless they
/// stand on one resting there (within [`STACKED`] meters of its top).
const RESTING: f32 = 0.35;
const STACKED: f32 = 0.1;
/// Meters the water must move anywhere before readers see it move: each
/// move makes the water of the tiles it reaches again.
const MOVED: f32 = 0.05;
/// Rounds settling held water's level a point upstream at the depth it
/// stands there.
const HELD_ROUNDS: usize = 6;
/// Meters above the river's own flow a pool no solid holds back still
/// stands before it has drained.
const DRAINED: f32 = 1.0e-3;
/// The most meters above its bed a pool may rise.
const DEEPEST: f32 = 40.0;
/// Foam pouring over a solid whitens the water this many points below it.
const SPILL: usize = 3;

/// A solid in a river's water: its river and point, the meters across the
/// river it spans (from and to, from the middle line, positive to the
/// river's right looking downstream), and its underside and top
/// (terrain-local meters).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block {
    pub river: u32,
    pub point: u32,
    pub from: f32,
    pub to: f32,
    pub bottom: f32,
    pub crest: f32,
}

/// The water held back at a river's point: its river and point, its level
/// (terrain-local meters), and what flowed into it and out of it at the
/// last step (cubic meters a second).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pool {
    pub river: u32,
    pub point: u32,
    pub level: f32,
    pub inflow: f32,
    pub outflow: f32,
}

/// Which river each river flows into (at its first point), the rivers
/// flowing into each, and every river after those flowing into it.
#[derive(Clone, Debug, Default)]
struct Network {
    into: Vec<Option<usize>>,
    sources: Vec<Vec<usize>>,
    order: Vec<usize>,
}

impl Network {
    fn new(rivers: &[River]) -> Self {
        let count = rivers.len();
        let mut into = vec![None; count];
        for (index, river) in rivers.iter().enumerate() {
            let (Some(&end), true) = (river.points.last(), river.points.len() >= 2) else {
                continue;
            };
            let end = Vec2::from(end);
            let mut nearest: Option<(f32, usize)> = None;
            for (other, next) in rivers.iter().enumerate() {
                if other == index || next.points.len() < 2 {
                    continue;
                }
                let distance = end.distance(Vec2::from(next.points[0]));
                if distance < 1.5 * RIVER_STEP && nearest.is_none_or(|(best, _)| distance < best) {
                    nearest = Some((distance, other));
                }
            }
            into[index] = nearest.map(|(_, other)| other);
        }
        let mut sources = vec![Vec::new(); count];
        let mut waiting = vec![0_usize; count];
        for (index, next) in into.iter().enumerate() {
            if let Some(next) = *next {
                sources[next].push(index);
                waiting[next] += 1;
            }
        }
        let mut order = Vec::with_capacity(count);
        let mut ready: Vec<usize> = (0..count)
            .rev()
            .filter(|&index| waiting[index] == 0)
            .collect();
        while let Some(index) = ready.pop() {
            order.push(index);
            if let Some(next) = into[index] {
                waiting[next] -= 1;
                if waiting[next] == 0 {
                    ready.push(next);
                }
            }
        }
        // A loop no river leaves, which no bake makes: its rivers last.
        if order.len() < count {
            let mut placed = vec![false; count];
            for &index in &order {
                placed[index] = true;
            }
            order.extend((0..count).filter(|&index| !placed[index]));
        }
        Self {
            into,
            sources,
            order,
        }
    }
}

/// Half a river's width where its water stands `rise` meters above its
/// bake's, up its banks.
fn widened(half: f32, rise: f32) -> f32 {
    half + min(max(rise / BANK_SLOPE, 0.0), BANK_REACH)
}

/// A terrain's rivers' flow at runtime: the bake's rivers, the water as
/// what stands in them now holds it back, and the pools behind them.
#[derive(Clone, Debug)]
pub struct RiverFlow {
    base: Arc<Rivers>,
    current: Arc<Rivers>,
    network: Network,
    beds: Vec<Vec<f32>>,
    along: Vec<Vec<f32>>,
    roughness: f32,
    blocks: Vec<Block>,
    pools: Vec<Pool>,
    // The water's level at each point of each river as the last step left
    // it, `None` where it ran as baked, and the discharge there: what each
    // pool's tailwater reads.
    last: Vec<Option<Vec<f32>>>,
    flows: Discharges,
    // The rivers whose water readers see differs from the bake's.
    shown: Vec<usize>,
}

/// The discharge at each point of each river pools change, `None` for those
/// carrying the bake's.
type Discharges = Vec<Option<Vec<f32>>>;

/// River `river`'s discharge at `point` in `flows`, or the bake's `line`.
fn carried(flows: &Discharges, river: usize, line: &River, point: usize) -> f32 {
    flows
        .get(river)
        .and_then(Option::as_ref)
        .map_or(line.discharge[point], |flows| flows[point])
}

/// Meters a pool must stand above the water just below it to hold its own:
/// lower, the water below has drowned it, and it passes on what reaches it.
const DROWNED: f32 = 1.0e-3;
/// The least share of a river's width solids must block to hold a pool
/// behind them: a lone log or stone in a wide river only parts it.
const JAM: f32 = 0.3;

impl RiverFlow {
    /// The flow of `base`, a bake's rivers with beds of Manning's
    /// `roughness`, as nothing holds it back.
    pub fn new(base: Arc<Rivers>, roughness: f32) -> Self {
        let rivers = base.rivers();
        Self {
            network: Network::new(rivers),
            beds: rivers
                .iter()
                .map(|river| {
                    river
                        .surface
                        .iter()
                        .zip(&river.depth)
                        .map(|(surface, depth)| surface - depth)
                        .collect()
                })
                .collect(),
            along: rivers
                .iter()
                .map(|river| {
                    let points: Vec<Vec2> = river
                        .points
                        .iter()
                        .map(|&point| Vec2::from(point))
                        .collect();
                    distances(&points)
                })
                .collect(),
            roughness,
            blocks: Vec::new(),
            pools: Vec::new(),
            last: Vec::new(),
            flows: Vec::new(),
            shown: Vec::new(),
            current: Arc::clone(&base),
            base,
        }
    }

    /// The bake's rivers.
    pub fn base(&self) -> &Arc<Rivers> {
        &self.base
    }

    /// The rivers' water now: the bake's where nothing holds it back.
    pub fn current(&self) -> &Arc<Rivers> {
        &self.current
    }

    /// The solids resting on the rivers' beds at the last step.
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// The water held back, by river and point.
    pub fn pools(&self) -> &[Pool] {
        &self.pools
    }

    /// What holds `pool` back: the share of its river's width the solids
    /// resting there block, the highest crest among them (terrain-local),
    /// and the water surface the pool covers (square meters).
    pub fn holding(&self, pool: &Pool) -> (f32, Option<f32>, f32) {
        let spans = self.spans(pool.river, pool.point);
        let line = &self.base.rivers()[pool.river as usize];
        let width = max(2.0 * line.half[pool.point as usize], 0.2);
        let blocked: f32 = spans.iter().map(|(span, _)| span).sum();
        let crest = spans.iter().map(|&(_, crest)| crest).reduce(max);
        (
            min(blocked / width, 1.0),
            crest,
            self.area(pool.river as usize, pool.point as usize, pool.level, &[]),
        )
    }

    /// Whether nothing holds the rivers back, nor has since they last ran
    /// as baked.
    pub fn still(&self) -> bool {
        self.pools.is_empty() && Arc::ptr_eq(&self.base, &self.current)
    }

    /// The solid about terrain-local `center` whose axes `axes` map the unit
    /// cube (or ball, when `round`) to it, as it stands in a river: `None`
    /// unless it reaches into a river's water across some of its width.
    pub fn block(&self, center: Vec3, axes: Mat3, round: bool) -> Option<Block> {
        let station = self.base.station(center.x, center.z)?;
        let (river, point) = (station.river as usize, station.point as usize);
        let line = &self.base.rivers()[river];
        let bed = self.beds[river][point];
        if !bed.is_finite() {
            return None;
        }
        let columns = [axes.x_axis, axes.y_axis, axes.z_axis];
        let side = Vec3::new(-station.direction[1], 0.0, station.direction[0]);
        let (rise, reach) = if round {
            (
                Vec3::new(columns[0].y, columns[1].y, columns[2].y).length(),
                Vec3::new(
                    columns[0].dot(side),
                    columns[1].dot(side),
                    columns[2].dot(side),
                )
                .length(),
            )
        } else {
            (
                columns.iter().map(|column| column.y.abs()).sum(),
                columns.iter().map(|column| column.dot(side).abs()).sum(),
            )
        };
        let (bottom, top) = (center.y - rise, center.y + rise);
        let surface = self.current.rivers()[river].surface[point];
        if !(bottom < surface && top > bed) {
            return None;
        }
        let half = line.half[point];
        let from = max(station.across - reach, -half);
        let to = min(station.across + reach, half);
        (to > from).then_some(Block {
            river: station.river,
            point: station.point,
            from,
            to,
            bottom,
            crest: top,
        })
    }

    /// Advances the flow `seconds` with `blocks` standing in the rivers:
    /// pools fill and drain and the rivers' water follows. True when the
    /// water readers see moved.
    pub fn step(&mut self, blocks: &[Block], seconds: f32) -> bool {
        self.blocks = blocks.to_vec();
        self.blocks.sort_by(|a, b| {
            (a.river, a.point)
                .cmp(&(b.river, b.point))
                .then(a.from.total_cmp(&b.from))
        });
        // A pool behind every point solids hold back, rising from the
        // river's own water there.
        for index in 0..self.blocks.len() {
            let block = self.blocks[index];
            let held = |pool: &Pool| pool.river == block.river && pool.point == block.point;
            if !self.pools.iter().any(held) && self.jammed(block.river, block.point) {
                let level = self.base.rivers()[block.river as usize].surface[block.point as usize];
                self.pools.push(Pool {
                    river: block.river,
                    point: block.point,
                    level,
                    inflow: 0.0,
                    outflow: 0.0,
                });
            }
        }
        self.pools.sort_by_key(|pool| (pool.river, pool.point));
        if self.pools.is_empty() {
            if Arc::ptr_eq(&self.base, &self.current) {
                return false;
            }
            self.current = Arc::clone(&self.base);
            self.shown.clear();
            return true;
        }
        let drowned: Vec<bool> = self
            .pools
            .iter()
            .map(|pool| {
                let (river, point) = (pool.river as usize, pool.point as usize);
                pool.level <= self.tail(river, point) + DROWNED || self.backed(river, point)
            })
            .collect();
        let flows = self.discharges(&drowned);
        self.fill(seconds, &drowned);
        let rivers = self.levels(&flows);
        self.last = rivers
            .iter()
            .map(|river| river.as_ref().map(|river| river.surface.clone()))
            .collect();
        self.flows = flows;
        self.publish(rivers)
    }

    /// The river and point just below river `river`'s point `point`: its
    /// next point, or where it joins the next river.
    fn next(&self, river: usize, point: usize) -> (usize, usize) {
        if point + 1 < self.base.rivers()[river].points.len() {
            (river, point + 1)
        } else if let Some(next) = self.network.into[river] {
            (next, 0)
        } else {
            (river, point)
        }
    }

    /// The water's level just below river `river`'s point `point` as the
    /// last step left it; below a fall, none (negative infinity).
    fn tail(&self, river: usize, point: usize) -> f32 {
        let (river, point) = self.next(river, point);
        let level = self
            .last
            .get(river)
            .and_then(Option::as_ref)
            .map_or(self.base.rivers()[river].surface[point], |levels| {
                levels[point]
            });
        if level.is_finite() {
            level
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Whether the water just below river `river`'s point `point` stands
    /// above the river's own flow there (with the discharge the last step
    /// carried), held back by what stands further down: a pool there stands
    /// in the pool below, which holds its water.
    fn backed(&self, river: usize, point: usize) -> bool {
        let tail = self.tail(river, point);
        let (river, point) = self.next(river, point);
        let line = &self.base.rivers()[river];
        let bed = self.beds[river][point];
        if !(tail.is_finite() && bed.is_finite()) {
            return false;
        }
        let discharge = carried(&self.flows, river, line, point);
        let own = bed
            + max(
                line.depth[point]
                    * libm::powf(discharge / max(line.discharge[point], TRICKLE), 0.6),
                SHALLOWEST,
            );
        tail > own + DROWNED
    }

    /// Whether solids at river `river`'s point `point` block enough of it to
    /// hold a pool behind them.
    fn jammed(&self, river: u32, point: u32) -> bool {
        let line = &self.base.rivers()[river as usize];
        let width = max(2.0 * line.half[point as usize], 0.2);
        let blocked: f32 = self.spans(river, point).iter().map(|(span, _)| span).sum();
        blocked >= JAM * width
    }

    /// What holds river `river`'s point `point` back: the meters of its
    /// width each span of solids resting on its bed covers, with its crest,
    /// raised by the solids stacked on them, overlapping spans merged under
    /// the higher crest.
    fn spans(&self, river: u32, point: u32) -> Vec<(f32, f32)> {
        let here: Vec<&Block> = self
            .blocks
            .iter()
            .filter(|block| block.river == river && block.point == point)
            .collect();
        let line = &self.base.rivers()[river as usize];
        let at = point as usize;
        let resting = self.beds[river as usize][at] + RESTING * max(line.depth[at], SHALLOWEST);
        let mut spans: Vec<(f32, f32, f32)> = Vec::new();
        for block in here.iter().filter(|block| block.bottom <= resting) {
            let mut crest = block.crest;
            loop {
                let raised = here
                    .iter()
                    .filter(|other| {
                        other.from < block.to
                            && other.to > block.from
                            && other.bottom <= crest + STACKED
                    })
                    .map(|other| other.crest)
                    .fold(crest, max);
                if raised <= crest {
                    break;
                }
                crest = raised;
            }
            spans.push((block.from, block.to, crest));
        }
        spans.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged: Vec<(f32, f32, f32)> = Vec::new();
        for span in spans {
            match merged.last_mut() {
                Some(last) if span.0 <= last.1 => {
                    last.1 = max(last.1, span.1);
                    last.2 = max(last.2, span.2);
                }
                _ => merged.push(span),
            }
        }
        merged
            .into_iter()
            .map(|(from, to, crest)| (to - from, crest))
            .collect()
    }

    /// What leaves a pool at `level` behind river `river`'s point `point`
    /// with `inflow` arriving: what passes the gap beside the solids there
    /// as the river's own flow would at that depth, and what pours over them,
    /// never more than the river would carry at that depth without them.
    fn outflow(&self, (river, point): (u32, u32), level: f32, inflow: f32) -> f32 {
        let line = &self.base.rivers()[river as usize];
        let at = point as usize;
        let bed = self.beds[river as usize][at];
        let width = max(2.0 * line.half[at], 0.2);
        let own = max(
            line.depth[at] * libm::powf(inflow / max(line.discharge[at], TRICKLE), 0.6),
            SHALLOWEST,
        );
        let spans = self.spans(river, point);
        let blocked: f32 = spans.iter().map(|(span, _)| span).sum();
        let open = max(width - blocked, 0.0);
        let depth = max(level - bed, 0.0);
        let unblocked = inflow * libm::powf(depth / own, 5.0 / 3.0);
        let gap = unblocked * (open / width);
        let over: f32 = spans
            .iter()
            .map(|&(span, crest)| WEIR * span * libm::powf(max(level - crest, 0.0), 1.5))
            .sum();
        min(gap + over, unblocked)
    }

    /// The discharge at each point of each river pools change: the bake's,
    /// less what pools upstream hold back and plus what they release, with
    /// each pool's inflow and outflow; pools the water below has drowned
    /// (`drowned`, by pool) pass on what reaches them. Rivers no pool's
    /// water reaches carry the bake's.
    fn discharges(&mut self, drowned: &[bool]) -> Discharges {
        let rivers = self.base.rivers();
        let mut flows: Discharges = vec![None; rivers.len()];
        let mut leaving = vec![0.0_f32; rivers.len()];
        for &river in &self.network.order {
            let mut change: f32 = self.network.sources[river]
                .iter()
                .map(|&source| leaving[source])
                .sum();
            let mut next = self
                .pools
                .partition_point(|pool| (pool.river as usize) < river);
            let pooled = self
                .pools
                .get(next)
                .is_some_and(|pool| pool.river as usize == river);
            if change == 0.0 && !pooled {
                continue;
            }
            let count = rivers[river].points.len();
            let mut changed = Vec::with_capacity(count);
            for point in 0..count {
                let discharge = max(rivers[river].discharge[point] + change, TRICKLE);
                changed.push(discharge);
                while next < self.pools.len()
                    && self.pools[next].river as usize == river
                    && self.pools[next].point as usize == point
                {
                    let key = (self.pools[next].river, self.pools[next].point);
                    let level = self.pools[next].level;
                    let outflow = if drowned[next] {
                        discharge
                    } else {
                        self.outflow(key, level, discharge)
                    };
                    let pool = &mut self.pools[next];
                    pool.inflow = discharge;
                    pool.outflow = outflow;
                    change += outflow - discharge;
                    next += 1;
                }
            }
            leaving[river] = change;
            flows[river] = Some(changed);
        }
        flows
    }

    /// The water surface a pool at `level` behind river `river`'s point
    /// `point` covers, up its river and the rivers flowing into it, as far
    /// as a pool upstream holding its own (one of `holding`).
    fn area(&self, river: usize, point: usize, level: f32, holding: &[(u32, u32)]) -> f32 {
        let rivers = self.base.rivers();
        let mut total = 0.0;
        let start = (river, point);
        let mut walks = vec![(river, point)];
        while let Some((river, mut point)) = walks.pop() {
            let line = &rivers[river];
            loop {
                let own = line.surface[point];
                let held =
                    (river, point) != start && holding.contains(&(river as u32, point as u32));
                if !own.is_finite() || level <= own || held {
                    break;
                }
                let run = if point + 1 < line.points.len() {
                    self.along[river][point + 1] - self.along[river][point]
                } else {
                    RIVER_STEP
                };
                total += 2.0 * widened(line.half[point], level - own) * run;
                if point == 0 {
                    for &source in &self.network.sources[river] {
                        let last = rivers[source].points.len().saturating_sub(1);
                        walks.push((source, last));
                    }
                    break;
                }
                point -= 1;
            }
        }
        let line = &rivers[river];
        max(total, 2.0 * max(line.half[point], 0.1) * RIVER_STEP)
    }

    /// Each pool rises by what flows in less what flows out over `seconds`,
    /// never past the level at which as much flows out as in nor below the
    /// river's own flow; pools no solid holds back drain away to it, and
    /// pools the water below has drowned (`drowned`) stand at its level.
    fn fill(&mut self, seconds: f32, drowned: &[bool]) {
        let mut pools = std::mem::take(&mut self.pools);
        let holding: Vec<(u32, u32)> = pools
            .iter()
            .zip(drowned)
            .filter(|(_, drowned)| !**drowned)
            .map(|(pool, _)| (pool.river, pool.point))
            .collect();
        let mut drained = Vec::new();
        for (pool, &drowned) in pools.iter_mut().zip(drowned) {
            let key = (pool.river, pool.point);
            let line = &self.base.rivers()[pool.river as usize];
            let at = pool.point as usize;
            let bed = self.beds[pool.river as usize][at];
            let own = bed
                + max(
                    line.depth[at]
                        * libm::powf(pool.inflow / max(line.discharge[at], TRICKLE), 0.6),
                    SHALLOWEST,
                );
            let blocked = self.jammed(pool.river, pool.point);
            if drowned {
                pool.level = max(own, self.tail(pool.river as usize, at));
                drained.push(!blocked);
                continue;
            }
            let area = self.area(pool.river as usize, at, pool.level, &holding);
            let rising = pool.inflow - pool.outflow;
            let mut level = pool.level + rising / area * seconds;
            // The level as much flows out as in at: past it the pool turns.
            let (mut low, mut high) = (bed, bed + DEEPEST);
            for _ in 0..40 {
                let middle = 0.5 * (low + high);
                if self.outflow(key, middle, pool.inflow) < pool.inflow {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            let balanced = 0.5 * (low + high);
            if (rising > 0.0 && level > balanced) || (rising < 0.0 && level < balanced) {
                level = balanced;
            }
            pool.level = min(max(level, own), bed + DEEPEST);
            drained.push(!blocked && pool.level <= own + DRAINED);
        }
        let mut drained = drained.into_iter();
        pools.retain(|_| !drained.next().unwrap_or(false));
        self.pools = pools;
    }

    /// The rivers' water with `flows` running and the pools standing,
    /// downstream first so each river meets the level of the river it joins:
    /// each river's water, or `None` where it runs as baked.
    fn levels(&self, flows: &Discharges) -> Vec<Option<River>> {
        let rivers = self.base.rivers();
        let mut out: Vec<Option<River>> = vec![None; rivers.len()];
        for &river in self.network.order.iter().rev() {
            let line = &rivers[river];
            let count = line.points.len();
            let outlet = self.network.into[river].and_then(|next| {
                let below = out[next].as_ref()?;
                let level = below.surface[0];
                (level > rivers[next].surface[0] + 1.0e-4).then_some(level)
            });
            let pooled = self.pools.iter().any(|pool| pool.river as usize == river);
            if outlet.is_none() && !pooled && flows[river].is_none() {
                continue;
            }
            let mut level = vec![f32::NAN; count];
            let mut held = outlet;
            for point in (0..count).rev() {
                let bed = self.beds[river][point];
                if !bed.is_finite() {
                    held = None;
                    continue;
                }
                let discharge = carried(flows, river, line, point);
                let own = if discharge == line.discharge[point] {
                    line.surface[point]
                } else {
                    bed + max(
                        line.depth[point]
                            * libm::powf(discharge / max(line.discharge[point], TRICKLE), 0.6),
                        SHALLOWEST,
                    )
                };
                let mut here = own;
                if let Some(held) = held
                    && held > here
                {
                    here = held;
                }
                for pool in self
                    .pools
                    .iter()
                    .filter(|pool| pool.river as usize == river && pool.point as usize == point)
                {
                    here = max(here, pool.level);
                }
                level[point] = here;
                // Held water reaches the point above while it stands over
                // that point's bed: there it stands higher by the friction
                // it meets on the way, at the depth it stands there (a few
                // rounds settle it).
                held = (here > own + 1.0e-4
                    && point > 0
                    && here > self.beds[river][point - 1] + SHALLOWEST)
                    .then(|| {
                        let above = point - 1;
                        let bed = self.beds[river][above];
                        let run = self.along[river][point] - self.along[river][above];
                        let discharge = carried(flows, river, line, above);
                        let mut level = here;
                        for _ in 0..HELD_ROUNDS {
                            let width =
                                2.0 * widened(line.half[above], level - line.surface[above]);
                            let depth = max(0.5 * (here + level) - bed, SHALLOWEST);
                            let loss =
                                friction_slope(discharge, width, depth, self.roughness) * run;
                            level = 0.5 * (level + here + loss);
                        }
                        level
                    });
            }
            let half: Vec<f32> = (0..count)
                .map(|point| {
                    if level[point].is_finite() {
                        widened(line.half[point], level[point] - line.surface[point])
                    } else {
                        line.half[point]
                    }
                })
                .collect();
            let depth: Vec<f32> = (0..count)
                .map(|point| {
                    if level[point].is_finite() {
                        max(level[point] - self.beds[river][point], SHALLOWEST)
                    } else {
                        line.depth[point]
                    }
                })
                .collect();
            let speed: Vec<f32> = (0..count)
                .map(|point| {
                    carried(flows, river, line, point)
                        / (2.0 * max(half[point], 0.1) * depth[point])
                })
                .collect();
            let mut foam = foam_along(&depth, &speed, (&self.along[river], &line.cut), &line.falls);
            self.whiten_spills(river, &mut foam);
            out[river] = Some(River {
                points: line.points.clone(),
                cut: cut_to(&level, &line.cut),
                surface: level,
                half,
                depth,
                channel: line.channel.clone(),
                fill: line.fill.clone(),
                speed,
                discharge: flows[river]
                    .clone()
                    .unwrap_or_else(|| line.discharge.clone()),
                foam,
                falls: line.falls.clone(),
                sheeted: line.sheeted.clone(),
                nappes: line.nappes.clone(),
                head: line.head,
            });
        }
        out
    }

    /// Whitens river `river`'s `foam` below its pools pouring over the solids
    /// holding them.
    fn whiten_spills(&self, river: usize, foam: &mut [f32]) {
        for pool in self
            .pools
            .iter()
            .filter(|pool| pool.river as usize == river)
        {
            let spans = self.spans(pool.river, pool.point);
            if spans.iter().any(|&(_, crest)| pool.level > crest) {
                let first = pool.point as usize + 1;
                for value in foam.iter_mut().skip(first).take(SPILL) {
                    *value = max(*value, 0.9);
                }
            }
        }
    }

    /// Hands readers `rivers` (the bake's where `None`) once the water moved
    /// far enough from what they read: whether it did.
    fn publish(&mut self, rivers: Vec<Option<River>>) -> bool {
        let shown = self.current.rivers();
        let held: Vec<usize> = (0..rivers.len())
            .filter(|&index| rivers[index].is_some())
            .collect();
        let moved = held.iter().chain(&self.shown).any(|&index| {
            let now = rivers[index].as_ref().unwrap_or(&self.base.rivers()[index]);
            let before = &shown[index];
            now.surface
                .iter()
                .zip(&before.surface)
                .chain(now.half.iter().zip(&before.half))
                .any(|(a, b)| (a.is_finite() != b.is_finite()) || (a - b).abs() > MOVED)
        });
        if !moved {
            return false;
        }
        self.shown = held;
        let whole: Vec<River> = rivers
            .into_iter()
            .enumerate()
            .map(|(index, river)| river.unwrap_or_else(|| self.base.rivers()[index].clone()))
            .collect();
        self.current = Arc::new(self.base.with_rivers(whole));
        true
    }
}

/// The levels a river's channel is cut to where its water stands at
/// `level`: that water, and beneath its jets (no water) as the bake cut it
/// (`baked`).
fn cut_to(level: &[f32], baked: &[f32]) -> Vec<f32> {
    level
        .iter()
        .zip(baked)
        .map(|(&level, &baked)| if level.is_finite() { level } else { baked })
        .collect()
}
