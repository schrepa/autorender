use std::sync::Arc;

use engine_math::{max, min};
use glam::{Mat3, Vec2, Vec3};

use crate::hydraulics::{SHALLOWEST, TRICKLE, distances, friction_slope};
use crate::rivers::{RIVER_STEP, River, Rivers};

pub const WEIR: f32 = 1.7;
const RESTING: f32 = 0.35;
const STACKED: f32 = 0.1;
const MOVED: f32 = 0.05;
const HELD_ROUNDS: usize = 6;
const DRAINED: f32 = 1.0e-3;
const DEEPEST: f32 = 40.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block {
    pub river: u32,
    pub point: u32,
    pub from: f32,
    pub to: f32,
    pub bottom: f32,
    pub crest: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pool {
    pub river: u32,
    pub point: u32,
    pub level: f32,
    pub inflow: f32,
    pub outflow: f32,
}

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

fn spread(line: &River, point: usize, level: f32) -> f32 {
    if level == line.surface[point] {
        return line.half[point];
    }
    let depth = max(line.depth[point], SHALLOWEST);
    line.half[point] * (max(level - line.bed[point], 0.0) / depth).sqrt()
}

fn own_depth(line: &River, point: usize, discharge: f32) -> f32 {
    max(
        line.depth[point] * libm::powf(discharge / max(line.discharge[point], TRICKLE), 0.6),
        SHALLOWEST,
    )
}

#[derive(Clone, Debug)]
pub struct RiverFlow {
    base: Arc<Rivers>,
    current: Arc<Rivers>,
    network: Network,
    along: Vec<Vec<f32>>,
    blocks: Vec<Block>,
    pools: Vec<Pool>,
    last: Vec<Option<Vec<f32>>>,
    flows: Discharges,
    shown: Vec<usize>,
}

type Discharges = Vec<Option<Vec<f32>>>;

fn carried(flows: &Discharges, river: usize, line: &River, point: usize) -> f32 {
    flows
        .get(river)
        .and_then(Option::as_ref)
        .map_or(line.discharge[point], |flows| flows[point])
}

const DROWNED: f32 = 1.0e-3;
const JAM: f32 = 0.3;

impl RiverFlow {
    pub fn new(base: Arc<Rivers>) -> Self {
        let rivers = base.rivers();
        Self {
            network: Network::new(rivers),
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
            blocks: Vec::new(),
            pools: Vec::new(),
            last: Vec::new(),
            flows: Vec::new(),
            shown: Vec::new(),
            current: Arc::clone(&base),
            base,
        }
    }

    pub fn base(&self) -> &Arc<Rivers> {
        &self.base
    }

    pub fn current(&self) -> &Arc<Rivers> {
        &self.current
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn pools(&self) -> &[Pool] {
        &self.pools
    }

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

    pub fn still(&self) -> bool {
        self.pools.is_empty() && Arc::ptr_eq(&self.base, &self.current)
    }

    pub fn block(&self, center: Vec3, axes: Mat3, round: bool) -> Option<Block> {
        let station = self.base.station(center.x, center.z)?;
        let (river, point) = (station.river as usize, station.point as usize);
        let line = &self.base.rivers()[river];
        let bed = line.bed[point];
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

    pub fn step(&mut self, blocks: &[Block], seconds: f32) -> bool {
        self.blocks = blocks.to_vec();
        self.blocks.sort_by(|a, b| {
            (a.river, a.point)
                .cmp(&(b.river, b.point))
                .then(a.from.total_cmp(&b.from))
        });
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

    fn next(&self, river: usize, point: usize) -> (usize, usize) {
        if point + 1 < self.base.rivers()[river].points.len() {
            (river, point + 1)
        } else if let Some(next) = self.network.into[river] {
            (next, 0)
        } else {
            (river, point)
        }
    }

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

    fn backed(&self, river: usize, point: usize) -> bool {
        let tail = self.tail(river, point);
        if !tail.is_finite() {
            return false;
        }
        let (river, point) = self.next(river, point);
        let line = &self.base.rivers()[river];
        let discharge = carried(&self.flows, river, line, point);
        tail > line.bed[point] + own_depth(line, point, discharge) + DROWNED
    }

    fn jammed(&self, river: u32, point: u32) -> bool {
        let line = &self.base.rivers()[river as usize];
        let width = max(2.0 * line.half[point as usize], 0.2);
        let blocked: f32 = self.spans(river, point).iter().map(|(span, _)| span).sum();
        blocked >= JAM * width
    }

    fn spans(&self, river: u32, point: u32) -> Vec<(f32, f32)> {
        let here: Vec<&Block> = self
            .blocks
            .iter()
            .filter(|block| block.river == river && block.point == point)
            .collect();
        let line = &self.base.rivers()[river as usize];
        let at = point as usize;
        let resting = line.bed[at] + RESTING * max(line.depth[at], SHALLOWEST);
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

    fn outflow(&self, (river, point): (u32, u32), level: f32, inflow: f32) -> f32 {
        let line = &self.base.rivers()[river as usize];
        let at = point as usize;
        let bed = line.bed[at];
        let width = max(2.0 * line.half[at], 0.2);
        let own = own_depth(line, at, inflow);
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
                total += 2.0 * spread(line, point, level) * run;
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
            let bed = line.bed[at];
            let own = bed + own_depth(line, at, pool.inflow);
            let blocked = self.jammed(pool.river, pool.point);
            if drowned {
                pool.level = max(own, self.tail(pool.river as usize, at));
                drained.push(!blocked);
                continue;
            }
            let area = self.area(pool.river as usize, at, pool.level, &holding);
            let rising = pool.inflow - pool.outflow;
            let mut level = pool.level + rising / area * seconds;
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

    fn levels(&self, flows: &Discharges) -> Vec<Option<River>> {
        let rivers = self.base.rivers();
        let mut out: Vec<Option<River>> = vec![None; rivers.len()];
        for &river in self.network.order.iter().rev() {
            let outlet = self.network.into[river].and_then(|next| {
                let below = out[next].as_ref()?;
                let level = below.surface[0];
                (level > rivers[next].surface[0] + 1.0e-4).then_some(level)
            });
            let pooled = self.pools.iter().any(|pool| pool.river as usize == river);
            if outlet.is_none() && !pooled && flows[river].is_none() {
                continue;
            }
            let level = self.held(river, outlet, flows);
            out[river] = Some(self.standing(river, level, flows));
        }
        out
    }

    fn held(&self, river: usize, outlet: Option<f32>, flows: &Discharges) -> Vec<f32> {
        let line = &self.base.rivers()[river];
        let count = line.points.len();
        let mut level = vec![f32::NAN; count];
        let mut held = outlet;
        for point in (0..count).rev() {
            if !line.surface[point].is_finite() {
                held = None;
                continue;
            }
            let discharge = carried(flows, river, line, point);
            let own = if discharge == line.discharge[point] {
                line.surface[point]
            } else {
                line.bed[point] + own_depth(line, point, discharge)
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
            held = (here > own + 1.0e-4 && point > 0 && here > line.bed[point - 1] + SHALLOWEST)
                .then(|| {
                    let above = point - 1;
                    let bed = line.bed[above];
                    let run = self.along[river][point] - self.along[river][above];
                    let discharge = carried(flows, river, line, above);
                    let mut level = here;
                    for _ in 0..HELD_ROUNDS {
                        let width = 2.0 * spread(line, above, level);
                        let depth = max(0.5 * (here + level) - bed, SHALLOWEST);
                        let loss = friction_slope(discharge, width, depth, line.d84[above]) * run;
                        level = 0.5 * (level + here + loss);
                    }
                    level
                });
        }
        level
    }

    fn standing(&self, river: usize, level: Vec<f32>, flows: &Discharges) -> River {
        let line = &self.base.rivers()[river];
        let (mut half, mut depth, mut speed) =
            (line.half.clone(), line.depth.clone(), line.speed.clone());
        for point in 0..line.points.len() {
            let discharge = carried(flows, river, line, point);
            let baked = level[point] == line.surface[point] && discharge == line.discharge[point];
            if baked || !level[point].is_finite() {
                continue;
            }
            half[point] = spread(line, point, level[point]);
            depth[point] = max(level[point] - line.bed[point], SHALLOWEST);
            speed[point] = discharge / (4.0 / 3.0 * max(half[point], 0.1) * depth[point]);
        }
        River {
            points: line.points.clone(),
            discharge: flows[river]
                .clone()
                .unwrap_or_else(|| line.discharge.clone()),
            half,
            depth,
            bed: line.bed.clone(),
            surface: level,
            speed,
            falls: line.falls.clone(),
            sheeted: line.sheeted.clone(),
            reach: line.reach.clone(),
            d50: line.d50.clone(),
            d84: line.d84.clone(),
            power: line.power.clone(),
            flood: line.flood.clone(),
            carved: line.carved.clone(),
            cut: line.cut.clone(),
            crest: line.crest.clone(),
            knobs: line.knobs.clone(),
            head: line.head,
            given: line.given,
        }
    }

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
