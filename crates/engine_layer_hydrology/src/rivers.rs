use std::f32::consts::{FRAC_PI_2, PI};

use engine_core::fields;
use engine_heightfield::bed::{keyed, place_key, unit_of};
use engine_heightfield::hydraulics::distances;
use engine_heightfield::hydrology::Grid;
use engine_heightfield::noise::smooth;
use engine_heightfield::rivers::{RIVER_STEP, RiverLine, bends, point_along};
use engine_math::collections::HashMap;
use engine_math::{clamp, max, min, sin};
use glam::Vec2;
use serde::{Deserialize, Serialize};

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct Meanders {
        pub wave: f32 = 11.0 => above_to(0.0, 1000.0),
        pub swing: f32 = 1.7 => range(0.0, 100.0),
        pub slopes: [f32; 2] = [0.01, 0.03] => range(0.0, 10.0),
        pub stretch: [f32; 2] = [0.9, 1.35] => above_to(0.0, 100.0),
        pub strength: [f32; 2] = [0.55, 1.0] => range(0.0, 1.0),
        pub ease: f32 = 0.25 => range(0.0, 10.0),
        pub straighten: f32 = 0.75 => range(0.0, 10.0),
        pub floor_rise: f32 = 0.8 => range(0.0, 100.0),
        pub floor_drop: f32 = 1.5 => range(0.0, 100.0),
        pub rival: f32 = 0.3 => range(0.0, 1.0),
        pub steeper: [f32; 2] = [2.0, 4.0] => above_to(0.0, 1000.0),
        pub clear: f32 = 4.0 => range(0.0, 100.0),
        pub plain_beside: f32 = 0.5 => range(0.0, 100.0),
        pub plain_most: f32 = 0.75 => range(0.0, 100.0),
        pub bar_chance: f32 = 0.4 => range(0.0, 1.0),
        pub bar_length: f32 = 1.75 => above_to(0.0, 100.0),
    }
}

engine_codec::record!(Meanders {
    wave,
    swing,
    slopes,
    stretch,
    strength,
    ease,
    straighten,
    floor_rise,
    floor_drop,
    rival,
    steeper,
    clear,
    plain_beside,
    plain_most,
    bar_chance,
    bar_length,
});

const END_SHARE: f32 = 0.3;
const SLOPE_WIDTHS: f32 = 3.0;
const HEADING_WIDTHS: f32 = 1.5;
const FOLD: f32 = 0.25;
const FOLD_PASSES: usize = 4;
const TRIM: f32 = 1.5;
const NEAR_CELL: f32 = 64.0;
const SWING_SEED: u32 = 0x6d2b_79f5;
const BAR_SEED: u32 = 0x9e37_79b1;

pub(crate) struct Channels<'a> {
    pub(crate) area: &'a [f32],
    pub(crate) receivers: &'a [u32],
    pub(crate) lakes: &'a [bool],
    pub(crate) rank: &'a [u32],
    pub(crate) within: Option<&'a [bool]>,
    pub(crate) continued: &'a [(u32, Vec2)],
    pub(crate) ground: Grid<'a>,
}

impl Channels<'_> {
    fn lead(&self, sample: usize) -> Option<Vec2> {
        self.continued
            .binary_search_by_key(&(sample as u32), |&(at, _)| at)
            .ok()
            .map(|found| self.continued[found].1)
    }

    fn place(&self, sample: usize) -> Vec2 {
        let (width, spacing) = (self.ground.width, self.ground.spacing);
        Vec2::new(
            (sample % width) as f32 * spacing,
            (sample / width) as f32 * spacing,
        )
    }
}

struct Course {
    cells: Vec<usize>,
    joins: bool,
    head: bool,
    outlet: bool,
}

pub(crate) struct Land<'a> {
    pub(crate) channels: &'a Channels<'a>,
    pub(crate) channel_area: f32,
    pub(crate) meanders: &'a Meanders,
    pub(crate) bend: f32,
    pub(crate) bars: f32,
    pub(crate) shortest: f32,
    pub(crate) lake: &'a dyn Fn(Vec2) -> Option<f32>,
    pub(crate) grade: &'a dyn Fn(f32) -> f32,
    pub(crate) bankfull: &'a dyn Fn(f32) -> (f32, f32),
}

#[derive(Default)]
struct Others {
    cells: HashMap<[i32; 2], Vec<(Vec2, f32)>>,
    widest: f32,
}

impl Others {
    fn cell(point: Vec2) -> [i32; 2] {
        [
            (point.x / NEAR_CELL).floor() as i32,
            (point.y / NEAR_CELL).floor() as i32,
        ]
    }

    fn add(&mut self, points: &[Vec2], widths: &[f32]) {
        for (&point, &width) in points.iter().zip(widths) {
            self.cells
                .entry(Self::cell(point))
                .or_default()
                .push((point, width));
            self.widest = max(self.widest, width);
        }
    }

    fn near(&self, point: Vec2, width: f32, clear: f32) -> bool {
        let reach = Vec2::splat(clear * max(width, self.widest));
        let (low, high) = (Self::cell(point - reach), Self::cell(point + reach));
        (low[1]..=high[1]).any(|row| {
            (low[0]..=high[0]).any(|column| {
                self.cells.get(&[column, row]).is_some_and(|held| {
                    held.iter().any(|&(other, wide)| {
                        let apart = clear * max(width, wide);
                        point.distance_squared(other) < apart * apart
                    })
                })
            })
        })
    }
}

type Valley = (Vec<Vec2>, Vec<f32>, Vec<usize>);

struct Shaped {
    points: Vec<Vec2>,
    area: Vec<f32>,
    level: Vec<f32>,
    marks: Vec<usize>,
    bend: Vec<f32>,
    plain: Vec<[f32; 2]>,
    bar: Vec<f32>,
}

fn courses(channels: &Channels<'_>, channel_area: f32) -> Vec<Course> {
    let Channels {
        area,
        receivers,
        lakes,
        rank,
        within,
        continued,
        ..
    } = *channels;
    let count = area.len();
    let given = |sample: usize| within.is_some_and(|within| within[sample]);
    let channel = |sample: usize| area[sample] >= channel_area && !lakes[sample] && !given(sample);
    let mut inflows = vec![0_u8; count];
    let mut spilled = vec![false; count];
    for (sample, &receiver) in receivers.iter().enumerate() {
        if receiver == u32::MAX {
            continue;
        }
        if channel(sample) {
            let entry = &mut inflows[receiver as usize];
            *entry = entry.saturating_add(1);
        } else if lakes[sample] && area[sample] >= channel_area {
            spilled[receiver as usize] = true;
        }
    }
    for &(sample, _) in continued {
        let entry = &mut inflows[sample as usize];
        *entry = entry.saturating_add(1);
    }
    let mut starts: Vec<usize> = (0..count)
        .filter(|&sample| {
            channel(sample) && (inflows[sample] != 1 || channels.lead(sample).is_some())
        })
        .collect();
    starts.sort_by_key(|&sample| std::cmp::Reverse(rank[sample]));
    let mut courses = Vec::with_capacity(starts.len());
    for start in starts {
        let mut cells = vec![start];
        let mut current = start;
        let mut joins = false;
        loop {
            let receiver = receivers[current];
            if receiver == u32::MAX {
                break;
            }
            let next = receiver as usize;
            cells.push(next);
            if lakes[next] || !channel(next) || inflows[next] >= 2 {
                joins = true;
                break;
            }
            current = next;
        }
        if cells.len() >= 2 {
            courses.push(Course {
                cells,
                joins,
                head: inflows[start] == 0,
                outlet: spilled[start],
            });
        }
    }
    courses
}

fn stems(channels: &Channels<'_>, courses: &[Course]) -> Vec<Vec<usize>> {
    let area = channels.area;
    let carried = |course: &Course| area[course.cells[course.cells.len() - 2]];
    let starting: HashMap<usize, usize> = courses
        .iter()
        .enumerate()
        .map(|(number, course)| (course.cells[0], number))
        .collect();
    let mut onward: Vec<Option<usize>> = vec![None; courses.len()];
    let mut fed = vec![0.0_f32; courses.len()];
    for (number, course) in courses.iter().enumerate() {
        let end = course.cells[course.cells.len() - 1];
        let Some(&next) = starting.get(&end) else {
            continue;
        };
        if course.joins && channels.lead(end).is_none() && carried(course) > fed[next] {
            fed[next] = carried(course);
            onward[number] = Some(next);
        }
    }
    let mut chained = vec![false; courses.len()];
    for (number, course) in courses.iter().enumerate() {
        match onward[number] {
            Some(next) if fed[next] == carried(course) && !chained[next] => chained[next] = true,
            Some(_) | None => onward[number] = None,
        }
    }
    let mut stems: Vec<(f32, Vec<usize>)> = (0..courses.len())
        .filter(|&number| !chained[number])
        .map(|first| {
            let mut stem = vec![first];
            while let Some(next) = onward[stem[stem.len() - 1]] {
                stem.push(next);
            }
            (carried(&courses[stem[stem.len() - 1]]), stem)
        })
        .collect();
    stems.sort_by(|a, b| b.0.total_cmp(&a.0));
    stems.into_iter().map(|(_, stem)| stem).collect()
}

fn valley(land: &Land<'_>, courses: &[Course], stem: &[usize], target: Option<Vec2>) -> Valley {
    let channels = land.channels;
    let mut points: Vec<Vec2> = Vec::new();
    let mut drained: Vec<f32> = Vec::new();
    let mut joints: Vec<usize> = Vec::with_capacity(stem.len());
    for (slot, &number) in stem.iter().enumerate() {
        let course = &courses[number];
        let last = course.cells.len() - 1;
        let drains = |index: usize| {
            let at = if course.joins && index == last {
                index - 1
            } else {
                index
            };
            channels.area[course.cells[at]]
        };
        if slot == 0 {
            if let Some(lead) = channels.lead(course.cells[0]) {
                points.push(lead);
                drained.push(drains(0));
            }
            points.push(channels.place(course.cells[0]));
            drained.push(drains(0));
        }
        let mut from = channels.place(course.cells[0]);
        for index in 1..last {
            let at = channels.place(course.cells[index]);
            if from.distance(at) >= (land.bankfull)(drains(index)).0 {
                points.push(at);
                drained.push(drains(index));
                from = at;
            }
        }
        points.push(channels.place(course.cells[last]));
        drained.push(drains(last));
        joints.push(points.len() - 1);
    }
    if let Some(target) = target {
        let old = points[points.len() - 1];
        let width = (land.bankfull)(drained[drained.len() - 1]).0;
        let clear = TRIM * target.distance(old) + width;
        let floor = if joints.len() >= 2 {
            joints[joints.len() - 2]
        } else {
            0
        };
        while points.len() >= floor + 3 && points[points.len() - 2].distance(old) < clear {
            points.remove(points.len() - 2);
            drained.remove(drained.len() - 2);
        }
        let last = points.len() - 1;
        points[last] = target;
    }
    smooth_marked(&points, &drained, &joints[..stem.len() - 1])
}

fn stub(land: &Land<'_>, course: &Course) -> bool {
    let channels = land.channels;
    let cells = &course.cells;
    let end = cells[cells.len() - 1];
    if course.outlet {
        let beside = |cell: usize| (land.lake)(channels.place(cell));
        return beside(cells[0]).is_some_and(|level| beside(end) == Some(level));
    }
    let length: f32 = cells
        .windows(2)
        .map(|pair| channels.place(pair[0]).distance(channels.place(pair[1])))
        .sum();
    let own = (land.bankfull)(channels.area[cells[cells.len() - 2]]).0;
    let joined = (land.bankfull)(channels.area[end]).0;
    length <= max(joined, land.shortest * own)
}

pub(crate) fn trace(land: &Land<'_>) -> Vec<RiverLine> {
    let channels = land.channels;
    let courses = courses(channels, land.channel_area);
    let mut moved: HashMap<usize, Vec2> = HashMap::default();
    let mut lines: Vec<Option<RiverLine>> = courses.iter().map(|_| None).collect();
    let mut others = Others::default();
    for stem in stems(channels, &courses) {
        let end = {
            let cells = &courses[stem[stem.len() - 1]].cells;
            cells[cells.len() - 1]
        };
        let first = &courses[stem[0]];
        if stem.len() == 1 && first.head && stub(land, first) {
            continue;
        }
        let valley = valley(land, &courses, &stem, moved.get(&end).copied());
        let key = place_key(channels.place(courses[stem[0]].cells[0]));
        let shaped = meander(land, valley, key, &others);
        let widths: Vec<f32> = shaped
            .area
            .iter()
            .map(|&drained| (land.bankfull)(drained).0)
            .collect();
        others.add(&shaped.points, &widths);
        let last = shaped.points.len() - 1;
        let mut from = 0;
        for (slot, &number) in stem.iter().enumerate() {
            let to = shaped.marks.get(slot).copied().unwrap_or(last);
            if slot + 1 < stem.len() {
                moved.insert(courses[stem[slot + 1]].cells[0], shaped.points[to]);
            }
            lines[number] = Some(RiverLine {
                points: shaped.points[from..=to].to_vec(),
                area: shaped.area[from..=to].to_vec(),
                level: shaped.level[from..=to].to_vec(),
                bend: shaped.bend[from..=to].to_vec(),
                plain: shaped.plain[from..=to].to_vec(),
                bar: shaped.bar[from..=to].to_vec(),
                head: courses[number].head,
                given: None,
            });
            from = to;
        }
    }
    lines.into_iter().flatten().collect()
}

fn mix([from, to]: [f32; 2], share: f32) -> f32 {
    from + (to - from) * share
}

fn eased(values: &[f32], along: &[f32], reach: impl Fn(usize) -> f32) -> Vec<f32> {
    let mut sums = Vec::with_capacity(values.len() + 1);
    sums.push(0.0_f64);
    for &value in values {
        sums.push(sums[sums.len() - 1] + f64::from(value));
    }
    (0..values.len())
        .map(|index| {
            let reach = reach(index);
            let first = along.partition_point(|&at| at < along[index] - reach);
            let end = along
                .partition_point(|&at| at <= along[index] + reach)
                .max(first + 1);
            ((sums[end] - sums[first]) / (end - first) as f64) as f32
        })
        .collect()
}

fn room(land: &Land<'_>, origin: Vec2, out: Vec2, channel: [f32; 3], reach: f32) -> f32 {
    let [width, depth, drained] = channel;
    let set = land.meanders;
    let Channels {
        area,
        lakes,
        ground,
        ..
    } = *land.channels;
    let (columns, rows) = (ground.width as f32, ground.rows() as f32);
    let level = ground.height(origin.x, origin.y);
    let step = max(ground.spacing, 0.25 * width);
    let rival = max(land.channel_area, set.rival * drained);
    let mut reached = 0.5 * width;
    let mut lateral = reached + step;
    while lateral <= reach {
        let at = origin + out * lateral;
        let (x, z) = (
            (at.x / ground.spacing).round(),
            (at.y / ground.spacing).round(),
        );
        if !(x >= 0.0 && z >= 0.0 && x < columns && z < rows) {
            break;
        }
        let cell = z as usize * ground.width + x as usize;
        let rise = ground.height(at.x, at.y) - level;
        if rise > set.floor_rise * depth
            || rise < -set.floor_drop * depth
            || lakes[cell]
            || (lateral > width && area[cell] >= rival)
        {
            break;
        }
        reached = lateral;
        lateral += step;
    }
    reached
}

fn carried(values: &[f32], places: &[f32]) -> Vec<f32> {
    let last = values.len() - 1;
    places
        .iter()
        .map(|&place| {
            let before = (place.floor() as usize).min(last);
            let after = (before + 1).min(last);
            values[before] + (values[after] - values[before]) * (place - before as f32)
        })
        .collect()
}

fn respaced(points: &[Vec2], marks: &[usize]) -> (Vec<Vec2>, Vec<f32>, Vec<usize>) {
    let along = distances(points);
    let last = points.len() - 1;
    let length = along[last];
    let steps = (max((length / RIVER_STEP).ceil(), 1.0) as usize).max(marks.len() + 1);
    let mut line = Vec::with_capacity(steps + 1);
    let mut places = Vec::with_capacity(steps + 1);
    let mut interval = 0;
    for step in 0..=steps {
        let wanted = length * step as f32 / steps as f32;
        while interval + 2 < along.len() && along[interval + 1] < wanted {
            interval += 1;
        }
        let run = along[interval + 1] - along[interval];
        let share = if run > 0.0 {
            clamp((wanted - along[interval]) / run, 0.0, 1.0)
        } else {
            0.0
        };
        line.push(points[interval] + (points[interval + 1] - points[interval]) * share);
        places.push(interval as f32 + share);
    }
    line[0] = points[0];
    places[0] = 0.0;
    line[steps] = points[last];
    places[steps] = last as f32;
    let placed = placed(marks.iter().map(|&mark| along[mark]), length, steps);
    (line, places, placed)
}

fn placed(marks: impl ExactSizeIterator<Item = f32>, length: f32, steps: usize) -> Vec<usize> {
    let many = marks.len();
    let mut placed: Vec<usize> = Vec::with_capacity(many);
    for (slot, wanted) in marks.enumerate() {
        let step = if length > 0.0 {
            (wanted / length * steps as f32).round() as usize
        } else {
            0
        };
        let least = placed.last().map_or(1, |&before| before + 1);
        placed.push(step.clamp(least, steps - (many - slot)));
    }
    placed
}

fn straight(land: &Land<'_>, (line, area, marks): Valley, level: Vec<f32>) -> Shaped {
    let widths: Vec<f32> = area
        .iter()
        .map(|&drained| (land.bankfull)(drained).0)
        .collect();
    Shaped {
        bend: bends(&line, &widths, land.bend),
        plain: vec![[0.0; 2]; line.len()],
        bar: vec![0.0; line.len()],
        points: line,
        area,
        level,
        marks,
    }
}

struct Reach<'a> {
    set: &'a Meanders,
    bars: f32,
    line: &'a [Vec2],
    along: &'a [f32],
    widths: &'a [f32],
}

impl Reach<'_> {
    fn span(&self, index: usize, reach: f32) -> std::ops::Range<usize> {
        let along = self.along;
        along.partition_point(|&at| at < along[index] - reach)
            ..along.partition_point(|&at| at <= along[index] + reach)
    }

    fn ends(&self, index: usize) -> f32 {
        let along = self.along;
        END_SHARE * min(along[index], along[along.len() - 1] - along[index])
    }

    fn ease(&self, index: usize) -> f32 {
        self.set.ease * self.set.wave * self.widths[index]
    }

    fn design(&self, carved: &[f32], grades: &[f32]) -> Vec<f32> {
        let (along, count) = (self.along, self.along.len());
        (0..count)
            .map(|index| {
                let span = self.span(index, SLOPE_WIDTHS * self.widths[index]);
                let before = span.start.min(index.saturating_sub(1));
                let after = span.end.saturating_sub(1).max((index + 1).min(count - 1));
                let run = max(along[after] - along[before], 1.0e-3);
                let slope = (carved[before] - carved[after]) / run;
                let graded = min(slope, grades[index]);
                let [from, to] = self.set.slopes;
                let gentle = 1.0 - smooth((graded - from) / max(to - from, 1.0e-6));
                let [open, shut] = self.set.steeper;
                let steeper = slope / max(grades[index], 1.0e-9);
                let free = 1.0 - smooth((steeper - open) / max(shut - open, 1.0e-6));
                min(
                    self.set.swing * self.widths[index] * gentle * free,
                    self.ends(index),
                )
            })
            .collect()
    }

    fn normals(&self, design: &[f32]) -> Vec<Vec2> {
        (0..self.line.len())
            .map(|index| {
                let reach = HEADING_WIDTHS * max(self.widths[index], RIVER_STEP) + design[index];
                let ahead = point_along(self.line, self.along, self.along[index] + reach);
                let behind = point_along(self.line, self.along, self.along[index] - reach);
                (ahead - behind).normalize_or(Vec2::X).perp()
            })
            .collect()
    }

    fn belt(
        &self,
        land: &Land<'_>,
        (depths, area): (&[f32], &[f32]),
        (design, normals): (&[f32], &[Vec2]),
    ) -> (Vec<f32>, Vec<f32>) {
        let count = self.line.len();
        let (mut swing, mut centre) = (vec![0.0; count], vec![0.0; count]);
        for index in 0..count {
            if design[index] <= 0.0 {
                continue;
            }
            let width = self.widths[index];
            let channel = [width, depths[index], area[index]];
            let reach = 2.0 * design[index] + width;
            let [low, high] = [-1.0, 1.0].map(|side| {
                room(
                    land,
                    self.line[index],
                    normals[index] * side,
                    channel,
                    reach,
                ) - 0.5 * width
            });
            let (belt, middle) = (0.5 * (high + low), 0.5 * (high - low));
            swing[index] = min(design[index], belt);
            let slack = belt - swing[index];
            centre[index] = clamp(0.0, middle - slack, middle + slack);
        }
        let ease = |index: usize| self.ease(index);
        (
            eased(&swing, self.along, ease)
                .iter()
                .enumerate()
                .map(|(index, &swing)| min(swing, self.ends(index)))
                .collect(),
            eased(&centre, self.along, ease)
                .iter()
                .enumerate()
                .map(|(index, &centre)| clamp(centre, -self.ends(index), self.ends(index)))
                .collect(),
        )
    }

    fn wander(&self, key: u32, (swing, centre): (&[f32], &[f32])) -> (Vec<f32>, Vec<f32>) {
        let (along, widths, count) = (self.along, self.widths, self.along.len());
        let set = self.set;
        let seed = keyed(key, SWING_SEED);
        let (mut turn, mut within) = (0_u32, PI * unit_of(seed));
        let mut bar = vec![0.0_f32; count];
        let mut offsets = Vec::with_capacity(count);
        for index in 0..count {
            if index > 0 {
                let stretch = mix(set.stretch, unit_of(keyed(seed, 2 * turn + 1)));
                let half_wave = 0.5 * set.wave * widths[index] * stretch;
                within += PI * (along[index] - along[index - 1]) / half_wave;
                while within.is_finite() && within >= PI {
                    within -= PI;
                    turn += 1;
                    let width = widths[index];
                    let drawn = unit_of(keyed(keyed(seed, BAR_SEED), turn));
                    if width < self.bars || drawn >= set.bar_chance {
                        continue;
                    }
                    let reach = set.bar_length * width;
                    for at in self.span(index, reach) {
                        let lens = smooth(1.0 - (along[at] - along[index]).abs() / reach)
                            * clamp(swing[at] / (0.5 * set.swing * widths[at]), 0.0, 1.0);
                        bar[at] = max(bar[at], lens);
                    }
                }
            }
            let strength = mix(set.strength, unit_of(keyed(seed, 2 * turn + 2)));
            let side = if turn % 2 == 0 { 1.0 } else { -1.0 };
            offsets.push(centre[index] + side * swing[index] * strength * sin(within));
        }
        (offsets, bar)
    }

    fn placed(
        &self,
        normals: &[Vec2],
        (swing, offsets): (&[f32], &[f32]),
        scale: &[f32],
    ) -> Vec<Vec2> {
        let (line, along) = (self.line, self.along);
        let (xs, zs): (Vec<f32>, Vec<f32>) = line.iter().map(|point| (point.x, point.y)).unzip();
        let reach = |index: usize| self.set.straighten * swing[index] * scale[index];
        let (eased_x, eased_z) = (eased(&xs, along, reach), eased(&zs, along, reach));
        let mut bent = line.to_vec();
        for index in 1..line.len() - 1 {
            bent[index] = Vec2::new(eased_x[index], eased_z[index])
                + normals[index] * (offsets[index] * scale[index]);
        }
        bent
    }

    fn crowded(&self, bent: &[Vec2], normals: &[Vec2], others: &Others) -> Vec<bool> {
        let (along, count) = (self.along, bent.len());
        let path = distances(bent);
        let mut held = vec![false; count];
        let window = |index: usize| {
            let reach = 0.5 * self.set.wave * self.widths[index] + self.ease(index);
            self.span(index, reach)
        };
        let mut longest = 1.0e-3_f32;
        for index in 0..count - 1 {
            let heading = (normals[index] + normals[index + 1]).perp();
            let run = along[index + 1] - along[index];
            if (bent[index + 1] - bent[index]).dot(heading) > -2.0 * FOLD * run {
                held[window(index)].fill(true);
            }
            longest = max(longest, path[index + 1] - path[index]);
        }
        for index in 0..count {
            let clear = self.set.clear * self.widths[index];
            let mut other = path.partition_point(|&at| at <= path[index] + FRAC_PI_2 * clear);
            while other < count {
                let apart = bent[index].distance(bent[other]);
                if apart < clear {
                    held[window(index)].fill(true);
                    held[window(other)].fill(true);
                    other += 1;
                } else {
                    other += (((apart - clear) / longest) as usize).max(1);
                }
            }
            if others.near(bent[index], self.widths[index], self.set.clear) {
                held[window(index)].fill(true);
            }
        }
        held
    }

    fn bent(
        &self,
        normals: &[Vec2],
        (swing, offsets): (&[f32], &[f32]),
        others: &Others,
    ) -> (Vec<Vec2>, Vec<f32>) {
        let mut scale = vec![1.0_f32; self.line.len()];
        for pass in 0..=FOLD_PASSES {
            let bent = self.placed(normals, (swing, offsets), &scale);
            let held = self.crowded(&bent, normals, others);
            if !held.contains(&true) {
                return (bent, scale);
            }
            let kept = if pass == FOLD_PASSES { 0.0 } else { 0.5 };
            let cut: Vec<f32> = scale
                .iter()
                .zip(&held)
                .map(|(&scale, &held)| if held { kept * scale } else { scale })
                .collect();
            scale = eased(&cut, self.along, |index| self.ease(index));
        }
        (self.placed(normals, (swing, offsets), &scale), scale)
    }
}

fn meander(land: &Land<'_>, (line, area, marks): Valley, key: u32, others: &Others) -> Shaped {
    let ground = land.channels.ground;
    let along = distances(&line);
    let (widths, depths): (Vec<f32>, Vec<f32>) =
        area.iter().map(|&drained| (land.bankfull)(drained)).unzip();
    let mut level = f32::INFINITY;
    let carved: Vec<f32> = line
        .iter()
        .map(|point| {
            level = min(level, ground.height(point.x, point.y));
            level
        })
        .collect();
    let set = land.meanders;
    let grades: Vec<f32> = area.iter().map(|&drained| (land.grade)(drained)).collect();
    let reach = Reach {
        set,
        bars: land.bars,
        line: &line,
        along: &along,
        widths: &widths,
    };
    let design = reach.design(&carved, &grades);
    if design.iter().all(|&swing| swing < 0.5 * RIVER_STEP) {
        return straight(land, (line, area, marks), carved);
    }
    let normals = reach.normals(&design);
    let (swing, centre) = reach.belt(land, (&depths, &area), (&design, &normals));
    let (offsets, bar) = reach.wander(key, (&swing, &centre));
    let (bent, scale) = reach.bent(&normals, (&swing, &offsets), others);
    let (low, high): (Vec<f32>, Vec<f32>) = (0..line.len())
        .map(|index| {
            let width = widths[index];
            let swung = swing[index] * scale[index];
            let beside =
                set.plain_beside * width * clamp(swung / (0.5 * set.swing * width), 0.0, 1.0);
            let off = (offsets[index] - centre[index]) * scale[index];
            (
                clamp(swung + off + beside, 0.0, set.plain_most * width),
                clamp(swung - off + beside, 0.0, set.plain_most * width),
            )
        })
        .unzip();
    let (points, places, marks) = respaced(&bent, &marks);
    let area = carried(&area, &places);
    let widths: Vec<f32> = area
        .iter()
        .map(|&drained| (land.bankfull)(drained).0)
        .collect();
    Shaped {
        bend: bends(&points, &widths, land.bend),
        plain: carried(&low, &places)
            .into_iter()
            .zip(carried(&high, &places))
            .map(|(low, high)| [low, high])
            .collect(),
        bar: carried(&bar, &places),
        level: carried(&carved, &places),
        points,
        area,
        marks,
    }
}

pub(crate) const MEASURED: usize = 8;

fn spline_weights(t: f32) -> [f32; 4] {
    let s = 1.0 - t;
    let (t2, t3) = (t * t, t * t * t);
    [
        s * s * s / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (3.0 * (t + t2 - t3) + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

pub(crate) fn smooth_line(points: &[Vec2], carried: &[f32]) -> (Vec<Vec2>, Vec<f32>) {
    let (line, values, _) = smooth_marked(points, carried, &[]);
    (line, values)
}

fn smooth_marked(points: &[Vec2], carried: &[f32], marks: &[usize]) -> Valley {
    let last = points.len() - 1;
    let mut control: Vec<(Vec2, f32)> = Vec::with_capacity(last + 3);
    control.push((2.0 * points[0] - points[1], 2.0 * carried[0] - carried[1]));
    control.extend(points.iter().copied().zip(carried.iter().copied()));
    control.push((
        2.0 * points[last] - points[last - 1],
        2.0 * carried[last] - carried[last - 1],
    ));
    let at = |u: f32| {
        let segment = (u.floor() as usize).min(last - 1);
        let (mut point, mut value) = (Vec2::ZERO, 0.0_f32);
        for (offset, weight) in spline_weights(u - segment as f32).into_iter().enumerate() {
            let (place, carries) = control[segment + offset];
            point += place * weight;
            value += carries * weight;
        }
        (point, value)
    };
    let mut measured: Vec<(f32, f32)> = Vec::with_capacity(last * MEASURED + 1);
    measured.push((0.0, 0.0));
    let (mut length, mut previous) = (0.0_f32, points[0]);
    for place in 1..=last * MEASURED {
        let u = place as f32 / MEASURED as f32;
        let (point, _) = at(u);
        length += point.distance(previous);
        previous = point;
        measured.push((u, length));
    }
    let steps = (max((length / RIVER_STEP).ceil(), 1.0) as usize).max(marks.len() + 1);
    let (mut line, mut values) = (Vec::with_capacity(steps + 1), Vec::with_capacity(steps + 1));
    let mut interval = 0;
    for step in 0..=steps {
        let wanted = length * step as f32 / steps as f32;
        while interval + 2 < measured.len() && measured[interval + 1].1 < wanted {
            interval += 1;
        }
        let ((from, before), (to, after)) = (measured[interval], measured[interval + 1]);
        let share = if after > before {
            min(max((wanted - before) / (after - before), 0.0), 1.0)
        } else {
            0.0
        };
        let (point, value) = at(from + (to - from) * share);
        line.push(point);
        values.push(value);
    }
    line[0] = points[0];
    values[0] = carried[0];
    line[steps] = points[last];
    values[steps] = carried[last];
    let placed = placed(
        marks.iter().map(|&mark| measured[mark * MEASURED].1),
        length,
        steps,
    );
    (line, values, placed)
}
