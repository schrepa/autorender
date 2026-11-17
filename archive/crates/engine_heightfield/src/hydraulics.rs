use engine_math::{GRAVITY, clamp, max, min};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::bed::{self, Bankfull, Crest, Forms, Knob, Reach, Shares};
use crate::hydrology::Fall;
use crate::noise::smooth;
use crate::rivers::{Form, Given, River, RiverLine, Valley};

pub const SHALLOWEST: f32 = 0.03;
pub const TRICKLE: f32 = 1.0e-3;
const FLATTEST: f32 = 2.0e-4;
const REACH_WIDTHS: f32 = 5.0;
pub const BESIDE: f32 = 1.0;
const FERGUSON: [f32; 2] = [6.5, 2.5];
const JET_STEP: f32 = 0.5;
const SLOWEST: f32 = 0.5;
const FACE_SAMPLES: f32 = 3.0;

engine_codec::fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct Shape {
        pub floods: f32 = 3.0 => range(1.0, 100.0),
        pub aspect: f32 = 12.0 => above_to(0.0, 1000.0),
        pub sunk: f32 = 0.25 => range(0.0, 4.0),
        pub blend: f32 = 0.2 => above_to(0.0, 0.9),
        pub narrowing: [f32; 2] = [0.02, 0.1] => range(0.0, 10.0),
        pub steep_narrow: f32 = 0.25 => range(0.0, 0.9),
        pub pool_narrow: f32 = 0.12 => range(0.0, 0.9),
        pub riffle_widen: f32 = 0.2 => range(0.0, 4.0),
        pub bar_widen: f32 = 0.45 => range(0.0, 4.0),
        pub bar_above: f32 = 0.2 => range(-1.0, 4.0),
        pub deepen: f32 = 0.6 => range(0.0, 2.0),
    }
}

engine_codec::fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct Falls {
        pub cling: f32 = 1.2 => above_to(0.0, 100.0),
        pub leap_depths: f32 = 2.0 => range(0.0, 100.0),
        pub cascade: f32 = 0.45 => above_to(0.0, 100.0),
        pub bridge: f32 = 2.0 => range(0.0, 100.0),
    }
}

engine_codec::fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct Profile {
        pub grade: [f32; 2] = [0.061, 0.46] => range(0.0, 10.0),
        pub drops: [f32; 2] = [1.0, 2.0] => range(0.0, 100.0),
        pub riser: f32 = 0.75 => above_to(0.0, 100.0),
        pub bench: f32 = 0.75 => range(0.0, 100.0),
    }
}

impl Profile {
    pub fn slope(&self, flood: f32) -> f32 {
        self.grade[0] * libm::powf(max(flood, 1.0e-6), -self.grade[1])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Flow {
    pub runoff: f32,
    pub finest: f32,
    pub form: Form,
}

pub struct Course<'a> {
    pub line: &'a RiverLine,
    pub ground: &'a dyn Fn(Vec2) -> f32,
    pub across: &'a dyn Fn(Vec2, Vec2, f32) -> Option<Across>,
    pub hard: &'a dyn Fn(Vec2, f32) -> f32,
    pub start: Option<f32>,
}

pub struct Surveyed {
    along: Vec<f32>,
    floods: Vec<f32>,
    widths: Vec<f32>,
    depths: Vec<f32>,
    across: Vec<Option<Across>>,
    holds: Vec<bool>,
    sides: Vec<[f32; 2]>,
    land: Vec<f32>,
    hard: Vec<f32>,
}

impl Surveyed {
    pub fn end(&self) -> f32 {
        self.land[self.land.len() - 1]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Mouth<'a> {
    pub level: f32,
    pub half: f32,
    pub rock: f32,
    pub wall: f32,
    pub joined: Option<[&'a [Vec2]; 2]>,
}

fn faces(vale: &Valley, wall: f32) -> [f32; 2] {
    let earth = max(min(vale.bank[0], vale.bank[1]), 1.0e-3);
    [max(wall, earth), earth]
}

fn climb(vale: &Valley, (rock, wall): (f32, f32), beyond: f32) -> f32 {
    let [wall, earth] = faces(vale, wall);
    let face = rock / wall;
    if beyond < face {
        wall * beyond
    } else {
        rock + earth * (beyond - face)
    }
}

fn reach(vale: &Valley, (rock, wall): (f32, f32), height: f32) -> f32 {
    let [wall, earth] = faces(vale, wall);
    let rock = min(rock, height);
    rock / wall + (height - rock) / earth
}

#[derive(Clone, Copy, Debug)]
pub struct Across {
    pub share: f32,
    pub lowest: f32,
    pub banks: f32,
    pub beside: f32,
}

pub fn resistance(relative: f32) -> f32 {
    let [deep, shallow] = FERGUSON.map(|coefficient| coefficient * coefficient);
    deep * shallow * relative * relative / (deep + shallow * libm::powf(relative, 5.0 / 3.0))
}

pub fn friction_slope(discharge: f32, width: f32, depth: f32, d84: f32) -> f32 {
    let radius = 2.0 / 3.0 * depth;
    let speed = discharge / (width * radius);
    speed * speed / (GRAVITY * radius * resistance(radius / d84))
}

fn critical_depth(discharge: f32, width: f32) -> f32 {
    let per_meter = discharge / width;
    libm::cbrtf(per_meter * per_meter / GRAVITY)
}

pub fn distances(points: &[Vec2]) -> Vec<f32> {
    let mut along = Vec::with_capacity(points.len());
    along.push(0.0);
    for pair in points.windows(2) {
        along.push(along[along.len() - 1] + pair[0].distance(pair[1]));
    }
    along
}

pub fn along_line(values: &[f32], along: &[f32], place: f32) -> f32 {
    let after = along
        .partition_point(|&at| at <= place)
        .clamp(1, along.len() - 1);
    let before = after - 1;
    let run = along[after] - along[before];
    let share = if run > 0.0 {
        clamp((place - along[before]) / run, 0.0, 1.0)
    } else {
        0.0
    };
    values[before] + (values[after] - values[before]) * share
}

fn profile(ground: &[f32], along: &[f32], start: Option<f32>) -> Vec<f32> {
    let mut banks: Vec<f32> = Vec::with_capacity(ground.len());
    for (index, &height) in ground.iter().enumerate() {
        let level = match banks.last() {
            Some(&above) => min(height, above - FLATTEST * (along[index] - along[index - 1])),
            None => start.unwrap_or(height),
        };
        banks.push(level);
    }
    banks
}

fn slopes(banks: &[f32], along: &[f32], widths: &[f32]) -> Vec<f32> {
    let count = banks.len();
    (0..count)
        .map(|index| {
            let reach = REACH_WIDTHS * widths[index];
            let before = along
                .partition_point(|&at| at < along[index] - reach)
                .min(index.saturating_sub(1));
            let after = along
                .partition_point(|&at| at <= along[index] + reach)
                .saturating_sub(1)
                .max((index + 1).min(count - 1));
            let run = max(along[after] - along[before], 1.0e-3);
            max((banks[before] - banks[after]) / run, FLATTEST)
        })
        .collect()
}

fn beside(points: &[Vec2], widths: &[f32], ground: &dyn Fn(Vec2) -> f32) -> Vec<[f32; 2]> {
    let last = points.len() - 1;
    (0..=last)
        .map(|index| {
            let tangent = points[(index + 1).min(last)] - points[index.saturating_sub(1)];
            let out = tangent.normalize_or(Vec2::X).perp() * (0.5 * widths[index] + BESIDE);
            [ground(points[index] - out), ground(points[index] + out)]
        })
        .collect()
}

fn given(line: &RiverLine, index: usize, value: usize) -> f32 {
    line.given
        .as_ref()
        .map_or(f32::NAN, |given| given.values[index][value])
}

pub fn surveyed(course: &Course<'_>, flow: &Flow) -> Surveyed {
    let line = course.line;
    let shape = &flow.form.shape;
    let count = line.points.len();
    let along = distances(&line.points);
    let floods: Vec<f32> = (0..count)
        .map(|index| {
            let flood = given(line, index, 2);
            if flood.is_finite() {
                flood
            } else {
                line.area[index] * flow.runoff
            }
        })
        .collect();
    let narrowest = bed::NARROWEST * flow.finest;
    let grid: Vec<f32> = line
        .points
        .iter()
        .map(|&point| (course.ground)(point))
        .collect();
    let (widths, depths): (Vec<f32>, Vec<f32>) = (0..count)
        .map(|index| {
            let (width, depth) = bed::bankfull(floods[index], shape.aspect);
            let (given_width, given_depth) = (given(line, index, 0), given(line, index, 1));
            (
                if given_width.is_finite() {
                    given_width
                } else {
                    max(width, narrowest)
                },
                if given_depth.is_finite() {
                    given_depth
                } else {
                    depth
                },
            )
        })
        .unzip();
    let across: Vec<Option<Across>> = (0..count)
        .map(|index| {
            let (before, after) = (index.saturating_sub(1), (index + 1).min(count - 1));
            let way = (line.points[after] - line.points[before]).normalize_or(Vec2::X);
            (course.across)(line.points[index], way, 0.5 * widths[index])
        })
        .collect();
    let finest = |values: &[f32], index: usize, of: fn(&Across) -> f32| match &across[index] {
        Some(found) => values[index] + (of(found) - values[index]) * found.share,
        None => values[index],
    };
    let ground: Vec<f32> = (0..count)
        .map(|index| finest(&grid, index, |found| found.lowest))
        .collect();
    let sides = beside(&line.points, &widths, course.ground);
    let lower: Vec<f32> = sides
        .iter()
        .map(|&[left, right]| min(left, right))
        .collect();
    let holds: Vec<bool> = (0..count)
        .map(|index| {
            across[index].is_some_and(|found| {
                found.share >= 1.0 && found.banks - found.lowest >= depths[index]
            })
        })
        .collect();
    let tops: Vec<f32> = (0..count)
        .map(|index| match across[index] {
            Some(found) if holds[index] => found.lowest + depths[index],
            _ => min(
                min(ground[index], line.level[index]),
                finest(&lower, index, |found| found.beside) - shape.sunk * depths[index],
            ),
        })
        .collect();
    let land = profile(&tops, &along, course.start);
    let hard = (0..count)
        .map(|index| (course.hard)(line.points[index], land[index]))
        .collect();
    Surveyed {
        along,
        floods,
        widths,
        depths,
        across,
        holds,
        sides,
        land,
        hard,
    }
}

fn apart<'a>(course: impl Iterator<Item = &'a Vec2>, point: Vec2, reach: f32) -> f32 {
    let mut course = course.copied();
    let Some(joint) = course.next() else {
        return f32::INFINITY;
    };
    let mut nearest = point.distance(joint);
    let mut from = joint;
    for to in course {
        let run = to - from;
        let share = clamp(
            (point - from).dot(run) / max(run.length_squared(), 1.0e-9),
            0.0,
            1.0,
        );
        nearest = min(nearest, point.distance(from + run * share));
        if to.distance(joint) > reach {
            break;
        }
        from = to;
    }
    nearest
}

fn entered(
    vale: &Valley,
    line: &RiverLine,
    land: &[f32],
    (mouth, [above, below]): (&Mouth<'_>, [&[Vec2]; 2]),
) -> Vec<f32> {
    let count = land.len();
    let joint = line.points[count - 1];
    let mut land = land.to_vec();
    for index in (0..count).rev() {
        let point = line.points[index];
        let reach = 2.0 * point.distance(joint);
        let off = min(
            apart(above.iter().rev(), point, reach),
            apart(below.iter(), point, reach),
        );
        let beyond = max(off - mouth.half, 0.0);
        let cap = mouth.level + climb(vale, (mouth.rock, mouth.wall), beyond);
        if cap >= land[index] {
            break;
        }
        land[index] = cap;
    }
    for index in 1..count {
        land[index] = min(land[index], land[index - 1]);
    }
    land
}

fn graded(
    set: &Profile,
    (land, along): (&[f32], &[f32]),
    (floods, depths): (&[f32], &[f32]),
    (pinned, hard): (&[bool], &[f32]),
    (held, end): (bool, Option<f32>),
) -> Vec<f32> {
    let last = land.len() - 1;
    let mut banks = land.to_vec();
    if let Some(end) = end {
        banks[last] = min(banks[last], end);
    }
    let [least, most] = [min(set.drops[0], set.drops[1]), set.drops[1]];
    let lip = |place: usize| hard[place] * (land[place] - land[place + 1]);
    let mut anchor = last;
    let mut index = last;
    while index > 0 {
        index -= 1;
        let rise = set.slope(floods[index]) * (along[index + 1] - along[index]);
        let want = banks[index + 1] + rise;
        let cut = land[index] - want;
        if cut <= 0.0 || pinned[index] {
            banks[index] = land[index];
            anchor = index;
        } else if cut <= most * depths[index] {
            banks[index] = want;
        } else {
            let mut drop = index;
            for place in index + 1..anchor {
                let tall = land[place] - banks[place + 1] >= least * depths[place];
                if tall && lip(place) > lip(drop) {
                    drop = place;
                }
            }
            banks[drop] = land[drop];
            anchor = drop;
            index = drop;
        }
    }
    if held {
        banks[0] = land[0];
    }
    banks
}

fn channel(
    course: &Course<'_>,
    flow: &Flow,
    surveyed: &Surveyed,
    mouth: Option<Mouth<'_>>,
) -> Bankfull {
    let line = course.line;
    let (shape, profiled) = (&flow.form.shape, &flow.form.profile);
    let Surveyed {
        along,
        floods,
        widths,
        depths,
        across,
        holds,
        land,
        hard,
        ..
    } = surveyed;
    let count = line.points.len();
    let entry = match mouth.and_then(|mouth| Some((mouth, mouth.joined?))) {
        Some((mouth, joined)) => entered(&flow.form.valley, line, land, (&mouth, joined)),
        None => land.clone(),
    };
    let mut banks = graded(
        profiled,
        (&entry, along),
        (floods, depths),
        (holds, hard),
        (
            line.head && course.start.is_some(),
            mouth.map(|mouth| mouth.level),
        ),
    );
    let least = min(profiled.drops[0], profiled.drops[1]);
    let falls: Vec<f32> = (0..count)
        .map(|index| {
            let next = (index + 1).min(count - 1);
            let rise = profiled.slope(floods[index]) * (along[next] - along[index]);
            let fall = banks[index] - banks[next] - rise;
            if fall > 0.0 && fall >= least * depths[index] {
                fall
            } else {
                0.0
            }
        })
        .collect();
    let (ledges, stairs) = bed::ledges(
        &flow.form,
        (bed::place_key(line.points[0]), flow.finest),
        (along, &falls),
        (widths, depths, floods),
        &mut banks,
    );
    let glide: Vec<f32> = banks
        .iter()
        .zip(bed::fallen(&ledges, along))
        .map(|(&bank, fallen)| bank + fallen)
        .collect();
    let slopes = slopes(&glide, along, widths);
    let shares: Vec<Shares> = slopes
        .iter()
        .map(|&slope| Shares::of(slope, shape.blend))
        .collect();
    let swings: Vec<f32> = (0..count)
        .map(|index| {
            let [low, high] = line.plain[index];
            clamp((low + high) / (2.0 * widths[index]), 0.0, 1.0)
        })
        .collect();
    let rock: Vec<f32> = land
        .iter()
        .zip(&banks)
        .map(|(&land, &bank)| max(land - bank, 0.0))
        .collect();
    let narrowest = bed::NARROWEST * flow.finest;
    let (widths, depths): (Vec<f32>, Vec<f32>) = (0..count)
        .map(|index| {
            let [from, to] = shape.narrowing;
            let steep = smooth((slopes[index] - from) / max(to - from, 1.0e-6));
            let bent = smooth(line.bend[index].abs() / flow.form.pools.bend);
            let shaped = (1.0
                + shape.riffle_widen * swings[index] * (1.0 - bent)
                + shape.bar_widen * line.bar[index])
                * (1.0 - shape.pool_narrow * bent)
                * (1.0 - shape.steep_narrow * steep);
            (
                if given(line, index, 0).is_finite() {
                    widths[index]
                } else {
                    max(widths[index] * shaped, narrowest)
                },
                if given(line, index, 1).is_finite() {
                    depths[index]
                } else {
                    depths[index] * libm::powf(shaped, -shape.deepen)
                },
            )
        })
        .unzip();
    let share = libm::powf(1.0 / shape.floods, 6.0 / 13.0);
    let cut = (0..count)
        .map(|index| {
            !(holds[index]
                && across[index].is_some_and(|found| {
                    banks[index] - (1.0 - share) * depths[index] > found.lowest
                }))
        })
        .collect();
    Bankfull {
        floods: floods.clone(),
        slopes,
        shares,
        widths,
        depths,
        banks,
        cut,
        bends: line.bend.clone(),
        swings,
        ledges,
        stairs,
        rock,
    }
}

fn drops(levels: &[f32], along: &[f32], (least, cling): (&[f32], f32)) -> Vec<(usize, usize)> {
    let steep = |index: usize| {
        levels[index] - levels[index + 1] > cling * (along[index + 1] - along[index])
    };
    let mut found = Vec::new();
    let mut index = 0;
    while index + 1 < levels.len() {
        if !steep(index) {
            index += 1;
            continue;
        }
        let lip = index;
        while index + 1 < levels.len() && steep(index) {
            index += 1;
        }
        if levels[lip] - levels[index] >= least[lip] {
            found.push((lip, index));
        }
    }
    found
}

fn jet(
    lip: usize,
    (points, along, targets): (&[Vec2], &[f32], &[f32]),
    surface: f32,
    speed: f32,
) -> Option<(usize, Vec<[f32; 3]>)> {
    let height = |run: f32| surface - GRAVITY * run * run / (2.0 * speed * speed);
    let landing = (lip + 1..targets.len()).find(|&index| {
        let run = along[index] - along[lip];
        height(run) <= targets[index]
    })?;
    if landing <= lip + 1 {
        return None;
    }
    let (from, to) = (along[landing - 1] - along[lip], along[landing] - along[lip]);
    let (above, below) = (
        height(from) - targets[landing - 1],
        height(to) - targets[landing],
    );
    let share = if above > below {
        above / (above - below)
    } else {
        1.0
    };
    let reach = from + (to - from) * min(max(share, 0.0), 1.0);
    let direction = (points[lip + 1] - points[lip]).normalize_or(Vec2::X);
    let drop = surface - height(reach);
    let steps = max(((reach + drop) / JET_STEP).ceil(), 1.0) as usize;
    let path = (0..=steps)
        .map(|step| {
            let run = reach * step as f32 / steps as f32;
            let at = points[lip] + direction * run;
            [at.x, height(run), at.y]
        })
        .collect();
    Some((landing, path))
}

struct Jet {
    lip: usize,
    landing: usize,
    speed: f32,
    path: Vec<[f32; 3]>,
    landing_speed: f32,
}

fn jets(
    drops: &[(usize, usize)],
    (points, along): (&[Vec2], &[f32]),
    (levels, speeds): (&[f32], &[f32]),
    (discharges, widths): (&[f32], &[f32]),
) -> Vec<Jet> {
    let mut found: Vec<Jet> = Vec::new();
    for &(lip, _) in drops {
        if found.last().is_some_and(|jet| lip <= jet.landing) {
            continue;
        }
        let critical = critical_depth(discharges[lip], widths[lip]);
        let speed = max(speeds[lip], (GRAVITY * critical).sqrt());
        let Some((landing, path)) = jet(lip, (points, along, levels), levels[lip], speed) else {
            continue;
        };
        let drop = levels[lip] - path[path.len() - 1][1];
        found.push(Jet {
            lip,
            landing,
            speed,
            path,
            landing_speed: (speed * speed + 2.0 * GRAVITY * max(drop, 0.0)).sqrt(),
        });
    }
    found
}

fn cascades(
    falls: &Falls,
    (drops, jets): (&[(usize, usize)], &[Jet]),
    (points, along, slopes): (&[Vec2], &[f32], &[f32]),
    (surface, speeds): (&[f32], &[f32]),
    (widths, discharges, areas): (&[f32], &[f32], &[f32]),
) -> (Vec<Fall>, Vec<bool>) {
    let count = points.len();
    let leaping = |at: usize| jets.iter().find(|jet| jet.lip == at);
    let steep = |at: usize| {
        slopes[at] >= falls.cascade || drops.iter().any(|&(lip, foot)| at >= lip && at <= foot)
    };
    let bridge = falls.bridge;
    let mut falls = Vec::new();
    let mut sheeted = vec![false; count];
    let mut first = 0;
    while first < count {
        if !(steep(first) || leaping(first).is_some()) {
            first += 1;
            continue;
        }
        let mut last = first;
        let mut at = first;
        while at < count {
            if let Some(jet) = leaping(at) {
                last = jet.landing.min(count - 1);
                at = jet.landing;
            } else if steep(at) {
                last = at;
                at += 1;
            } else if let Some(next) = (at + 1..count)
                .take_while(|&next| along[next] - along[at] <= bridge * widths[at])
                .find(|&next| steep(next) || leaping(next).is_some())
            {
                last = next - 1;
                at = next;
            } else {
                break;
            }
        }
        let (mut path, mut times, mut aloft, mut across) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut time, mut previous, mut landing_speed) = (0.0, None, None);
        let mut at = first;
        while at <= last {
            if let Some(jet) = leaping(at) {
                let lip = Vec3::from_array(jet.path[0]);
                if let Some(previous) = previous {
                    time += lip.distance(previous) / max(speeds[at], SLOWEST);
                }
                let leaving = time;
                for point in &jet.path {
                    let run = Vec2::new(point[0] - lip.x, point[2] - lip.z).length();
                    time = leaving + run / jet.speed;
                    path.push(*point);
                    times.push(time);
                    aloft.push(true);
                    across.push(widths[at]);
                }
                if let Some(landed) = aloft.last_mut() {
                    *landed = false;
                }
                previous = path.last().copied().map(Vec3::from_array);
                landing_speed = Some(jet.landing_speed);
                at = jet.landing;
                continue;
            }
            let point = Vec3::new(points[at].x, surface[at], points[at].y);
            if let Some(previous) = previous {
                time += point.distance(previous) / max(speeds[at], SLOWEST);
            }
            path.push(point.to_array());
            times.push(time);
            aloft.push(false);
            across.push(widths[at]);
            previous = Some(point);
            landing_speed = None;
            at += 1;
        }
        let dropping = drops.iter().any(|&(lip, _)| lip >= first && lip <= last);
        if path.len() >= 2 && dropping {
            falls.push(Fall {
                top: path[0],
                bottom: path[path.len() - 1],
                path,
                times,
                aloft,
                widths: across,
                sky: Vec::new(),
                width: widths[first],
                area: areas[first],
                speed: landing_speed.unwrap_or(speeds[last]),
                discharge: discharges[first],
            });
            sheeted[first..=last].fill(true);
        }
        first = last + 1;
    }
    (falls, sheeted)
}

fn leaping_heights(falls: &Falls, discharges: &[f32], widths: &[f32]) -> Vec<f32> {
    discharges
        .iter()
        .zip(widths)
        .map(|(&discharge, &width)| falls.leap_depths * critical_depth(discharge, width))
        .collect()
}

fn cut_beneath(values: &mut [f32], jets: &[Jet], along: &[f32]) {
    for jet in jets {
        let (high, low) = (values[jet.lip], values[jet.landing]);
        let run = max(along[jet.landing] - along[jet.lip], 1.0e-3);
        for index in jet.lip + 1..jet.landing {
            values[index] = high + (low - high) * (along[index] - along[jet.lip]) / run;
        }
    }
}

fn uncarved(mut forms: Forms, cut: &[bool]) -> Forms {
    for (index, &carved) in cut.iter().enumerate() {
        if !carved {
            forms.dips[index] = 0.0;
            forms.held[index] = f32::NAN;
            forms.stepped[index] = false;
        }
    }
    forms
        .crests
        .retain(|crest| cut[crest.segment as usize] && cut[crest.segment as usize + 1]);
    forms
        .knobs
        .retain(|knob| cut[knob.segment as usize] && cut[knob.segment as usize + 1]);
    forms
}

struct Beds {
    reach: Vec<Reach>,
    d50: Vec<f32>,
    d84: Vec<f32>,
    power: Vec<f32>,
    carved: Vec<bool>,
    crest: Vec<Crest>,
    knobs: Vec<Knob>,
}

fn beds(
    bankfull: &Bankfull,
    forms: Forms,
    (jets, airborne, sheeted): (&[Jet], &[bool], &[bool]),
) -> Beds {
    let Forms {
        stepped: mut carved,
        crests: mut crest,
        mut knobs,
        ..
    } = forms;
    for jet in jets {
        crest.retain(|crest| !(jet.lip..jet.landing).contains(&(crest.segment as usize)));
        carved[jet.lip + 1..jet.landing].fill(false);
    }
    knobs.retain(|knob| !airborne[knob.segment as usize]);
    for knob in &knobs {
        let segment = knob.segment as usize;
        carved[segment] = true;
        carved[segment + 1] = true;
    }
    let count = carved.len();
    let (d50, d84) = (0..count)
        .map(|index| {
            bed::grains(
                bankfull.depths[index],
                bankfull.slopes[index],
                &bankfull.shares[index],
            )
        })
        .unzip();
    Beds {
        reach: (0..count)
            .map(|index| {
                if sheeted[index] || airborne[index] {
                    Reach::Falls
                } else {
                    Reach::of(bankfull.slopes[index])
                }
            })
            .collect(),
        d50,
        d84,
        power: (0..count)
            .map(|index| {
                bed::power(
                    bankfull.floods[index],
                    bankfull.slopes[index],
                    bankfull.widths[index],
                )
            })
            .collect(),
        carved,
        crest,
        knobs,
    }
}

pub fn river(
    course: &Course<'_>,
    flow: &Flow,
    surveyed: Surveyed,
    mouth: Option<Mouth<'_>>,
) -> (River, Vec<Fall>) {
    let line = course.line;
    let count = line.points.len();
    let bankfull = channel(course, flow, &surveyed, mouth);
    let Surveyed { along, sides, .. } = surveyed;
    let Bankfull {
        floods,
        slopes,
        widths,
        depths,
        banks,
        ..
    } = &bankfull;
    let form = &flow.form;
    let forms = uncarved(
        bed::forms((&line.points, &along), &bankfull, form, flow.finest),
        &bankfull.cut,
    );
    let mut bed: Vec<f32> = (0..count)
        .map(|index| banks[index] - depths[index] - forms.pools[index] - forms.dips[index])
        .collect();
    let mut depth: Vec<f32> = banks
        .iter()
        .zip(&bed)
        .map(|(&banks, &bed)| banks - bed)
        .collect();
    let discharges: Vec<f32> = floods
        .iter()
        .map(|&flood| flood / form.shape.floods)
        .collect();
    let share = libm::powf(1.0 / form.shape.floods, 6.0 / 13.0);
    let over = |place: f32| {
        along_line(banks, &along, place) - (1.0 - share) * along_line(depths, &along, place)
    };
    let mut surface: Vec<f32> = (0..count)
        .map(|index| {
            let own = banks[index] - (1.0 - share) * depths[index] - forms.dips[index];
            let crest = forms.held[index];
            if crest.is_finite() {
                max(own, over(crest))
            } else {
                own
            }
        })
        .collect();
    let speed_of = |index: usize, surface: f32, bed: f32, depth: f32| {
        let water = max(surface - bed, SHALLOWEST);
        let across = widths[index] * (water / max(depth, water)).sqrt();
        discharges[index] / (2.0 / 3.0 * across * water)
    };
    let mut speed: Vec<f32> = (0..count)
        .map(|index| speed_of(index, surface[index], bed[index], depth[index]))
        .collect();
    let stairs = &bankfull.stairs;
    let mut drops = drops(
        &surface,
        &along,
        (
            &leaping_heights(&form.falls, &discharges, widths),
            form.falls.cling,
        ),
    );
    drops.retain(|&(lip, foot)| !stairs.iter().any(|&(top, base)| lip < base && foot > top));
    let jets = jets(
        &drops,
        (&line.points, &along),
        (&surface, &speed),
        (&discharges, widths),
    );
    drops.extend(stairs);
    cut_beneath(&mut bed, &jets, &along);
    cut_beneath(&mut depth, &jets, &along);
    let mut airborne = vec![false; count];
    for jet in &jets {
        airborne[jet.lip..jet.landing].fill(true);
        for index in jet.lip + 1..jet.landing {
            speed[index] = speed_of(index, bed[index] + depth[index], bed[index], depth[index]);
            surface[index] = f32::NAN;
        }
    }
    let (falls, sheeted) = cascades(
        &form.falls,
        (&drops, &jets),
        (&line.points, &along, slopes),
        (&surface, &speed),
        (widths, &discharges, &line.area),
    );
    let beds = beds(&bankfull, forms, (&jets, &airborne, &sheeted));
    let bar = (0..count)
        .map(|index| {
            let lens = line.bar[index];
            if lens > 0.0 && surface[index].is_finite() {
                lens * (max(surface[index] - bed[index], 0.0)
                    + form.shape.bar_above * depths[index])
            } else {
                0.0
            }
        })
        .collect();
    let vale = &form.valley;
    let wall: Vec<f32> = (0..count)
        .map(|index| {
            let face = (1.0 - share) * depths[index] + bankfull.rock[index];
            min(vale.wall, face / (FACE_SAMPLES * flow.finest))
        })
        .collect();
    let margin = (0..count)
        .map(|index| {
            sides[index].map(|side| {
                let height = max(side - banks[index], 0.0);
                let climbed = reach(vale, (bankfull.rock[index], wall[index]), height);
                vale.reach * widths[index] + max(climbed, height / max(vale.fade, 1.0e-3))
            })
        })
        .collect();
    let given = line.given.as_ref().map_or_else(Given::default, |given| {
        let first = given.values.first().copied().unwrap_or([f32::NAN; 3]);
        Given {
            source: Some(given.source),
            width: first[0].is_finite(),
            depth: first[1].is_finite(),
            flood: first[2].is_finite(),
        }
    });
    let river = River {
        points: line.points.iter().map(|point| point.to_array()).collect(),
        discharge: discharges,
        half: widths.iter().map(|&width| 0.5 * width).collect(),
        depth,
        bed,
        surface,
        speed,
        falls: airborne,
        sheeted,
        reach: beds.reach,
        d50: beds.d50,
        d84: beds.d84,
        power: beds.power,
        flood: banks.clone(),
        carved: beds.carved,
        cut: bankfull.cut.clone(),
        crest: beds.crest,
        knobs: beds.knobs,
        bend: line.bend.clone(),
        plain: line.plain.clone(),
        bar,
        rock: bankfull.rock.clone(),
        wall,
        margin,
        head: line.head && course.start.is_none(),
        given,
    };
    (river, falls)
}

engine_codec::record!(Shape {
    floods,
    aspect,
    sunk,
    blend,
    narrowing,
    steep_narrow,
    pool_narrow,
    riffle_widen,
    bar_widen,
    bar_above,
    deepen,
});
engine_codec::record!(Profile {
    grade,
    drops,
    riser,
    bench,
});
engine_codec::record!(Falls {
    cling,
    leap_depths,
    cascade,
    bridge,
});
