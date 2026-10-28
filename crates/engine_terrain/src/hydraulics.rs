use engine_math::{clamp, max, min};
use glam::{Vec2, Vec3};

use crate::bed::{self, Bankfull, Crest, Forms, Knob, Reach, Shares};
use crate::hydrology::Fall;
use crate::rivers::{Given, River, RiverLine};

pub const GRAVITY: f32 = 9.81;
pub(crate) const SHALLOWEST: f32 = 0.03;
pub(crate) const TRICKLE: f32 = 1.0e-3;
const FLATTEST: f32 = 2.0e-4;
const REACH_WIDTHS: f32 = 5.0;
pub(crate) const FLOODS: f32 = 3.0;
const HELD_DEPTH: f32 = 0.25;
pub(crate) const BESIDE: f32 = 1.0;
const FERGUSON: [f32; 2] = [6.5, 2.5];
const CLING_SLOPE: f32 = 1.2;
const LEAP_DEPTHS: f32 = 2.0;
const LEAP_LEAST: f32 = 1.5;
const JET_STEP: f32 = 0.5;
const CASCADE_SLOPE: f32 = 0.45;
const BRIDGE: f32 = 2.0;
const SLOWEST: f32 = 0.5;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Flow {
    pub(crate) runoff: f32,
    pub(crate) finest: f32,
}

pub(crate) struct Course<'a> {
    pub(crate) line: &'a RiverLine,
    pub(crate) ground: &'a dyn Fn(Vec2) -> f32,
    pub(crate) across: &'a dyn Fn(Vec2, Vec2, f32) -> Option<Across>,
    pub(crate) start: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Across {
    pub(crate) share: f32,
    pub(crate) lowest: f32,
    pub(crate) banks: f32,
    pub(crate) beside: f32,
}

pub(crate) fn friction_slope(discharge: f32, width: f32, depth: f32, d84: f32) -> f32 {
    let radius = 2.0 / 3.0 * depth;
    let speed = discharge / (width * radius);
    let relative = radius / d84;
    let [deep, shallow] = FERGUSON.map(|coefficient| coefficient * coefficient);
    let ratio =
        deep * shallow * relative * relative / (deep + shallow * libm::powf(relative, 5.0 / 3.0));
    speed * speed / (GRAVITY * radius * ratio)
}

fn critical_depth(discharge: f32, width: f32) -> f32 {
    let per_meter = discharge / width;
    libm::cbrtf(per_meter * per_meter / GRAVITY)
}

pub(crate) fn distances(points: &[Vec2]) -> Vec<f32> {
    let mut along = Vec::with_capacity(points.len());
    along.push(0.0);
    for pair in points.windows(2) {
        along.push(along[along.len() - 1] + pair[0].distance(pair[1]));
    }
    along
}

pub(crate) fn along_line(values: &[f32], along: &[f32], place: f32) -> f32 {
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

fn beside(points: &[Vec2], widths: &[f32], ground: &dyn Fn(Vec2) -> f32) -> Vec<f32> {
    let last = points.len() - 1;
    (0..=last)
        .map(|index| {
            let tangent = points[(index + 1).min(last)] - points[index.saturating_sub(1)];
            let out = tangent.normalize_or(Vec2::X).perp() * (0.5 * widths[index] + BESIDE);
            min(ground(points[index] + out), ground(points[index] - out))
        })
        .collect()
}

fn channel(course: &Course<'_>, flow: &Flow, along: &[f32]) -> Bankfull {
    let line = course.line;
    let count = line.points.len();
    let given = |index: usize, value: usize| {
        line.given
            .as_ref()
            .map_or(f32::NAN, |given| given.values[index][value])
    };
    let floods: Vec<f32> = (0..count)
        .map(|index| {
            let flood = given(index, 2);
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
            let (width, depth) = bed::bankfull(floods[index]);
            let (given_width, given_depth) = (given(index, 0), given(index, 1));
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
    let slopes = slopes(&profile(&ground, along, course.start), along, &widths);
    let shares: Vec<Shares> = slopes.iter().map(|&slope| Shares::of(slope)).collect();
    let sides = beside(&line.points, &widths, course.ground);
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
                ground[index],
                finest(&sides, index, |found| found.beside) - HELD_DEPTH * depths[index],
            ),
        })
        .collect();
    let banks = profile(&tops, along, course.start);
    let share = libm::powf(1.0 / FLOODS, 6.0 / 13.0);
    let cut = (0..count)
        .map(|index| {
            !(holds[index]
                && across[index].is_some_and(|found| {
                    banks[index] - (1.0 - share) * depths[index] > found.lowest
                }))
        })
        .collect();
    Bankfull {
        floods,
        slopes,
        shares,
        widths,
        depths,
        banks,
        cut,
    }
}

fn drops(levels: &[f32], along: &[f32], least: &[f32]) -> Vec<(usize, usize)> {
    let steep = |index: usize| {
        levels[index] - levels[index + 1] > CLING_SLOPE * (along[index + 1] - along[index])
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
    (drops, jets): (&[(usize, usize)], &[Jet]),
    (points, along, slopes): (&[Vec2], &[f32], &[f32]),
    (surface, speeds): (&[f32], &[f32]),
    (widths, discharges, areas): (&[f32], &[f32], &[f32]),
) -> (Vec<Fall>, Vec<bool>) {
    let count = points.len();
    let leaping = |at: usize| jets.iter().find(|jet| jet.lip == at);
    let steep = |at: usize| {
        slopes[at] >= CASCADE_SLOPE || drops.iter().any(|&(lip, foot)| at >= lip && at <= foot)
    };
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
                .take_while(|&next| along[next] - along[at] <= BRIDGE * widths[at])
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

fn leaping_heights(discharges: &[f32], widths: &[f32]) -> Vec<f32> {
    discharges
        .iter()
        .zip(widths)
        .map(|(&discharge, &width)| max(LEAP_DEPTHS * critical_depth(discharge, width), LEAP_LEAST))
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
    for (crest, pair) in forms.crests.iter_mut().zip(cut.windows(2)) {
        if !(pair[0] && pair[1]) {
            *crest = None;
        }
    }
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
    crest: Vec<Option<Crest>>,
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
        crest[jet.lip..jet.landing].fill(None);
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

pub(crate) fn river(course: &Course<'_>, flow: &Flow) -> (River, Vec<Fall>) {
    let line = course.line;
    let count = line.points.len();
    let along = distances(&line.points);
    let bankfull = channel(course, flow, &along);
    let Bankfull {
        floods,
        slopes,
        widths,
        depths,
        banks,
        ..
    } = &bankfull;
    let forms = uncarved(
        bed::forms((&line.points, &along), &bankfull, flow.finest),
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
    let discharges: Vec<f32> = floods.iter().map(|&flood| flood / FLOODS).collect();
    let share = libm::powf(1.0 / FLOODS, 6.0 / 13.0);
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
    let drops = drops(&surface, &along, &leaping_heights(&discharges, widths));
    let jets = jets(
        &drops,
        (&line.points, &along),
        (&surface, &speed),
        (&discharges, widths),
    );
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
        (&drops, &jets),
        (&line.points, &along, slopes),
        (&surface, &speed),
        (widths, &discharges, &line.area),
    );
    let beds = beds(&bankfull, forms, (&jets, &airborne, &sheeted));
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
        head: line.head && course.start.is_none(),
        given,
    };
    (river, falls)
}
