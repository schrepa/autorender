use glam::Vec3;

use crate::bots::aim::{flat, flat_distance, max, radians, sin_cos};
use crate::bots::numbers::BotNumbers;
use crate::bots::places::{Places, Site, ZoneId};
use crate::bots::sense::World;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hold {
    pub at: Vec3,
    pub face: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Way {
    pub choke: usize,
    pub mid: Vec3,
    pub inward: Vec3,
    pub inside: Vec3,
    pub outside: Vec3,
    pub attack_stage: Vec3,
    pub defend_stage: Vec3,
    pub attack_length: f32,
    pub defend_length: f32,
    pub retake_stage: Vec3,
    pub retake_length: f32,
    pub retake_clear: bool,
    pub hold: Hold,
    pub cross: Hold,
    pub post: Hold,
    pub smoke_at: Vec3,
    pub flash_at: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SiteTactics {
    pub ways: Vec<Way>,
    pub plant: Vec3,
    pub defend_order: Vec<usize>,
    pub retake_order: Vec<usize>,
    pub back: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tactics {
    pub sites: Vec<SiteTactics>,
    pub mid_hold: Hold,
    pub mid_spot: Hold,
    pub attack_home: Vec3,
    pub defend_home: Vec3,
    pub safe: Vec<bool>,
}

pub fn path_length(world: &dyn World, from: Vec3, to: Vec3, buffer: &mut Vec<Vec3>) -> f32 {
    if !world.path(from, to, buffer) || buffer.is_empty() {
        return flat_distance(from, to);
    }
    let mut length = flat_distance(from, buffer[0]);
    for pair in buffer.windows(2) {
        length += flat_distance(pair[0], pair[1]);
    }
    length
}

pub fn along(start: Vec3, corners: &[Vec3], distance: f32) -> Vec3 {
    let mut at = start;
    let mut left = max(distance, 0.0);
    for &corner in corners {
        let step = flat_distance(at, corner);
        if step >= left && step > 0.0 {
            return at + (corner - at) * (left / step);
        }
        left -= step;
        at = corner;
    }
    at
}

fn mean(points: &[Vec3]) -> Vec3 {
    if points.is_empty() {
        return Vec3::ZERO;
    }
    points.iter().copied().sum::<Vec3>() / points.len() as f32
}

fn stage_on_path(
    world: &dyn World,
    from: Vec3,
    to: Vec3,
    back: f32,
    buffer: &mut Vec<Vec3>,
) -> (Vec3, f32) {
    let length = path_length(world, from, to, buffer);
    if buffer.is_empty() {
        let towards = flat(from - to).normalize_or_zero();
        return (to + towards * back, length);
    }
    (along(from, buffer, length - back), length)
}

pub fn safe_path(
    world: &dyn World,
    safe: &[bool],
    from: Vec3,
    to: Vec3,
    allowed: [Option<ZoneId>; 2],
    step: f32,
    buffer: &mut Vec<Vec3>,
) -> bool {
    if !world.path(from, to, buffer) {
        return false;
    }
    let places = world.places();
    let mut at = from;
    for &corner in buffer.iter() {
        let length = flat_distance(at, corner);
        let samples = (length / max(step, 0.1)).ceil() as usize;
        for k in 0..=samples {
            let point = at + (corner - at) * (k as f32 / samples.max(1) as f32);
            if let Some(zone) = places.zone_at(point)
                && !safe.get(zone as usize).copied().unwrap_or(true)
                && !allowed.contains(&Some(zone))
            {
                return false;
            }
        }
        at = corner;
    }
    true
}

pub fn crosses(
    world: &dyn World,
    from: Vec3,
    corners: &[Vec3],
    zone: Option<ZoneId>,
    step: f32,
) -> bool {
    let Some(zone) = zone else {
        return false;
    };
    let places = world.places();
    let mut at = from;
    for &corner in corners {
        let length = flat_distance(at, corner);
        let samples = (length / max(step, 0.1)).ceil() as usize;
        for k in 0..=samples {
            let point = at + (corner - at) * (k as f32 / samples.max(1) as f32);
            if places.zone_at(point) == Some(zone) {
                return true;
            }
        }
        at = corner;
    }
    false
}

fn open_between(world: &dyn World, from: Vec3, to: Vec3, step: f32) -> Vec3 {
    let length = flat_distance(from, to);
    let share = step / max(length, step);
    let rings = (0.5 / share).ceil() as u32;
    (0..=rings)
        .flat_map(|ring| [0.5 - ring as f32 * share, 0.5 + ring as f32 * share])
        .filter(|share| (0.0..=1.0).contains(share))
        .map(|share| from.lerp(to, share))
        .find(|point| world.open(*point))
        .unwrap_or_else(|| from.lerp(to, 0.5))
}

fn cover_or(world: &dyn World, threat: Vec3, near: Vec3, search: f32) -> Vec3 {
    world.cover_from(threat, near, search).unwrap_or(near)
}

fn watch_spot(
    world: &dyn World,
    far: Vec3,
    watch: Vec3,
    nears: [Vec3; 3],
    search: f32,
    up: Vec3,
) -> Vec3 {
    for near in nears {
        let spot = world.cover_from(far, near, search);
        if let Some(spot) = spot.filter(|s| world.sees(*s + up, watch)) {
            return spot;
        }
    }
    cover_or(world, far, nears[0], search)
}

fn cross_spot(
    world: &dyn World,
    numbers: &BotNumbers,
    far: Vec3,
    hold: Vec3,
    sides: [Vec3; 3],
) -> Vec3 {
    let [inside, center, inward] = sides;
    let play = &numbers.play;
    let up = Vec3::new(0.0, numbers.player.eye_height_m, 0.0);
    let watch = inside + up;
    let lateral = Vec3::new(-inward.z, 0.0, inward.x);
    let first = flat(hold - inside).normalize_or_zero();
    let apart = sin_cos(radians(play.crossfire_deg)).1;
    for side in [1.0, -1.0] {
        for share in [0.5, 0.3, 0.75] {
            let near = inside.lerp(center, share) + lateral * (side * play.gate_step_m * 2.0);
            let Some(spot) = world.cover_from(far, near, play.cover_search_m) else {
                continue;
            };
            let angle = flat(spot - inside).normalize_or_zero().dot(first);
            if angle <= apart
                && flat_distance(spot, hold) >= play.spacing_m * 2.0
                && world.sees(spot + up, watch)
            {
                return spot;
            }
        }
    }
    hold
}

struct WayInput<'a> {
    site: &'a Site,
    choke: usize,
    attack_home: Vec3,
    defend_home: Vec3,
    plant: Vec3,
}

fn build_way(
    world: &dyn World,
    numbers: &BotNumbers,
    input: &WayInput<'_>,
    buffer: &mut Vec<Vec3>,
) -> Way {
    let play = &numbers.play;
    let places = world.places();
    let gate = &places.chokes[input.choke];
    let mid = gate.middle();
    let along_gate = flat(gate.b - gate.a);
    let mut inward = Vec3::new(-along_gate.z, 0.0, along_gate.x).normalize_or_zero();
    if places.zone_at(mid + inward * play.gate_step_m) != Some(input.site.zone) {
        inward = -inward;
    }
    let inside = mid + inward * play.gate_step_m;
    let outside = mid - inward * play.gate_step_m;
    let back = play.gate_step_m + play.stage_back_m;
    let (attack_stage, attack_length) =
        stage_on_path(world, input.attack_home, inside, back, buffer);
    let (defend_stage, defend_length) =
        stage_on_path(world, input.defend_home, inside, back, buffer);
    let (retake_stage, retake_length) =
        stage_on_path(world, input.defend_home, outside, play.stage_back_m, buffer);
    let retake_clear = !crosses(
        world,
        input.defend_home,
        buffer,
        Some(input.site.zone),
        play.gate_step_m,
    );
    let up = Vec3::new(0.0, numbers.player.eye_height_m, 0.0);
    let face = mid + up;
    let far = mid - inward * back + up;
    let watch = inside + up * 0.75;
    let search = play.cover_search_m;
    let center = input.site.center;
    let hold_nears = [
        inside.lerp(center, 0.5),
        inside.lerp(center, 0.3),
        inside.lerp(center, 0.75),
    ];
    let hold = watch_spot(world, far, watch, hold_nears, search, up);
    let cross = cross_spot(world, numbers, far, hold, [inside, center, inward]);
    let post_nears = [
        input.plant.lerp(inside, 0.5),
        input.plant.lerp(inside, 0.35),
        input.plant.lerp(inside, 0.65),
    ];
    let post = watch_spot(world, far, watch, post_nears, search, up);
    let post = if flat_distance(post, input.plant) < play.post_clear_m {
        hold
    } else {
        post
    };
    Way {
        choke: input.choke,
        mid,
        inward,
        inside,
        outside,
        attack_stage,
        defend_stage,
        attack_length,
        defend_length,
        retake_stage,
        retake_length,
        retake_clear,
        hold: Hold { at: hold, face },
        cross: Hold { at: cross, face },
        post: Hold { at: post, face },
        smoke_at: open_between(world, hold, inside, play.smoke_open_step_m),
        flash_at: inside + inward * play.flash_inside_m + up,
    }
}

fn back_way(places: &Places, ways: &[Way], safe: &[bool]) -> Option<usize> {
    (0..ways.len())
        .filter(|&w| {
            places
                .zone_at(ways[w].outside)
                .is_some_and(|zone| safe.get(zone as usize).copied().unwrap_or(false))
        })
        .min_by(|&a, &b| {
            ways[a]
                .defend_length
                .total_cmp(&ways[b].defend_length)
                .then(a.cmp(&b))
        })
}

fn zone_sides(
    world: &dyn World,
    numbers: &BotNumbers,
    homes: [Vec3; 2],
    buffer: &mut Vec<Vec3>,
) -> Vec<bool> {
    let places = world.places();
    places
        .zones
        .iter()
        .enumerate()
        .map(|(index, zone)| {
            let centre = (zone.min + zone.max) * 0.5;
            let site = places.sites.iter().any(|s| s.zone as usize == index);
            let attack = path_length(world, homes[0], centre, buffer);
            let defend = path_length(world, homes[1], centre, buffer);
            !site && defend + numbers.play.safe_margin_m < attack
        })
        .collect()
}

fn build_site(
    world: &dyn World,
    numbers: &BotNumbers,
    site_index: usize,
    homes: [Vec3; 2],
    safe: &[bool],
    buffer: &mut Vec<Vec3>,
) -> SiteTactics {
    let places: &Places = world.places();
    let site = &places.sites[site_index];
    let mut ways: Vec<Way> = Vec::new();
    for (choke, gate) in places.chokes.iter().enumerate() {
        if !gate.zones.contains(&site.zone) || gate.zones[0] == gate.zones[1] {
            continue;
        }
        let input = WayInput {
            site,
            choke,
            attack_home: homes[0],
            defend_home: homes[1],
            plant: site.center,
        };
        ways.push(build_way(world, numbers, &input, buffer));
    }
    ways.sort_by(|a, b| {
        a.attack_length
            .total_cmp(&b.attack_length)
            .then(a.choke.cmp(&b.choke))
    });
    let mut defend_order: Vec<usize> = (0..ways.len()).collect();
    defend_order.sort_by(|&a, &b| {
        ways[a]
            .defend_length
            .total_cmp(&ways[b].defend_length)
            .then(a.cmp(&b))
    });
    let mut retake_order: Vec<usize> = (0..ways.len()).filter(|&w| ways[w].retake_clear).collect();
    retake_order.sort_by(|&a, &b| {
        ways[a]
            .retake_length
            .total_cmp(&ways[b].retake_length)
            .then(a.cmp(&b))
    });
    let up = Vec3::new(0.0, numbers.player.eye_height_m, 0.0);
    let plant = match ways.first() {
        Some(main) => {
            let spot = cover_or(
                world,
                main.outside + up,
                site.center,
                numbers.play.reach_m * 2.0,
            );
            if places.site_at(spot) == Some(site_index) {
                spot
            } else {
                site.center
            }
        }
        None => site.center,
    };
    SiteTactics {
        back: back_way(places, &ways, safe),
        ways,
        plant,
        defend_order,
        retake_order,
    }
}

fn mid_hold(
    world: &dyn World,
    numbers: &BotNumbers,
    homes: [Vec3; 2],
    safe: &[bool],
    buffer: &mut Vec<Vec3>,
) -> Hold {
    let play = &numbers.play;
    let places = world.places();
    let length = path_length(world, homes[1], homes[0], buffer);
    let step = max(play.gate_step_m, 0.1);
    let mut edge = 0.0;
    let mut distance = 0.0;
    while distance <= length {
        let point = along(homes[1], buffer, distance);
        if let Some(zone) = places.zone_at(point)
            && !safe.get(zone as usize).copied().unwrap_or(true)
        {
            break;
        }
        edge = distance;
        distance += step;
    }
    let up = Vec3::new(0.0, numbers.player.eye_height_m, 0.0);
    let watch = along(homes[1], buffer, edge + step) + up * 0.75;
    let far = along(homes[1], buffer, edge + play.stage_back_m) + up;
    let nears = [
        along(homes[1], buffer, edge - step),
        along(homes[1], buffer, edge - 2.0 * step),
        along(homes[1], buffer, edge),
    ];
    Hold {
        at: watch_spot(world, far, watch, nears, play.cover_search_m, up),
        face: watch,
    }
}

pub fn build(world: &dyn World, numbers: &BotNumbers, buffer: &mut Vec<Vec3>) -> Tactics {
    let places = world.places();
    let attack_home = mean(&places.attack_spawns);
    let defend_home = mean(&places.defend_spawns);
    let homes = [attack_home, defend_home];
    let safe = zone_sides(world, numbers, homes, buffer);
    let sites = (0..places.sites.len())
        .map(|s| build_site(world, numbers, s, homes, &safe, buffer))
        .collect();
    let play = &numbers.play;
    let mid_hold = mid_hold(world, numbers, homes, &safe, buffer);
    let length = path_length(world, attack_home, defend_home, buffer);
    let middle = length * play.mid_share;
    let up = Vec3::new(0.0, numbers.player.eye_height_m, 0.0);
    let middle_point = along(attack_home, buffer, middle);
    let spot = along(attack_home, buffer, middle - 2.0 * play.stage_back_m);
    Tactics {
        sites,
        mid_hold,
        mid_spot: Hold {
            at: spot,
            face: middle_point + up,
        },
        attack_home,
        defend_home,
        safe,
    }
}
