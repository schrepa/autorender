use glam::Vec3;

use crate::bots::aim::{flat, flat_distance, max};
use crate::bots::numbers::BotNumbers;
use crate::bots::places::{Places, Site};
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
    pub hold: Hold,
    pub post: Hold,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SiteTactics {
    pub ways: Vec<Way>,
    pub plant: Vec3,
    pub defend_order: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tactics {
    pub sites: Vec<SiteTactics>,
    pub mid_hold: Hold,
    pub mid_spot: Hold,
    pub attack_home: Vec3,
    pub defend_home: Vec3,
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
    let post_nears = [
        input.plant.lerp(inside, 0.35),
        input.plant,
        input.plant.lerp(inside, 0.6),
    ];
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
        hold: Hold {
            at: watch_spot(world, far, watch, hold_nears, search, up),
            face,
        },
        post: Hold {
            at: watch_spot(world, far, watch, post_nears, search, up),
            face,
        },
    }
}

fn build_site(
    world: &dyn World,
    numbers: &BotNumbers,
    site_index: usize,
    homes: [Vec3; 2],
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
        ways,
        plant,
        defend_order,
    }
}

pub fn build(world: &dyn World, numbers: &BotNumbers, buffer: &mut Vec<Vec3>) -> Tactics {
    let places = world.places();
    let attack_home = mean(&places.attack_spawns);
    let defend_home = mean(&places.defend_spawns);
    let sites = (0..places.sites.len())
        .map(|s| build_site(world, numbers, s, [attack_home, defend_home], buffer))
        .collect();
    let play = &numbers.play;
    let length = path_length(world, attack_home, defend_home, buffer);
    let middle = length * play.mid_share;
    let up = Vec3::new(0.0, numbers.player.eye_height_m, 0.0);
    let threat = along(attack_home, buffer, middle - play.stage_back_m);
    let near = along(attack_home, buffer, middle + play.gate_step_m);
    let middle_point = along(attack_home, buffer, middle);
    let spot = along(attack_home, buffer, middle - 2.0 * play.stage_back_m);
    Tactics {
        sites,
        mid_hold: Hold {
            at: cover_or(world, threat + up, near, play.cover_search_m),
            face: threat + up,
        },
        mid_spot: Hold {
            at: spot,
            face: middle_point + up,
        },
        attack_home,
        defend_home,
    }
}
