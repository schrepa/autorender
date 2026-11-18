use std::cell::Cell;

use glam::Vec3;
use serde::Deserialize;

use super::peers;
use super::shapes::{least, most, within};
use super::world::{Footing, Support, Work};
use super::{Capsule, Ground, Hit, Kind, Solids, Surface};

const SKIN: f32 = 0.003;
const PROBE: f32 = 0.006;
const MAX_PLANES: usize = 4;
const REST: f32 = 1e-6;
const PROGRESS: f32 = 1e-4;
const COSINE_TERMS: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct Player {
    pub height_m: f32,
    pub crouch_height_m: f32,
    pub radius_m: f32,
    pub jump_height_m: f32,
    pub jump_apex_seconds: f32,
    pub ground_accel_mps2: f32,
    pub ground_friction: f32,
    pub braking_friction: f32,
    pub braking_decel_mps2: f32,
    pub air_accel_per_s: f32,
    pub air_wish_cap_mps: f32,
    pub jump_speed_cap_mps: f32,
    pub crouch_jump_lift_m: f32,
    pub step_height_m: f32,
    pub slope_limit_deg: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub position: Vec3,
    pub velocity: Vec3,
    pub grounded: bool,
    pub crouched: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wish {
    pub direction: Vec3,
    pub speed: f32,
    pub jump: bool,
    pub crouch: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landed {
    pub speed: f32,
    pub surface: Surface,
    pub kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moved {
    pub body: Body,
    pub landed: Option<Landed>,
    pub ground: Option<Ground>,
    pub wall: Option<Hit>,
    pub stand_refused: bool,
    pub shift: f32,
    pub work: Work,
}

struct Tally<'a> {
    solids: &'a Solids,
    peers: &'a [Capsule],
    work: Cell<Work>,
}

#[derive(Clone, Copy)]
struct Contact {
    distance: f32,
    normal: Vec3,
    hit: Option<Hit>,
}

impl Tally<'_> {
    fn count<R>(&self, change: impl FnOnce(&mut Work) -> R) -> R {
        let mut work = self.work.get();
        let result = change(&mut work);
        self.work.set(work);
        result
    }

    fn capsule_sweep(&self, capsule: Capsule, motion: Vec3) -> Option<Contact> {
        let map = self
            .count(|work| self.solids.capsule_sweep_counted(capsule, motion, work))
            .map(|hit| Contact {
                distance: hit.distance,
                normal: hit.normal,
                hit: Some(hit),
            });
        let peer = peers::sweep(capsule, motion, self.peers).map(|(fraction, normal)| Contact {
            distance: fraction * motion.length(),
            normal,
            hit: None,
        });
        match (map, peer) {
            (Some(map), Some(peer)) if peer.distance < map.distance => Some(peer),
            (Some(map), _) => Some(map),
            (None, peer) => peer,
        }
    }

    fn capsule_overlaps(&self, capsule: Capsule) -> bool {
        self.count(|work| self.solids.capsule_overlaps_counted(capsule, work))
    }

    fn capsule_push_out(&self, capsule: Capsule) -> Vec3 {
        self.count(|work| self.solids.capsule_push_out_counted(capsule, work))
    }

    fn support(&self, footing: &Footing) -> Option<Support> {
        self.count(|work| self.solids.support(footing, work))
    }
}

struct Rules {
    height: f32,
    radius: f32,
    step: f32,
    cos_limit: f32,
    gravity: f32,
    takeoff: f32,
}

#[derive(Clone, Copy)]
struct Slid {
    position: Vec3,
    planes: [Vec3; MAX_PLANES],
    count: usize,
    wall: Option<Hit>,
}

fn cos_degrees(degrees: f32) -> f32 {
    let x = within(degrees, 0.0, 180.0) * (std::f32::consts::PI / 180.0);
    let x2 = x * x;
    let mut sum = 0.0f32;
    for n in (0..COSINE_TERMS).rev() {
        let k = (2 * n + 1) as f32 * (2 * n + 2) as f32;
        sum = 1.0 - x2 * sum / k;
    }
    sum
}

impl Rules {
    fn new(numbers: &Player, height: f32) -> Rules {
        let apex = numbers.jump_apex_seconds;
        Rules {
            height,
            radius: numbers.radius_m,
            step: numbers.step_height_m,
            cos_limit: cos_degrees(numbers.slope_limit_deg),
            gravity: 2.0 * numbers.jump_height_m / (apex * apex),
            takeoff: 2.0 * numbers.jump_height_m / apex,
        }
    }

    fn capsule(&self, feet: Vec3) -> Capsule {
        Capsule {
            feet,
            height: self.height,
            radius: self.radius,
        }
    }

    fn support(&self, solids: &Tally, feet: Vec3, highest: f32) -> Option<Support> {
        solids.support(&Footing {
            bottom: feet + Vec3::Y * self.radius,
            radius: self.radius,
            reach: PROBE,
            cos_limit: self.cos_limit,
            highest,
        })
    }

    fn walkable(&self, normal: Vec3) -> bool {
        normal.y >= self.cos_limit
    }
}

pub fn capsule_of(body: &Body, numbers: &Player) -> Capsule {
    Capsule {
        feet: body.position,
        height: if body.crouched {
            numbers.crouch_height_m
        } else {
            numbers.height_m
        },
        radius: numbers.radius_m,
    }
}

fn flat(v: Vec3) -> Vec3 {
    Vec3::new(v.x, 0.0, v.z)
}

fn clip(v: Vec3, normal: Vec3) -> Vec3 {
    let into = v.dot(normal);
    if into < 0.0 { v - normal * into } else { v }
}

fn constrain(v: Vec3, planes: &[Vec3]) -> Vec3 {
    if planes.iter().all(|plane| v.dot(*plane) >= -REST) {
        return v;
    }
    for (i, plane) in planes.iter().enumerate() {
        let clipped = clip(v, *plane);
        if planes
            .iter()
            .enumerate()
            .all(|(j, other)| j == i || clipped.dot(*other) >= -REST)
        {
            return clipped;
        }
    }
    for i in 0..planes.len() {
        for j in i + 1..planes.len() {
            let crease = planes[i].cross(planes[j]);
            let length = crease.length();
            if length <= REST {
                continue;
            }
            let direction = crease / length;
            let along = direction * direction.dot(v);
            if planes.iter().all(|plane| along.dot(*plane) >= -REST) {
                return along;
            }
        }
    }
    Vec3::ZERO
}

fn follow_ground(v: Vec3, normal: Vec3) -> Vec3 {
    Vec3::new(v.x, -(v.x * normal.x + v.z * normal.z) / normal.y, v.z)
}

fn wall_plane(normal: Vec3) -> Vec3 {
    let level = flat(normal);
    if level.length_squared() > REST {
        level.normalize()
    } else {
        normal
    }
}

fn slide(solids: &Tally, rules: &Rules, start: Vec3, motion: Vec3, grounded: bool) -> Slid {
    let mut slid = Slid {
        position: start,
        planes: [Vec3::ZERO; MAX_PLANES],
        count: 0,
        wall: None,
    };
    let mut remaining = motion;
    for _ in 0..MAX_PLANES {
        let length = remaining.length();
        if length <= REST {
            break;
        }
        let Some(hit) = solids.capsule_sweep(rules.capsule(slid.position), remaining) else {
            slid.position += remaining;
            break;
        };
        let fraction = hit.distance / length;
        slid.position += remaining * fraction + hit.normal * SKIN;
        remaining *= 1.0 - fraction;
        let walkable = rules.walkable(hit.normal);
        if !walkable && slid.wall.is_none() {
            slid.wall = hit.hit;
        }
        if grounded && walkable {
            remaining = follow_ground(remaining, hit.normal);
            continue;
        }
        let plane = if grounded {
            wall_plane(hit.normal)
        } else {
            hit.normal
        };
        if slid.count < MAX_PLANES {
            slid.planes[slid.count] = plane;
            slid.count += 1;
        }
        remaining = constrain(remaining, &slid.planes[..slid.count]);
    }
    slid
}

fn drop_onto(solids: &Tally, rules: &Rules, from: Vec3, depth: f32) -> Option<Vec3> {
    let drop = Vec3::new(0.0, -depth, 0.0);
    let hit = solids.capsule_sweep(rules.capsule(from), drop)?;
    Some(from + drop * (hit.distance / depth) + hit.normal * SKIN)
}

fn step_up(solids: &Tally, rules: &Rules, start: Vec3, motion: Vec3) -> Option<Slid> {
    let rise = Vec3::Y * rules.step;
    let raised = match solids.capsule_sweep(rules.capsule(start), rise) {
        Some(hit) => start + Vec3::Y * most(hit.distance - SKIN, 0.0),
        None => start + rise,
    };
    let lifted = raised.y - start.y;
    if lifted <= SKIN {
        return None;
    }
    let mut moved = slide(solids, rules, raised, flat(motion), true);
    let landing = drop_onto(solids, rules, moved.position, lifted + PROBE)?;
    rules.support(solids, landing, start.y + rules.step)?;
    moved.position = landing;
    Some(moved)
}

fn travel(solids: &Tally, rules: &Rules, start: Vec3, motion: Vec3, grounded: bool) -> Slid {
    let plain = slide(solids, rules, start, motion, grounded);
    if !grounded || plain.wall.is_none() {
        return plain;
    }
    let Some(climbed) = step_up(solids, rules, start, motion) else {
        return plain;
    };
    let reach = |slid: &Slid| flat(slid.position - start).length();
    if reach(&climbed) > reach(&plain) + PROGRESS {
        climbed
    } else {
        plain
    }
}

fn settle_ground(
    solids: &Tally,
    rules: &Rules,
    position: Vec3,
    rising: bool,
    was_grounded: bool,
) -> (Vec3, Option<Support>) {
    if rising {
        return (position, None);
    }
    let highest = position.y + rules.step;
    if let Some(support) = rules.support(solids, position, highest) {
        return (position, Some(support));
    }
    if !was_grounded {
        return (position, None);
    }
    let Some(landing) = drop_onto(solids, rules, position, rules.step + PROBE) else {
        return (position, None);
    };
    match rules.support(solids, landing, highest) {
        Some(support) => (landing, Some(support)),
        None => (position, None),
    }
}

fn accelerate(current: Vec3, target: Vec3, (rate, cap): (f32, f32), dt: f32) -> Vec3 {
    let wished = target.length();
    if wished == 0.0 {
        return current;
    }
    let direction = target / wished;
    let missing = least(wished, cap) - current.dot(direction);
    if missing <= 0.0 {
        return current;
    }
    current + direction * least(missing, rate * wished * dt)
}

fn braked(velocity: Vec3, numbers: &Player, dt: f32) -> Vec3 {
    let slowed = velocity
        - (velocity * numbers.braking_friction
            + velocity.normalize_or_zero() * numbers.braking_decel_mps2)
            * dt;
    if slowed.dot(velocity) <= 0.0 {
        Vec3::ZERO
    } else {
        slowed
    }
}

fn ground_velocity(current: Vec3, target: Vec3, numbers: &Player, dt: f32) -> Vec3 {
    let wished = target.length();
    let over = current.length() > wished;
    let mut velocity = current;
    if wished == 0.0 || over {
        velocity = braked(current, numbers, dt);
        if wished > 0.0 && velocity.length() < wished && target.dot(current) > 0.0 {
            velocity = current.normalize_or_zero() * wished;
        }
    } else {
        let direction = target / wished;
        let turned = direction * velocity.length();
        velocity -= (velocity - turned) * least(dt * numbers.ground_friction, 1.0);
    }
    if wished == 0.0 {
        return velocity;
    }
    let limit = if over { velocity.length() } else { wished };
    capped(
        velocity + target / wished * numbers.ground_accel_mps2 * dt,
        limit,
    )
}

fn air_velocity(current: Vec3, target: Vec3, numbers: &Player, dt: f32) -> Vec3 {
    accelerate(
        current,
        target,
        (numbers.air_accel_per_s, numbers.air_wish_cap_mps),
        dt,
    )
}

fn capped(velocity: Vec3, cap: f32) -> Vec3 {
    let speed = velocity.length();
    if speed > cap && speed > 0.0 {
        velocity * (cap / speed)
    } else {
        velocity
    }
}

fn settle_crouch(solids: &Tally, body: &Body, wants: bool, numbers: &Player) -> (bool, bool, f32) {
    let gap = least(
        numbers.crouch_jump_lift_m,
        most(numbers.height_m - numbers.crouch_height_m, 0.0),
    );
    let airborne = !body.grounded;
    if wants {
        let shift = if airborne && !body.crouched { gap } else { 0.0 };
        return (true, false, shift);
    }
    if !body.crouched {
        return (false, false, 0.0);
    }
    let standing = |feet: Vec3| Capsule {
        feet,
        height: numbers.height_m,
        radius: numbers.radius_m,
    };
    if airborne && !solids.capsule_overlaps(standing(body.position - Vec3::Y * gap)) {
        return (false, false, -gap);
    }
    if solids.capsule_overlaps(standing(body.position)) {
        (true, true, 0.0)
    } else {
        (false, false, 0.0)
    }
}

fn wish_target(wish: &Wish) -> Vec3 {
    let mut direction = flat(wish.direction);
    if direction.length_squared() > 1.0 {
        direction = direction.normalize();
    }
    direction * wish.speed
}

pub fn step(
    (world, peers): (&Solids, &[Capsule]),
    body: Body,
    wish: Wish,
    numbers: &Player,
    dt: f32,
) -> Moved {
    let start = capsule_of(&body, numbers);
    let clear: Vec<Capsule> = (peers.iter().copied())
        .filter(|peer| peers::overlap(start, *peer).is_none())
        .collect();
    let tally = Tally {
        solids: world,
        peers: &clear,
        work: Cell::new(Work::default()),
    };
    let solids = &tally;
    let (crouched, stand_refused, shift) = settle_crouch(solids, &body, wish.crouch, numbers);
    let from = body.position + Vec3::Y * shift;
    let height = if crouched {
        numbers.crouch_height_m
    } else {
        numbers.height_m
    };
    let rules = Rules::new(numbers, height);
    let start_support = if body.grounded {
        rules.support(solids, body.position, body.position.y + rules.step)
    } else {
        None
    };
    let target = wish_target(&wish);
    let current = flat(body.velocity);
    let mut velocity = match start_support {
        Some(_) => ground_velocity(current, target, numbers, dt),
        None => air_velocity(current, target, numbers, dt),
    };
    let jumped = start_support.is_some() && wish.jump;
    if jumped {
        velocity = capped(velocity, numbers.jump_speed_cap_mps);
    }
    let grounded = start_support.is_some() && !jumped;
    let upward = if jumped {
        rules.takeoff
    } else {
        body.velocity.y
    };
    let rise = if grounded {
        0.0
    } else {
        upward * dt - 0.5 * rules.gravity * dt * dt
    };
    velocity.y = if grounded {
        0.0
    } else {
        upward - rules.gravity * dt
    };
    let fall = velocity.y;
    let mut motion = Vec3::new(velocity.x * dt, rise, velocity.z * dt);
    if let Some(support) =
        start_support.filter(|support| grounded && rules.walkable(support.normal))
    {
        motion = follow_ground(motion, support.normal);
    }
    let moved = travel(solids, &rules, from, motion, grounded);
    velocity = constrain(velocity, &moved.planes[..moved.count]);
    let (settled, support) =
        settle_ground(solids, &rules, moved.position, velocity.y > 0.0, grounded);
    let mut position = settled + solids.capsule_push_out(rules.capsule(settled));
    if solids.capsule_overlaps(rules.capsule(position))
        && !solids.capsule_overlaps(rules.capsule(from))
    {
        position = from;
        velocity = Vec3::ZERO;
    }
    if support.is_some() {
        velocity.y = 0.0;
    }
    let landed = support.filter(|_| !body.grounded).map(|support| Landed {
        speed: most(-fall, 0.0),
        surface: support.ground.surface,
        kind: support.ground.kind,
    });
    Moved {
        body: Body {
            position,
            velocity,
            grounded: support.is_some(),
            crouched,
        },
        landed,
        ground: support.map(|support| support.ground),
        wall: moved.wall,
        stand_refused,
        shift,
        work: tally.work.get(),
    }
}
