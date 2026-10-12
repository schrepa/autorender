//! The wild city: a deterministic procedural city of diverse low-poly shapes,
//! built through the public SDK from the same configuration as the tower
//! field.
//!
//! Static renderables fill 20-unit blocks between 4-unit streets outward from
//! the center: a skyscraper core, midtown, low-rise residential streets and
//! industrial outskirts, with parks, plazas, construction sites, stadiums, a
//! river with bridges, an elevated rail ring and a landmark tower. Moving
//! renderables are traffic (cars, buses, trains, boats), drones and
//! helicopters, crane jibs with their loads, Ferris wheels, rooftop beacons
//! and façade elevators. With `hierarchy`, each block's parts are children of
//! a transform-only block node, and moving assemblies hang from moving pivot
//! nodes; without it, every part is a root whose motion reproduces the same
//! poses. Both layouts display the same city.

mod world;

use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::ops::DerefMut;

use engine::prelude::*;

use crate::scene::{Animation, hash, position_for_index};
use crate::shapes::{Library, Paint, Shape, Tex};
use crate::{FrameMetrics, StressConfig, metrics};
pub(crate) use world::CityWorld;

/// Street grid: 20-unit blocks between 4-unit streets, centered on multiples
/// of `PERIOD`.
const PERIOD: f32 = 24.0;
const HALF_BLOCK: f32 = 10.0;
const ROAD_TOP: f32 = 0.15;
/// Static renderables of an average block, for sizing districts.
const PER_BLOCK: f32 = 80.0;
const RAIL_HEIGHT: f32 = 9.5;
const WATER_TOP: f32 = -0.2;
const INV_TAU: f32 = 1.0 / TAU;

const NEON: [[f32; 3]; 8] = [
    [0.09, 0.88, 0.78],
    [0.10, 0.49, 1.0],
    [0.55, 0.20, 1.0],
    [1.0, 0.22, 0.52],
    [1.0, 0.61, 0.13],
    [0.25, 0.93, 0.54],
    [0.45, 0.72, 1.0],
    [0.92, 0.37, 0.97],
];
const GLASS: [[f32; 3]; 5] = [
    [0.35, 0.62, 0.95],
    [0.25, 0.75, 0.85],
    [0.6, 0.7, 0.85],
    [0.45, 0.5, 0.9],
    [0.3, 0.85, 0.7],
];
const STONE: [[f32; 3]; 6] = [
    [0.85, 0.72, 0.55],
    [0.75, 0.45, 0.35],
    [0.9, 0.85, 0.75],
    [0.65, 0.6, 0.55],
    [0.8, 0.55, 0.4],
    [0.95, 0.9, 0.8],
];
const PASTEL: [[f32; 3]; 6] = [
    [0.95, 0.6, 0.55],
    [0.55, 0.75, 0.95],
    [0.6, 0.9, 0.6],
    [0.98, 0.85, 0.5],
    [0.75, 0.6, 0.95],
    [1.0, 0.7, 0.45],
];
const RUST: [[f32; 3]; 4] = [
    [0.6, 0.6, 0.62],
    [0.72, 0.5, 0.35],
    [0.5, 0.55, 0.6],
    [0.8, 0.75, 0.6],
];
const ROOFS: [[f32; 3]; 4] = [
    [0.8, 0.25, 0.15],
    [0.25, 0.3, 0.42],
    [0.55, 0.2, 0.2],
    [0.2, 0.45, 0.3],
];
const CARS: [[f32; 3]; 8] = [
    [0.9, 0.1, 0.1],
    [0.95, 0.95, 0.95],
    [0.1, 0.1, 0.12],
    [0.1, 0.35, 0.9],
    [0.95, 0.75, 0.1],
    [0.2, 0.7, 0.3],
    [0.6, 0.6, 0.65],
    [1.0, 0.45, 0.1],
];
const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const ASPHALT: [f32; 3] = [0.6, 0.6, 0.66];
const CRANE: [f32; 3] = [1.0, 0.72, 0.1];
const DIRT: [f32; 3] = [0.55, 0.42, 0.3];

/// Deterministic draws from a seed.
pub(crate) struct Rng(u32);

impl Rng {
    pub(crate) fn new(seed: u32, a: i32, b: i32) -> Self {
        Self(hash(
            seed ^ hash((a as u32).wrapping_mul(73_856_093) ^ (b as u32).wrapping_mul(19_349_663)),
        ))
    }
    pub(crate) fn next(&mut self) -> u32 {
        self.0 = hash(self.0.wrapping_add(0x9e37_79b9));
        self.0
    }
    pub(crate) fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 16_777_216.0
    }
    pub(crate) fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }
    pub(crate) fn below(&mut self, count: u32) -> u32 {
        self.next() % count
    }
    pub(crate) fn chance(&mut self, percent: u32) -> bool {
        self.below(100) < percent
    }
    pub(crate) fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u32) as usize]
    }
    pub(crate) fn sign(&mut self) -> f32 {
        if self.chance(50) { 1.0 } else { -1.0 }
    }
}

// Motions: pure functions of simulated time, so both layouts agree.

/// Around a rectangle in the XZ plane: counterclockwise seen from above for
/// a positive speed (units per second), clockwise for a negative one.
#[derive(Component, Clone, Copy, Debug)]
pub(crate) struct Travel {
    pub(crate) center: Vec3,
    pub(crate) half: Vec2,
    pub(crate) speed: f32,
    pub(crate) phase: f32,
}

/// Around a horizontal circle (angle `a`: `center + radius (cos a, 0, -sin a)`,
/// as `Quat::from_rotation_y(a)` moves +X), bobbing by `bob sin 2a`, turned
/// by its own facing angle about Y.
#[derive(Component, Clone, Copy, Debug)]
pub(crate) struct Orbit {
    pub(crate) center: Vec3,
    pub(crate) radius: f32,
    pub(crate) speed: f32,
    pub(crate) phase: f32,
    pub(crate) bob: f32,
    pub(crate) facing_speed: f32,
    pub(crate) facing_phase: f32,
}

/// Around the circle `center + u cos a + v sin a`, keeping its rotation.
#[derive(Component, Clone, Copy, Debug)]
pub(crate) struct Wheel {
    pub(crate) center: Vec3,
    pub(crate) u: Vec3,
    pub(crate) v: Vec3,
    pub(crate) speed: f32,
    pub(crate) phase: f32,
}

/// `base` then a turn about local Y (or Z), in place.
#[derive(Component, Clone, Copy, Debug)]
pub(crate) struct Spin {
    pub(crate) base: Quat,
    pub(crate) axis_z: bool,
    pub(crate) speed: f32,
    pub(crate) phase: f32,
}

/// Along `axis` from `anchor` by `amplitude sin a`.
#[derive(Component, Clone, Copy, Debug)]
pub(crate) struct Slide {
    pub(crate) anchor: Vec3,
    pub(crate) axis: Vec3,
    pub(crate) amplitude: f32,
    pub(crate) speed: f32,
    pub(crate) phase: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Motion {
    Travel(Travel),
    Orbit(Orbit),
    Wheel(Wheel),
    Spin(Spin),
    Slide(Slide),
}

/// `x` reduced to `[0, TAU)`, keeping trigonometry on its fast path.
fn wrap(x: f32) -> f32 {
    x - (x * INV_TAU).floor() * TAU
}

fn wrap4(x: Vec4) -> Vec4 {
    x - (x * INV_TAU).floor() * TAU
}

fn yaws() -> [Quat; 8] {
    [
        0.0, -FRAC_PI_2, PI, FRAC_PI_2, PI, FRAC_PI_2, 0.0, -FRAC_PI_2,
    ]
    .map(Quat::from_rotation_y)
}

impl Travel {
    fn at(&self, t: f32, yaws: &[Quat; 8]) -> (Vec3, Quat) {
        let (hx, hz) = (self.half.x, self.half.y);
        let perimeter = 4.0 * (hx + hz);
        let run = t * self.speed.abs() + self.phase;
        let mut distance = run - (run / perimeter).floor() * perimeter;
        let forward = self.speed >= 0.0;
        if !forward {
            distance = perimeter - distance;
        }
        let (a, b) = (2.0 * hx, 2.0 * hz);
        let (x, z, segment) = if distance < a {
            (-hx + distance, -hz, 0)
        } else if distance < a + b {
            (hx, -hz + distance - a, 1)
        } else if distance < 2.0 * a + b {
            (hx - (distance - a - b), hz, 2)
        } else {
            (-hx, hz - (distance - 2.0 * a - b), 3)
        };
        let rotation = yaws[segment + if forward { 0 } else { 4 }];
        (self.center + Vec3::new(x, 0.0, z), rotation)
    }
}

impl Orbit {
    fn at(&self, t: f32) -> (Vec3, Quat) {
        let (sin, cos) = math::sin_cos(wrap(self.speed * t + self.phase));
        let (facing_sin, facing_cos) =
            math::sin_cos(wrap(self.facing_speed * t + self.facing_phase) * 0.5);
        (
            self.center
                + Vec3::new(
                    self.radius * cos,
                    self.bob * 2.0 * sin * cos,
                    -self.radius * sin,
                ),
            Quat::from_xyzw(0.0, facing_sin, 0.0, facing_cos),
        )
    }
}

impl Wheel {
    fn at(&self, t: f32) -> Vec3 {
        let (sin, cos) = math::sin_cos(wrap(self.speed * t + self.phase));
        self.center + self.u * cos + self.v * sin
    }
}

impl Spin {
    fn at(&self, t: f32) -> Quat {
        let (sin, cos) = math::sin_cos(wrap(self.speed * t + self.phase) * 0.5);
        self.base * self.turn(sin, cos)
    }
    fn turn(&self, sin: f32, cos: f32) -> Quat {
        if self.axis_z {
            Quat::from_xyzw(0.0, 0.0, sin, cos)
        } else {
            Quat::from_xyzw(0.0, sin, 0.0, cos)
        }
    }
}

impl Slide {
    fn at(&self, t: f32) -> Vec3 {
        self.anchor + self.axis * (self.amplitude * math::sin(wrap(self.speed * t + self.phase)))
    }
}

impl Motion {
    /// The transform this motion gives `rest` at time `t`.
    pub(crate) fn apply(&self, rest: Transform, t: f32) -> Transform {
        let mut transform = rest;
        match self {
            Motion::Travel(travel) => {
                (transform.translation, transform.rotation) = travel.at(t, &yaws())
            }
            Motion::Orbit(orbit) => (transform.translation, transform.rotation) = orbit.at(t),
            Motion::Wheel(wheel) => transform.translation = wheel.at(t),
            Motion::Spin(spin) => transform.rotation = spin.at(t),
            Motion::Slide(slide) => transform.translation = slide.at(t),
        }
        transform
    }
}

fn simulated(time: &FixedTime) -> f32 {
    (time.elapsed_seconds % 10_000.0) as f32
}

// Four movers at a time: the scalar functions above, lane by lane, with the
// same bits (`sin_cos4` matches `sin_cos`).

fn orbit_lanes<'a, T: DerefMut<Target = Transform>>(
    t: f32,
    movers: impl Iterator<Item = (&'a Orbit, T)>,
) {
    math::for_each4(movers, |lanes| {
        let angle =
            wrap4(lanes.load(|(orbit, _)| orbit.speed) * t + lanes.load(|(orbit, _)| orbit.phase));
        let (sin, cos) = math::sin_cos4(angle);
        let facing = wrap4(
            lanes.load(|(orbit, _)| orbit.facing_speed) * t
                + lanes.load(|(orbit, _)| orbit.facing_phase),
        );
        let (facing_sin, facing_cos) = math::sin_cos4(facing * 0.5);
        lanes.for_each(|lane, (orbit, transform)| {
            let (sin, cos) = (sin[lane], cos[lane]);
            transform.translation = orbit.center
                + Vec3::new(
                    orbit.radius * cos,
                    orbit.bob * 2.0 * sin * cos,
                    -orbit.radius * sin,
                );
            transform.rotation = Quat::from_xyzw(0.0, facing_sin[lane], 0.0, facing_cos[lane]);
        });
    });
}

fn wheel_lanes<'a, T: DerefMut<Target = Transform>>(
    t: f32,
    movers: impl Iterator<Item = (&'a Wheel, T)>,
) {
    math::for_each4(movers, |lanes| {
        let angle =
            wrap4(lanes.load(|(wheel, _)| wheel.speed) * t + lanes.load(|(wheel, _)| wheel.phase));
        let (sin, cos) = math::sin_cos4(angle);
        lanes.for_each(|lane, (wheel, transform)| {
            transform.translation = wheel.center + wheel.u * cos[lane] + wheel.v * sin[lane];
        });
    });
}

fn spin_lanes<'a, T: DerefMut<Target = Transform>>(
    t: f32,
    movers: impl Iterator<Item = (&'a Spin, T)>,
) {
    math::for_each4(movers, |lanes| {
        let angle =
            wrap4(lanes.load(|(spin, _)| spin.speed) * t + lanes.load(|(spin, _)| spin.phase));
        let (sin, cos) = math::sin_cos4(angle * 0.5);
        lanes.for_each(|lane, (spin, transform)| {
            transform.rotation = spin.base * spin.turn(sin[lane], cos[lane]);
        });
    });
}

fn slide_lanes<'a, T: DerefMut<Target = Transform>>(
    t: f32,
    movers: impl Iterator<Item = (&'a Slide, T)>,
) {
    math::for_each4(movers, |lanes| {
        let angle =
            wrap4(lanes.load(|(slide, _)| slide.speed) * t + lanes.load(|(slide, _)| slide.phase));
        let (sin, _) = math::sin_cos4(angle);
        lanes.for_each(|lane, (slide, transform)| {
            transform.translation = slide.anchor + slide.axis * (slide.amplitude * sin[lane]);
        });
    });
}

/// Moves every mover to its pose at the tick's simulated time.
#[expect(clippy::type_complexity, reason = "a query per mover kind")]
fn animate(
    time: Res<FixedTime>,
    mut movers: ParamSet<(
        Query<(&Travel, &mut Transform)>,
        Query<(&Orbit, &mut Transform)>,
        Query<(&Wheel, &mut Transform)>,
        Query<(&Spin, &mut Transform)>,
        Query<(&Slide, &mut Transform)>,
    )>,
    mut metrics: ResMut<FrameMetrics>,
) {
    let start = metrics::now_ms();
    let (t, yaws) = (simulated(&time), yaws());
    for (travel, mut transform) in &mut movers.p0() {
        (transform.translation, transform.rotation) = travel.at(t, &yaws);
    }
    orbit_lanes(t, movers.p1().iter_mut());
    wheel_lanes(t, movers.p2().iter_mut());
    spin_lanes(t, movers.p3().iter_mut());
    slide_lanes(t, movers.p4().iter_mut());
    metrics.animation_ms += metrics::now_ms() - start;
}

/// Registers the city's motion system.
pub(crate) fn add_systems(app: &mut App) {
    app.add_systems(FixedSimulation, animate.in_set(Animation));
}

/// A renderable's shape and material.
#[derive(Clone, Copy, Debug)]
struct Look {
    shape: Shape,
    paint: Paint,
}

fn look(shape: Shape, tex: Tex, color: [f32; 3]) -> Look {
    Look {
        shape,
        paint: Paint { tex, color },
    }
}

#[derive(Debug)]
struct Node {
    parent: Option<u32>,
    transform: Transform,
    look: Option<Look>,
    motion: Option<Motion>,
    moving: bool,
}

/// Where parts attach: a transform-only node (with `hierarchy`) or the world,
/// through `frame`, which is never scaled.
#[derive(Clone, Copy)]
struct Anchor {
    node: Option<u32>,
    frame: Transform,
}

/// `frame` then `local`, for an unscaled `frame`.
pub(crate) fn compose(frame: &Transform, local: &Transform) -> Transform {
    Transform {
        translation: frame.translation + frame.rotation * local.translation,
        rotation: frame.rotation * local.rotation,
        scale: local.scale,
    }
}

pub(crate) fn part_at(translation: Vec3, scale: Vec3) -> Transform {
    Transform {
        translation,
        rotation: Quat::IDENTITY,
        scale,
    }
}

pub(crate) fn turned(translation: Vec3, yaw: f32, scale: Vec3) -> Transform {
    Transform {
        translation,
        rotation: Quat::from_rotation_y(yaw),
        scale,
    }
}

/// A roof moving parts can use: its block anchor, the building's base
/// center in that frame, its height and half width.
#[derive(Clone, Copy)]
struct Rooftop {
    anchor: Anchor,
    base: Vec3,
    height: f32,
    half: f32,
}

#[derive(Clone, Copy)]
struct Site {
    anchor: Anchor,
    at: Vec3,
    size: f32,
    yaw: f32,
}

struct City {
    seed: u32,
    hierarchy: bool,
    nodes: Vec<Node>,
    static_left: u32,
    moving_left: u32,
    node_room: u32,
    blocks: Vec<(i32, i32, f32)>,
    rooftops: Vec<Rooftop>,
    cranes: Vec<Site>,
    wheels: Vec<Site>,
    river: Option<(i32, f32)>,
    rail: Option<i32>,
    radius: f32,
    /// Skyline centers in block units: position, reach and peak intensity.
    centers: Vec<(Vec2, f32, f32)>,
}

impl City {
    /// How tall the block at (`bx`, `bz`) builds, in `[0, 1]`: highest at
    /// the core and the secondary centers.
    fn skyline(&self, bx: i32, bz: i32) -> f32 {
        let at = Vec2::new(bx as f32, bz as f32);
        self.centers
            .iter()
            .fold(0.0, |tallest, &(center, reach, peak)| {
                math::max(tallest, peak * (1.0 - at.distance(center) / reach))
            })
    }

    fn anchor(&mut self, frame: Transform) -> Anchor {
        if self.hierarchy && self.node_room > 0 {
            self.node_room -= 1;
            self.nodes.push(Node {
                parent: None,
                transform: frame,
                look: None,
                motion: None,
                moving: false,
            });
            Anchor {
                node: Some(self.nodes.len() as u32 - 1),
                frame,
            }
        } else {
            Anchor { node: None, frame }
        }
    }

    /// A static part; false once the static budget is spent.
    fn part(&mut self, anchor: &Anchor, local: Transform, look: Look) -> bool {
        if self.static_left == 0 {
            return false;
        }
        self.static_left -= 1;
        self.nodes.push(Node {
            parent: anchor.node,
            transform: if anchor.node.is_some() {
                local
            } else {
                compose(&anchor.frame, &local)
            },
            look: Some(look),
            motion: None,
            moving: false,
        });
        true
    }

    /// A moving part with its motion as a child of `anchor` (`nested`, in
    /// the anchor's frame) and as a root (`flat`, in world space).
    fn mover(
        &mut self,
        anchor: &Anchor,
        local: Transform,
        look: Look,
        nested: Option<Motion>,
        flat: Option<Motion>,
    ) -> bool {
        if self.moving_left == 0 {
            return false;
        }
        self.moving_left -= 1;
        let node = if anchor.node.is_some() {
            Node {
                parent: anchor.node,
                transform: local,
                look: Some(look),
                motion: nested,
                moving: true,
            }
        } else {
            Node {
                parent: None,
                transform: compose(&anchor.frame, &local),
                look: Some(look),
                motion: flat,
                moving: true,
            }
        };
        self.nodes.push(node);
        true
    }

    /// A transform-only moving pivot for an assembly, when hierarchies are
    /// on and nodes remain.
    fn pivot(&mut self, frame: Transform, motion: Motion) -> Option<Anchor> {
        let anchor = self.anchor(frame);
        let node = anchor.node?;
        self.nodes[node as usize].motion = Some(motion);
        Some(anchor)
    }

    fn world(&self) -> Anchor {
        Anchor {
            node: None,
            frame: Transform::default(),
        }
    }
}

/// Transform nodes the city may add beyond its renderables.
fn node_room(renderables: u32) -> u32 {
    (engine::MAX_TRANSFORM_NODES as u32).saturating_sub(renderables)
}

/// Moving and static renderables of `config`'s city, and its radius in
/// blocks of `per_block` static renderables each.
pub(crate) fn budget(config: &StressConfig, per_block: f32) -> (u32, u32, f32) {
    let moving = config.renderables / 100 * config.moving_percent
        + (config.renderables % 100).min(config.moving_percent);
    let dense = if config.dense_colliders {
        config.colliders
    } else {
        0
    };
    let static_total = config.renderables - moving - dense.min(config.renderables - moving);
    let radius = math::max(
        (math::max(static_total as f32, moving as f32 * 0.5) / per_block / PI).sqrt(),
        1.0,
    );
    (moving, static_total, radius)
}

/// The block row the river runs along, for cities wide enough to have one.
pub(crate) fn river_row(radius: f32) -> Option<i32> {
    (radius >= 4.0).then(|| (radius * 0.45).round() as i32)
}

fn plan(config: &StressConfig) -> City {
    let (moving, static_total, radius) = budget(config, PER_BLOCK);
    // The core, and a few secondary downtowns in the outer half.
    let mut centers = vec![(Vec2::ZERO, radius * 0.32, 1.0)];
    let mut rng = Rng::new(config.seed ^ 0x5c1, 0, 0);
    if radius >= 6.0 {
        for _ in 0..3 + rng.below(3) {
            let (sin, cos) = math::sin_cos(rng.range(0.0, TAU));
            let distance = radius * rng.range(0.45, 0.75);
            centers.push((
                Vec2::new(cos, sin) * distance,
                radius * rng.range(0.1, 0.17),
                rng.range(0.55, 0.9),
            ));
        }
    }
    City {
        seed: config.seed,
        hierarchy: config.hierarchy,
        nodes: Vec::with_capacity(config.renderables as usize + 4096),
        static_left: static_total,
        moving_left: moving,
        node_room: node_room(config.renderables),
        blocks: Vec::new(),
        rooftops: Vec::new(),
        cranes: Vec::new(),
        wheels: Vec::new(),
        river: river_row(radius).map(|row| (row, 0.0)),
        rail: (radius >= 6.0 && moving >= 400)
            .then(|| math::max((radius * 0.32).round(), 2.0) as i32),
        radius,
        centers,
    }
}

/// Builds the city's nodes: dense collider crates, blocks outward from the
/// center until the static budget is spent, then every moving renderable.
fn build(config: &StressConfig) -> City {
    let mut city = plan(config);
    if config.dense_colliders {
        let world = city.world();
        let crates = config.colliders.min(config.renderables - city.moving_left);
        for index in 0..crates {
            let random = hash(index ^ config.seed);
            let transform = Transform {
                translation: position_for_index(index, config),
                rotation: Quat::from_rotation_y((random % 8) as f32 * 0.16),
                scale: Vec3::new(
                    0.7 + (random % 5) as f32 * 0.08,
                    1.0,
                    0.7 + ((random >> 4) % 5) as f32 * 0.08,
                ),
            };
            city.nodes.push(Node {
                parent: world.node,
                transform,
                look: Some(look(
                    Shape::Block,
                    Tex::Metal,
                    CARS[index as usize % CARS.len()],
                )),
                motion: None,
                moving: false,
            });
        }
    }
    let reach = (city.radius * 1.8).ceil() as i32 + 2;
    let mut order = Vec::with_capacity(((2 * reach + 1) * (2 * reach + 1)) as usize);
    for bz in -reach..=reach {
        for bx in -reach..=reach {
            let jitter = Rng::new(config.seed ^ 0x51ed, bx, bz).range(0.0, 0.9);
            order.push((((bx * bx + bz * bz) as f32).sqrt() + jitter, bx, bz));
        }
    }
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (distance, bx, bz) in order {
        if city.static_left == 0 {
            break;
        }
        let rho = distance / city.radius;
        block(&mut city, bx, bz, rho);
    }
    traffic(&mut city, config);
    city
}

fn block(city: &mut City, bx: i32, bz: i32, rho: f32) {
    let mut rng = Rng::new(city.seed, bx, bz);
    let center = Vec3::new(
        bx as f32 * PERIOD + PERIOD * 0.5,
        0.0,
        bz as f32 * PERIOD + PERIOD * 0.5,
    );
    let anchor = city.anchor(part_at(center, Vec3::ONE));
    let river = city.river.is_some_and(|(row, _)| row == bz);
    // Streets on the block's west and north sides and their corner, in the
    // block frame; the west street bridges the river.
    let west = Vec3::new(-HALF_BLOCK - 2.0, 0.0, 0.0);
    let north = Vec3::new(0.0, 0.0, -HALF_BLOCK - 2.0);
    if river {
        city.part(
            &anchor,
            turned(west + Vec3::Y * -0.3, FRAC_PI_2, Vec3::new(20.0, 0.6, 4.5)),
            look(Shape::Deck, Tex::Road, ASPHALT),
        );
    } else {
        city.part(
            &anchor,
            turned(
                west + Vec3::Y * (ROAD_TOP * 0.5),
                FRAC_PI_2,
                Vec3::new(20.0, ROAD_TOP, 4.0),
            ),
            look(Shape::Block, Tex::Road, ASPHALT),
        );
    }
    city.part(
        &anchor,
        part_at(
            north + Vec3::Y * (ROAD_TOP * 0.5),
            Vec3::new(20.0, ROAD_TOP, 4.0),
        ),
        look(Shape::Block, Tex::Road, ASPHALT),
    );
    city.part(
        &anchor,
        part_at(
            west + north + Vec3::Y * (ROAD_TOP * 0.5),
            Vec3::new(4.0, ROAD_TOP, 4.0),
        ),
        look(Shape::Block, Tex::Crossing, ASPHALT),
    );
    if let Some(ring) = city.rail {
        rail(city, &anchor, bx, bz, ring);
    }
    if river {
        city.part(
            &anchor,
            part_at(Vec3::Y * (WATER_TOP - 0.1), Vec3::new(20.0, 0.2, 20.0)),
            look(Shape::Block, Tex::Water, WHITE),
        );
        if let Some((_, span)) = &mut city.river {
            *span = math::max(*span, (bx.abs() as f32 + 1.0) * PERIOD);
        }
        return;
    }
    city.blocks.push((bx, bz, rho));
    let skyline = city.skyline(bx, bz);
    // The block's ground.
    let ground = if rho < 0.55 || skyline > 0.0 {
        Tex::Paving
    } else {
        Tex::Grass
    };
    let ground_color = if ground == Tex::Paving {
        [0.45, 0.45, 0.5]
    } else {
        [1.0, 1.0, 1.0]
    };
    city.part(
        &anchor,
        part_at(Vec3::Y * 0.05, Vec3::new(20.0, 0.1, 20.0)),
        look(Shape::Block, ground, ground_color),
    );
    if bx == 0 && bz == 0 {
        landmark(city, &anchor, &mut rng);
        return;
    }
    let roll = rng.below(1000);
    let (park, plaza, construction, stadium) = (
        if rho > 0.6 && skyline == 0.0 { 150 } else { 60 },
        if rho < 0.5 || skyline > 0.0 { 40 } else { 0 },
        if (0.15..0.9).contains(&rho) { 35 } else { 0 },
        if (0.3..0.8).contains(&rho) { 12 } else { 0 },
    );
    if roll < park {
        park_block(city, &anchor, &mut rng);
    } else if roll < park + plaza {
        plaza_block(city, &anchor, &mut rng);
    } else if roll < park + plaza + construction {
        construction_block(city, &anchor, &mut rng);
    } else if roll < park + plaza + construction + stadium {
        stadium_block(city, &anchor, &mut rng);
    } else if skyline >= 0.3 {
        downtown(city, &anchor, &mut rng, skyline);
    } else if skyline > 0.0 || rho < 0.55 {
        let scale = math::clamp(math::max(skyline / 0.3, 1.0 - (rho - 0.3) / 0.3), 0.15, 1.0);
        midtown(city, &anchor, &mut rng, scale);
    } else if rho < 0.85 {
        residential(city, &anchor, &mut rng);
    } else {
        industrial(city, &anchor, &mut rng);
    }
    // Street lamps at two corners.
    for (x, z) in [
        (-HALF_BLOCK + 0.6, -HALF_BLOCK + 0.6),
        (HALF_BLOCK - 0.6, -HALF_BLOCK + 0.6),
    ] {
        city.part(
            &anchor,
            turned(
                Vec3::new(x, 2.6, z),
                rng.range(0.0, TAU),
                Vec3::new(1.6, 5.2, 1.6),
            ),
            look(Shape::Lamp, Tex::Lamp, [1.0, 1.0, 1.0]),
        );
    }
}

fn rail(city: &mut City, anchor: &Anchor, bx: i32, bz: i32, ring: i32) {
    if bx.abs() > ring || bz.abs() > ring || (bx.abs() != ring && bz.abs() != ring) {
        return;
    }
    let corner = Vec3::new(-HALF_BLOCK - 2.0, 0.0, -HALF_BLOCK - 2.0);
    let metal = [0.62, 0.64, 0.7];
    city.part(
        anchor,
        part_at(
            corner + Vec3::Y * (RAIL_HEIGHT * 0.5 - 0.25),
            Vec3::new(4.0, RAIL_HEIGHT - 0.5, 3.0),
        ),
        look(Shape::Pillar, Tex::Paving, metal),
    );
    let deck = Vec3::new(20.0 + 4.0, 1.0, 3.2);
    if bz.abs() == ring && bx < ring {
        city.part(
            anchor,
            part_at(corner + Vec3::new(PERIOD * 0.5, RAIL_HEIGHT, 0.0), deck),
            look(Shape::Deck, Tex::Metal, metal),
        );
    }
    if bx.abs() == ring && bz < ring {
        city.part(
            anchor,
            turned(
                corner + Vec3::new(0.0, RAIL_HEIGHT, PERIOD * 0.5),
                FRAC_PI_2,
                deck,
            ),
            look(Shape::Deck, Tex::Metal, metal),
        );
    }
}

/// A building of `shape` standing at `base` (its ground center), `size`
/// wide and deep and `height` tall, turned by `yaw`.
fn building(
    city: &mut City,
    anchor: &Anchor,
    base: Vec3,
    size: Vec2,
    height: f32,
    yaw: f32,
    look: Look,
) -> bool {
    city.part(
        anchor,
        turned(
            base + Vec3::Y * (height * 0.5),
            yaw,
            Vec3::new(size.x, height, size.y),
        ),
        look,
    )
}

fn tower_look(rng: &mut Rng) -> Look {
    let shape = rng.pick(&[
        Shape::Setback4,
        Shape::Setback,
        Shape::Setback2,
        Shape::Twisted,
        Shape::TwistedTall,
        Shape::Crowned,
        Shape::Round,
        Shape::Tapered,
        Shape::Tower,
        Shape::Octa,
        Shape::Hex,
        Shape::Ziggurat,
        Shape::Pagoda,
    ]);
    let (tex, color) = if rng.chance(60) {
        (
            Tex::Glass,
            if rng.chance(60) {
                rng.pick(&GLASS)
            } else {
                rng.pick(&NEON)
            },
        )
    } else {
        (
            Tex::Office,
            if rng.chance(50) {
                rng.pick(&STONE)
            } else {
                rng.pick(&NEON)
            },
        )
    };
    look(shape, tex, color)
}

/// A tall building with a crown and, sometimes, a helipad; its roof is
/// offered to moving parts.
fn skyscraper(
    city: &mut City,
    anchor: &Anchor,
    rng: &mut Rng,
    base: Vec3,
    width: f32,
    height: f32,
) {
    let body = tower_look(rng);
    let yaw = if matches!(
        body.shape,
        Shape::Twisted | Shape::TwistedTall | Shape::Round | Shape::Crowned
    ) {
        rng.range(0.0, TAU)
    } else {
        rng.below(4) as f32 * FRAC_PI_2
    };
    if !building(city, anchor, base, Vec2::splat(width), height, yaw, body) {
        return;
    }
    let top = base + Vec3::Y * height;
    // Tiers narrow toward the top: crowns sit on the top tier.
    let crown = match body.shape {
        Shape::Setback | Shape::Setback4 | Shape::Ziggurat | Shape::Pagoda | Shape::Crowned => {
            width * 0.25
        }
        Shape::Setback2 | Shape::Tapered => width * 0.5,
        _ => width * 0.8,
    };
    match rng.below(10) {
        0..=3 => {
            city.part(
                anchor,
                part_at(
                    top + Vec3::Y * (height * 0.09),
                    Vec3::new(crown * 0.5, height * 0.18, crown * 0.5),
                ),
                look(Shape::Spire, Tex::Metal, rng.pick(&NEON)),
            );
        }
        4..=6 => {
            city.part(
                anchor,
                part_at(
                    top + Vec3::Y * 4.0,
                    Vec3::new(crown * 0.6, 8.0, crown * 0.6),
                ),
                look(Shape::Antenna, Tex::Metal, [0.8, 0.82, 0.86]),
            );
        }
        7 => {
            city.part(
                anchor,
                part_at(top + Vec3::Y * 0.1, Vec3::new(crown, 0.2, crown)),
                look(Shape::Round, Tex::Crossing, WHITE),
            );
        }
        _ => {
            city.part(
                anchor,
                part_at(
                    top + Vec3::Y * (crown * 0.3),
                    Vec3::new(crown, crown * 0.6, crown),
                ),
                look(Shape::Dome, Tex::Glass, rng.pick(&GLASS)),
            );
        }
    }
    city.rooftops.push(Rooftop {
        anchor: *anchor,
        base,
        height,
        half: crown * 0.5,
    });
}

/// Four skyscrapers per block, `scale` (0.3 to 1) setting their heights.
fn downtown(city: &mut City, anchor: &Anchor, rng: &mut Rng, scale: f32) {
    for (x, z) in [(-6.0, -6.0), (6.0, -6.0), (-6.0, 6.0), (6.0, 6.0)] {
        let height = (45.0 + 95.0 * scale) * rng.range(0.55, 1.1);
        let width = rng.range(5.6, 7.4);
        skyscraper(city, anchor, rng, Vec3::new(x, 0.1, z), width, height);
    }
    // The cross between the towers: a small tower, trees and a kiosk.
    let height = rng.range(10.0, 26.0);
    let body = look(
        rng.pick(&[Shape::Hex, Shape::Midrise, Shape::Octa, Shape::Tapered]),
        Tex::Office,
        rng.pick(&STONE),
    );
    building(
        city,
        anchor,
        Vec3::new(0.0, 0.1, 0.0),
        Vec2::splat(3.2),
        height,
        0.0,
        body,
    );
    for (x, z) in [(0.0, -6.5), (0.0, 6.5), (-6.5, 0.0), (6.5, 0.0)] {
        tree(city, anchor, rng, Vec3::new(x, 0.1, z), 1.0);
    }
}

/// Nine buildings per block, `scale` (0 to 1) setting their heights.
fn midtown(city: &mut City, anchor: &Anchor, rng: &mut Rng, scale: f32) {
    for row in 0..3 {
        for column in 0..3 {
            let base = Vec3::new(-6.6 + column as f32 * 6.6, 0.1, -6.6 + row as f32 * 6.6);
            let width = rng.range(4.2, 5.6);
            let depth = rng.range(4.2, 5.6);
            let height = (8.0 + 30.0 * scale) * rng.range(0.5, 1.2);
            if height > 24.0 && rng.chance(40) {
                skyscraper(city, anchor, rng, base, width, height);
                continue;
            }
            let shape = rng.pick(&[
                Shape::Midrise,
                Shape::Midrise,
                Shape::Tower,
                Shape::Ell,
                Shape::Tee,
                Shape::Court,
                Shape::Setback2,
                Shape::Hex,
                Shape::Octa,
                Shape::Arch,
            ]);
            let (tex, color) = match rng.below(3) {
                0 => (Tex::Office, rng.pick(&STONE)),
                1 => (Tex::Residential, rng.pick(&STONE)),
                _ => (Tex::Glass, rng.pick(&GLASS)),
            };
            let yaw = rng.below(4) as f32 * FRAC_PI_2;
            if !building(
                city,
                anchor,
                base,
                Vec2::new(width, depth),
                height,
                yaw,
                look(shape, tex, color),
            ) {
                return;
            }
            let top = base + Vec3::Y * height;
            // Rooftop clutter.
            for _ in 0..rng.below(3) {
                let at = top + Vec3::new(rng.range(-1.4, 1.4), 0.5, rng.range(-1.4, 1.4));
                city.part(
                    anchor,
                    turned(at, rng.range(0.0, TAU), Vec3::new(1.2, 1.0, 0.9)),
                    look(Shape::Block, Tex::Metal, [0.7, 0.72, 0.75]),
                );
            }
            match rng.below(20) {
                0..=4 => {
                    city.part(
                        anchor,
                        part_at(top + Vec3::new(0.8, 1.5, -0.8), Vec3::new(1.8, 3.0, 1.8)),
                        look(Shape::WaterTower, Tex::Tiles, [0.6, 0.4, 0.3]),
                    );
                }
                5..=7 => {
                    city.part(
                        anchor,
                        part_at(top + Vec3::Y * 3.0, Vec3::new(1.4, 6.0, 1.4)),
                        look(Shape::Antenna, Tex::Metal, [0.85, 0.85, 0.9]),
                    );
                }
                8 => {
                    city.part(
                        anchor,
                        turned(
                            top + Vec3::Y * 1.5,
                            rng.range(0.0, TAU),
                            Vec3::new(4.0, 3.0, 0.6),
                        ),
                        look(Shape::Billboard, Tex::Neon, rng.pick(&NEON)),
                    );
                }
                9 | 10 => tree(city, anchor, rng, top, 0.7),
                _ => {}
            }
            if height > 18.0 {
                city.rooftops.push(Rooftop {
                    anchor: *anchor,
                    base,
                    height,
                    half: math::min(width, depth) * 0.3,
                });
            }
        }
    }
}

fn residential(city: &mut City, anchor: &Anchor, rng: &mut Rng) {
    for row in 0..4 {
        for column in 0..4 {
            let base = Vec3::new(-7.5 + column as f32 * 5.0, 0.1, -7.5 + row as f32 * 5.0);
            if rng.chance(15) {
                tree(city, anchor, rng, base, 1.1);
                continue;
            }
            let (width, depth) = (rng.range(2.8, 3.8), rng.range(2.8, 3.8));
            let height = rng.range(2.5, 7.0);
            let yaw = rng.below(4) as f32 * FRAC_PI_2;
            let apartment = height > 5.5 && rng.chance(30);
            let body = look(
                if apartment {
                    Shape::Midrise
                } else {
                    Shape::House
                },
                Tex::Residential,
                rng.pick(&PASTEL),
            );
            if !building(
                city,
                anchor,
                base,
                Vec2::new(width, depth),
                height,
                yaw,
                body,
            ) {
                return;
            }
            let top = base + Vec3::Y * height;
            let roof = rng.pick(&[
                Shape::Roof,
                Shape::Roof,
                Shape::Wedge,
                Shape::Wedge,
                Shape::Dome,
            ]);
            let roof_height = if roof == Shape::Dome {
                1.2
            } else {
                rng.range(1.2, 2.2)
            };
            city.part(
                anchor,
                turned(
                    top + Vec3::Y * (roof_height * 0.5),
                    yaw,
                    Vec3::new(width, roof_height, depth),
                ),
                look(roof, Tex::Tiles, rng.pick(&ROOFS)),
            );
            if rng.chance(30) {
                city.part(
                    anchor,
                    part_at(
                        top + Vec3::new(width * 0.25, 0.9, 0.0),
                        Vec3::new(0.5, 1.8, 0.5),
                    ),
                    look(Shape::Chimney, Tex::Industrial, [0.6, 0.35, 0.3]),
                );
            }
            if rng.chance(35) {
                city.part(
                    anchor,
                    part_at(base + Vec3::new(1.9, 0.4, 1.9), Vec3::splat(0.9)),
                    look(Shape::Bush, Tex::Foliage, WHITE),
                );
            }
        }
    }
}

fn industrial(city: &mut City, anchor: &Anchor, rng: &mut Rng) {
    for row in 0..2 {
        let z = -5.0 + row as f32 * 10.0;
        if rng.chance(60) {
            let (length, height) = (rng.range(12.0, 17.0), rng.range(3.5, 7.0));
            let base = Vec3::new(rng.range(-2.0, 2.0), 0.1, z);
            if !building(
                city,
                anchor,
                base,
                Vec2::new(length, 7.0),
                height,
                0.0,
                look(Shape::Midrise, Tex::Industrial, rng.pick(&RUST)),
            ) {
                return;
            }
            city.part(
                anchor,
                part_at(base + Vec3::Y * (height + 1.0), Vec3::new(length, 2.0, 7.0)),
                look(Shape::Sawtooth, Tex::Metal, rng.pick(&RUST)),
            );
        } else {
            for index in 0..3 {
                let at = Vec3::new(-6.0 + index as f32 * 6.0, 0.1, z);
                let height = rng.range(5.0, 11.0);
                building(
                    city,
                    anchor,
                    at,
                    Vec2::splat(4.5),
                    height,
                    0.0,
                    look(Shape::Tank, Tex::Metal, rng.pick(&RUST)),
                );
            }
        }
    }
    if rng.chance(50) {
        let height = rng.range(18.0, 34.0);
        building(
            city,
            anchor,
            Vec3::new(8.0, 0.1, 0.0),
            Vec2::splat(1.6),
            height,
            0.0,
            look(Shape::Chimney, Tex::Industrial, [0.7, 0.4, 0.35]),
        );
    }
    for index in 0..rng.below(5) {
        let at = Vec3::new(-8.0 + index as f32 * 2.6, 1.3, 8.8);
        city.part(
            anchor,
            part_at(at, Vec3::new(2.4, 2.4, 1.2)),
            look(Shape::Block, Tex::Metal, rng.pick(&CARS)),
        );
    }
}

fn tree(city: &mut City, anchor: &Anchor, rng: &mut Rng, base: Vec3, scale: f32) {
    let shape = rng.pick(&[
        Shape::Tree,
        Shape::Tree,
        Shape::Pine,
        Shape::Poplar,
        Shape::Bush,
    ]);
    let (width, height) = match shape {
        Shape::Pine => (2.2, 4.5),
        Shape::Poplar => (1.6, 5.0),
        Shape::Bush => (1.6, 1.2),
        _ => (2.6, 4.0),
    };
    let size = scale * rng.range(0.7, 1.25);
    city.part(
        anchor,
        turned(
            base + Vec3::Y * (height * size * 0.5),
            rng.range(0.0, TAU),
            Vec3::new(width * size, height * size, width * size),
        ),
        look(shape, Tex::Foliage, WHITE),
    );
}

fn park_block(city: &mut City, anchor: &Anchor, rng: &mut Rng) {
    let wheel = rng.chance(25);
    for _ in 0..rng.below(8) + 10 {
        let at = Vec3::new(rng.range(-8.5, 8.5), 0.1, rng.range(-8.5, 8.5));
        if wheel && at.length() < 5.5 {
            continue;
        }
        tree(city, anchor, rng, at, 1.0);
    }
    if wheel {
        // A Ferris wheel's supports; the wheel itself moves.
        let (size, yaw) = (rng.range(12.0, 16.0), rng.range(0.0, PI));
        let hub = size * 0.5 + 1.0;
        for side in [-1.0, 1.0] {
            let leg = Transform {
                translation: Vec3::new(0.0, hub * 0.5, side * 0.9),
                rotation: Quat::from_rotation_y(yaw) * Quat::from_rotation_x(side * 0.12),
                scale: Vec3::new(0.5, hub + 0.4, 0.5),
            };
            let leg = Transform {
                translation: Quat::from_rotation_y(yaw) * leg.translation,
                ..leg
            };
            city.part(
                anchor,
                leg,
                look(Shape::Block, Tex::Metal, [0.9, 0.9, 0.95]),
            );
        }
        city.wheels.push(Site {
            anchor: *anchor,
            at: Vec3::new(0.0, hub, 0.0),
            size,
            yaw,
        });
    } else if rng.chance(60) {
        city.part(
            anchor,
            part_at(Vec3::Y * 1.1, Vec3::new(5.0, 2.0, 5.0)),
            look(Shape::Fountain, Tex::Paving, [0.85, 0.9, 1.0]),
        );
    } else {
        city.part(
            anchor,
            part_at(Vec3::Y * 0.12, Vec3::new(7.0, 0.1, 5.0)),
            look(Shape::Block, Tex::Water, WHITE),
        );
    }
}

fn plaza_block(city: &mut City, anchor: &Anchor, rng: &mut Rng) {
    match rng.below(4) {
        0 => {
            city.part(
                anchor,
                part_at(Vec3::Y * 9.0, Vec3::new(2.2, 18.0, 2.2)),
                look(Shape::Obelisk, Tex::Paving, rng.pick(&STONE)),
            );
        }
        1 => {
            city.part(
                anchor,
                part_at(Vec3::Y * 4.5, Vec3::new(12.0, 9.0, 4.0)),
                look(Shape::Arch, Tex::Office, rng.pick(&STONE)),
            );
        }
        2 => {
            city.part(
                anchor,
                part_at(Vec3::Y * 3.1, Vec3::new(9.0, 6.0, 9.0)),
                look(Shape::Dome, Tex::Glass, rng.pick(&GLASS)),
            );
        }
        _ => {
            city.part(
                anchor,
                part_at(Vec3::Y * 1.6, Vec3::new(7.0, 3.0, 7.0)),
                look(Shape::Fountain, Tex::Paving, [0.85, 0.9, 1.0]),
            );
        }
    }
    for index in 0..8 {
        let (sin, cos) = math::sin_cos(TAU * index as f32 / 8.0);
        tree(city, anchor, rng, Vec3::new(cos * 8.0, 0.1, sin * 8.0), 0.9);
    }
}

fn construction_block(city: &mut City, anchor: &Anchor, rng: &mut Rng) {
    city.part(
        anchor,
        part_at(Vec3::Y * 0.12, Vec3::new(18.0, 0.1, 18.0)),
        look(Shape::Block, Tex::Paving, DIRT),
    );
    let height = rng.range(12.0, 40.0);
    building(
        city,
        anchor,
        Vec3::new(-3.0, 0.1, -3.0),
        Vec2::new(9.0, 9.0),
        height,
        0.0,
        look(Shape::Midrise, Tex::Metal, [0.75, 0.75, 0.78]),
    );
    // The crane's mast; its jib moves.
    let mast = height + rng.range(8.0, 16.0);
    let at = Vec3::new(5.5, 0.1, 5.5);
    if building(
        city,
        anchor,
        at,
        Vec2::splat(1.4),
        mast,
        0.0,
        look(Shape::Tower, Tex::Metal, CRANE),
    ) {
        city.cranes.push(Site {
            anchor: *anchor,
            at: at + Vec3::Y * mast,
            size: rng.range(14.0, 22.0),
            yaw: rng.range(0.0, TAU),
        });
    }
    for index in 0..3 {
        city.part(
            anchor,
            part_at(
                Vec3::new(-8.0 + index as f32 * 2.7, 1.3, 8.5),
                Vec3::new(2.4, 2.4, 1.2),
            ),
            look(Shape::Block, Tex::Metal, rng.pick(&CARS)),
        );
    }
}

fn stadium_block(city: &mut City, anchor: &Anchor, rng: &mut Rng) {
    city.part(
        anchor,
        part_at(Vec3::Y * 4.1, Vec3::new(19.0, 8.0, 19.0)),
        look(Shape::Stadium, Tex::Office, rng.pick(&STONE)),
    );
    city.part(
        anchor,
        part_at(Vec3::Y * 0.6, Vec3::new(10.0, 0.1, 10.0)),
        look(Shape::Block, Tex::Grass, WHITE),
    );
}

fn landmark(city: &mut City, anchor: &Anchor, rng: &mut Rng) {
    city.part(
        anchor,
        part_at(Vec3::Y * 100.1, Vec3::new(14.0, 200.0, 14.0)),
        look(Shape::TvTower, Tex::Metal, [0.9, 0.92, 1.0]),
    );
    for index in 0..4 {
        let (sin, cos) = math::sin_cos(TAU * (index as f32 + 0.5) / 4.0);
        city.part(
            anchor,
            part_at(
                Vec3::new(cos * 6.5, 1.1, sin * 6.5),
                Vec3::new(3.5, 2.0, 3.5),
            ),
            look(Shape::Fountain, Tex::Paving, [0.85, 0.9, 1.0]),
        );
    }
    for index in 0..12 {
        let (sin, cos) = math::sin_cos(TAU * index as f32 / 12.0);
        tree(city, anchor, rng, Vec3::new(cos * 9.0, 0.1, sin * 9.0), 0.8);
    }
}

/// Every moving renderable: assemblies at the sites the blocks offered, then
/// trains, boats, aircraft, rooftop parts, buses and cars for the rest.
fn traffic(city: &mut City, config: &StressConfig) {
    let total = city.moving_left;
    let mut rng = Rng::new(config.seed ^ 0x7a11, 0, 0);
    let share = |percent: u32| total / 100 * percent + (total % 100) * percent / 100;
    // Cranes: a jib and its load.
    let cranes = std::mem::take(&mut city.cranes);
    for site in cranes.iter().take((share(3) / 2) as usize) {
        crane(city, &mut rng, site);
    }
    let wheels = std::mem::take(&mut city.wheels);
    for site in wheels.iter().take((share(3) / 9) as usize) {
        ferris(city, &mut rng, site);
    }
    if let Some(ring) = city.rail {
        let half = ring as f32 * PERIOD;
        let perimeter = 8.0 * half;
        let trains = math::min((share(8) / 5) as f32, perimeter / 60.0) as u32;
        let speed = rng.range(12.0, 16.0);
        let world = city.world();
        for train in 0..trains {
            let phase = perimeter * train as f32 / trains as f32;
            let color = rng.pick(&[[0.9, 0.92, 0.95], [0.95, 0.3, 0.2], [0.2, 0.5, 0.95]]);
            for wagon in 0..5 {
                let motion = Motion::Travel(Travel {
                    center: Vec3::new(0.0, RAIL_HEIGHT + 1.4, 0.0),
                    half: Vec2::splat(half),
                    speed,
                    phase: phase - wagon as f32 * 5.3,
                });
                let rest = part_at(Vec3::ZERO, Vec3::new(5.0, 1.8, 1.8));
                city.mover(
                    &world,
                    rest,
                    look(Shape::Wagon, Tex::Vehicle, color),
                    Some(motion),
                    Some(motion),
                );
            }
        }
    }
    if let Some((row, span)) = city.river {
        let world = city.world();
        let boats = share(4).min((span * 0.3) as u32);
        for _ in 0..boats {
            let half = Vec2::new(rng.range(12.0, math::max(span * 0.6, 14.0)), 3.5);
            let center = Vec3::new(
                rng.range(-span, span) * 0.5,
                WATER_TOP + 0.45,
                row as f32 * PERIOD + PERIOD * 0.5,
            );
            let motion = Motion::Travel(Travel {
                center,
                half,
                speed: rng.range(3.0, 7.0) * rng.sign(),
                phase: rng.range(0.0, 8.0 * (half.x + half.y)),
            });
            let color = rng.pick(&[WHITE, [0.95, 0.4, 0.3], [0.3, 0.6, 0.95], [0.95, 0.85, 0.3]]);
            city.mover(
                &world,
                part_at(Vec3::ZERO, Vec3::new(4.5, 1.2, 1.8)),
                look(Shape::Boat, Tex::Vehicle, color),
                Some(motion),
                Some(motion),
            );
        }
    }
    for _ in 0..share(3) / 2 {
        helicopter(city, &mut rng);
    }
    for _ in 0..share(14) / 2 {
        drone(city, &mut rng);
    }
    let rooftops = std::mem::take(&mut city.rooftops);
    if !rooftops.is_empty() {
        for index in 0..share(4) {
            let roof = rooftops[(hash(index ^ config.seed) as usize) % rooftops.len()];
            let at = roof.base + Vec3::Y * (roof.height + 1.3);
            let spin = Spin {
                base: Quat::IDENTITY,
                axis_z: false,
                speed: rng.range(1.5, 4.0) * rng.sign(),
                phase: rng.range(0.0, TAU),
            };
            let flat = Spin {
                base: roof.anchor.frame.rotation,
                ..spin
            };
            let size = math::min(roof.half * 1.6, 5.0);
            city.mover(
                &roof.anchor,
                part_at(at, Vec3::new(size, 2.6, size)),
                look(Shape::Beacon, Tex::Neon, rng.pick(&NEON)),
                Some(Motion::Spin(spin)),
                Some(Motion::Spin(flat)),
            );
        }
        for index in 0..share(4) {
            let roof = rooftops[(hash(index ^ config.seed ^ 0xe1e) as usize) % rooftops.len()];
            let side = Vec3::new(roof.half + 0.9, 0.0, 0.0);
            let anchor = roof.base + side + Vec3::Y * (roof.height * 0.5);
            let slide = Slide {
                anchor,
                axis: Vec3::Y,
                amplitude: roof.height * 0.35,
                speed: rng.range(0.3, 0.8),
                phase: rng.range(0.0, TAU),
            };
            let flat = Slide {
                anchor: compose(&roof.anchor.frame, &part_at(anchor, Vec3::ONE)).translation,
                axis: roof.anchor.frame.rotation * Vec3::Y,
                ..slide
            };
            city.mover(
                &roof.anchor,
                part_at(anchor, Vec3::new(1.4, 2.2, 1.4)),
                look(Shape::Block, Tex::Glass, rng.pick(&NEON)),
                Some(Motion::Slide(slide)),
                Some(Motion::Slide(flat)),
            );
        }
    }
    let buses = share(5);
    let cars = city.moving_left.saturating_sub(buses);
    for index in 0..buses + cars {
        vehicle(city, &mut rng, index < buses);
    }
}

/// A random street loop around one or two of `blocks` (any block within
/// `radius` when there are none): its center and half extents.
pub(crate) fn street_loop(blocks: &[(i32, i32, f32)], radius: f32, rng: &mut Rng) -> (Vec3, Vec2) {
    let (bx, bz) = if blocks.is_empty() {
        let reach = radius.ceil() as i32;
        (
            rng.below(2 * reach as u32 + 1) as i32 - reach,
            rng.below(2 * reach as u32 + 1) as i32 - reach,
        )
    } else {
        let (bx, bz, _) = blocks[rng.below(blocks.len() as u32) as usize];
        (bx, bz)
    };
    let (wide, deep) = (1 + rng.below(2), 1 + rng.below(2));
    let center = Vec3::new(
        bx as f32 * PERIOD + wide as f32 * PERIOD * 0.5,
        0.0,
        bz as f32 * PERIOD + deep as f32 * PERIOD * 0.5,
    );
    (
        center,
        Vec2::new(wide as f32 * PERIOD * 0.5, deep as f32 * PERIOD * 0.5),
    )
}

fn vehicle(city: &mut City, rng: &mut Rng, bus: bool) {
    let (center, half) = street_loop(&city.blocks, city.radius, rng);
    let direction = rng.sign();
    // Counterclockwise traffic keeps to the outer lane, clockwise to the inner.
    let lane = direction * 1.1;
    let half = half + Vec2::splat(lane);
    let (size, shape, color) = if bus {
        (
            Vec3::new(5.0, 1.7, 1.3),
            Shape::Bus,
            rng.pick(&[[0.95, 0.8, 0.1], [0.2, 0.5, 0.9], [0.9, 0.2, 0.2]]),
        )
    } else {
        (
            Vec3::new(rng.range(1.8, 2.3), rng.range(0.8, 1.0), 0.95),
            Shape::Car,
            rng.pick(&CARS),
        )
    };
    let motion = Motion::Travel(Travel {
        center: center + Vec3::Y * (ROAD_TOP + size.y * 0.5),
        half,
        speed: rng.range(6.0, 14.0) * direction,
        phase: rng.range(0.0, 4.0 * (half.x + half.y)),
    });
    let world = city.world();
    city.mover(
        &world,
        part_at(Vec3::ZERO, size),
        look(shape, Tex::Vehicle, color),
        Some(motion),
        Some(motion),
    );
}

fn drone(city: &mut City, rng: &mut Rng) {
    let (center, _) = street_loop(&city.blocks, city.radius, rng);
    let downtown = center.length() < city.radius * PERIOD * 0.3;
    let height = if downtown {
        rng.range(40.0, 130.0)
    } else {
        rng.range(15.0, 50.0)
    };
    let direction = rng.sign();
    let speed = rng.range(0.25, 0.9) * direction;
    let phase = rng.range(0.0, TAU);
    let orbit = Orbit {
        center: center + Vec3::Y * height,
        radius: rng.range(4.0, 26.0),
        speed,
        phase,
        bob: rng.range(0.3, 1.2),
        facing_speed: speed,
        facing_phase: phase + direction * FRAC_PI_2,
    };
    let body = look(Shape::Drone, Tex::Vehicle, rng.pick(&NEON));
    let cargo = look(Shape::Block, Tex::Metal, rng.pick(&CARS));
    let (body_size, cargo_size, drop) = (Vec3::new(1.6, 0.6, 1.6), Vec3::splat(0.6), 0.8);
    let world = city.world();
    match city.pivot(Transform::default(), Motion::Orbit(orbit)) {
        Some(pivot) => {
            city.mover(&pivot, part_at(Vec3::ZERO, body_size), body, None, None);
            city.mover(
                &pivot,
                part_at(Vec3::NEG_Y * drop, cargo_size),
                cargo,
                None,
                None,
            );
        }
        None => {
            city.mover(
                &world,
                part_at(Vec3::ZERO, body_size),
                body,
                None,
                Some(Motion::Orbit(orbit)),
            );
            let lowered = Orbit {
                center: orbit.center - Vec3::Y * drop,
                ..orbit
            };
            city.mover(
                &world,
                part_at(Vec3::ZERO, cargo_size),
                cargo,
                None,
                Some(Motion::Orbit(lowered)),
            );
        }
    }
}

fn helicopter(city: &mut City, rng: &mut Rng) {
    let direction = rng.sign();
    let speed = rng.range(0.04, 0.12) * direction;
    let phase = rng.range(0.0, TAU);
    let orbit = Orbit {
        center: Vec3::new(
            rng.range(-60.0, 60.0),
            rng.range(70.0, 150.0),
            rng.range(-60.0, 60.0),
        ),
        radius: rng.range(30.0, math::max(city.radius * PERIOD * 0.7, 40.0)),
        speed,
        phase,
        bob: rng.range(0.5, 2.0),
        facing_speed: speed,
        facing_phase: phase + direction * FRAC_PI_2,
    };
    let (rotor_speed, rotor_phase) = (rng.range(16.0, 24.0), rng.range(0.0, TAU));
    let body = look(
        Shape::Helicopter,
        Tex::Vehicle,
        rng.pick(&[[0.9, 0.2, 0.15], [0.95, 0.95, 0.95], [0.15, 0.2, 0.3]]),
    );
    let rotor = look(Shape::Rotor, Tex::Metal, [0.3, 0.3, 0.32]);
    let (body_size, rotor_size, lift) = (Vec3::new(5.0, 2.2, 2.0), Vec3::new(8.0, 0.6, 8.0), 0.9);
    let world = city.world();
    match city.pivot(Transform::default(), Motion::Orbit(orbit)) {
        Some(pivot) => {
            city.mover(&pivot, part_at(Vec3::ZERO, body_size), body, None, None);
            let spin = Spin {
                base: Quat::IDENTITY,
                axis_z: false,
                speed: rotor_speed,
                phase: rotor_phase,
            };
            city.mover(
                &pivot,
                part_at(Vec3::Y * lift, rotor_size),
                rotor,
                Some(Motion::Spin(spin)),
                None,
            );
        }
        None => {
            city.mover(
                &world,
                part_at(Vec3::ZERO, body_size),
                body,
                None,
                Some(Motion::Orbit(orbit)),
            );
            let turning = Orbit {
                center: orbit.center + Vec3::Y * lift,
                facing_speed: orbit.facing_speed + rotor_speed,
                facing_phase: orbit.facing_phase + rotor_phase,
                ..orbit
            };
            city.mover(
                &world,
                part_at(Vec3::ZERO, rotor_size),
                rotor,
                None,
                Some(Motion::Orbit(turning)),
            );
        }
    }
}

fn crane(city: &mut City, rng: &mut Rng, site: &Site) {
    let top = compose(&site.anchor.frame, &part_at(site.at, Vec3::ONE)).translation;
    let (speed, phase) = (rng.range(0.1, 0.35) * rng.sign(), rng.range(0.0, TAU));
    let base = Quat::from_rotation_y(site.yaw);
    let spin = Spin {
        base,
        axis_z: false,
        speed,
        phase,
    };
    let jib = look(Shape::Jib, Tex::Metal, CRANE);
    let load = look(Shape::Block, Tex::Metal, rng.pick(&CARS));
    let (jib_size, load_size) = (Vec3::new(site.size, 3.0, 1.2), Vec3::new(2.0, 1.4, 1.2));
    let (reach, drop) = (site.size * 0.35, rng.range(4.0, 9.0));
    let world = city.world();
    match city.pivot(
        Transform {
            translation: top,
            rotation: base,
            scale: Vec3::ONE,
        },
        Motion::Spin(spin),
    ) {
        Some(pivot) => {
            city.mover(&pivot, part_at(Vec3::ZERO, jib_size), jib, None, None);
            city.mover(
                &pivot,
                part_at(Vec3::new(reach, -drop, 0.0), load_size),
                load,
                None,
                None,
            );
        }
        None => {
            let rest = Transform {
                translation: top,
                rotation: base,
                scale: jib_size,
            };
            city.mover(&world, rest, jib, None, Some(Motion::Spin(spin)));
            // The load circles the mast at the jib's angle: yaw + turn.
            let orbit = Orbit {
                center: top - Vec3::Y * drop,
                radius: reach,
                speed,
                phase: phase + site.yaw,
                bob: 0.0,
                facing_speed: speed,
                facing_phase: phase + site.yaw,
            };
            city.mover(
                &world,
                part_at(Vec3::ZERO, load_size),
                load,
                None,
                Some(Motion::Orbit(orbit)),
            );
        }
    }
}

fn ferris(city: &mut City, rng: &mut Rng, site: &Site) {
    let hub = compose(&site.anchor.frame, &part_at(site.at, Vec3::ONE)).translation;
    let base = Quat::from_rotation_y(site.yaw);
    let (speed, phase) = (rng.range(0.08, 0.2), rng.range(0.0, TAU));
    let spin = Spin {
        base,
        axis_z: true,
        speed,
        phase,
    };
    let rim = look(Shape::Rim, Tex::Metal, rng.pick(&NEON));
    let radius = site.size * 0.45;
    let world = city.world();
    let pivot = city.pivot(
        Transform {
            translation: hub,
            rotation: base,
            scale: Vec3::ONE,
        },
        Motion::Spin(spin),
    );
    let rim_size = Vec3::new(site.size, site.size, 1.2);
    match pivot {
        Some(pivot) => {
            city.mover(&pivot, part_at(Vec3::ZERO, rim_size), rim, None, None);
        }
        None => {
            let rest = Transform {
                translation: hub,
                rotation: base,
                scale: rim_size,
            };
            city.mover(&world, rest, rim, None, Some(Motion::Spin(spin)));
        }
    }
    for cabin in 0..8 {
        let angle = TAU * cabin as f32 / 8.0;
        let (sin, cos) = math::sin_cos(angle);
        let paint = look(Shape::Cabin, Tex::Vehicle, rng.pick(&NEON));
        let size = Vec3::new(1.4, 1.4, 1.2);
        match pivot {
            Some(pivot) => {
                // Turned back as the wheel turns: cabins stay upright.
                let upright = Spin {
                    base: Quat::IDENTITY,
                    axis_z: true,
                    speed: -speed,
                    phase: -phase,
                };
                city.mover(
                    &pivot,
                    part_at(Vec3::new(radius * cos, radius * sin, 0.0), size),
                    paint,
                    Some(Motion::Spin(upright)),
                    None,
                );
            }
            None => {
                let wheel = Wheel {
                    center: hub,
                    u: base * Vec3::X * radius,
                    v: Vec3::Y * radius,
                    speed,
                    phase: phase + angle,
                };
                let rest = Transform {
                    translation: hub,
                    rotation: base,
                    scale: size,
                };
                city.mover(&world, rest, paint, None, Some(Motion::Wheel(wheel)));
            }
        }
    }
}

/// A street lamp's warm light: a wide cone down from the lamp head, in the
/// lamp's local space (its unit mesh is 5.2 units tall once placed).
fn street_light() -> LocalLight {
    LocalLight::spot(
        Vec3::new(1.0, 0.72, 0.42),
        60.0,
        10.0,
        Vec3::NEG_Y,
        40.0_f32.to_radians(),
        72.0_f32.to_radians(),
    )
    .with_offset(Vec3::new(0.4, 0.32, 0.0))
}

/// Smoke rising from a chimney top and drifting with the wind.
fn chimney_smoke(seed: u32) -> ParticleEmitter {
    ParticleEmitter {
        rate: 5.0,
        lifetime: 7.0,
        speed: 1.4,
        direction: Vec3::Y,
        spread: 12.0_f32.to_radians(),
        acceleration: Vec3::new(0.7, 0.25, 0.3),
        size: [1.2, 6.5],
        color: [[0.36, 0.36, 0.38, 0.7], [0.62, 0.62, 0.65, 0.0]],
        emission: 1.0,
        blend: ParticleBlend::Alpha,
        stretch: 0.0,
        offset: Vec3::new(0.0, 0.5, 0.0),
        radius: 0.25,
        seed,
    }
}

/// Water thrown up from a fountain and falling back.
fn fountain_spray(seed: u32) -> ParticleEmitter {
    ParticleEmitter {
        rate: 70.0,
        lifetime: 1.3,
        speed: 5.5,
        direction: Vec3::Y,
        spread: 9.0_f32.to_radians(),
        acceleration: Vec3::new(0.0, -9.8, 0.0),
        size: [0.14, 0.07],
        color: [[0.78, 0.9, 1.0, 0.6], [0.85, 0.93, 1.0, 0.0]],
        emission: 1.0,
        blend: ParticleBlend::Alpha,
        stretch: 0.05,
        offset: Vec3::new(0.0, 0.5, 0.0),
        radius: 0.05,
        seed,
    }
}

/// The spawned city, for the workload's observations.
pub(crate) struct Spawned {
    pub transform_nodes: u32,
    pub moving_nodes: u32,
    pub sensor_nodes: u32,
    pub probes: Vec<Entity>,
    /// World region holding every block and the air above them, for the
    /// shadows.
    pub bounds: (Vec3, Vec3),
}

/// Builds and spawns the city of `config` into `world`.
pub(crate) fn populate(world: &mut World, config: &StressConfig) -> Result<Spawned, String> {
    let city = build(config);
    let mut assets = world.resource_mut::<AssetLibrary>();
    let mut library = Library::new(&mut assets, config.mesh_variety, config.textured)?;
    let meshes = city
        .nodes
        .iter()
        .map(|node| {
            node.look
                .map(|look| library.get(&mut assets, look.shape, look.paint))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let renderables = meshes.iter().filter(|mesh| mesh.is_some()).count() as u32;
    debug_assert_eq!(renderables, config.renderables, "renderables as configured");
    let dense = if config.dense_colliders {
        config.colliders
    } else {
        0
    };
    let mut entities = Vec::with_capacity(city.nodes.len());
    let mut parents = Vec::new();
    let mut probes = Vec::with_capacity(config.colliders as usize);
    let (mut leaf, mut moving_nodes, mut sensor_nodes) = (0_u32, 0_u32, 0_u32);
    for (node, mesh) in city.nodes.iter().zip(meshes) {
        let transform = node
            .motion
            .map_or(node.transform, |motion| motion.apply(node.transform, 0.0));
        let mut entity = world.spawn(transform);
        if let Some(motion) = node.motion {
            match motion {
                Motion::Travel(travel) => entity.insert(travel),
                Motion::Orbit(orbit) => entity.insert(orbit),
                Motion::Wheel(wheel) => entity.insert(wheel),
                Motion::Spin(spin) => entity.insert(spin),
                Motion::Slide(slide) => entity.insert(slide),
            };
            if config.interpolation {
                entity.insert(InterpolateTransform);
            }
        }
        if let (Some(mesh), Some(look)) = (mesh, node.look) {
            entity.insert(mesh);
            // Engine lights and particles on
            // the parts that carry them; the renderables stay the same.
            match look.shape {
                Shape::Lamp => {
                    entity.insert(street_light());
                }
                Shape::Chimney => {
                    entity.insert(chimney_smoke(leaf));
                }
                Shape::Fountain => {
                    entity.insert(fountain_spray(leaf));
                }
                _ => {}
            }
            moving_nodes += u32::from(node.moving);
            // Colliders spread evenly over the renderables, or on the dense
            // crates that lead them.
            let collider = if dense > 0 {
                leaf < dense
            } else {
                u64::from(leaf + 1) * u64::from(config.colliders) / u64::from(config.renderables)
                    > u64::from(leaf) * u64::from(config.colliders) / u64::from(config.renderables)
            };
            if collider {
                let sensor = probes.len() as u32 % 100 < config.sensor_percent;
                let shape = if look.shape.round() {
                    Collider::sphere(0.5)
                } else {
                    Collider::cuboid(Vec3::splat(0.5))
                }
                .map_err(|error| error.to_string())?
                .with_sensor(sensor);
                entity.insert(shape);
                probes.push(entity.id());
                sensor_nodes += u32::from(sensor);
            }
            leaf += 1;
        }
        let id = entity.id();
        if let Some(parent) = node.parent {
            parents.push((id, Some(entities[parent as usize])));
        }
        entities.push(id);
    }
    set_parents(world, &parents).map_err(|error| error.to_string())?;
    // The outermost block ring, one ring of margin for streets and traffic,
    // and the tallest spires.
    let ring = city
        .blocks
        .iter()
        .map(|&(bx, bz, _)| bx.abs().max(bz.abs()))
        .max()
        .unwrap_or(0);
    let half = (ring as f32 + 1.5) * PERIOD;
    Ok(Spawned {
        transform_nodes: entities.len() as u32,
        moving_nodes,
        sensor_nodes,
        probes,
        bounds: (Vec3::new(-half, -2.0, -half), Vec3::new(half, 300.0, half)),
    })
}
