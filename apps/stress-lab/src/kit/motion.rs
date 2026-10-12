//! What moves the worlds' moving objects: the movers the city and the
//! metro run (pure functions of simulated time, so both layouts agree),
//! and the set every world's motion runs in.

use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::ops::DerefMut;

use engine::prelude::*;

use crate::{FrameMetrics, metrics};

/// The systems moving the worlds' objects; the workload's queries run
/// after them.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Animation;

const INV_TAU: f32 = 1.0 / TAU;

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

/// Registers the movers' system.
pub(crate) fn add_systems(app: &mut App) {
    app.add_systems(FixedSimulation, animate.in_set(Animation));
}
