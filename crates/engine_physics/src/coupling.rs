//! What the world's water and wind do to bodies. Water buoys a body by the
//! weight of the water it displaces and drags it toward the water's own
//! velocity, so lighter bodies float, currents carry them and falls drop
//! them; the wind, gusting as it sways the placed objects, drags what stands
//! out of the water. A body's box divides into cells water presses on one by
//! one, so a floating body rights itself and a log turns with the current;
//! round bodies take the exact share of a sphere below a surface.

use bevy_ecs::prelude::World;
use engine_core::math::clamp;
use engine_core::{Physics, WaterSample, WaterSources, Wind};
use glam::{Quat, Vec3};

use crate::body::BodyWater;
use crate::shapes::BodyShape;

/// The most cells a body's box divides into along an axis.
const MOST_CELLS: f32 = 6.0;
/// The spin damping a body takes in air, and the more it takes wholly in water.
pub(crate) const AIR_SPIN: f32 = 0.05;
const WATER_SPIN: f32 = 1.5;
/// Meters a second of current that wake a body asleep in it.
const STILL_WATER: f32 = 0.05;

/// How water presses on a body: the half extents of its box, the cells it
/// divides into along each local axis, the axis it is longest along, whether
/// it is round, and its volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Form {
    pub half: Vec3,
    pub cells: [u8; 3],
    pub long: usize,
    pub round: bool,
    pub volume: f32,
}

impl Form {
    pub fn new(shape: &BodyShape) -> Self {
        let half = shape.half;
        let least = half.min_element();
        let long = if half.x >= half.y && half.x >= half.z {
            0
        } else if half.y >= half.z {
            1
        } else {
            2
        };
        let cells = if shape.round {
            [1; 3]
        } else {
            [0, 1, 2].map(|axis| clamp((2.0 * half[axis] / least).round(), 2.0, MOST_CELLS) as u8)
        };
        Self {
            half,
            cells,
            long,
            round: shape.round,
            volume: shape.volume,
        }
    }

    /// Where along its long axis water is sampled, from -1 to 1: its middle,
    /// and for a long body its end cells too.
    fn stations(&self) -> &'static [f32] {
        if self.half[self.long] >= 2.0 * self.half.min_element() {
            &[-1.0, 0.0, 1.0]
        } else {
            &[0.0]
        }
    }

    /// The area it turns to a flow: a face of its box on average, or a
    /// round body's cross-section.
    fn area(&self) -> f32 {
        let h = self.half;
        if self.round {
            let radius = (h.x + h.y + h.z) / 3.0;
            std::f32::consts::PI * radius * radius
        } else {
            4.0 * (h.x * h.y + h.y * h.z + h.z * h.x) / 3.0
        }
    }
}

/// A body as the forces read it: its pose, center of mass, velocities and
/// mass.
pub(crate) struct Motion {
    pub translation: Vec3,
    pub rotation: Quat,
    pub center: Vec3,
    pub velocity: Vec3,
    pub spin: Vec3,
    pub mass: f32,
}

impl Motion {
    /// The velocity of its point `point`.
    fn at(&self, point: Vec3) -> Vec3 {
        self.velocity + self.spin.cross(point - self.center)
    }
}

/// What water and wind do to a body this tick: the force and the torque
/// about its center of mass, its spin damping, the water it is in and
/// whether that water runs fast enough to wake it.
pub(crate) struct Push {
    pub force: Vec3,
    pub torque: Vec3,
    pub damping: f32,
    pub water: Option<BodyWater>,
    pub running: bool,
}

/// `force` limited to what changes `mass`'s velocity by at most `speed` over
/// `seconds`: drag slows toward the flow and never past it.
fn limited(force: Vec3, mass: f32, speed: f32, seconds: f32) -> Vec3 {
    let impulse = force.length() * seconds;
    let most = mass * speed;
    if impulse > most && impulse > 0.0 {
        force * (most / impulse)
    } else {
        force
    }
}

/// The water sampled along a body: each station's position along its long
/// axis (-1 to 1) and the water there.
struct Along {
    stations: [(f32, Option<WaterSample>); 3],
    count: usize,
}

impl Along {
    /// The world's water over `motion`'s body at each of `form`'s
    /// stations; `samples` counts them.
    fn sample(world: &World, form: &Form, motion: &Motion, samples: &mut u32) -> Self {
        let mut axis = Vec3::ZERO;
        axis[form.long] = form.half[form.long];
        let cells = f32::from(form.cells[form.long]);
        let reach = 1.0 - 1.0 / cells;
        let mut along = Self {
            stations: [(0.0, None); 3],
            count: 0,
        };
        for &station in form.stations() {
            let at = station * reach;
            let point = motion.translation + motion.rotation * (axis * at);
            *samples += 1;
            along.stations[along.count] = (at, WaterSources::sample_in(world, point));
            along.count += 1;
        }
        along
    }

    /// The water at `at` along the body, between the stations around it.
    fn at(&self, at: f32) -> Option<WaterSample> {
        let stations = &self.stations[..self.count];
        let (low, high) = match stations {
            [only] => return only.1,
            [first, middle, last] => {
                if at <= middle.0 {
                    (first, middle)
                } else {
                    (middle, last)
                }
            }
            _ => return None,
        };
        match (low.1, high.1) {
            (Some(a), Some(b)) => {
                let t = clamp((at - low.0) / (high.0 - low.0), 0.0, 1.0);
                let blend = |x: f32, y: f32| x + (y - x) * t;
                Some(WaterSample {
                    surface: blend(a.surface, b.surface),
                    depth: blend(a.depth, b.depth),
                    velocity: [0, 1, 2].map(|i| blend(a.velocity[i], b.velocity[i])),
                    foam: blend(a.foam, b.foam),
                    kind: if t < 0.5 { a.kind } else { b.kind },
                })
            }
            (one, other) => one.or(other),
        }
    }

    /// The water at the body's middle, or any station's.
    fn middle(&self) -> Option<WaterSample> {
        self.at(0.0).or_else(|| {
            self.stations[..self.count]
                .iter()
                .find_map(|station| station.1)
        })
    }
}

/// What the world's water and wind do to a body of `form` and `drag`
/// coefficient moving as `motion` this tick of `seconds` at time `time`.
#[expect(
    clippy::too_many_arguments,
    reason = "each is one of the tick's inputs"
)]
pub(crate) fn push(
    world: &World,
    settings: &Physics,
    wind: Option<&Wind>,
    time: f32,
    seconds: f32,
    form: &Form,
    drag: f32,
    motion: &Motion,
    samples: &mut u32,
) -> Push {
    let mut push = Push {
        force: Vec3::ZERO,
        torque: Vec3::ZERO,
        damping: AIR_SPIN,
        water: None,
        running: false,
    };
    let mut submerged = 0.0;
    if settings.water_density > 0.0 {
        let along = Along::sample(world, form, motion, samples);
        if let Some(middle) = along.middle() {
            let gravity = Vec3::from_array(settings.gravity);
            submerged = if form.round {
                round(
                    &along, settings, gravity, seconds, form, drag, motion, &mut push,
                )
            } else {
                cells(
                    &along, settings, gravity, seconds, form, drag, motion, &mut push,
                )
            };
            let share = clamp(submerged / form.volume, 0.0, 1.0);
            if share > 0.0 {
                let current = Vec3::from_array(middle.velocity);
                push.running = current.length() > STILL_WATER;
                push.water = Some(BodyWater {
                    kind: middle.kind,
                    submerged: share,
                    current: middle.velocity,
                });
            }
        }
    }
    let share = clamp(submerged / form.volume, 0.0, 1.0);
    push.damping = AIR_SPIN + WATER_SPIN * share;
    // The wind on what stands out of the water.
    if let Some(wind) = wind
        && settings.air_density > 0.0
        && share < 1.0
    {
        let blowing = wind.at(motion.center.x, motion.center.z, time);
        let relative = Vec3::new(blowing.x, 0.0, blowing.y) - motion.velocity;
        let speed = relative.length();
        let force =
            relative * (0.5 * settings.air_density * drag * form.area() * speed * (1.0 - share));
        push.force += limited(force, motion.mass, speed, seconds);
    }
    push
}

/// Water's push on a box body, cell by cell, added to `push`: the volume
/// below the surface.
#[expect(
    clippy::too_many_arguments,
    reason = "each is one of the tick's inputs"
)]
fn cells(
    along: &Along,
    settings: &Physics,
    gravity: Vec3,
    seconds: f32,
    form: &Form,
    drag: f32,
    motion: &Motion,
    push: &mut Push,
) -> f32 {
    let counts = form.cells.map(f32::from);
    let cell = form.half / Vec3::from_array(counts);
    let axes = [Vec3::X, Vec3::Y, Vec3::Z].map(|axis| motion.rotation * axis);
    // Each cell's half height in the world.
    let rise = axes[0].y.abs() * cell.x + axes[1].y.abs() * cell.y + axes[2].y.abs() * cell.z;
    let count = counts[0] * counts[1] * counts[2];
    let volume = form.volume / count;
    let mass = motion.mass / count;
    let area = 4.0 * (cell.x * cell.y + cell.y * cell.z + cell.z * cell.x) / 3.0;
    let mut submerged = 0.0;
    for i in 0..form.cells[0] {
        for j in 0..form.cells[1] {
            for k in 0..form.cells[2] {
                let local = Vec3::new(
                    ((f32::from(i) + 0.5) / counts[0] * 2.0 - 1.0) * form.half.x,
                    ((f32::from(j) + 0.5) / counts[1] * 2.0 - 1.0) * form.half.y,
                    ((f32::from(k) + 0.5) / counts[2] * 2.0 - 1.0) * form.half.z,
                );
                let Some(water) = along.at(local[form.long] / form.half[form.long]) else {
                    continue;
                };
                let point = motion.translation + motion.rotation * local;
                let wet = clamp((water.surface - (point.y - rise)) / (2.0 * rise), 0.0, 1.0);
                if wet <= 0.0 {
                    continue;
                }
                let relative = Vec3::from_array(water.velocity) - motion.at(point);
                let speed = relative.length();
                let buoyancy = -gravity * (settings.water_density * volume * wet);
                let dragged = limited(
                    relative * (0.5 * settings.water_density * drag * area * speed * wet),
                    mass,
                    speed,
                    seconds,
                );
                let force = buoyancy + dragged;
                push.force += force;
                push.torque += (point - motion.center).cross(force);
                submerged += volume * wet;
            }
        }
    }
    submerged
}

/// Water's push on a round body, added to `push`: the volume below the
/// surface, the share of a sphere as tall as the body stands.
#[expect(
    clippy::too_many_arguments,
    reason = "each is one of the tick's inputs"
)]
fn round(
    along: &Along,
    settings: &Physics,
    gravity: Vec3,
    seconds: f32,
    form: &Form,
    drag: f32,
    motion: &Motion,
    push: &mut Push,
) -> f32 {
    let Some(water) = along.at(0.0) else {
        return 0.0;
    };
    let axes = [Vec3::X, Vec3::Y, Vec3::Z].map(|axis| motion.rotation * axis);
    let rise = Vec3::new(
        axes[0].y * form.half.x,
        axes[1].y * form.half.y,
        axes[2].y * form.half.z,
    )
    .length();
    let depth = clamp(
        water.surface - (motion.translation.y - rise),
        0.0,
        2.0 * rise,
    );
    let share = depth * depth * (3.0 * rise - depth) / (4.0 * rise * rise * rise);
    if share <= 0.0 {
        return 0.0;
    }
    let relative = Vec3::from_array(water.velocity) - motion.velocity;
    let speed = relative.length();
    let buoyancy = -gravity * (settings.water_density * form.volume * share);
    let dragged = limited(
        relative * (0.5 * settings.water_density * drag * form.area() * speed * share),
        motion.mass,
        speed,
        seconds,
    );
    push.force += buoyancy + dragged;
    push.torque += (motion.translation - motion.center).cross(buoyancy + dragged);
    form.volume * share
}
