use engine::math;
use engine::prelude::*;
use glam::Mat3;

use super::gather::{Setup, Spawns, SprayKind, device_spot, view_direction};
use super::numbers::{BarrierLook, DeviceLook, EffectsNumbers, PropsLook, Spray};
use super::{
    Device, EffectsDrawing, EffectsFrame, EffectsShow, EffectsStats, Item, ScreenStats, Show,
};
use crate::play::actors::{Body, Human, Wish};
use crate::play::camera::TestCamera;
use crate::play::effects::{EffectsFeed, Form, GroundMarks, Look, PropPlace, Shown};
use crate::play::puppet::DeviceLamp;
use bevy_ecs::system::SystemParam;

use super::previews::preview_items;
use crate::play::actors::Actor;
use crate::play::flow::{Flow, Stage};
use crate::play::light::Lighting;
use crate::play::powers::Powers;
use crate::play::round::Round;
use crate::play::view::Watched;
use crate::play::world::{Barrier, BarrierFace};
use crate::rules::Side;
use crate::rules::Team;

const TRACER: f32 = 0.0;
const FLASH: f32 = 1.0;
const SPRAY: f32 = 2.0;
const HALO: f32 = 3.0;

const PANEL: f32 = 0.0;
const GLOWING: f32 = 1.0;
const BODY: f32 = 2.0;
const STRAND: f32 = 3.0;

const DECAL: f32 = 0.0;
const DISC: f32 = 1.0;
const STRIKE: f32 = 2.0;
const PULSE: f32 = 3.0;
const NEAR: f32 = 4.0;
const BLOB: f32 = 5.0;

pub(super) const BOX_VERTICES: u32 = 36;
const QUAD_VERTICES: u32 = 6;

fn row(v: Vec3, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}

fn rgb(c: [f32; 3], w: f32) -> [f32; 4] {
    [c[0], c[1], c[2], w]
}

fn seeded(seed: u32) -> f32 {
    (seed >> 8) as f32
}

fn age_of(now: f64, born: f64) -> f32 {
    math::max((now - born) as f32, 0.0)
}

fn frame_of(normal: Vec3) -> (Vec3, Vec3, Vec3) {
    let z = normal.try_normalize().unwrap_or(Vec3::Y);
    let helper = if z.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let x = helper.cross(z).normalize();
    (x, z.cross(x), z)
}

fn spray_of(look: &EffectsNumbers, kind: SprayKind) -> (&Spray, [f32; 3]) {
    match kind {
        SprayKind::Impact(kind) => {
            let spray = look.impacts.of(kind);
            (spray, spray.color)
        }
        SprayKind::Juice(color) => (&look.juice.spray, color),
        SprayKind::Blast => (&look.blast.spray, look.blast.spray.color),
    }
}

fn glow_items(
    look: &EffectsNumbers,
    spawns: &Spawns,
    now: f64,
    items: &mut Vec<Item>,
    stats: &mut EffectsStats,
) {
    let tracer = &look.tracer;
    for shot in spawns.tracers.iter() {
        let age = age_of(now, shot.born);
        if age < tracer.seconds {
            stats.tracers += 1;
            items.push(Item {
                rows: [
                    row(shot.from, age),
                    row(shot.to, tracer.seconds),
                    rgb(tracer.core, tracer.width_m),
                    [TRACER, 0.0, 1.0, 0.0],
                    [0.0; 4],
                    [tracer.halo, 0.0, 0.0, 0.0],
                    [0.0; 4],
                    [0.0; 4],
                ],
            });
        }
    }
    let flash = &look.flash;
    for burst in spawns.flashes.iter() {
        let age = age_of(now, burst.born);
        if age < burst.seconds {
            stats.flashes += 1;
            items.push(Item {
                rows: [
                    row(burst.at, age),
                    row(burst.toward, burst.seconds),
                    rgb(flash.color, flash.size_m * burst.scale),
                    [FLASH, seeded(burst.seed), 1.0, 0.0],
                    [0.0; 4],
                    [0.0, flash.points, 0.0, 0.0],
                    [0.0; 4],
                    [0.0; 4],
                ],
            });
        }
    }
    for burst in spawns.sprays.iter() {
        let (spray, color) = spray_of(look, burst.kind);
        let age = age_of(now, burst.born);
        if age < spray.seconds {
            stats.sprays += 1;
            items.push(Item {
                rows: [
                    row(burst.at, age),
                    row(burst.normal, spray.seconds),
                    rgb(color, spray.size_m * burst.scale),
                    [
                        SPRAY,
                        seeded(burst.seed),
                        spray.sprites as f32,
                        spray.speed_mps * burst.scale,
                    ],
                    [
                        spray.gravity_mps2,
                        spray.opacity,
                        spray.spread,
                        spray.streak_seconds,
                    ],
                    [0.0; 4],
                    [0.0; 4],
                    [0.0; 4],
                ],
            });
        }
    }
}

struct Boxed {
    centre: Vec3,
    half: Vec3,
    axes: Mat3,
}

fn boxed(form: &Form, size: f32) -> Boxed {
    match *form {
        Form::Box {
            centre,
            half_extents,
            axes,
        } => Boxed {
            centre,
            half: half_extents,
            axes,
        },
        Form::Sphere { centre, .. } => Boxed {
            centre,
            half: Vec3::splat(size * 0.5),
            axes: Mat3::IDENTITY,
        },
        Form::Segment { a, b } => {
            let along = b - a;
            let (x, y, z) = frame_of(along);
            Boxed {
                centre: (a + b) * 0.5,
                half: Vec3::new(size * 0.5, size * 0.5, along.length() * 0.5),
                axes: Mat3::from_cols(x, y, z),
            }
        }
    }
}

fn centre_radius(form: &Form) -> (Vec3, f32) {
    match *form {
        Form::Sphere { centre, radius } => (centre, radius),
        Form::Box {
            centre,
            half_extents,
            ..
        } => (centre, half_extents.max_element()),
        Form::Segment { a, b } => ((a + b) * 0.5, a.distance(b) * 0.5),
    }
}

struct Solid {
    shape: Boxed,
    kind: f32,
    fill: [f32; 3],
    glow: f32,
    edge: [f32; 3],
    armed: bool,
    strength: f32,
    pulse_hz: f32,
}

fn solid_item(solid: &Solid, look: &EffectsNumbers) -> Item {
    let shape = &solid.shape;
    Item {
        rows: [
            row(shape.centre, 0.0),
            row(shape.half, solid.pulse_hz),
            rgb(solid.fill, solid.glow),
            [
                solid.kind,
                0.0,
                if solid.armed { 1.0 } else { 0.0 },
                solid.strength,
            ],
            row(shape.axes.x_axis, 0.0),
            row(shape.axes.y_axis, 0.0),
            rgb(solid.edge, look.panels.edge_power),
            [look.panels.edge_glow, 0.0, 0.0, 0.0],
        ],
    }
}

struct Mark {
    centre: Vec3,
    radius: f32,
    half: Vec3,
    axes: Mat3,
    color: [f32; 3],
    fill: f32,
    kind: f32,
    seed: f32,
    rim: f32,
    progress: f32,
    fade: f32,
    pattern: (f32, f32),
}

fn mark_item(mark: &Mark) -> Item {
    Item {
        rows: [
            row(mark.centre, mark.radius),
            row(mark.half, 0.0),
            rgb(mark.color, mark.fill),
            [mark.kind, mark.seed, 0.0, mark.rim],
            row(mark.axes.x_axis, mark.pattern.0),
            row(mark.axes.y_axis, mark.pattern.1),
            [mark.progress, mark.fade, 0.0, 0.0],
            [0.0; 4],
        ],
    }
}

fn disc(look: &EffectsNumbers, centre: Vec3, radius: f32, color: [f32; 3], kind: f32) -> Mark {
    let zones = &look.zones;
    Mark {
        centre,
        radius,
        half: Vec3::new(radius, zones.height_m, radius),
        axes: Mat3::IDENTITY,
        color,
        fill: zones.fill,
        kind,
        seed: 0.0,
        rim: zones.rim_m,
        progress: 0.0,
        fade: 1.0,
        pattern: (zones.pattern_m, zones.pattern_speed),
    }
}

fn team_color(look: &EffectsNumbers, team: Team) -> [f32; 3] {
    look.teams[team.index()]
}

fn progress(shown: &Shown) -> f32 {
    shown
        .lifetime_ticks
        .filter(|life| *life > 0)
        .map_or(0.0, |life| {
            math::clamp(shown.age_ticks as f32 / life as f32, 0.0, 1.0)
        })
}

#[derive(Default)]
struct Sorted {
    solids: Vec<Item>,
    smokes: Vec<Item>,
    marks: Vec<Item>,
    cover: f32,
}

fn smoke_item(look: &EffectsNumbers, centre: Vec3, radius: f32, strength: f32, seed: f32) -> Item {
    let smoke = &look.smoke;
    let outer = radius + smoke.rim_m;
    Item {
        rows: [
            row(centre, radius),
            row(Vec3::splat(outer), 0.0),
            rgb(smoke.color, smoke.rim_m),
            [0.0, seed, smoke.sight_m, strength],
            row(Vec3::X, smoke.noise_scale),
            row(Vec3::Y, smoke.noise_speed),
            [0.0; 4],
            [0.0; 4],
        ],
    }
}

fn sort_shown(
    look: &EffectsNumbers,
    (shown, meshed): (&[Shown], &[u32]),
    eye: Option<Vec3>,
    out: &mut Sorted,
    stats: &mut EffectsStats,
) {
    let traps = &look.traps;
    for (index, thing) in shown.iter().enumerate() {
        if meshed.contains(&thing.effect)
            && matches!(
                thing.look,
                Look::Projectile | Look::Trap | Look::Sensor | Look::Wall
            )
        {
            stats.meshed += 1;
            continue;
        }
        let (centre, radius) = centre_radius(&thing.form);
        let edge = team_color(look, thing.team);
        let zone = |color: [f32; 3]| mark_item(&disc(look, centre, radius, color, DISC));
        let glowing = |size: f32| Solid {
            shape: boxed(&thing.form, size),
            kind: GLOWING,
            fill: edge,
            glow: if thing.armed {
                traps.armed_glow
            } else {
                traps.idle_glow
            },
            edge,
            armed: thing.armed,
            strength: thing.strength,
            pulse_hz: traps.pulse_hz,
        };
        match thing.look {
            Look::Smoke => {
                stats.smokes += 1;
                out.smokes.push(smoke_item(
                    look,
                    centre,
                    radius,
                    thing.strength,
                    index as f32,
                ));
                if let Some(eye) = eye {
                    let inside = 1.0 - (eye.distance(centre) - radius) / look.smoke.rim_m;
                    out.cover =
                        math::max(out.cover, math::clamp(inside, 0.0, 1.0) * thing.strength);
                }
            }
            Look::DamageZone => out.marks.push(zone(look.zones.damage)),
            Look::SlowZone => out.marks.push(zone(look.zones.slow)),
            Look::HealZone => out.marks.push(zone(look.zones.heal)),
            Look::SightZone => out.marks.push(zone(look.zones.sight)),
            Look::RevealZone => out.marks.push(zone(look.zones.reveal)),
            Look::StrikeMark => {
                let mut mark = disc(look, centre, radius, look.strike.color, STRIKE);
                mark.fill = look.strike.fill;
                mark.rim = look.strike.rim_m;
                mark.half.y = look.strike.height_m;
                mark.progress = progress(thing);
                out.marks.push(mark_item(&mark));
            }
            Look::Curtain | Look::Wall => {
                stats.panels += 1;
                let fill = if thing.look == Look::Wall {
                    look.panels.wall
                } else {
                    look.panels.curtain
                };
                let panel = Solid {
                    shape: boxed(&thing.form, radius),
                    kind: PANEL,
                    fill,
                    glow: 1.0,
                    edge,
                    armed: thing.armed,
                    strength: thing.strength,
                    pulse_hz: 0.0,
                };
                out.solids.push(solid_item(&panel, look));
            }
            Look::Trap | Look::Sensor => {
                stats.glowing += 1;
                out.solids.push(solid_item(&glowing(traps.size_m), look));
            }
            Look::Wire => {
                stats.glowing += 1;
                let strand = Solid {
                    kind: STRAND,
                    glow: look.props.wire_glow,
                    edge: look.props.wire_color,
                    ..glowing(traps.wire_width_m)
                };
                out.solids.push(solid_item(&strand, look));
            }
            Look::Projectile => {
                stats.glowing += 1;
                out.solids
                    .push(solid_item(&glowing(traps.projectile_size_m), look));
            }
        }
        if matches!(
            thing.look,
            Look::DamageZone | Look::SlowZone | Look::HealZone | Look::SightZone | Look::RevealZone
        ) {
            stats.zones += 1;
        }
    }
}

fn pooled_marks(
    look: &EffectsNumbers,
    spawns: &Spawns,
    now: f64,
    out: &mut Sorted,
    stats: &mut EffectsStats,
) {
    let decal = &look.decal;
    for mark in spawns.decals.iter() {
        let age = age_of(now, mark.born);
        if age >= decal.seconds {
            continue;
        }
        stats.decals += 1;
        let (x, y, z) = frame_of(mark.normal);
        let fade = math::clamp((decal.seconds - age) / decal.fade_seconds, 0.0, 1.0);
        out.marks.push(mark_item(&Mark {
            centre: mark.at,
            radius: decal.size_m,
            half: Vec3::new(decal.size_m, decal.size_m, decal.depth_m),
            axes: Mat3::from_cols(x, y, z),
            color: decal.color,
            fill: decal.opacity,
            kind: DECAL,
            seed: seeded(mark.seed),
            rim: 0.0,
            progress: age / decal.seconds,
            fade,
            pattern: (0.0, 0.0),
        }));
    }
    let pulse = &look.pulse;
    for ring in spawns.rings.iter() {
        let age = age_of(now, ring.born);
        if age >= ring.seconds {
            continue;
        }
        stats.rings += 1;
        let mut mark = disc(look, ring.at, ring.radius, pulse.color, PULSE);
        mark.fill = pulse.fill;
        mark.rim = pulse.rim_m;
        mark.half.y = pulse.height_m;
        mark.progress = age / ring.seconds;
        out.marks.push(mark_item(&mark));
    }
}

fn showcase_shown(
    look: &EffectsNumbers,
    show: Show,
    human: Option<(&Body, &Wish)>,
    now: f64,
) -> Vec<Shown> {
    let Some((body, wish)) = human else {
        return Vec::new();
    };
    let toward = view_direction(wish);
    let flat = Vec3::new(toward.x, 0.0, toward.z).normalize_or_zero();
    let side = flat.cross(Vec3::Y);
    let case = &look.showcase;
    let ahead = body.feet + flat * case.distance_m;
    let cycle = (now / f64::from(case.every_seconds * 4.0)).fract() as f32;
    let life = 1000;
    let shown = |look: Look, form: Form, team: Team| Shown {
        effect: u32::MAX,
        look,
        form,
        team,
        age_ticks: (cycle * life as f32) as u32,
        lifetime_ticks: Some(life),
        strength: 1.0,
        armed: cycle > 0.5,
    };
    let sphere = |at: Vec3, radius: f32| Form::Sphere { centre: at, radius };
    let all = show == Show::All;
    let mut list = Vec::new();
    if all || show == Show::Smoke {
        list.push(shown(
            Look::Smoke,
            sphere(ahead + Vec3::Y * case.radius_m * 0.5, case.radius_m),
            Team::First,
        ));
    }
    if show == Show::InsideSmoke {
        list.push(shown(
            Look::Smoke,
            sphere(body.eye, case.radius_m),
            Team::First,
        ));
    }
    if all || show == Show::Walls {
        let [length, height, thickness] = case.wall_m;
        for (offset, look_kind, team) in [
            (-1.0, Look::Wall, Team::First),
            (1.0, Look::Curtain, Team::Second),
        ] {
            let centre = ahead + side * (offset * length * 0.6) + Vec3::Y * height * 0.5;
            let axes = Mat3::from_cols(side, Vec3::Y, flat);
            let half = Vec3::new(length * 0.5, height * 0.5, thickness * 0.5);
            list.push(shown(
                look_kind,
                Form::Box {
                    centre,
                    half_extents: half,
                    axes,
                },
                team,
            ));
        }
    }
    if all || show == Show::Zones {
        let kinds = [
            Look::DamageZone,
            Look::SlowZone,
            Look::HealZone,
            Look::SightZone,
            Look::RevealZone,
        ];
        for (index, kind) in kinds.into_iter().enumerate() {
            let offset = (index as f32 - 2.0) * case.radius_m * 2.2;
            list.push(shown(
                kind,
                sphere(ahead + side * offset, case.radius_m),
                Team::First,
            ));
        }
    }
    if all || show == Show::Traps {
        let near = body.feet + flat * case.distance_m * 0.5;
        list.push(shown(
            Look::Trap,
            sphere(near - side, look.traps.size_m),
            Team::Second,
        ));
        list.push(shown(
            Look::Sensor,
            sphere(near, look.traps.size_m),
            Team::First,
        ));
        let wire = Form::Segment {
            a: near + side + Vec3::Y * 0.2,
            b: near + side * 3.0 + Vec3::Y * 0.2,
        };
        list.push(shown(Look::Wire, wire, Team::Second));
        list.push(shown(
            Look::Projectile,
            sphere(near + Vec3::Y * (1.0 + cycle), 0.1),
            Team::First,
        ));
    }
    if show == Show::Strike {
        list.push(shown(
            Look::StrikeMark,
            sphere(ahead, case.radius_m),
            Team::Second,
        ));
    }
    list
}

fn lamp_glow(device: &DeviceLook, left_seconds: Option<f32>, fuse_seconds: f32) -> f32 {
    let Some(left) = left_seconds else {
        return device.dim_glow;
    };
    let fast = math::max(device.blink_fast_seconds, 0.01);
    let fuse = math::max(fuse_seconds, 0.01);
    let left = math::clamp(left, 0.0, fuse);
    let slope = (device.blink_slow_seconds - fast) / fuse;
    let cycles = if slope.abs() < 1.0e-6 {
        left / fast
    } else {
        math::ln(1.0 + slope * left / fast) / slope
    };
    let pulse = 0.5 + 0.5 * math::cos(cycles * std::f32::consts::TAU);
    device.dim_glow + (device.peak_glow - device.dim_glow) * math::pow(pulse, device.pulse_power)
}

fn halo_item(at: Vec3, color: [f32; 3], size: f32) -> Item {
    Item {
        rows: [
            row(at, 0.0),
            row(Vec3::ZERO, 1.0),
            rgb(color, size),
            [HALO, 0.0, 1.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
        ],
    }
}

struct DeviceSeen {
    spot: Option<(Vec3, Option<u32>)>,
    lamp: Option<Vec3>,
    showcase: bool,
    stand_in: bool,
}

fn device_items(
    look: &EffectsNumbers,
    seen: &DeviceSeen,
    (fuse_seconds, tick_seconds, alpha): (f32, f32, f32),
    now: f64,
) -> (Vec<Item>, Option<(Item, f32)>) {
    let Some((at, left)) = seen.spot else {
        return (Vec::new(), None);
    };
    let stand_in = seen.stand_in;
    let device = &look.device;
    let [width, height, depth] = device.size_m;
    let mut items = Vec::new();
    let base = Solid {
        shape: Boxed {
            centre: at + Vec3::Y * height * 0.5,
            half: Vec3::new(width, height, depth) * 0.5,
            axes: Mat3::IDENTITY,
        },
        kind: BODY,
        fill: device.body,
        glow: 1.0,
        edge: device.light,
        armed: false,
        strength: 1.0,
        pulse_hz: 0.0,
    };
    if stand_in {
        items.push(solid_item(&base, look));
    }
    let left_seconds = match left {
        Some(ticks) => Some((ticks as f32 - alpha) * tick_seconds),
        None if seen.showcase => Some(fuse_seconds - (now % f64::from(fuse_seconds)) as f32),
        None => None,
    };
    let glow = lamp_glow(device, left_seconds, fuse_seconds);
    let lamp = seen
        .lamp
        .unwrap_or(at + Vec3::Y * (height + device.light_size_m * 0.5));
    let light = Solid {
        shape: Boxed {
            centre: lamp,
            half: Vec3::splat(device.light_size_m * 0.5),
            axes: Mat3::IDENTITY,
        },
        kind: GLOWING,
        fill: device.light,
        glow,
        ..base
    };
    items.push(solid_item(&light, look));
    let halo = device.light.map(|value| value * glow * 0.5);
    (items, Some((halo_item(lamp, halo, device.halo_m), glow)))
}

struct Raised {
    face: BarrierFace,
    friendly: bool,
    drop: f32,
    near: Option<Vec3>,
}

fn barrier_item(look: &BarrierLook, raised: &Raised, seed: f32) -> Item {
    let [a, b, _, top] = raised.face.corners;
    let along = (b - a).normalize_or(Vec3::X);
    let rise = top - a;
    let start = a - along * look.extend_m;
    let width = a.distance(b) + 2.0 * look.extend_m;
    let color = if raised.friendly {
        look.friendly
    } else {
        look.enemy
    };
    Item {
        rows: [
            row(start, width),
            row(along, rise.length()),
            row(rise.normalize_or(Vec3::Y), raised.drop),
            rgb(color.map(|value| value * look.glow), look.near_boost),
            raised.near.map_or([0.0; 4], |near| row(near, look.near_m)),
            [look.cell_m, look.band_m, look.band_mps, look.line_px],
            [look.fill, look.line, look.edge, look.edge_m],
            [seed, look.band_share, 0.0, 0.0],
        ],
    }
}

fn nearest(face: &BarrierFace, bodies: &[Vec3], reach: f32) -> Option<Vec3> {
    let [a, b, _, _] = face.corners;
    let flat = |point: Vec3| Vec3::new(point.x, 0.0, point.z);
    let (a, b) = (flat(a), flat(b));
    bodies
        .iter()
        .map(|body| {
            let at = flat(*body);
            let along = b - a;
            let share = math::clamp(
                (at - a).dot(along) / math::max(along.length_squared(), 1.0e-6),
                0.0,
                1.0,
            );
            (at.distance(a + along * share), *body)
        })
        .filter(|(distance, _)| *distance <= reach)
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, body)| body)
}

fn showcase_barrier(look: &EffectsNumbers, human: Option<(&Body, &Wish)>, now: f64) -> Vec<Raised> {
    let Some((body, wish)) = human else {
        return Vec::new();
    };
    let case = &look.showcase;
    let every = f64::from(case.every_seconds * 2.0);
    let phase = ((now / every).fract() * every) as f32;
    let count = (now / every) as u64;
    let toward = view_direction(wish);
    let ahead = Vec3::new(toward.x, 0.0, toward.z).normalize_or(Vec3::NEG_Z);
    let along = ahead.cross(Vec3::Y);
    let centre = body.feet + ahead * case.distance_m;
    let [width, height, _] = case.wall_m;
    let a = centre - along * width * 0.5;
    let b = centre + along * width * 0.5;
    let held = case.every_seconds * 1.5;
    vec![Raised {
        face: BarrierFace {
            side: Side::Attack,
            corners: [a, b, b + Vec3::Y * height, a + Vec3::Y * height],
            normal: -ahead,
        },
        friendly: count.is_multiple_of(2),
        drop: math::max(phase - held, 0.0) / look.barrier.drop_seconds,
        near: Some(body.feet + Vec3::Y),
    }]
}

fn raised_faces(
    look: &BarrierLook,
    barrier: &Barrier,
    (viewer, bodies): (Option<Side>, &[Vec3]),
    since_drop: Option<f32>,
) -> Vec<Raised> {
    let drop = if barrier.up {
        Some(0.0)
    } else {
        since_drop.map(|seconds| seconds / look.drop_seconds)
    };
    let Some(drop) = drop else {
        return Vec::new();
    };
    barrier
        .faces
        .iter()
        .map(|face| Raised {
            face: *face,
            friendly: viewer.is_none_or(|side| side == face.side),
            drop,
            near: nearest(face, bodies, look.near_m),
        })
        .collect()
}

fn barrier_part(
    look: &EffectsNumbers,
    (seen, round): (&Seen<'_, '_>, Option<&Round>),
    (fixed, alpha): (&FixedTime, f32),
    showcase: Vec<Raised>,
    stats: &mut EffectsStats,
) -> Vec<Item> {
    let viewer = seen
        .slot()
        .zip(round)
        .and_then(|(slot, round)| round.side_of_slot(slot as usize));
    let bodies: Vec<Vec3> = seen.bodies.iter().map(|body| body.feet + Vec3::Y).collect();
    let mut raised = showcase;
    if let Some(barrier) = seen.barrier.as_deref() {
        let since = barrier
            .dropped_tick
            .map(|tick| (fixed.tick.saturating_sub(tick) as f32 + alpha) * fixed.delta_seconds);
        raised.extend(raised_faces(
            &look.barrier,
            barrier,
            (viewer, &bodies),
            since,
        ));
    }
    raised.retain(|face| face.drop < 1.0);
    stats.barrier_faces = raised.len() as u32;
    stats.barrier_drop = raised.iter().map(|face| face.drop).fold(0.0, math::max);
    raised
        .iter()
        .enumerate()
        .map(|(index, face)| barrier_item(&look.barrier, face, index as f32))
        .collect()
}

fn trail_item(look: &PropsLook, color: [f32; 3], from: Vec3, to: Vec3) -> Item {
    Item {
        rows: [
            row(from, 0.0),
            row(to, 1.0),
            rgb(
                color.map(|value| value * look.trail_glow),
                look.trail_width_m,
            ),
            [TRACER, 0.0, 1.0, 0.0],
            [0.0; 4],
            [0.5, 0.0, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
        ],
    }
}

fn prop_items(
    look: &EffectsNumbers,
    feed: &EffectsFeed,
    (glows, marks): (&mut Vec<Item>, &mut Vec<Item>),
    stats: &mut EffectsStats,
) {
    let props = &look.props;
    for prop in &feed.props {
        match prop.place {
            PropPlace::Thrown {
                centre,
                velocity_mps,
                resting: false,
            } if velocity_mps.length_squared() > 0.25 => {
                stats.trails += 1;
                let from = centre - velocity_mps * props.trail_seconds;
                glows.push(trail_item(props, team_color(look, prop.team), from, centre));
            }
            PropPlace::Drop {
                target,
                fall_starts_tick,
                impact_tick,
                ..
            } if (fall_starts_tick..impact_tick).contains(&feed.tick) => {
                stats.shadows += 1;
                let span = (impact_tick - fall_starts_tick) as f32;
                let share = (feed.tick - fall_starts_tick) as f32 / math::max(span, 1.0);
                let radius = props.shadow_radius_m;
                marks.push(mark_item(&Mark {
                    centre: target,
                    radius,
                    half: Vec3::splat(radius),
                    axes: Mat3::IDENTITY,
                    color: [0.0; 3],
                    fill: props.shadow_opacity * share * share,
                    kind: BLOB,
                    seed: 0.0,
                    rim: 0.0,
                    progress: 0.0,
                    fade: 1.0,
                    pattern: (0.0, 0.0),
                }));
            }
            PropPlace::Thrown { .. }
            | PropPlace::Placed { .. }
            | PropPlace::Wall { .. }
            | PropPlace::Wire { .. }
            | PropPlace::Drop { .. } => {}
        }
    }
}

fn screen_frame(
    look: &EffectsNumbers,
    feed: &EffectsFeed,
    show: Show,
    spawns: &Spawns,
    now: f64,
    cover: f32,
) -> (EffectsFrame, ScreenStats) {
    let screen = &look.screen;
    let cycle = 1.0 - (now / f64::from(look.showcase.every_seconds)).fract() as f32;
    let flash = if show == Show::Flash {
        cycle
    } else {
        feed.human.blindness
    };
    let damage = spawns.hurt.now(now, screen.damage_seconds);
    let frame = EffectsFrame {
        clock: [(now % 3600.0) as f32, screen.damage_edge, 0.0, 0.0],
        flash: rgb(screen.flash, math::clamp(flash, 0.0, 1.0)),
        damage: rgb(screen.damage, damage),
        cover: rgb(look.smoke.color, 0.0),
        sun: [0.0; 4],
        sun_color: [0.0; 4],
        sky_color: [0.0; 4],
        field: [0.0; 4],
        field_shape: [0.0; 4],
    };
    let stats = ScreenStats {
        flash: frame.flash[3],
        damage,
        cover,
        near_sight: 0.0,
    };
    (frame, stats)
}

fn near_sight(look: &EffectsNumbers, eye: Vec3, limit: f32) -> Item {
    let reach = limit + look.screen.sight_fade_m;
    mark_item(&Mark {
        centre: eye,
        radius: limit,
        half: Vec3::splat(reach),
        axes: Mat3::IDENTITY,
        color: look.screen.sight,
        fill: 1.0,
        kind: NEAR,
        seed: 0.0,
        rim: look.screen.sight_fade_m,
        progress: 0.0,
        fade: 1.0,
        pattern: (0.0, 0.0),
    })
}

fn blob_marks(marks: &GroundMarks, out: &mut Sorted, stats: &mut EffectsStats) {
    for blob in &marks.blobs {
        if blob.strength <= 0.0 {
            continue;
        }
        stats.blobs += 1;
        out.marks.push(mark_item(&Mark {
            centre: blob.at,
            radius: blob.radius,
            half: Vec3::new(blob.radius, blob.height, blob.radius),
            axes: Mat3::IDENTITY,
            color: [0.0; 3],
            fill: blob.strength,
            kind: BLOB,
            seed: 0.0,
            rim: 0.0,
            progress: 0.0,
            fade: 1.0,
            pattern: (0.0, 0.0),
        }));
    }
}

fn lit_frame(
    frame: &mut EffectsFrame,
    lighting: Option<&Lighting>,
) -> Option<engine_core::TextureRef> {
    let lighting = lighting.filter(|lighting| lighting.lit)?;
    frame.sun = row(lighting.to_sun, 1.0);
    frame.sun_color = rgb(lighting.sun, 0.0);
    frame.sky_color = rgb(lighting.sky, 0.0);
    let field = lighting.field?;
    let [min_x, min_z, max_x, max_z] = field.rect;
    frame.field = [min_x, min_z, 1.0 / (max_x - min_x), 1.0 / (max_z - min_z)];
    frame.field_shape = [field.height_m, field.scale, 1.0, 0.0];
    Some(field.map)
}

fn span(items: &[Item], start: usize) -> std::ops::Range<u32> {
    start as u32..items.len() as u32
}

#[derive(SystemParam)]
pub(super) struct Seen<'w, 's> {
    lighting: Option<Res<'w, Lighting>>,
    lamp: Option<Res<'w, DeviceLamp>>,
    barrier: Option<Res<'w, Barrier>>,
    camera: Option<Res<'w, TestCamera>>,
    watched: Option<Res<'w, Watched>>,
    powers: Option<Res<'w, Powers>>,
    flow: Option<Res<'w, Flow>>,
    bodies: Query<'w, 's, &'static Body>,
    humans: Query<'w, 's, &'static Actor, With<Human>>,
}

impl Seen<'_, '_> {
    fn viewer<'a>(&'a self, human: Option<&'a Body>) -> Option<&'a Body> {
        self.watched
            .as_deref()
            .and_then(|watched| watched.entity)
            .and_then(|entity| self.bodies.get(entity).ok())
            .or(human)
    }

    fn slot(&self) -> Option<u32> {
        self.watched
            .as_deref()
            .and_then(|watched| watched.slot)
            .or_else(|| self.humans.iter().next().map(|actor| actor.slot))
    }

    fn playing(&self) -> bool {
        self.flow
            .as_deref()
            .is_none_or(|flow| flow.stage == Stage::Playing)
            && !self.camera.as_deref().is_some_and(|camera| camera.outside)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "one frame's drawing reads every effect source"
)]
pub(super) fn assemble(
    time: Res<FrameTime>,
    fixed: Res<FixedTime>,
    setup: Res<Setup>,
    show: Res<EffectsShow>,
    spawns: Res<Spawns>,
    feed: Res<EffectsFeed>,
    ground: Option<Res<GroundMarks>>,
    round: Option<Res<Round>>,
    humans: Query<(&Body, &Wish), With<Human>>,
    seen: Seen,
    devices: Query<&Device>,
    mut drawing: ResMut<EffectsDrawing>,
    mut stats: ResMut<EffectsStats>,
) {
    let Some(parsed) = setup.parsed.as_ref() else {
        *drawing = EffectsDrawing::default();
        stats.set_if_neq(EffectsStats {
            refusal: setup
                .refusal
                .clone()
                .or(Some("numbers.json not read yet".into())),
            ..EffectsStats::default()
        });
        return;
    };
    let look = &parsed.look;
    let now = time.elapsed;
    let human = humans.iter().next();
    let eye = seen
        .viewer(human.map(|(body, _)| body))
        .map(|body| body.eye_before.lerp(body.eye, time.alpha));
    let mut next = EffectsStats::default();
    let mut items = Vec::new();
    let mut sorted = Sorted::default();
    glow_items(look, &spawns, now, &mut items, &mut next);
    next.previews = preview_items(
        look,
        seen.powers.as_deref(),
        seen.slot(),
        &mut items,
        &mut sorted.marks,
    );
    prop_items(look, &feed, (&mut items, &mut sorted.marks), &mut next);
    next.puffs = spawns.puffs;
    let seen_device = DeviceSeen {
        spot: device_spot(round.as_deref(), show.show, human, look),
        lamp: seen.lamp.as_deref().and_then(|lamp| lamp.position),
        showcase: matches!(show.show, Show::Device | Show::All),
        stand_in: devices.iter().any(|device| device.stand_in),
    };
    let timing = (parsed.fuse_seconds, fixed.delta_seconds, time.alpha);
    let (device_solids, halo) = device_items(look, &seen_device, timing, now);
    if let Some((item, glow)) = halo {
        items.push(item);
        next.lamp = glow;
    }
    let glow = span(&items, 0);
    let mut shown = feed.shown.clone();
    if show.show != Show::Off {
        shown.extend(showcase_shown(look, show.show, human, now));
    }
    if let Some(ground) = ground.as_deref() {
        blob_marks(ground, &mut sorted, &mut next);
    }
    let meshed: Vec<u32> = feed.props.iter().map(|prop| prop.effect).collect();
    sort_shown(look, (&shown, &meshed), eye, &mut sorted, &mut next);
    pooled_marks(look, &spawns, now, &mut sorted, &mut next);
    let start = items.len();
    items.extend(sorted.solids);
    let solid = span(&items, start);
    let start = items.len();
    items.extend(device_solids);
    let device = span(&items, start);
    let start = items.len();
    items.extend(sorted.smokes);
    let smoke = span(&items, start);
    let (mut frame, mut screen) = screen_frame(look, &feed, show.show, &spawns, now, sorted.cover);
    let field = lit_frame(&mut frame, seen.lighting.as_deref());
    let limit = if show.show == Show::NearSight {
        Some(look.showcase.radius_m)
    } else {
        feed.human.sight_limit_m
    };
    let playing = seen.playing();
    if let (Some(limit), Some(eye), true) = (limit, eye, playing) {
        sorted.marks.push(near_sight(look, eye, limit));
        screen.near_sight = 1.0;
    }
    if !playing {
        frame.flash[3] = 0.0;
        frame.damage[3] = 0.0;
        screen = ScreenStats::default();
    }
    let start = items.len();
    items.extend(sorted.marks);
    let marks = span(&items, start);
    let shown_barrier = if show.show == Show::Barrier || show.show == Show::All {
        showcase_barrier(look, human, now)
    } else {
        Vec::new()
    };
    let start = items.len();
    let sources = (&seen, round.as_deref());
    let faces = barrier_part(
        look,
        sources,
        (&fixed, time.alpha),
        shown_barrier,
        &mut next,
    );
    items.extend(faces);
    let barrier = span(&items, start);
    let screened = frame.flash[3] > 0.0 || frame.damage[3] > 0.0 || frame.cover[3] > 0.0;
    let ranges = [&glow, &solid, &device, &smoke, &marks, &barrier];
    next.draws =
        ranges.iter().filter(|range| !range.is_empty()).count() as u32 + u32::from(screened);
    next.instances = items.len() as u32;
    next.bytes = (items.len() * size_of::<Item>() + size_of::<EffectsFrame>()) as u64;
    next.device = !device.is_empty();
    next.screen = screen;
    *drawing = EffectsDrawing {
        items,
        glow,
        solid,
        device,
        smoke,
        marks,
        barrier,
        glow_vertices: QUAD_VERTICES * look.most_sprites(),
        frame,
        screen: screened,
        field,
    };
    stats.set_if_neq(next);
}
