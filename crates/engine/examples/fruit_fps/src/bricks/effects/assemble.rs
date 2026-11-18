use engine::math;
use engine::prelude::*;
use glam::Mat3;

use super::gather::{Setup, Spawns, SprayKind, device_spot, view_direction};
use super::models::{ShowcaseModels, traps_layout};
use super::numbers::{
    BarrierLook, DeviceLook, EffectsNumbers, OrbLook, PropsLook, SmokeLook, Spray,
};
use super::{
    Device, EffectsDrawing, EffectsFrame, EffectsShow, EffectsStats, Item, ScreenStats, Show,
};
use crate::abilities::OrbState;
use crate::play::actors::{Body, Human, Wish};
use crate::play::camera::TestCamera;
use crate::play::effects::{EffectsFeed, Form, GroundMarks, Look, PropPlace, Shown};
use crate::play::puppet::DeviceLamp;
use crate::play::screen::Covered;
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

const RIND: f32 = 4.0;
const CURTAIN: f32 = 5.0;
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
        SprayKind::Pieces(color) => (&look.shatter.pieces, color),
        SprayKind::Sparks(color) => (&look.shatter.sparks, color),
    }
}

fn spray_item(
    spray: &Spray,
    color: [f32; 3],
    (at, normal): (Vec3, Vec3),
    (age, seed, scale): (f32, u32, f32),
) -> Item {
    Item {
        rows: [
            row(at, age),
            row(normal, spray.seconds),
            rgb(color, spray.size_m * scale),
            [
                SPRAY,
                seeded(seed),
                spray.sprites as f32,
                spray.speed_mps * scale,
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
    }
}

fn flash_item(
    look: &EffectsNumbers,
    (at, toward): (Vec3, Vec3),
    (age, seconds): (f32, f32),
    (seed, scale): (u32, f32),
) -> Item {
    let flash = &look.flash;
    Item {
        rows: [
            row(at, age),
            row(toward, seconds),
            rgb(flash.color, flash.size_m * scale),
            [FLASH, seeded(seed), 1.0, 0.0],
            [0.0; 4],
            [0.0, flash.points, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
        ],
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
    for burst in spawns.flashes.iter() {
        let age = age_of(now, burst.born);
        if age < burst.seconds {
            stats.flashes += 1;
            items.push(flash_item(
                look,
                (burst.at, burst.toward),
                (age, burst.seconds),
                (burst.seed, burst.scale),
            ));
        }
    }
    for burst in spawns.sprays.iter() {
        let (spray, color) = spray_of(look, burst.kind);
        let age = age_of(now, burst.born);
        if age < spray.seconds {
            stats.sprays += 1;
            items.push(spray_item(
                spray,
                color,
                (burst.at, burst.normal),
                (age, burst.seed, burst.scale),
            ));
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

fn relative(team: Team, viewer: Option<Team>) -> Team {
    if viewer.is_none_or(|viewer| viewer == team) {
        Team::First
    } else {
        Team::Second
    }
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
            [0.0, seed, 0.0, strength],
            row(Vec3::X, smoke.noise_scale),
            row(Vec3::Y, smoke.noise_speed),
            rgb(smoke.tint, smoke.tint_share),
            [smoke.under, 0.0, 0.0, 0.0],
        ],
    }
}

fn panel_item(look: &EffectsNumbers, thing: &Shown, tick_seconds: f32) -> Item {
    let (_, radius) = centre_radius(&thing.form);
    let rise = math::clamp(
        thing.age_ticks as f32 * tick_seconds / math::max(look.panels.rise_seconds, 1.0e-3),
        0.0,
        1.0,
    );
    let mut shape = boxed(&thing.form, radius);
    let sunk = shape.half.y * (1.0 - rise);
    shape.centre -= shape.axes.y_axis * sunk;
    shape.half.y -= sunk;
    let edge = team_color(look, thing.team);
    let panel = Solid {
        shape,
        kind: if thing.look == Look::Wall {
            RIND
        } else {
            CURTAIN
        },
        fill: edge,
        glow: 1.0,
        edge,
        armed: thing.armed,
        strength: thing.strength,
        pulse_hz: 0.0,
    };
    solid_item(&panel, look)
}

fn trim_ring(look: &EffectsNumbers, thing: &Shown) -> Item {
    let (centre, _) = centre_radius(&thing.form);
    let radius = look.traps.size_m * 1.2;
    let mut ring = disc(look, centre, radius, team_color(look, thing.team), DISC);
    ring.fill = 0.0;
    ring.half.y = look.traps.size_m;
    mark_item(&ring)
}

type Things<'a> = (
    &'a [Shown],
    &'a EffectsFeed,
    &'a Spawns,
    (Option<Team>, Show, &'a ShowcaseModels),
);

fn shown_things(
    look: &EffectsNumbers,
    (shown, feed, spawns, (team, show, models)): Things<'_>,
    (eye, tick_seconds, now): (Option<Vec3>, f32, f64),
    (sorted, stats): (&mut Sorted, &mut EffectsStats),
) {
    let mut meshed: Vec<u32> = feed.props.iter().map(|prop| prop.effect).collect();
    if models.traps {
        meshed.push(u32::MAX);
    }
    stats.showcase_model.clone_from(&models.shown);
    let faded = faded_wires(look, feed, (team, show));
    let sided: Vec<Shown> = shown
        .iter()
        .map(|thing| Shown {
            team: relative(thing.team, team),
            ..*thing
        })
        .collect();
    sort_shown(
        look,
        (&sided, &meshed, &faded),
        (eye, tick_seconds),
        sorted,
        stats,
    );
    pooled_marks(look, spawns, now, sorted, stats);
}

fn cover_color(smoke: &SmokeLook) -> [f32; 3] {
    let share = math::clamp(smoke.tint_share, 0.0, 1.0);
    std::array::from_fn(|at| smoke.color[at] + (smoke.tint[at] - smoke.color[at]) * share)
}

fn sort_shown(
    look: &EffectsNumbers,
    (shown, meshed, faded): (&[Shown], &[u32], &[(u32, f32)]),
    (eye, tick_seconds): (Option<Vec3>, f32),
    out: &mut Sorted,
    stats: &mut EffectsStats,
) {
    let traps = &look.traps;
    for (index, thing) in shown.iter().enumerate() {
        if meshed.contains(&thing.effect)
            && matches!(
                thing.look,
                Look::Projectile | Look::Trap | Look::Sensor | Look::Wall | Look::Curtain
            )
        {
            stats.meshed += 1;
            if matches!(thing.look, Look::Trap | Look::Sensor) {
                out.marks.push(trim_ring(look, thing));
            }
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
                out.solids.push(panel_item(look, thing, tick_seconds));
            }
            Look::Trap | Look::Sensor => {
                stats.glowing += 1;
                out.solids.push(solid_item(&glowing(traps.size_m), look));
            }
            Look::Wire => {
                stats.glowing += 1;
                let width = eye.map_or(traps.wire_width_m, |eye| {
                    math::max(
                        traps.wire_width_m,
                        eye.distance(centre) * look.props.wire_least_per_m,
                    )
                });
                let strand = Solid {
                    kind: STRAND,
                    glow: look.props.wire_glow,
                    edge: look.props.wire_color,
                    ..glowing(width)
                };
                let share = faded
                    .iter()
                    .find(|(effect, _)| *effect == thing.effect)
                    .map_or(1.0, |&(_, share)| math::clamp(share, 0.0, 1.0));
                stats.faded_wires += u32::from(share < 1.0);
                let mut item = solid_item(&strand, look);
                item.rows[3][1] = share;
                item.rows[7][1] = look.props.wire_pulse_mps;
                item.rows[7][2] = look.props.wire_pulse_m;
                out.solids.push(item);
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
        owner: None,
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
        let at = traps_layout(body, wish, case);
        let size = look.traps.size_m;
        list.push(shown(Look::Trap, sphere(at.trap, size), Team::Second));
        list.push(shown(Look::Sensor, sphere(at.sensor, size), Team::First));
        let wire = Form::Segment {
            a: at.wire.0 + Vec3::Y * 0.2,
            b: at.wire.1 + Vec3::Y * 0.2,
        };
        list.push(shown(Look::Wire, wire, Team::Second));
        list.push(shown(
            Look::Projectile,
            sphere(at.projectile, 0.1),
            Team::First,
        ));
    }
    if show == Show::Wire {
        for (lift, effect, team) in [
            (0.3, u32::MAX, Team::First),
            (0.6, u32::MAX - 1, Team::Second),
        ] {
            let wire = Form::Segment {
                a: ahead - side * 1.5 + Vec3::Y * lift,
                b: ahead + side * 1.5 + Vec3::Y * lift,
            };
            list.push(Shown {
                effect,
                ..shown(Look::Wire, wire, team)
            });
        }
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

fn device_part(
    look: &EffectsNumbers,
    seen: &Seen<'_, '_>,
    (round, show, human, stand_in): (Option<&Round>, Show, Option<(&Body, &Wish)>, bool),
    (timing, now): ((f32, f32, f32), f64),
    (items, stats): (&mut Vec<Item>, &mut EffectsStats),
) -> Vec<Item> {
    let seen_device = DeviceSeen {
        spot: device_spot(round, show, human, look),
        lamp: seen.lamp.as_deref().and_then(|lamp| lamp.position),
        showcase: matches!(show, Show::Device | Show::All),
        stand_in,
    };
    let (solids, halo) = device_items(look, &seen_device, timing, now);
    if let Some((item, glow)) = halo {
        items.push(item);
        stats.lamp = glow;
    }
    solids
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

struct OrbSeen {
    at: Vec3,
    waiting: bool,
    progress: f32,
    taker: Option<Vec3>,
    taken: Option<f32>,
    fading: Option<f32>,
    seed: u32,
}

fn beam_item(orb: &OrbLook, from: Vec3, to: Vec3, share: f32) -> Item {
    Item {
        rows: [
            row(from, 0.0),
            row(to, 1.0),
            rgb(
                orb.color.map(|value| value * orb.beam_glow * share),
                orb.beam_width_m * share,
            ),
            [TRACER, 0.0, 1.0, 0.0],
            [0.0; 4],
            [0.8, 0.0, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
        ],
    }
}

fn orb_items(
    look: &EffectsNumbers,
    orbs: &[OrbSeen],
    now: f64,
    items: &mut Vec<Item>,
    stats: &mut EffectsStats,
) {
    let orb = &look.orbs;
    let turn = (now * f64::from(orb.pulse_hz)).fract() as f32;
    let beat = 0.5 + 0.5 * math::cos(turn * std::f32::consts::TAU);
    for seen in orbs {
        let centre = seen.at + Vec3::Y * orb.lift_m;
        let strength = match (seen.waiting, seen.fading) {
            (true, _) => 1.0,
            (false, Some(age)) => 1.0 - age / math::max(orb.fade_seconds, 0.01),
            (false, None) => 0.0,
        };
        if strength > 0.0 {
            stats.orbs += 1;
            let glow = (orb.dim + (1.0 - orb.dim) * beat)
                * (1.0 + orb.channel_boost * seen.progress)
                * strength;
            let core = orb.color.map(|value| value * orb.glow * glow);
            let halo = orb.color.map(|value| value * orb.halo_glow * glow);
            items.push(halo_item(centre, core, orb.core_m));
            items.push(halo_item(
                centre,
                halo,
                orb.halo_m * (1.0 + 0.3 * seen.progress),
            ));
        }
        if let (true, Some(taker)) = (seen.waiting, seen.taker) {
            stats.orb_beams += 1;
            let share = orb.beam_least + (1.0 - orb.beam_least) * seen.progress;
            items.push(beam_item(orb, centre, taker, share));
        }
        if let Some(age) = seen.taken.filter(|age| *age < orb.burst.seconds) {
            stats.orb_bursts += 1;
            let burst = &orb.burst;
            items.push(spray_item(
                burst,
                burst.color,
                (centre, Vec3::Y),
                (age, seen.seed, 1.0),
            ));
            if age < look.flash.seconds {
                items.push(flash_item(
                    look,
                    (centre, Vec3::Y),
                    (age, look.flash.seconds),
                    (seen.seed, orb.burst_flash),
                ));
            }
        }
    }
}

fn showcase_orbs(look: &EffectsNumbers, human: Option<(&Body, &Wish)>, now: f64) -> Vec<OrbSeen> {
    let Some((body, wish)) = human else {
        return Vec::new();
    };
    let toward = view_direction(wish);
    let flat = Vec3::new(toward.x, 0.0, toward.z).normalize_or(Vec3::NEG_Z);
    let every = f64::from(math::max(look.showcase.every_seconds, 0.1));
    let count = (now / (every * 2.0)) as u64;
    let phase = (now - count as f64 * every * 2.0) as f32;
    let every = every as f32;
    let channel = phase < every;
    let after = (!channel).then_some(phase - every);
    vec![OrbSeen {
        at: body.feet + flat * look.showcase.distance_m,
        waiting: channel,
        progress: if channel { phase / every } else { 0.0 },
        taker: Some(body.feet.lerp(body.eye, 0.75)),
        taken: after.filter(|_| count.is_multiple_of(2)),
        fading: after.filter(|_| !count.is_multiple_of(2)),
        seed: count as u32,
    }]
}

fn powers_orbs(
    seen: &Seen<'_, '_>,
    (fixed, alpha): (&FixedTime, f32),
    vanished: &[(usize, f64)],
    now: f64,
) -> Vec<OrbSeen> {
    let Some(powers) = seen.powers.as_deref() else {
        return Vec::new();
    };
    let since = |tick: u64| (fixed.tick.saturating_sub(tick) as f32 + alpha) * fixed.delta_seconds;
    let mut list = Vec::new();
    for (index, orb) in powers.orbs().iter().enumerate() {
        let best = orb
            .channels
            .iter()
            .filter_map(|channel| {
                powers
                    .orb_progress(channel.actor.0, fixed.tick)
                    .filter(|(at, _)| *at == index)
                    .map(|(_, share)| (channel.actor.0, share))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));
        list.push(OrbSeen {
            at: orb.at,
            waiting: orb.state == OrbState::Waiting,
            progress: best.map_or(0.0, |(_, share)| math::clamp(share, 0.0, 1.0)),
            taker: best.and_then(|(slot, _)| seen.chest(slot)),
            taken: match orb.state {
                OrbState::Taken { tick, .. } => Some(since(tick)),
                OrbState::Waiting | OrbState::Gone => None,
            },
            fading: vanished
                .iter()
                .find(|(known, _)| *known == index)
                .filter(|_| orb.state == OrbState::Gone)
                .map(|&(_, born)| age_of(now, born)),
            seed: index as u32 * 7919 + 13,
        });
    }
    list
}

fn faded_wires(
    look: &EffectsNumbers,
    feed: &EffectsFeed,
    (viewer, show): (Option<Team>, Show),
) -> Vec<(u32, f32)> {
    let mut faded: Vec<(u32, f32)> = feed
        .props
        .iter()
        .filter(|prop| viewer.is_some_and(|team| team != prop.team))
        .map(|prop| (prop.effect, prop.enemy_share))
        .collect();
    if show == Show::Wire {
        faded.push((u32::MAX - 1, look.showcase.wire_share));
    }
    faded
}

fn power_glows(
    look: &EffectsNumbers,
    (seen, feed, spawns): (&Seen<'_, '_>, &EffectsFeed, &Spawns),
    (fixed, alpha, now): (&FixedTime, f32, f64),
    (show, human): (Show, Option<(&Body, &Wish)>),
    (glows, marks): (&mut Vec<Item>, &mut Vec<Item>),
    stats: &mut EffectsStats,
) {
    stats.previews = preview_items(look, seen.powers.as_deref(), seen.slot(), glows, marks);
    prop_items(look, (feed, seen.team()), (glows, marks), stats);
    let mut orbs = powers_orbs(seen, (fixed, alpha), &spawns.vanished, now);
    if show == Show::Orbs {
        orbs.extend(showcase_orbs(look, human, now));
    }
    orb_items(look, &orbs, now, glows, stats);
    (stats.puffs, stats.shatters) = (spawns.puffs, spawns.shatters);
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
    (feed, viewer): (&EffectsFeed, Option<Team>),
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
                let side = team_color(look, relative(prop.team, viewer));
                glows.push(trail_item(props, side, from, centre));
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
    let panels = &look.panels;
    let (rind, curtain) = (&panels.rind, &panels.curtain);
    let frame = EffectsFrame {
        clock: [(now % 3600.0) as f32, screen.damage_edge, 0.0, 0.0],
        flash: rgb(screen.flash, math::clamp(flash, 0.0, 1.0)),
        damage: rgb(screen.damage, damage),
        cover: rgb(cover_color(&look.smoke), 0.0),
        sun: [0.0; 4],
        sun_color: [0.0; 4],
        sky_color: [0.0; 4],
        field: [0.0; 4],
        field_shape: [0.0; 4],
        rind_skin: rgb(rind.skin, rind.stripe_m),
        rind_stripe: rgb(rind.stripe, rind.pith_m),
        rind_pith: rgb(rind.pith, rind.seed_m),
        rind_flesh: rgb(rind.flesh, 0.0),
        rind_seed: rgb(rind.seed, 0.0),
        curtain_peel: rgb(curtain.peel, curtain.cell_m),
        curtain_pulp: rgb(curtain.pulp, curtain.glow),
        panel: [panels.trim_m, panels.rim_glow, curtain.pleat_m, 0.0],
        zones: [look.zones.dashes, look.zones.dash_speed, 0.0, 0.0],
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
    let mut map = field.map;
    map.sampler = lighting.filtering.light_field.applied(map.sampler);
    Some(map)
}

fn covered_frame(drawing: &mut EffectsDrawing, stats: &mut EffectsStats) {
    *drawing = EffectsDrawing {
        covered: true,
        ..EffectsDrawing::default()
    };
    *stats = EffectsStats {
        covered: true,
        ..EffectsStats::default()
    };
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
    covered: Option<Res<'w, Covered>>,
    watched: Option<Res<'w, Watched>>,
    powers: Option<Res<'w, Powers>>,
    flow: Option<Res<'w, Flow>>,
    bodies: Query<'w, 's, &'static Body>,
    actors: Query<'w, 's, (&'static Actor, &'static Body)>,
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

    fn team(&self) -> Option<Team> {
        let slot = self.slot()?;
        self.actors
            .iter()
            .find(|(actor, _)| actor.slot == slot)
            .map(|(actor, _)| actor.team)
    }

    fn chest(&self, slot: u32) -> Option<Vec3> {
        self.actors
            .iter()
            .find(|(actor, _)| actor.slot == slot)
            .map(|(_, body)| body.feet.lerp(body.eye, 0.75))
    }

    fn covered(&self) -> bool {
        self.covered.as_deref().is_some_and(|covered| covered.0)
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
    (setup, models): (Res<Setup>, Res<ShowcaseModels>),
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
    if seen.covered() {
        return covered_frame(&mut drawing, &mut stats);
    }
    let (look, now) = (&parsed.look, time.elapsed);
    let human = humans.iter().next();
    let eye = seen
        .viewer(human.map(|(body, _)| body))
        .map(|body| body.eye_before.lerp(body.eye, time.alpha));
    let mut next = EffectsStats::default();
    let mut items = Vec::new();
    let mut sorted = Sorted::default();
    glow_items(look, &spawns, now, &mut items, &mut next);
    let sources = (&seen, &*feed, &*spawns);
    let timing = (&*fixed, time.alpha, now);
    power_glows(
        look,
        sources,
        timing,
        (show.show, human),
        (&mut items, &mut sorted.marks),
        &mut next,
    );
    let stand_in = devices.iter().any(|device| device.stand_in);
    let timing = (parsed.fuse_seconds, fixed.delta_seconds, time.alpha);
    let place = (round.as_deref(), show.show, human, stand_in);
    let device_solids = device_part(look, &seen, place, (timing, now), (&mut items, &mut next));
    let glow = span(&items, 0);
    let mut shown = feed.shown.clone();
    if show.show != Show::Off {
        shown.extend(showcase_shown(look, show.show, human, now));
    }
    if let Some(ground) = ground.as_deref() {
        blob_marks(ground, &mut sorted, &mut next);
    }
    let viewer = (seen.team(), show.show, &*models);
    let timing = (eye, fixed.delta_seconds, now);
    let things = (&shown[..], &*feed, &*spawns, viewer);
    shown_things(look, things, timing, (&mut sorted, &mut next));
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
        covered: false,
    };
    stats.set_if_neq(next);
}
