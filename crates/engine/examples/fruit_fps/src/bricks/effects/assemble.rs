use engine::math;
use engine::prelude::*;
use engine_core::{DevSetting, Poses, Stats, ViewSkip};
use glam::{Mat3, Quat};

use super::gather::{Setup, Spawns, SprayKind, device_spot, view_direction};
use super::models::{ShowcaseModels, traps_layout};
use super::numbers::{
    BarrierLook, DeviceLook, EffectsNumbers, OrbLook, PropsLook, RindLook, Spray,
};
use super::{EffectsDrawing, EffectsFrame, EffectsShow, EffectsStats, Item, ScreenStats, Show};
use crate::abilities::OrbState;
use crate::play::actors::{Body, Eye, Human, Wish};
use crate::play::camera::TestCamera;
use crate::play::effects::{EffectsFeed, Form, GroundMarks, Look, PropPlace, Shown};
use crate::play::puppet::DeviceLamp;
use bevy_ecs::system::SystemParam;

use super::previews::preview_items;
use crate::play::actors::Actor;
use crate::play::flow::{Flow, Stage};
use crate::play::light::Lighting;
use crate::play::lobby::LobbyView;
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

const GLOWING: f32 = 1.0;
const STRAND: f32 = 3.0;

const DECAL: f32 = 0.0;
const DISC: f32 = 1.0;
const STRIKE: f32 = 2.0;
const PULSE: f32 = 3.0;
const NEAR: f32 = 4.0;
const BLOB: f32 = 5.0;
const CONTACT: f32 = 8.0;

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

fn unit_hash(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277_803_737);
    (((x >> 22) ^ x) >> 8) as f32 / 16_777_216.0
}

fn eased(share: f32) -> f32 {
    let rest = 1.0 - math::clamp(share, 0.0, 1.0);
    1.0 - rest * rest * rest
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
    stats: &mut Stats<'_, EffectsStats>,
) {
    let tracer = &look.tracer;
    for shot in spawns.tracers.iter() {
        let age = age_of(now, shot.born);
        if age < tracer.seconds {
            stats.edit(|stats| stats.tracers += 1);
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
            stats.edit(|stats| stats.flashes += 1);
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
            stats.edit(|stats| stats.sprays += 1);
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
    pulse_hz: f32,
}

fn solid_item(solid: &Solid) -> Item {
    let shape = &solid.shape;
    Item {
        rows: [
            row(shape.centre, 0.0),
            row(shape.half, solid.pulse_hz),
            rgb(solid.fill, solid.glow),
            [solid.kind, 0.0, if solid.armed { 1.0 } else { 0.0 }, 0.0],
            row(shape.axes.x_axis, 0.0),
            row(shape.axes.y_axis, 0.0),
            rgb(solid.edge, 0.0),
            [0.0; 4],
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
        half: Vec3::new(radius + zones.rim_m, zones.height_m, radius + zones.rim_m),
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
    slabs: Vec<Item>,
    sheets: Vec<Item>,
    smokes: Vec<Item>,
    marks: Vec<Item>,
}

#[derive(Default)]
pub(super) struct Lists {
    sorted: Sorted,
    shown: Vec<Shown>,
    meshed: Vec<u32>,
    faded: Vec<(u32, f32)>,
    orbs: Vec<OrbSeen>,
    raised: Vec<Raised>,
    bodies: Vec<Vec3>,
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

struct Wall {
    base: Vec3,
    along: Vec3,
    up: Vec3,
    half_length: f32,
    half_thickness: f32,
    height: f32,
}

fn wall_of(form: &Form) -> Option<Wall> {
    let Form::Box {
        centre,
        half_extents,
        axes,
    } = *form
    else {
        return None;
    };
    let (along, half_length, half_thickness) = if half_extents.x >= half_extents.z {
        (axes.x_axis, half_extents.x, half_extents.z)
    } else {
        (axes.z_axis, half_extents.z, half_extents.x)
    };
    Some(Wall {
        base: centre - axes.y_axis * half_extents.y,
        along,
        up: axes.y_axis,
        half_length,
        half_thickness,
        height: half_extents.y * 2.0,
    })
}

fn life_seconds(thing: &Shown, tick_seconds: f32) -> (f32, f32) {
    let left = thing.lifetime_ticks.map_or(f32::MAX, |life| {
        life.saturating_sub(thing.age_ticks) as f32 * tick_seconds
    });
    (thing.age_ticks as f32 * tick_seconds, left)
}

fn contact_mark(rind: &RindLook, wall: &Wall, share: f32) -> Item {
    mark_item(&Mark {
        centre: wall.base,
        radius: rind.contact_m,
        half: Vec3::new(
            wall.half_length + rind.contact_m,
            rind.contact_height_m,
            wall.half_thickness + rind.contact_m,
        ),
        axes: Mat3::from_cols(wall.along, wall.up, wall.along.cross(wall.up)),
        color: [0.0; 3],
        fill: rind.contact * share,
        kind: CONTACT,
        seed: 0.0,
        rim: rind.contact_m,
        progress: 0.0,
        fade: 1.0,
        pattern: (0.0, 0.0),
    })
}

fn rind_items(look: &EffectsNumbers, thing: &Shown, tick_seconds: f32, out: &mut Sorted) -> u32 {
    let Some(wall) = wall_of(&thing.form) else {
        return 0;
    };
    let rind = &look.panels.rind;
    let count = math::max(
        (wall.half_length * 2.0 / math::max(rind.slab_m, 0.1)).round(),
        1.0,
    ) as u32;
    let half_width = wall.half_length / count as f32;
    let middle = (count as f32 - 1.0) * 0.5;
    let (age, left) = life_seconds(thing, tick_seconds);
    let team = team_color(look, thing.team);
    let mut risen = 0.0;
    for slab in 0..count {
        let seed = thing.effect.wrapping_mul(7919).wrapping_add(slab * 104_729);
        let random = |salt: u32| unit_hash(seed.wrapping_add(salt));
        let order = if middle > 0.0 {
            (slab as f32 - middle).abs() / middle
        } else {
            0.0
        };
        let delay = rind.stagger_seconds * (0.6 * order + 0.4 * random(0));
        let rising = eased((age - delay) / math::max(rind.rise_seconds, 1.0e-3));
        let sinking =
            eased((rind.sink_seconds + delay - left) / math::max(rind.sink_seconds, 1.0e-3));
        let shown = rising * (1.0 - sinking);
        risen += shown;
        let tops = [
            wall.height + rind.bevel_m + rind.top_m * random(1),
            wall.height + rind.bevel_m + rind.top_m * random(2),
        ];
        let sink = (math::max(tops[0], tops[1]) + rind.bury_m) * (1.0 - shown);
        let yaw = (random(3) * 2.0 - 1.0) * rind.yaw_degrees.to_radians();
        let lean = (random(4) * 2.0 - 1.0) * rind.lean_degrees.to_radians();
        let x = Quat::from_axis_angle(wall.up, yaw) * wall.along;
        let y = Quat::from_axis_angle(x, lean) * wall.up;
        let offset = (slab as f32 - middle) * half_width * 2.0;
        out.slabs.push(Item {
            rows: [
                row(wall.base + wall.along * offset - wall.up * sink, half_width),
                [wall.half_thickness, tops[0], tops[1], rind.bevel_m],
                [random(5) * 64.0, thing.strength, sink, 0.0],
                rgb(team, 0.0),
                row(x, 0.0),
                row(y, rind.bury_m),
                [0.0; 4],
                [0.0; 4],
            ],
        });
    }
    out.marks
        .push(contact_mark(rind, &wall, risen / count as f32));
    count
}

fn curtain_item(look: &EffectsNumbers, thing: &Shown, tick_seconds: f32) -> Option<Item> {
    let wall = wall_of(&thing.form)?;
    let curtain = &look.panels.curtain;
    let (age, left) = life_seconds(thing, tick_seconds);
    let fall = math::max(curtain.fall_seconds, 1.0e-3);
    Some(Item {
        rows: [
            row(wall.base, wall.half_length),
            [
                wall.half_thickness,
                wall.height,
                curtain.end_m,
                curtain.top_m,
            ],
            [
                unit_hash(thing.effect) * 64.0,
                math::clamp(age / fall, 0.0, 1.0),
                math::clamp(1.0 - left / fall, 0.0, 1.0),
                0.0,
            ],
            rgb(team_color(look, thing.team), 0.0),
            row(wall.along, 0.0),
            row(wall.up, 0.0),
            [0.0; 4],
            [0.0; 4],
        ],
    })
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
    &'a EffectsFeed,
    &'a Spawns,
    (Option<Team>, Show, &'a ShowcaseModels),
);

fn shown_things(
    look: &EffectsNumbers,
    (feed, spawns, (team, show, models)): Things<'_>,
    (eye, tick_seconds, now): (Option<Vec3>, f32, f64),
    (lists, stats): (&mut Lists, &mut Stats<'_, EffectsStats>),
) {
    let Lists {
        sorted,
        shown,
        meshed,
        faded,
        ..
    } = lists;
    meshed.clear();
    meshed.extend(feed.props.iter().map(|prop| prop.effect));
    if models.traps {
        meshed.push(u32::MAX);
    }
    stats.edit(|stats| stats.showcase_model.clone_from(&models.shown));
    faded_wires(look, feed, (team, show), faded);
    sort_shown(
        look,
        (shown, meshed, faded),
        (eye, tick_seconds),
        sorted,
        stats,
    );
    pooled_marks(look, spawns, now, sorted, stats);
}

fn sort_shown(
    look: &EffectsNumbers,
    (shown, meshed, faded): (&[Shown], &[u32], &[(u32, f32)]),
    (eye, tick_seconds): (Option<Vec3>, f32),
    out: &mut Sorted,
    stats: &mut Stats<'_, EffectsStats>,
) {
    let traps = &look.traps;
    for (index, thing) in shown.iter().enumerate() {
        if meshed.contains(&thing.effect)
            && matches!(thing.look, Look::Projectile | Look::Trap | Look::Sensor)
        {
            stats.edit(|stats| stats.meshed += 1);
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
            pulse_hz: traps.pulse_hz,
        };
        match thing.look {
            Look::Smoke => {
                stats.edit(|stats| stats.smokes += 1);
                out.smokes.push(smoke_item(
                    look,
                    centre,
                    radius,
                    thing.strength,
                    index as f32,
                ));
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
                mark.half.x = radius + mark.rim;
                mark.half.z = radius + mark.rim;
                mark.half.y = look.strike.height_m;
                mark.progress = progress(thing);
                out.marks.push(mark_item(&mark));
            }
            Look::Wall => {
                let slabs = rind_items(look, thing, tick_seconds, out);
                stats.edit(|stats| {
                    stats.panels += 1;
                    stats.slabs += slabs;
                });
            }
            Look::Curtain => {
                stats.edit(|stats| stats.panels += 1);
                out.sheets.extend(curtain_item(look, thing, tick_seconds));
            }
            Look::Trap | Look::Sensor => {
                stats.edit(|stats| stats.glowing += 1);
                out.solids.push(solid_item(&glowing(traps.size_m)));
            }
            Look::Wire => {
                stats.edit(|stats| stats.glowing += 1);
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
                stats.edit(|stats| stats.faded_wires += u32::from(share < 1.0));
                let mut item = solid_item(&strand);
                item.rows[3][1] = share;
                item.rows[7][1] = look.props.wire_pulse_mps;
                item.rows[7][2] = look.props.wire_pulse_m;
                out.solids.push(item);
            }
            Look::Projectile => {
                stats.edit(|stats| stats.glowing += 1);
                out.solids
                    .push(solid_item(&glowing(traps.projectile_size_m)));
            }
        }
        if matches!(
            thing.look,
            Look::DamageZone | Look::SlowZone | Look::HealZone | Look::SightZone | Look::RevealZone
        ) {
            stats.edit(|stats| stats.zones += 1);
        }
    }
}

fn pooled_marks(
    look: &EffectsNumbers,
    spawns: &Spawns,
    now: f64,
    out: &mut Sorted,
    stats: &mut Stats<'_, EffectsStats>,
) {
    let decal = &look.decal;
    for mark in spawns.decals.iter() {
        let age = age_of(now, mark.born);
        if age >= decal.seconds {
            continue;
        }
        stats.edit(|stats| stats.decals += 1);
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
        stats.edit(|stats| stats.rings += 1);
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
}

fn device_items(
    look: &EffectsNumbers,
    seen: &DeviceSeen,
    (fuse_seconds, tick_seconds, alpha): (f32, f32, f32),
    now: f64,
) -> Option<(Item, Item, f32)> {
    let (at, left) = seen.spot?;
    let device = &look.device;
    let left_seconds = match left {
        Some(ticks) => Some((ticks as f32 - alpha) * tick_seconds),
        None if seen.showcase => Some(fuse_seconds - (now % f64::from(fuse_seconds)) as f32),
        None => None,
    };
    let glow = lamp_glow(device, left_seconds, fuse_seconds);
    let lamp = seen.lamp.unwrap_or(at + Vec3::Y * device.lamp_m);
    let light = Solid {
        shape: Boxed {
            centre: lamp,
            half: Vec3::splat(device.light_size_m * 0.5),
            axes: Mat3::IDENTITY,
        },
        kind: GLOWING,
        fill: device.light,
        glow,
        edge: device.light,
        armed: false,
        pulse_hz: 0.0,
    };
    let halo = device.light.map(|value| value * glow * 0.5);
    Some((
        solid_item(&light),
        halo_item(lamp, halo, device.halo_m),
        glow,
    ))
}

fn device_part(
    look: &EffectsNumbers,
    seen: &Seen<'_, '_>,
    (round, show, human): (Option<&Round>, Show, Option<(&Body, &Wish)>),
    (timing, now): ((f32, f32, f32), f64),
    (items, stats): (&mut Vec<Item>, &mut Stats<'_, EffectsStats>),
) -> Option<Item> {
    let seen_device = DeviceSeen {
        spot: device_spot(round, show, human, look),
        lamp: seen.lamp.as_deref().and_then(|lamp| lamp.position),
        showcase: matches!(show, Show::Device | Show::All),
    };
    let (solid, halo, glow) = device_items(look, &seen_device, timing, now)?;
    items.push(halo);
    stats.edit(|stats| stats.lamp = glow);
    Some(solid)
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
            [look.cell_m, look.band_m, look.band_mps, look.band_share],
            [look.fill, look.line, look.ground, look.ground_m],
            [seed, look.side_m, look.fade_from, 0.0],
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
    raised: &mut Vec<Raised>,
) {
    let drop = if barrier.up {
        Some(0.0)
    } else {
        since_drop.map(|seconds| seconds / look.drop_seconds)
    };
    let Some(drop) = drop else {
        return;
    };
    raised.extend(barrier.faces.iter().map(|face| Raised {
        face: *face,
        friendly: viewer.is_none_or(|side| side == face.side),
        drop,
        near: nearest(face, bodies, look.near_m),
    }));
}

fn barrier_part(
    look: &EffectsNumbers,
    (seen, round): (&Seen<'_, '_>, Option<&Round>),
    (fixed, alpha): (&FixedTime, f32),
    (raised, bodies, items): (&mut Vec<Raised>, &mut Vec<Vec3>, &mut Vec<Item>),
    stats: &mut Stats<'_, EffectsStats>,
) -> std::ops::Range<u32> {
    let viewer = seen
        .slot()
        .zip(round)
        .and_then(|(slot, round)| round.side_of_slot(slot as usize));
    if let Some(barrier) = seen.barrier.as_deref() {
        bodies.clear();
        bodies.extend(seen.actors.iter().map(|(_, body)| body.feet + Vec3::Y));
        let since = barrier
            .dropped_tick
            .map(|tick| (fixed.tick.saturating_sub(tick) as f32 + alpha) * fixed.delta_seconds);
        raised_faces(&look.barrier, barrier, (viewer, &bodies[..]), since, raised);
    }
    raised.retain(|face| face.drop < 1.0);
    stats.edit(|stats| {
        stats.barrier_faces = raised.len() as u32;
        stats.barrier_drop = raised.iter().map(|face| face.drop).fold(0.0, math::max);
    });
    put(
        items,
        raised
            .iter()
            .enumerate()
            .map(|(index, face)| barrier_item(&look.barrier, face, index as f32)),
    )
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
    stats: &mut Stats<'_, EffectsStats>,
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
            stats.edit(|stats| stats.orbs += 1);
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
            stats.edit(|stats| stats.orb_beams += 1);
            let share = orb.beam_least + (1.0 - orb.beam_least) * seen.progress;
            items.push(beam_item(orb, centre, taker, share));
        }
        if let Some(age) = seen.taken.filter(|age| *age < orb.burst.seconds) {
            stats.edit(|stats| stats.orb_bursts += 1);
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
    (vanished, now): (&[(usize, f64)], f64),
    list: &mut Vec<OrbSeen>,
) {
    list.clear();
    let Some(powers) = seen.powers.as_deref() else {
        return;
    };
    let since = |tick: u64| (fixed.tick.saturating_sub(tick) as f32 + alpha) * fixed.delta_seconds;
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
}

fn faded_wires(
    look: &EffectsNumbers,
    feed: &EffectsFeed,
    (viewer, show): (Option<Team>, Show),
    faded: &mut Vec<(u32, f32)>,
) {
    faded.clear();
    faded.extend(
        feed.props
            .iter()
            .filter(|prop| viewer.is_some_and(|team| team != prop.team))
            .map(|prop| (prop.effect, prop.enemy_share)),
    );
    if show == Show::Wire {
        faded.push((u32::MAX - 1, look.showcase.wire_share));
    }
}

fn power_glows(
    look: &EffectsNumbers,
    (seen, feed, spawns): (&Seen<'_, '_>, &EffectsFeed, &Spawns),
    (fixed, alpha, now): (&FixedTime, f32, f64),
    (show, human): (Show, Option<(&Body, &Wish)>),
    (glows, marks, orbs): (&mut Vec<Item>, &mut Vec<Item>, &mut Vec<OrbSeen>),
    stats: &mut Stats<'_, EffectsStats>,
) {
    let previews = preview_items(look, seen.powers.as_deref(), seen.slot(), glows, marks);
    prop_items(look, (feed, seen.team()), (glows, marks), stats);
    powers_orbs(seen, (fixed, alpha), (&spawns.vanished[..], now), orbs);
    if show == Show::Orbs {
        orbs.extend(showcase_orbs(look, human, now));
    }
    orb_items(look, orbs, now, glows, stats);
    stats.edit(|stats| {
        (stats.previews, stats.puffs, stats.shatters) = (previews, spawns.puffs, spawns.shatters);
    });
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
    stats: &mut Stats<'_, EffectsStats>,
) {
    let props = &look.props;
    for prop in &feed.props {
        match prop.place {
            PropPlace::Thrown {
                centre,
                velocity_mps,
                resting: false,
            } if velocity_mps.length_squared() > 0.25 => {
                stats.edit(|stats| stats.trails += 1);
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
                stats.edit(|stats| stats.shadows += 1);
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
    let (rind, curtain, barrier) = (&panels.rind, &panels.curtain, &look.barrier);
    let frame = EffectsFrame {
        clock: [(now % 3600.0) as f32, screen.damage_edge, 0.0, 0.0],
        flash: rgb(screen.flash, math::clamp(flash, 0.0, 1.0)),
        damage: rgb(screen.damage, damage),
        sun: [0.0; 4],
        sun_color: [0.0; 4],
        sky_color: [0.0; 4],
        field: [0.0; 4],
        field_shape: [0.0; 4],
        rind_skin: rgb(rind.skin, rind.stripe_m),
        rind_stripe: rgb(rind.stripe, rind.pith_m),
        rind_pith: rgb(rind.pith, rind.seed_m),
        rind_flesh: rgb(rind.flesh, rind.skin_m),
        rind_seed: rgb(rind.seed, rind.round),
        curtain_peel: rgb(curtain.peel, curtain.streak_m),
        curtain_pulp: rgb(curtain.pulp, curtain.glow),
        curtain_foam: rgb(curtain.foam, curtain.fleck_m),
        curtain_shape: [
            curtain.fall_mps,
            curtain.splash_m,
            curtain.near_m,
            curtain.hop_m,
        ],
        panel: [
            panels.trim_m,
            panels.rim_glow,
            panels.edge_power,
            panels.edge_glow,
        ],
        zones: [look.zones.dashes, look.zones.dash_speed, 0.0, 0.0],
        barrier: [
            barrier.eye_m,
            barrier.eye_least,
            barrier.drift_mps,
            barrier.line_px,
        ],
    };
    let stats = ScreenStats {
        flash: frame.flash[3],
        damage,
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

fn blob_marks(marks: &GroundMarks, out: &mut Sorted, stats: &mut Stats<'_, EffectsStats>) {
    for blob in &marks.blobs {
        if blob.strength <= 0.0 {
            continue;
        }
        stats.edit(|stats| stats.blobs += 1);
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
    let (rect, [height, scale]) = field.uniform();
    frame.field = rect;
    frame.field_shape = [height, scale, 1.0, 0.0];
    let mut map = field.map;
    map.sampler = Some(lighting.filtering.light_field.applied(map.sampler));
    Some(map)
}

fn unread_frame(
    drawing: &mut EffectsDrawing,
    stats: &mut Stats<EffectsStats>,
    refusal: Option<&str>,
) {
    *drawing = EffectsDrawing::default();
    stats.show(|| EffectsStats {
        refusal: Some(refusal.unwrap_or("numbers.json not read yet").to_owned()),
        ..EffectsStats::default()
    });
}

fn covered_frame(drawing: &mut EffectsDrawing, stats: &mut Stats<EffectsStats>) {
    *drawing = EffectsDrawing::default();
    stats.show(|| EffectsStats {
        covered: true,
        ..EffectsStats::default()
    });
}

fn lobby_frame(drawing: &mut EffectsDrawing, stats: &mut Stats<EffectsStats>) {
    *drawing = EffectsDrawing::default();
    stats.show(|| EffectsStats {
        lobby: true,
        ..EffectsStats::default()
    });
}

fn put(items: &mut Vec<Item>, list: impl IntoIterator<Item = Item>) -> std::ops::Range<u32> {
    let start = items.len() as u32;
    items.extend(list);
    start..items.len() as u32
}

fn glow_runs(glows: &[Item], runs: &mut Vec<(std::ops::Range<u32>, u32)>) {
    runs.clear();
    for (at, item) in (0_u32..).zip(glows) {
        let vertices = QUAD_VERTICES * (item.rows[3][2] as u32).max(1);
        match runs.last_mut() {
            Some((range, most)) if (*most > QUAD_VERTICES) == (vertices > QUAD_VERTICES) => {
                range.end = at + 1;
                *most = (*most).max(vertices);
            }
            _ => runs.push((at..at + 1, vertices)),
        }
    }
}

#[derive(SystemParam)]
pub(super) struct Seen<'w, 's> {
    lighting: Option<Res<'w, Lighting>>,
    lamp: Option<Res<'w, DeviceLamp>>,
    barrier: Option<Res<'w, Barrier>>,
    camera: Option<Res<'w, TestCamera>>,
    skipped: Query<'w, 's, (), With<ViewSkip>>,
    lobby: Option<Res<'w, LobbyView>>,
    watched: Option<Res<'w, Watched>>,
    powers: Option<Res<'w, Powers>>,
    flow: Option<Res<'w, Flow>>,
    poses: Poses<'w>,
    eyes: Query<'w, 's, &'static Eye>,
    actors: Query<'w, 's, (&'static Actor, &'static Body)>,
    humans: Query<'w, 's, &'static Actor, With<Human>>,
}

impl Seen<'_, '_> {
    fn eye(&self, human: Option<Entity>) -> Option<Vec3> {
        let eye = self
            .watched
            .as_deref()
            .and_then(|watched| watched.entity)
            .and_then(|viewer| self.eyes.get(viewer).ok())
            .or_else(|| self.eyes.get(human?).ok())?;
        self.poses
            .drawn_pose(eye.node)
            .map(|pose| pose.w_axis.truncate())
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
        !self.skipped.is_empty()
    }

    fn lobby(&self) -> bool {
        self.lobby
            .as_deref()
            .is_some_and(|lobby| lobby.camera.is_some())
            && !self.camera.as_deref().is_some_and(|camera| camera.outside)
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
    show: DevSetting<EffectsShow>,
    spawns: Res<Spawns>,
    feed: Res<EffectsFeed>,
    ground: Option<Res<GroundMarks>>,
    round: Option<Res<Round>>,
    humans: Query<(Entity, &Body, &Wish), With<Human>>,
    seen: Seen,
    mut drawing: ResMut<EffectsDrawing>,
    mut lists: Local<Lists>,
    mut stats: Stats<EffectsStats>,
) {
    let Some(parsed) = setup.parsed.as_ref() else {
        return unread_frame(&mut drawing, &mut stats, setup.refusal.as_deref());
    };
    if seen.covered() {
        return covered_frame(&mut drawing, &mut stats);
    }
    if seen.lobby() {
        return lobby_frame(&mut drawing, &mut stats);
    }
    let (look, now) = (&parsed.look, time.elapsed);
    let show = show.get().map_or(Show::Off, |show| show.show);
    let human = humans.iter().next();
    let eye = seen.eye(human.map(|(entity, ..)| entity));
    let human = human.map(|(_, body, wish)| (body, wish));
    stats.show(EffectsStats::default);
    let lists = &mut *lists;
    let mut items = std::mem::take(&mut drawing.items);
    items.clear();
    glow_items(look, &spawns, now, &mut items, &mut stats);
    let sources = (&seen, &*feed, &*spawns);
    let timing = (&*fixed, time.alpha, now);
    power_glows(
        look,
        sources,
        timing,
        (show, human),
        (&mut items, &mut lists.sorted.marks, &mut lists.orbs),
        &mut stats,
    );
    let timing = (parsed.fuse_seconds, fixed.delta_seconds, time.alpha);
    let place = (round.as_deref(), show, human);
    let device_solid = device_part(look, &seen, place, (timing, now), (&mut items, &mut stats));
    let mut glows = std::mem::take(&mut drawing.glows);
    glow_runs(&items, &mut glows);
    let team = seen.team();
    let showcase = if show == Show::Off {
        Vec::new()
    } else {
        showcase_shown(look, show, human, now)
    };
    lists.shown.clear();
    lists
        .shown
        .extend(feed.shown.iter().chain(&showcase).map(|thing| Shown {
            team: relative(thing.team, team),
            ..*thing
        }));
    if let Some(ground) = ground.as_deref() {
        blob_marks(ground, &mut lists.sorted, &mut stats);
    }
    let viewer = (team, show, &*models);
    let timing = (eye, fixed.delta_seconds, now);
    let things = (&*feed, &*spawns, viewer);
    shown_things(look, things, timing, (&mut *lists, &mut stats));
    let sorted = &mut lists.sorted;
    let solid = put(&mut items, sorted.solids.drain(..));
    let slab = put(&mut items, sorted.slabs.drain(..));
    let sheet = put(&mut items, sorted.sheets.drain(..));
    let device = put(&mut items, device_solid);
    let smoke = put(&mut items, sorted.smokes.drain(..));
    let (mut frame, mut screen) = screen_frame(look, &feed, show, &spawns, now);
    let field = lit_frame(&mut frame, seen.lighting.as_deref());
    let limit = if show == Show::NearSight {
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
    let marks = put(&mut items, sorted.marks.drain(..));
    lists.raised.clear();
    if show == Show::Barrier || show == Show::All {
        lists.raised.extend(showcase_barrier(look, human, now));
    }
    let sources = (&seen, round.as_deref());
    let barrier = barrier_part(
        look,
        sources,
        (&fixed, time.alpha),
        (&mut lists.raised, &mut lists.bodies, &mut items),
        &mut stats,
    );
    let screened = frame.flash[3] > 0.0 || frame.damage[3] > 0.0;
    let pools = &look.pools;
    stats.edit(|next| {
        let ranges = [&solid, &slab, &sheet, &device, &smoke, &marks, &barrier];
        next.draws = ranges.iter().filter(|range| !range.is_empty()).count() as u32
            + glows.len() as u32
            + u32::from(screened);
        next.instances = items.len() as u32;
        next.bytes = (items.len() * size_of::<Item>() + size_of::<EffectsFrame>()) as u64;
        next.device = !device.is_empty();
        next.screen = screen;
    });
    *drawing = EffectsDrawing {
        items,
        pooled: pools.tracers + pools.flashes + pools.sprays + pools.decals + pools.rings,
        glows,
        solid,
        slab,
        sheet,
        device,
        smoke,
        marks,
        barrier,
        frame,
        screen: screened,
        field,
    };
}
