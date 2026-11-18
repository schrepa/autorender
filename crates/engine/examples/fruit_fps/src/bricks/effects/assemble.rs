use engine::math;
use engine::prelude::*;
use glam::Mat3;

use super::gather::{Setup, Spawns, SprayKind, device_spot, view_direction};
use super::numbers::{EffectsNumbers, Spray};
use super::{
    Device, EffectsDrawing, EffectsFrame, EffectsShow, EffectsStats, Item, ScreenStats, Show,
};
use crate::play::actors::{Body, Human, Wish};
use crate::play::effects::{EffectsFeed, Form, GroundMarks, Look, Shown};
use bevy_ecs::system::SystemParam;

use super::previews::preview_items;
use crate::play::actors::Actor;
use crate::play::flow::{Flow, Stage};
use crate::play::light::Lighting;
use crate::play::powers::Powers;
use crate::play::round::Round;
use crate::play::view::Watched;
use crate::rules::Team;

const TRACER: f32 = 0.0;
const FLASH: f32 = 1.0;
const SPRAY: f32 = 2.0;

const PANEL: f32 = 0.0;
const GLOWING: f32 = 1.0;
const BODY: f32 = 2.0;

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
        SprayKind::Impact(surface) => {
            let spray = look.impacts.of(surface);
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
    shown: &[Shown],
    eye: Option<Vec3>,
    out: &mut Sorted,
    stats: &mut EffectsStats,
) {
    let traps = &look.traps;
    for (index, thing) in shown.iter().enumerate() {
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
                out.solids
                    .push(solid_item(&glowing(traps.wire_width_m), look));
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

fn device_items(
    look: &EffectsNumbers,
    spot: Option<(Vec3, Option<u32>)>,
    stand_in: bool,
    fuse: (f32, f32),
    now: f64,
) -> Vec<Item> {
    let Some((at, left)) = spot else {
        return Vec::new();
    };
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
    let (fuse_seconds, tick_seconds) = fuse;
    let left_seconds = match left {
        Some(ticks) => ticks as f32 * tick_seconds,
        None => fuse_seconds - (now % f64::from(fuse_seconds)) as f32,
    };
    let share = math::clamp(left_seconds / fuse_seconds, 0.0, 1.0);
    let period =
        device.blink_fast_seconds + (device.blink_slow_seconds - device.blink_fast_seconds) * share;
    let lit = (now / f64::from(period)).fract() < f64::from(device.lit_share);
    let light = Solid {
        shape: Boxed {
            centre: at + Vec3::Y * (height + device.light_size_m * 0.5),
            half: Vec3::splat(device.light_size_m * 0.5),
            axes: Mat3::IDENTITY,
        },
        kind: GLOWING,
        fill: device.light,
        glow: if lit { 1.0 } else { 0.0 },
        ..base
    };
    items.push(solid_item(&light, look));
    items
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

fn span(items: &[Item], start: usize) -> std::ops::Range<u32> {
    start as u32..items.len() as u32
}

#[derive(SystemParam)]
pub(super) struct Seen<'w, 's> {
    lighting: Option<Res<'w, Lighting>>,
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
    let glow = span(&items, 0);
    let mut shown = feed.shown.clone();
    if show.show != Show::Off {
        shown.extend(showcase_shown(look, show.show, human, now));
    }
    if let Some(ground) = ground.as_deref() {
        blob_marks(ground, &mut sorted, &mut next);
    }
    sort_shown(look, &shown, eye, &mut sorted, &mut next);
    pooled_marks(look, &spawns, now, &mut sorted, &mut next);
    let start = items.len();
    items.extend(sorted.solids);
    let solid = span(&items, start);
    let stand_in = devices.iter().any(|device| device.stand_in);
    let spot = device_spot(round.as_deref(), show.show, human, look);
    let start = items.len();
    items.extend(device_items(
        look,
        spot,
        stand_in,
        (parsed.fuse_seconds, fixed.delta_seconds),
        now,
    ));
    let device = span(&items, start);
    let start = items.len();
    items.extend(sorted.smokes);
    let smoke = span(&items, start);
    let (mut frame, mut screen) = screen_frame(look, &feed, show.show, &spawns, now, sorted.cover);
    if let Some(lighting) = seen.lighting.as_deref().filter(|lighting| lighting.lit) {
        frame.sun = row(lighting.to_sun, 1.0);
        frame.sun_color = rgb(lighting.sun, 0.0);
        frame.sky_color = rgb(lighting.sky, 0.0);
        if let Some(field) = lighting.field {
            let [min_x, min_z, max_x, max_z] = field.rect;
            frame.field = [min_x, min_z, 1.0 / (max_x - min_x), 1.0 / (max_z - min_z)];
            frame.field_shape = [field.height_m, field.scale, 1.0, 0.0];
        }
    }
    let field = seen
        .lighting
        .as_deref()
        .filter(|lighting| lighting.lit)
        .and_then(|lighting| lighting.field)
        .map(|field| field.map);
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
    let screened = frame.flash[3] > 0.0 || frame.damage[3] > 0.0 || frame.cover[3] > 0.0;
    let ranges = [&glow, &solid, &device, &smoke, &marks];
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
        glow_vertices: QUAD_VERTICES * look.most_sprites(),
        frame,
        screen: screened,
        field,
    };
    stats.set_if_neq(next);
}
