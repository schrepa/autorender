use glam::{Mat3, Vec3};

use super::effects::{Actor, Aim, CueKind, Effects, Event, World, WorldHit};
use super::numbers::{AbilityId, AbilitySpec, Action, Delivery, Onto};
use super::queries::{DrawKind, DrawShape};

const LOOK_AHEAD_TICKS: u64 = 300;

#[derive(Clone, Debug, PartialEq)]
pub struct Preview {
    pub ability: AbilityId,
    pub team: usize,
    pub at: Vec3,
    pub shape: DrawShape,
    pub path: Vec<Vec3>,
    pub valid: bool,
}

struct Sandbox<'a, W: World>(&'a W);

impl<W: World> World for Sandbox<'_, W> {
    fn ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<WorldHit> {
        self.0.ray(origin, direction, max)
    }

    fn sphere_sweep(&self, centre: Vec3, radius: f32, motion: Vec3) -> Option<WorldHit> {
        self.0.sphere_sweep(centre, radius, motion)
    }

    fn add_box(
        &mut self,
        _id: u32,
        _ability: AbilityId,
        _centre: Vec3,
        _half_extents: Vec3,
        _axes: Mat3,
    ) {
    }

    fn remove_box(&mut self, _id: u32) {}
}

pub fn needs_aim(spec: &AbilitySpec) -> bool {
    match spec.delivery {
        Delivery::Caster => matches!(spec.action, Action::Wall(_)),
        Delivery::Throw(_) | Delivery::Place(_) => true,
    }
}

fn landing(cue: CueKind) -> bool {
    match cue {
        CueKind::SmokeBurst
        | CueKind::ZoneStart
        | CueKind::StrikeWarning
        | CueKind::Explosion
        | CueKind::FlashBurst
        | CueKind::Pulse
        | CueKind::WallUp
        | CueKind::TrapPlaced
        | CueKind::SensorPlaced
        | CueKind::StrikeImpact
        | CueKind::Glare
        | CueKind::HealStart => true,
        CueKind::Cast
        | CueKind::Throw
        | CueKind::Bounce
        | CueKind::WallDown
        | CueKind::TrapArmed
        | CueKind::TrapTriggered
        | CueKind::Destroyed
        | CueKind::Dash
        | CueKind::TransformStart
        | CueKind::TransformEnd => false,
    }
}

fn placed(world: &impl World, spec: &AbilitySpec, aim: Aim) -> bool {
    let Delivery::Place(place) = spec.delivery else {
        return true;
    };
    let hit = world.ray(aim.origin, aim.direction, place.range_m);
    match place.onto {
        Onto::Surface => hit.is_some(),
        Onto::Ground => {
            let point = hit.map_or(aim.origin + aim.direction * place.range_m, |hit| {
                aim.origin + aim.direction * hit.distance + hit.normal * place.lift_m
            });
            world.ray(point, Vec3::NEG_Y, place.range_m).is_some()
        }
    }
}

fn footprint(effects: &Effects, cue: CueKind, at: Vec3, size: f32, throw_radius: f32) -> DrawShape {
    let built = effects
        .drawables()
        .into_iter()
        .find(|drawable| match drawable.kind {
            DrawKind::Wall | DrawKind::Curtain | DrawKind::Trap | DrawKind::Wire => true,
            DrawKind::Projectile
            | DrawKind::Smoke
            | DrawKind::Zone
            | DrawKind::Sensor
            | DrawKind::StrikeMark => false,
        });
    if let Some(drawable) = built {
        return drawable.shape;
    }
    let radius = if cue == CueKind::FlashBurst {
        throw_radius
    } else {
        size
    };
    DrawShape::Sphere { centre: at, radius }
}

pub fn preview(
    world: &impl World,
    caster: &Actor,
    spec: &AbilitySpec,
    aim: Aim,
) -> Option<Preview> {
    if !needs_aim(spec) {
        return None;
    }
    let throw_radius = match spec.delivery {
        Delivery::Throw(throw) => throw.radius_m,
        Delivery::Caster | Delivery::Place(_) => 0.0,
    };
    let mut sandbox = Sandbox(world);
    let mut effects = Effects::new(0);
    effects.cast(caster, spec, aim, 0);
    let mut path = vec![aim.origin];
    for tick in 1..=LOOK_AHEAD_TICKS {
        let events = effects.tick(&mut sandbox, &[], tick);
        for drawable in effects.drawables() {
            if let (DrawKind::Projectile, DrawShape::Sphere { centre, .. }) =
                (drawable.kind, drawable.shape)
            {
                path.push(centre);
            }
        }
        let landed = events.iter().find_map(|event| {
            if let Event::Cue { cue, at, size, .. } = event
                && landing(*cue)
            {
                Some((*cue, *at, *size))
            } else {
                None
            }
        });
        if let Some((cue, at, size)) = landed {
            path.push(at);
            return Some(Preview {
                ability: spec.id,
                team: caster.team,
                at,
                shape: footprint(&effects, cue, at, size, throw_radius),
                path,
                valid: placed(world, spec, aim),
            });
        }
    }
    let at = path.last().copied().unwrap_or(aim.origin);
    Some(Preview {
        ability: spec.id,
        team: caster.team,
        at,
        shape: DrawShape::Sphere {
            centre: at,
            radius: throw_radius,
        },
        path,
        valid: false,
    })
}
