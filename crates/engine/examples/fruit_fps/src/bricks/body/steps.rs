use crate::play::actors::{Body, Gait, Wish};
use crate::play::events::body::{BodyEvents, Footstep, Jump, Landing, Stride};
use crate::play::numbers::PlayerNumbers;
use crate::solids::Moved;

pub(super) struct Step<'a> {
    pub slot: u32,
    pub before: &'a Body,
    pub next: &'a Body,
    pub moved: &'a Moved,
    pub wish: &'a Wish,
}

fn stride_of(next: &Body, wish: &Wish) -> Stride {
    if next.crouched {
        Stride::CrouchWalk
    } else {
        match wish.gait {
            Gait::Walk => Stride::Walk,
            Gait::Run => Stride::Run,
        }
    }
}

fn length_of(player: &PlayerNumbers, stride: Stride) -> f32 {
    match stride {
        Stride::Run => player.stride_run_m,
        Stride::Walk => player.stride_walk_m,
        Stride::CrouchWalk => player.stride_crouch_m,
    }
}

pub(super) fn record(
    player: &PlayerNumbers,
    step: &Step<'_>,
    travel: &mut f32,
    events: &mut BodyEvents,
) {
    let Step {
        slot,
        before,
        next,
        moved,
        wish,
    } = *step;
    if before.grounded && !next.grounded && next.velocity.y > 0.0 {
        events.jumps.push(Jump {
            slot,
            position: before.feet,
            surface: before.ground.map(|ground| ground.surface),
            kind: before.ground.map(|ground| ground.kind),
        });
    }
    if let Some(landed) = moved.landed {
        events.landings.push(Landing {
            slot,
            position: next.feet,
            surface: landed.surface,
            kind: landed.kind,
            fall_speed: landed.speed,
            hard: landed.speed >= player.hard_landing_mps,
        });
    }
    let across = (next.feet - before.feet).with_y(0.0).length();
    let Some(ground) = next.ground.filter(|_| next.grounded && across > 0.0) else {
        *travel = 0.0;
        return;
    };
    let stride = stride_of(next, wish);
    let length = length_of(player, stride);
    *travel += across;
    if *travel >= length {
        *travel -= length;
        events.footsteps.push(Footstep {
            slot,
            position: next.feet,
            surface: ground.surface,
            kind: ground.kind,
            stride,
            loud: stride == Stride::Run || !player.walk_is_silent,
        });
    }
}
