use engine::prelude::*;

use crate::play::numbers::{Numbers, PlayerNumbers};
use crate::play::round::Round;
use crate::play::world::{Arena, Barrier, BarrierFace, Zone, ZoneBox};
use crate::rules::Side;
use crate::solids::{self, Probe};

fn faces(arena: &Arena, player: &PlayerNumbers) -> Vec<BarrierFace> {
    let probe = Probe {
        column_m: player.barrier_column_m,
        radius_m: player.moving.radius_m,
        height_m: player.moving.height_m,
        step_m: player.moving.step_height_m,
    };
    let sided = |zone: &ZoneBox| match zone.zone {
        Zone::BuyAttack => Some((Side::Attack, *zone)),
        Zone::BuyDefend => Some((Side::Defend, *zone)),
        Zone::SiteA | Zone::SiteB => None,
    };
    (arena.zones().iter())
        .filter_map(sided)
        .flat_map(|(side, zone)| {
            solids::openings(
                arena.solids(),
                (zone.centre, zone.rotation, zone.half),
                probe,
            )
            .into_iter()
            .map(move |opening| BarrierFace {
                side,
                corners: opening.corners,
                normal: opening.normal,
            })
        })
        .collect()
}

pub(super) fn publish(
    time: Res<FixedTime>,
    arena: Res<Arena>,
    round: Option<Res<Round>>,
    numbers: Res<Numbers>,
    mut barrier: ResMut<Barrier>,
) {
    let Some(numbers) = numbers.get() else {
        return;
    };
    if !arena.ready() {
        return;
    }
    if barrier.revision != arena.revision() {
        barrier.faces = faces(&arena, &numbers.player);
        barrier.revision = arena.revision();
    }
    let up = round.as_deref().is_some_and(|round| round.buy_lock);
    if barrier.up != up {
        barrier.dropped_tick = (!up).then_some(time.tick);
        barrier.up = up;
    }
}
