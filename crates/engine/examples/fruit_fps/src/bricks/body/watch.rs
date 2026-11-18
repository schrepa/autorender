use engine::prelude::*;

use crate::play::actors::{Actor, Alive, Body, Human, Wish};
use crate::play::numbers::Numbers;
use crate::play::view::{CorpseView, Watch, Watched};

type Seen<'a> = (Entity, &'a Actor, Has<Alive>, Has<Human>);

type Own<'a> = (Entity, &'a Actor, &'a Body, &'a Wish, Has<Alive>);

fn teammate_after(listed: &[(Entity, u32, bool)], after: Option<u32>) -> Option<(Entity, u32)> {
    let living = || listed.iter().filter(|(_, _, alive)| *alive);
    after
        .and_then(|slot| living().find(|(_, other, _)| *other > slot))
        .or_else(|| living().next())
        .map(|&(entity, slot, _)| (entity, slot))
}

fn teammate(entity: Entity, slot: u32) -> Watched {
    Watched {
        slot: Some(slot),
        entity: Some(entity),
        watch: Watch::Teammate,
        corpse: None,
    }
}

pub(super) fn watch(
    time: Res<FixedTime>,
    numbers: Res<Numbers>,
    actors: Query<Seen<'_>>,
    humans: Query<Own<'_>, With<Human>>,
    mut watched: ResMut<Watched>,
) {
    let Some(numbers) = numbers.get() else {
        return;
    };
    let Some((human, own, body, wish, alive)) = humans.iter().next() else {
        return;
    };
    if alive {
        watched.set_if_neq(Watched {
            slot: Some(own.slot),
            entity: Some(human),
            watch: Watch::Own,
            corpse: None,
        });
        return;
    }
    let mut listed: Vec<(Entity, u32, bool)> = actors
        .iter()
        .filter(|(_, actor, _, is_human)| !is_human && actor.team == own.team)
        .map(|(entity, actor, alive, _)| (entity, actor.slot, alive))
        .collect();
    listed.sort_by_key(|&(_, slot, _)| slot);
    let corpse_ticks = (numbers.spectate.corpse_seconds / time.delta_seconds).round() as u64;
    let next = match watched.watch {
        Watch::Own => Watched {
            slot: None,
            entity: None,
            watch: Watch::Corpse,
            corpse: Some(CorpseView {
                eye: body.eye,
                yaw_radians: wish.yaw_radians,
                tick: time.tick,
            }),
        },
        Watch::Corpse => {
            let over = watched
                .corpse
                .is_none_or(|corpse| time.tick >= corpse.tick + corpse_ticks);
            match teammate_after(&listed, None).filter(|_| over) {
                Some((entity, slot)) => teammate(entity, slot),
                None => *watched,
            }
        }
        Watch::Teammate => {
            let still = listed
                .iter()
                .any(|&(entity, _, alive)| alive && Some(entity) == watched.entity);
            if still && !wish.fire_pressed {
                *watched
            } else {
                match teammate_after(&listed, watched.slot) {
                    Some((entity, slot)) => teammate(entity, slot),
                    None => Watched {
                        slot: None,
                        entity: None,
                        watch: Watch::Corpse,
                        corpse: watched.corpse,
                    },
                }
            }
        }
    };
    watched.set_if_neq(next);
}
