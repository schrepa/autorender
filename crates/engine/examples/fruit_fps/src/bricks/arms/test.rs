use engine::prelude::*;
use engine_core::DevSetting;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{ArmsBrick, Book, Setup, drop_weapon};
use crate::play::actors::{Actor, Alive, Body, Health};
use crate::play::arms::{Arms, Drops};
use crate::play::events::arms::{ArmsEvents, DeathEvent};
use crate::weapons::{BodyPart, GunKind, Vitals, ticks, wound};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Hurt {
    pub slot: u32,
    pub damage: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Given {
    Sidearm,
    Smg,
    Rifle,
    Sniper,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Give {
    pub slot: u32,
    pub weapon: Given,
}

settings! {
    #[dev]
    #[section("arms_test")]
    #[serde(deny_unknown_fields)]
    pub struct ArmsTest {
        pub hurt: Option<Hurt> = None => any,
        pub give: Option<Give> = None => any,
    }
}

fn kind_of(given: Given) -> GunKind {
    match given {
        Given::Sidearm => GunKind::Sidearm,
        Given::Smg => GunKind::Smg,
        Given::Rifle => GunKind::Rifle,
        Given::Sniper => GunKind::Sniper,
    }
}

#[derive(SystemParam)]
pub(super) struct Outputs<'w> {
    events: ResMut<'w, ArmsEvents>,
    book: ResMut<'w, Book>,
    drops: ResMut<'w, Drops>,
}

type Tested<'a> = (
    Entity,
    &'a Actor,
    &'a Body,
    &'a mut Arms,
    &'a mut Health,
    Has<Alive>,
);

pub(super) fn door(
    test: DevSetting<ArmsTest>,
    time: Res<FixedTime>,
    setup: Res<Setup>,
    mut actors: Query<Tested<'_>>,
    mut out: Outputs,
    mut spawner: Spawner<ArmsBrick>,
) {
    let Some(test) = test.changed() else {
        return;
    };
    let Some(numbers) = setup.numbers.as_ref() else {
        return;
    };
    let tps = time.rate;
    for (_, actor, _, mut arms, _, alive) in &mut actors {
        if let Some(give) = test.give
            && give.slot == actor.slot
            && alive
        {
            arms.loadout
                .give(&numbers.weapons, kind_of(give.weapon), tps);
        }
    }
    let Some(hurt) = test.hurt else {
        return;
    };
    let Some((entity, _, body, mut arms, mut health, _)) = actors
        .iter_mut()
        .find(|(_, actor, .., alive)| actor.slot == hurt.slot && *alive)
    else {
        return;
    };
    let before = Vitals {
        health: health.health,
        armor: health.armour,
    };
    let after = wound(before, hurt.damage, BodyPart::Body, 0.0);
    health.health = after.after.health;
    if !after.killed {
        return;
    }
    let window = u64::from(ticks(numbers.combat.assist_seconds, tps));
    let Outputs {
        events,
        book,
        drops,
    } = &mut out;
    events.deaths.push(DeathEvent {
        victim: hurt.slot,
        killer: None,
        assistants: book.ledger.assistants(hurt.slot, None, time.tick, window),
        headshot: false,
        weapon: None,
        ability: None,
    });
    book.ledger.forget_victim(hurt.slot);
    drop_weapon(&mut arms.loadout, body.feet, drops);
    spawner.remove::<Alive>(entity);
}
