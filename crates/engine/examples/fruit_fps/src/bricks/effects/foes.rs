use engine::prelude::*;
use engine_core::{Drawn, InstanceValues};

use super::EffectsBrick;
use super::gather::Setup;
use crate::play::actors::{Actor, Human};
use crate::play::figures::Figure;
use crate::play::powers::Powers;

const PLAIN_TINT: [f32; 4] = [1.0; 4];

type Figures<'w, 's> = Query<'w, 's, (Entity, &'static Actor, &'static Drawn), With<Figure>>;

pub(super) fn mark_figures(
    setup: Res<Setup>,
    powers: Option<Res<Powers>>,
    humans: Query<&Actor, With<Human>>,
    figures: Figures,
    mut spawner: Spawner<EffectsBrick>,
) {
    let enemy = setup.parsed.as_ref().and_then(|parsed| parsed.look.enemy);
    let human = humans.iter().next().map(|actor| actor.team);
    for (entity, actor, drawn) in &figures {
        let foe = enemy.filter(|_| human.is_some_and(|team| team != actor.team));
        let tint = foe.map_or(PLAIN_TINT, |look| {
            [
                look.rim[0],
                look.rim[1],
                look.rim[2],
                1.0 - look.rim_strength,
            ]
        });
        let shown = foe.is_some()
            && human.is_some_and(|team| {
                powers
                    .as_deref()
                    .is_some_and(|powers| powers.revealed(actor.slot, team))
            });
        let emission = match foe {
            Some(look) if shown => look.reveal,
            Some(_) | None => [0.0; 3],
        };
        if drawn.values.tint != tint || drawn.values.emission != emission {
            spawner.insert(
                entity,
                Drawn {
                    values: InstanceValues {
                        tint,
                        emission,
                        ..drawn.values.clone()
                    },
                    ..drawn.clone()
                },
            );
        }
    }
}
