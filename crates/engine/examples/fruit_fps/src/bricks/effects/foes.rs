use std::collections::BTreeMap;

use engine::prelude::*;

use super::gather::Setup;
use crate::play::actors::{Actor, Human};
use crate::play::effects::{FigureMarks, Mark};
use crate::play::powers::Powers;

pub(super) fn mark_figures(
    setup: Res<Setup>,
    powers: Option<Res<Powers>>,
    humans: Query<&Actor, With<Human>>,
    actors: Query<&Actor>,
    mut marks: ResMut<FigureMarks>,
) {
    let enemy = setup.parsed.as_ref().and_then(|parsed| parsed.look.enemy);
    let human = humans.iter().next().map(|actor| actor.team);
    let mut by_actor = BTreeMap::new();
    for actor in &actors {
        let foe = enemy.filter(|_| human.is_some_and(|team| team != actor.team));
        let tint = foe.map_or(Mark::PLAIN.tint, |look| {
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
            Some(_) | None => Mark::PLAIN.emission,
        };
        by_actor.insert((actor.slot, actor.team), Mark { tint, emission });
    }
    marks.set_if_neq(FigureMarks { by_actor });
}
