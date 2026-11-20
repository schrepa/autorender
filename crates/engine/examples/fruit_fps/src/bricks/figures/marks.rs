use engine::prelude::*;
use engine_core::{Drawn, InstanceValues, Spawner};

use super::FiguresBrick;
use crate::play::actors::Actor;
use crate::play::effects::{FigureMarks, Mark};
use crate::play::skinned::Figure;

type Figures<'w, 's> = Query<'w, 's, (Entity, &'static Actor, &'static Drawn), With<Figure>>;

pub(super) fn fold(
    marks: Option<Res<FigureMarks>>,
    figures: Figures,
    mut spawner: Spawner<FiguresBrick>,
) {
    for (entity, actor, drawn) in &figures {
        let mark = marks
            .as_deref()
            .and_then(|marks| marks.by_actor.get(&(actor.slot, actor.team)))
            .copied()
            .unwrap_or(Mark::PLAIN);
        if drawn.values.tint != mark.tint || drawn.values.emission != mark.emission {
            spawner.insert(
                entity,
                Drawn {
                    values: InstanceValues {
                        tint: mark.tint,
                        emission: mark.emission,
                        ..drawn.values.clone()
                    },
                    ..drawn.clone()
                },
            );
        }
    }
}
