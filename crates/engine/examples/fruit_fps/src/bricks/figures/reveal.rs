use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::{Drawn, Mobility, Spawner, Transform};

use super::FiguresBrick;
use crate::play::figures::{FIGURE, Figure, REVEALED};

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct RevealOf(Entity);

type Shown<'w, 's> = Query<'w, 's, (Entity, &'static Drawn), With<Figure>>;

type Outlines<'w, 's> = Query<'w, 's, (Entity, &'static RevealOf, &'static Drawn)>;

fn riding() -> Node {
    Node {
        visible: true,
        mobility: Some(Mobility::Moving),
        ..Node::default()
    }
}

fn outlined(drawn: &Drawn) -> Option<Drawn> {
    (drawn.kind == FIGURE && drawn.values.emission != [0.0; 3]).then(|| Drawn {
        kind: REVEALED,
        ..drawn.clone()
    })
}

pub(super) fn outline(figures: Shown, outlines: Outlines, mut spawner: Spawner<FiguresBrick>) {
    let mut held: BTreeMap<Entity, (Entity, &Drawn)> = outlines
        .iter()
        .map(|(entity, of, drawn)| (of.0, (entity, drawn)))
        .collect();
    for (entity, drawn) in &figures {
        let found = held.remove(&entity);
        match (outlined(drawn), found) {
            (Some(wanted), None) => {
                let outline =
                    spawner.spawn((Transform::default(), riding(), wanted, RevealOf(entity)));
                spawner.set_parent(outline, Some(entity));
            }
            (Some(wanted), Some((outline, was))) => {
                if *was != wanted {
                    spawner.insert(outline, wanted);
                }
            }
            (None, Some((outline, _))) => spawner.despawn(outline),
            (None, None) => {}
        }
    }
    for (outline, _) in held.into_values() {
        spawner.despawn(outline);
    }
}
