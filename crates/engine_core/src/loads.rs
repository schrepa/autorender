use std::collections::BTreeMap;
use std::sync::{Arc, Weak};

use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_place::OperationScene;
use serde::Serialize;

use crate::{EntityIds, Event, Feed, FeedSettings, Level, LevelView, Raises, Readiness, Refusal};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Loaded {
    Scene,
    Level,
}

#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
pub struct Load {
    pub what: Loaded,
    pub name: String,
    pub failed: Option<Refusal>,
}

impl Event for Load {
    const NAME: &'static str = "load";
    type Row = Self;

    fn row(&self, _: &dyn EntityIds) -> Self {
        self.clone()
    }
}

#[derive(Resource)]
pub struct Loads {
    raised: Feed<Load>,
    scene: Option<Weak<BTreeMap<String, Entity>>>,
    waiting: Option<String>,
    level: LevelView,
}

impl Default for Loads {
    fn default() -> Self {
        Self {
            raised: Feed::new(FeedSettings::default().events as usize),
            scene: None,
            waiting: None,
            level: LevelView::Idle,
        }
    }
}

impl Raises<Load> for Loads {
    fn raised(&self) -> &Feed<Load> {
        &self.raised
    }
}

fn size(world: &World) -> usize {
    world
        .get_resource::<FeedSettings>()
        .map_or(FeedSettings::default().events, |feeds| feeds.events) as usize
}

pub(crate) fn settle_level(world: &mut World) {
    let Some(view) = world.get_resource::<Level>().map(Level::view) else {
        return;
    };
    let size = size(world);
    let Some(mut loads) = world.get_resource_mut::<Loads>() else {
        return;
    };
    loads.raised.resize(size);
    if loads.level == view {
        return;
    }
    let ended = match &view {
        LevelView::Ready { path } => Some((path.clone(), None)),
        LevelView::Failed { path, error } => Some((path.clone(), Some(error.clone()))),
        LevelView::Idle | LevelView::Asked { .. } | LevelView::Missing { .. } => None,
    };
    if let Some((name, failed)) = ended {
        loads.raised.push(Load {
            what: Loaded::Level,
            name,
            failed,
        });
    }
    loads.level = view;
}

pub(crate) fn settle_scene(world: &mut World) {
    let readiness = world.get_resource::<Readiness>();
    let settled = readiness.is_some_and(Readiness::settled);
    let failed = readiness
        .and_then(|readiness| readiness.failed.first())
        .map(|(_, error)| Refusal::of(error));
    let current = world
        .get_resource::<OperationScene>()
        .map(|scene| (Arc::downgrade(&scene.entities), scene.id.clone()));
    let size = size(world);
    let Some(mut loads) = world.get_resource_mut::<Loads>() else {
        return;
    };
    loads.raised.resize(size);
    if let Some((held, id)) = current {
        let fresh = loads
            .scene
            .as_ref()
            .is_none_or(|known| !Weak::ptr_eq(known, &held));
        if fresh {
            loads.scene = Some(held);
            loads.waiting = Some(id);
        }
    }
    if !settled {
        return;
    }
    if let Some(name) = loads.waiting.take() {
        loads.raised.push(Load {
            what: Loaded::Scene,
            name,
            failed,
        });
    }
}
