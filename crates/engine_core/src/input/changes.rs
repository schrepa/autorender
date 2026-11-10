use bevy_ecs::prelude::{Resource, World};
use serde::Serialize;

use super::{Buttons, Devices, MouseButton, PadButton, TouchPhase};
use crate::{EntityIds, Event, Feed, FeedSettings, Raises};

#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
#[serde(tag = "device", rename_all = "snake_case")]
pub enum InputChange {
    Key {
        code: String,
        down: bool,
    },
    Mouse {
        button: MouseButton,
        down: bool,
    },
    Pad {
        pad: u32,
        button: PadButton,
        down: bool,
    },
    Pointer {
        position: Option<[f32; 2]>,
        inside: bool,
    },
    Touch {
        id: u32,
        position: [f32; 2],
        phase: TouchPhase,
    },
}

impl Event for InputChange {
    const NAME: &'static str = "input";
    type Row = Self;

    fn row(&self, _: &dyn EntityIds) -> Self {
        self.clone()
    }
}

#[derive(Resource)]
pub struct InputChanges {
    raised: Feed<InputChange>,
}

impl Default for InputChanges {
    fn default() -> Self {
        Self {
            raised: Feed::new(FeedSettings::default().events as usize),
        }
    }
}

impl Raises<InputChange> for InputChanges {
    fn raised(&self) -> &Feed<InputChange> {
        &self.raised
    }
}

fn edges<B: Ord + Clone>(
    buttons: &Buttons<B>,
    change: impl Fn(B, bool) -> InputChange,
) -> Vec<InputChange> {
    buttons
        .pressed
        .iter()
        .map(|button| change(button.clone(), true))
        .chain(
            buttons
                .released
                .iter()
                .map(|button| change(button.clone(), false)),
        )
        .collect()
}

fn changes(before: Option<&Devices>, after: &Devices) -> Vec<InputChange> {
    let mut changed = edges(&after.keyboard, |code, down| InputChange::Key {
        code,
        down,
    });
    changed.extend(edges(&after.mouse.buttons, |button, down| {
        InputChange::Mouse { button, down }
    }));
    for pad in &after.pads {
        let index = pad.index;
        changed.extend(edges(&pad.buttons, |button, down| InputChange::Pad {
            pad: index,
            button,
            down,
        }));
    }
    let moved = before.is_none_or(|before| {
        before.mouse.position_pixels != after.mouse.position_pixels
            || before.mouse.inside != after.mouse.inside
    });
    if moved {
        changed.push(InputChange::Pointer {
            position: after
                .mouse
                .position_pixels
                .map(|position| position.to_array()),
            inside: after.mouse.inside,
        });
    }
    changed.extend(
        after
            .touches
            .iter()
            .filter(|touch| touch.phase != TouchPhase::Moved)
            .map(|touch| InputChange::Touch {
                id: touch.id,
                position: touch.position_pixels.to_array(),
                phase: touch.phase,
            }),
    );
    changed
}

pub(super) fn raise_changes(world: &mut World, devices: &Devices) {
    let changed = changes(world.get_resource::<Devices>(), devices);
    let size = world
        .get_resource::<FeedSettings>()
        .map_or(FeedSettings::default().events, |feeds| feeds.events);
    if let Some(mut raised) = world.get_resource_mut::<InputChanges>() {
        raised.raised.resize(size as usize);
        for change in changed {
            raised.raised.push(change);
        }
    }
}
