use engine::prelude::*;
use engine_core::{Background, Kept, Landed};
use serde::{Deserialize, Serialize};

use crate::play::flow::Flow;
use crate::play::screen::Options;
use crate::play::sound::Volume;
use crate::play::view::Controls;

const NAME: &str = "fruit_fps_settings";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Blob {
    controls: Controls,
    volume: Volume,
    options: Options,
    character: String,
    level: String,
}

#[derive(Resource, Default)]
pub(super) struct KeepBook {
    reading: Option<Background<Option<Vec<u8>>, u64>>,
    writing: Option<Background<(), u64>>,
    pub restored: bool,
    last: Option<Blob>,
    pub writes: u32,
    pub problem: Option<String>,
}

fn current(world: &World) -> Blob {
    let flow = world.resource::<Flow>();
    Blob {
        controls: world.resource::<Controls>().clone(),
        volume: world.resource::<Volume>().clone(),
        options: world.resource::<Options>().clone(),
        character: flow.character.clone(),
        level: flow.level.clone(),
    }
}

fn same(world: &World, last: &Blob) -> bool {
    let flow = world.resource::<Flow>();
    last.controls == *world.resource::<Controls>()
        && last.volume == *world.resource::<Volume>()
        && last.options == *world.resource::<Options>()
        && last.character == flow.character
        && last.level == flow.level
}

fn restore(world: &mut World, blob: Blob) -> Result<(), String> {
    world
        .section::<Controls>()
        .set(|held| *held = blob.controls)
        .map_err(|error| error.to_string())?;
    world
        .section::<Volume>()
        .set(|held| *held = blob.volume)
        .map_err(|error| error.to_string())?;
    world
        .section::<Options>()
        .set(|held| *held = blob.options)
        .map_err(|error| error.to_string())?;
    let mut flow = world.resource_mut::<Flow>();
    flow.character = blob.character;
    flow.level = blob.level;
    Ok(())
}

fn read_kept(book: &mut KeepBook, kept: &Kept) -> Option<Result<Option<Blob>, String>> {
    let reading = book.reading.as_mut()?;
    let read = match reading.poll(kept) {
        Landed::Waiting => return None,
        Landed::Ready(None) => Ok(None),
        Landed::Ready(Some(bytes)) => serde_json::from_slice::<Blob>(bytes)
            .map(Some)
            .map_err(|error| format!("saved settings {NAME}: {error}")),
        Landed::Failed(error) => Err(error.to_string()),
    };
    book.reading = None;
    Some(read)
}

pub(super) fn keep_settings(world: &mut World) {
    world.resource_scope(|world, mut kept: Mut<Kept>| {
        world.resource_scope(|world, mut book: Mut<KeepBook>| {
            let book = &mut *book;
            if !book.restored && book.reading.is_none() {
                book.reading = Some(kept.kept(NAME));
            }
            if let Some(read) = read_kept(book, &kept) {
                book.restored = true;
                match read {
                    Ok(Some(blob)) => book.problem = restore(world, blob).err(),
                    Ok(None) => {}
                    Err(problem) => book.problem = Some(problem),
                }
                book.last = Some(current(world));
                world.resource_mut::<Flow>().restored = true;
            }
            if let Some(writing) = book.writing.as_mut() {
                match writing.poll(&*kept) {
                    Landed::Waiting => {}
                    Landed::Ready(()) => book.writing = None,
                    Landed::Failed(error) => {
                        book.problem = Some(error.to_string());
                        book.writing = None;
                    }
                }
            }
            if !book.restored || book.last.as_ref().is_some_and(|last| same(world, last)) {
                return;
            }
            let now = current(world);
            match serde_json::to_vec(&now) {
                Ok(bytes) => {
                    book.writing = Some(kept.keep(NAME, bytes));
                    book.writes += 1;
                }
                Err(error) => book.problem = Some(error.to_string()),
            }
            book.last = Some(now);
        });
    });
}
