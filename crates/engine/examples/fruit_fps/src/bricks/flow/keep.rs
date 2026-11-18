use engine::prelude::*;
use engine_core::{Background, Kept, Landed};
use serde::{Deserialize, Serialize};

use crate::bricks::screen::Options;
use crate::play::flow::Flow;
use crate::play::sound::Volume;
use crate::play::view::Controls;

const NAME: &str = "fruit_fps_settings";
const FORMAT: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Blob {
    format: u32,
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
        format: FORMAT,
        controls: world.resource::<Controls>().clone(),
        volume: world.resource::<Volume>().clone(),
        options: world.resource::<Options>().clone(),
        character: flow.character.clone(),
        level: flow.level.clone(),
    }
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
        Landed::Ready(Some(bytes)) => Ok(serde_json::from_slice::<Blob>(bytes)
            .ok()
            .filter(|blob| blob.format == FORMAT)),
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
            if !book.restored {
                return;
            }
            let now = current(world);
            if book.last.as_ref() == Some(&now) {
                return;
            }
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
