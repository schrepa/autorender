use std::sync::{Arc, Mutex, PoisonError};

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;

use crate::{
    Background, Delivers, Error, ErrorCode, Expected, Interest, MAX_ID_BYTES, Requests, Ticket,
};

const FILE_PREFIX: &str = "kept-";
const PRESENT: u8 = 1;

type Done = Result<Vec<u8>, String>;
type Landing = Arc<Mutex<Vec<(u64, Ticket, String, Done)>>>;

enum Order {
    Keep(Vec<u8>),
    Read,
}

struct Sent {
    place: String,
    name: String,
    order: Order,
    serial: u64,
    ticket: Ticket,
    landing: Landing,
}

#[derive(Resource, Default)]
pub struct Kept {
    requests: Requests<u64, Arc<[u8]>>,
    place: Option<String>,
    serial: u64,
    sent: u64,
    landed: u64,
    landing: Landing,
    mover: Mover,
}

impl Kept {
    pub fn keep_in(&mut self, place: impl Into<String>) {
        self.place = Some(place.into());
    }

    pub fn place(&self) -> Option<&str> {
        self.place.as_deref()
    }

    pub fn keep(&mut self, name: &str, bytes: Vec<u8>) -> Background<(), u64> {
        self.ask(name, Order::Keep(bytes), |_| Ok(()))
    }

    pub fn kept(&mut self, name: &str) -> Background<Option<Vec<u8>>, u64> {
        self.ask(name, Order::Read, |landed| {
            Ok(landed
                .split_first()
                .filter(|(present, _)| **present == PRESENT)
                .map(|(_, bytes)| bytes.to_vec()))
        })
    }

    pub fn in_flight(&self) -> bool {
        self.sent != self.landed
    }

    pub fn listing(&self) -> Result<Vec<(String, u64)>, Error> {
        let place = self.place.as_deref().ok_or_else(unplaced)?;
        let mut listed = list(place).map_err(|error| refused("$.kept", error))?;
        listed.retain(|(name, _)| named(name));
        listed.sort();
        Ok(listed)
    }

    fn ask<T>(
        &mut self,
        name: &str,
        order: Order,
        decode: impl FnOnce(Arc<[u8]>) -> Result<T, Error> + Send + Sync + 'static,
    ) -> Background<T, u64> {
        if !named(name) {
            return Background::failed(
                Error::new(ErrorCode::InvalidName, "$.kept")
                    .expected(Expected::Matching)
                    .got(name),
            );
        }
        let Some(place) = self.place.clone() else {
            return Background::failed(unplaced());
        };
        self.serial += 1;
        let serial = self.serial;
        let held = self.requests.ask(serial);
        let Some(ticket) = self.requests.ticket(&serial) else {
            return Background::failed(unplaced());
        };
        self.sent += 1;
        self.mover.send(Sent {
            place,
            name: name.to_owned(),
            order,
            serial,
            ticket,
            landing: Arc::clone(&self.landing),
        });
        Background::asked(held, decode)
    }
}

impl Delivers<u64> for Kept {
    fn landed(&self, held: &Interest<u64>) -> Option<Result<Arc<[u8]>, Error>> {
        self.requests.landed(held)
    }

    fn cause(&self, held: &Interest<u64>) -> engine_profile::Cause {
        self.requests.cause(held.key())
    }
}

fn named(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_ID_BYTES
        && name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
}

fn unplaced() -> Error {
    Error::new(ErrorCode::UnsupportedFeature, "$.kept").got("this host keeps nothing")
}

fn refused(path: &str, error: String) -> Error {
    Error::new(ErrorCode::SourceIo, path).got(error)
}

fn lock<T>(held: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    held.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn exchange_kept(world: &mut World) {
    let now = world
        .get_resource::<crate::FrameTime>()
        .map_or(0.0, |time| time.real_elapsed);
    let Some(mut kept) = world.get_resource_mut::<Kept>() else {
        return;
    };
    let kept = &mut *kept;
    kept.requests.settle(now);
    let landed = std::mem::take(&mut *lock(&kept.landing));
    for (serial, ticket, name, done) in landed {
        kept.landed += 1;
        match done {
            Ok(bytes) => {
                kept.requests.ready(&serial, ticket, Arc::from(bytes));
            }
            Err(error) => {
                let error = refused(&format!("$.kept.{name}"), error);
                kept.requests.fail(&serial, ticket, error);
            }
        }
    }
}

fn run(sent: Sent) {
    let Sent {
        place,
        name,
        order,
        serial,
        ticket,
        landing,
    } = sent;
    let done = match order {
        Order::Keep(bytes) => store(&place, &name, &bytes).map(|()| Vec::new()),
        Order::Read => load(&place, &name).map(|found| match found {
            Some(bytes) => [&[PRESENT][..], &bytes].concat(),
            None => vec![0],
        }),
    };
    lock(&landing).push((serial, ticket, name, done));
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct Mover(Option<std::sync::mpsc::Sender<Sent>>);

#[cfg(not(target_arch = "wasm32"))]
impl Mover {
    fn send(&mut self, sent: Sent) {
        let sent = match &self.0 {
            Some(sender) => match sender.send(sent) {
                Ok(()) => return,
                Err(returned) => returned.0,
            },
            None => sent,
        };
        let (sender, receiver) = std::sync::mpsc::channel::<Sent>();
        let spawned = std::thread::Builder::new()
            .name("engine_kept".to_owned())
            .spawn(move || {
                while let Ok(sent) = receiver.recv() {
                    run(sent);
                }
            });
        match spawned {
            Ok(_) => {
                if let Err(returned) = sender.send(sent) {
                    run(returned.0);
                }
                self.0 = Some(sender);
            }
            Err(error) => {
                let Sent {
                    name,
                    serial,
                    ticket,
                    landing,
                    ..
                } = sent;
                lock(&landing).push((serial, ticket, name, Err(error.to_string())));
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn file(place: &str, name: &str) -> std::path::PathBuf {
    std::path::Path::new(place).join(format!("{FILE_PREFIX}{name}"))
}

#[cfg(not(target_arch = "wasm32"))]
fn store(place: &str, name: &str, bytes: &[u8]) -> Result<(), String> {
    let at = file(place, name);
    let writing = file(place, &format!("{name}.writing"));
    std::fs::create_dir_all(place)
        .and_then(|()| std::fs::write(&writing, bytes))
        .and_then(|()| std::fs::rename(&writing, &at))
        .map_err(|error| format!("{}: {error}", at.display()))
}

#[cfg(not(target_arch = "wasm32"))]
fn load(place: &str, name: &str) -> Result<Option<Vec<u8>>, String> {
    let at = file(place, name);
    match std::fs::read(&at) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("{}: {error}", at.display())),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn list(place: &str) -> Result<Vec<(String, u64)>, String> {
    let entries = match std::fs::read_dir(place) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("{place}: {error}")),
    };
    Ok(entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let name = name.strip_prefix(FILE_PREFIX)?.to_owned();
            let bytes = entry.metadata().ok()?.len();
            Some((name, bytes))
        })
        .collect())
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct Mover;

#[cfg(target_arch = "wasm32")]
impl Mover {
    fn send(&mut self, sent: Sent) {
        wasm_bindgen_futures::spawn_local(async move { run(sent) });
    }
}

#[cfg(target_arch = "wasm32")]
fn storage() -> Result<web_sys::Storage, String> {
    web_sys::window()
        .ok_or("no window")?
        .local_storage()
        .map_err(|error| format!("{error:?}"))?
        .ok_or_else(|| "no local storage".to_owned())
}

#[cfg(target_arch = "wasm32")]
fn key(place: &str, name: &str) -> String {
    format!("{place}/{FILE_PREFIX}{name}")
}

#[cfg(target_arch = "wasm32")]
fn store(place: &str, name: &str, bytes: &[u8]) -> Result<(), String> {
    let text: String = bytes.iter().map(|&byte| char::from(byte)).collect();
    storage()?
        .set_item(&key(place, name), &text)
        .map_err(|error| format!("{}: {error:?}", key(place, name)))
}

#[cfg(target_arch = "wasm32")]
fn load(place: &str, name: &str) -> Result<Option<Vec<u8>>, String> {
    let at = key(place, name);
    let Some(text) = storage()?
        .get_item(&at)
        .map_err(|error| format!("{at}: {error:?}"))?
    else {
        return Ok(None);
    };
    text.chars()
        .map(|unit| u8::try_from(u32::from(unit)).map_err(|_| format!("{at}: not bytes")))
        .collect::<Result<Vec<u8>, String>>()
        .map(Some)
}

#[cfg(target_arch = "wasm32")]
fn list(place: &str) -> Result<Vec<(String, u64)>, String> {
    let storage = storage()?;
    let prefix = key(place, "");
    let count = storage.length().map_err(|error| format!("{error:?}"))?;
    let mut listed = Vec::new();
    for at in 0..count {
        let Some(found) = storage.key(at).map_err(|error| format!("{error:?}"))? else {
            continue;
        };
        let Some(name) = found.strip_prefix(&prefix) else {
            continue;
        };
        let bytes = storage
            .get_item(&found)
            .map_err(|error| format!("{error:?}"))?
            .map_or(0, |text| text.chars().count() as u64);
        listed.push((name.to_owned(), bytes));
    }
    Ok(listed)
}
