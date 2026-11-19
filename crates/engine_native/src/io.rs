use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};

use engine::host::SceneLoad;
use engine::scene::{SceneDocument, SceneFiles, SceneKinds};
use engine::{Game, GameHost};
use engine_core::{IoAct, IoOrder, IoReply, Ticket};

const KEPT_PREFIX: &str = "kept-";
const WRITING: &str = ".writing";

type Done = Result<Option<Vec<u8>>, String>;

pub(crate) struct Io {
    reply: Sender<IoReply>,
    replies: Receiver<IoReply>,
    kept: Option<Sender<IoOrder>>,
}

impl Default for Io {
    fn default() -> Self {
        let (reply, replies) = channel();
        Self {
            reply,
            replies,
            kept: None,
        }
    }
}

impl Io {
    pub(crate) fn exchange<G: Game>(&mut self, host: &mut GameHost<G>) {
        for order in host.take_io() {
            self.start(order);
        }
        while let Ok(reply) = self.replies.try_recv() {
            host.land_io(reply);
        }
    }

    fn start(&mut self, order: IoOrder) {
        if let IoAct::Load(..) | IoAct::Store(..) = order.act {
            self.keep(order);
            return;
        }
        let reply = self.reply.clone();
        let id = order.id;
        let spawned = std::thread::Builder::new()
            .name("engine_file_read".to_owned())
            .spawn(move || {
                let _ = reply.send(answer(order));
            });
        if let Err(error) = spawned {
            self.failed(id, &error);
        }
    }

    fn keep(&mut self, order: IoOrder) {
        let order = match &self.kept {
            Some(kept) => match kept.send(order) {
                Ok(()) => return,
                Err(returned) => returned.0,
            },
            None => order,
        };
        let (kept, orders) = channel::<IoOrder>();
        let reply = self.reply.clone();
        let spawned = std::thread::Builder::new()
            .name("engine_kept".to_owned())
            .spawn(move || {
                for order in orders {
                    if reply.send(answer(order)).is_err() {
                        return;
                    }
                }
            });
        match spawned {
            Ok(_) => {
                let _ = kept.send(order);
                self.kept = Some(kept);
            }
            Err(error) => self.failed(order.id, &error),
        }
    }

    fn failed(&self, id: Ticket, error: &std::io::Error) {
        let _ = self.reply.send(IoReply {
            id,
            done: Err(error.to_string()),
        });
    }
}

pub(crate) fn scene_opened(
    path: &Path,
    source: &str,
    kinds: &SceneKinds,
) -> engine::Result<(SceneDocument, SceneFiles)> {
    let mut load = SceneLoad::new(source, kinds)?;
    let folder = path.parent().unwrap_or(Path::new("."));
    for read in load.take_reads() {
        let path = folder.join(&read.uri);
        let bytes = std::fs::read(&path).map_err(|error| crate::file_error("$", &path, &error))?;
        load.read(&read.uri, bytes)?;
    }
    load.opened()
}

fn answer(IoOrder { id, act }: IoOrder) -> IoReply {
    let done = match act {
        IoAct::Read(path) => std::fs::read(&path)
            .map(Some)
            .map_err(|error| format!("{path}: {error}")),
        IoAct::Load(place, name) => load(&kept_file(&place, &name)),
        IoAct::Store(place, name, bytes) => store(&place, &name, &bytes),
    };
    IoReply { id, done }
}

fn kept_file(place: &str, name: &str) -> PathBuf {
    Path::new(place).join(format!("{KEPT_PREFIX}{name}"))
}

fn load(at: &Path) -> Done {
    match std::fs::read(at) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("{}: {error}", at.display())),
    }
}

fn store(place: &str, name: &str, bytes: &[u8]) -> Done {
    let at = kept_file(place, name);
    let writing = kept_file(place, &format!("{name}{WRITING}"));
    std::fs::create_dir_all(place)
        .and_then(|()| std::fs::write(&writing, bytes))
        .and_then(|()| std::fs::rename(&writing, &at))
        .map(|()| None)
        .map_err(|error| format!("{}: {error}", at.display()))
}

#[cfg(feature = "dev")]
pub(crate) fn kept_listing(place: &str) -> Result<Vec<(String, u64)>, String> {
    let entries = match std::fs::read_dir(place) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("{place}: {error}")),
    };
    let mut listed: Vec<(String, u64)> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let name = name.strip_prefix(KEPT_PREFIX)?;
            if name.ends_with(WRITING) {
                return None;
            }
            Some((name.to_owned(), entry.metadata().ok()?.len()))
        })
        .collect();
    listed.sort();
    Ok(listed)
}
