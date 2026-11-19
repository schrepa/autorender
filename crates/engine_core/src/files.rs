use std::sync::Arc;

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;

use crate::{Background, Delivers, Error, ErrorCode, Interest, Kept, Requests, Ticket};

pub enum IoAct {
    Read(String),
    Load(String, String),
    Store(String, String, Vec<u8>),
}

pub struct IoOrder {
    pub id: Ticket,
    pub act: IoAct,
}

pub struct IoReply {
    pub id: Ticket,
    pub done: Result<Option<Vec<u8>>, String>,
}

struct Reading;

#[derive(Resource, Default)]
pub struct Files {
    requests: Requests<String, Arc<[u8]>, Reading>,
    root: String,
}

impl Files {
    pub fn set_root(&mut self, root: impl Into<String>) {
        self.root = root.into();
    }

    pub fn read<T>(
        &mut self,
        path: &str,
        decode: impl FnOnce(&[u8]) -> Result<T, Error> + Send + Sync + 'static,
    ) -> Background<T, String> {
        Background::asked(self.requests.ask(path.to_owned()), move |bytes| {
            decode(&bytes[..])
        })
    }

    pub fn census(&self) -> crate::Census {
        self.requests.census()
    }

    fn orders(&mut self, now: f64, orders: &mut Vec<IoOrder>) {
        self.requests.settle(now);
        let fresh: Vec<(String, Ticket)> = self
            .requests
            .asked()
            .filter(|(path, _)| self.requests.work(path).is_none())
            .map(|(path, ticket)| (path.clone(), ticket))
            .collect();
        for (path, id) in fresh {
            self.requests.set_work(&path, Reading);
            orders.push(IoOrder {
                id,
                act: IoAct::Read(joined(&self.root, &path)),
            });
        }
    }

    fn land(&mut self, reply: IoReply) {
        let Some(path) = self.requests.key_of(reply.id).cloned() else {
            return;
        };
        match reply.done.and_then(|read| read.ok_or_else(|| path.clone())) {
            Ok(bytes) => {
                self.requests.ready(&path, reply.id, Arc::from(bytes));
            }
            Err(error) => {
                let missing = Error::new(ErrorCode::MissingFile, "$.files").got(error);
                self.requests.fail(&path, reply.id, missing);
            }
        }
    }
}

impl Delivers<String> for Files {
    fn landed(&self, held: &Interest<String>) -> Option<Result<Arc<[u8]>, Error>> {
        self.requests.landed(held)
    }
}

pub(crate) fn take_io(world: &mut World) -> Vec<IoOrder> {
    let now = world
        .get_resource::<crate::FrameTime>()
        .map_or(0.0, |time| time.real_elapsed);
    let mut orders = Vec::new();
    if let Some(mut kept) = world.get_resource_mut::<Kept>() {
        kept.orders(now, &mut orders);
    }
    if let Some(mut files) = world.get_resource_mut::<Files>() {
        files.orders(now, &mut orders);
    }
    orders
}

pub(crate) fn land_io(world: &mut World, reply: IoReply) {
    let reply = match world.get_resource_mut::<Kept>() {
        Some(mut kept) => match kept.land(reply) {
            Some(theirs) => theirs,
            None => return,
        },
        None => reply,
    };
    if let Some(mut files) = world.get_resource_mut::<Files>() {
        files.land(reply);
    }
}

fn joined(root: &str, path: &str) -> String {
    if root.is_empty() || root.ends_with('/') {
        format!("{root}{path}")
    } else {
        format!("{root}/{path}")
    }
}
