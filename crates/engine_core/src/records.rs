use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{AssetLibrary, TextureHandle};
use engine_settings::{Error, ErrorCode};
pub use engine_textures::{
    Family as TextureFamily, Source as TextureSource, Usage as TextureUsage,
};

use crate::jobs::record_key;
use crate::{
    Census, Interest, JobExchange, JobId, JobRef, RecordKey, Requests, TEXTURE_COOK_NAME,
    TEXTURE_COOK_SOURCE, URGENT_JOBS, job_priority,
};

const TEXTURE_RECORD: &str = "texture";

type MakeRecord = Box<dyn FnOnce() -> Result<Vec<u8>, String> + Send + Sync>;
type DeliverRecord = Box<dyn FnOnce(&mut World, &[u8]) -> Result<(), String> + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureSampling {
    Family(TextureFamily),
    Decompressed,
}

impl TextureSampling {
    pub fn family(self) -> TextureFamily {
        match self {
            Self::Family(family) => family,
            Self::Decompressed => TextureFamily::Bc7,
        }
    }
}

pub type MakeTextures = Arc<dyn Fn() -> Result<Vec<TextureSource>, String> + Send + Sync>;

#[derive(Clone)]
pub struct TextureSet {
    pub source: [u8; 32],
    pub name: String,
    pub textures: Vec<(String, TextureUsage)>,
    pub make: MakeTextures,
}

impl TextureSet {
    pub fn key(&self, index: usize, family: TextureFamily) -> RecordKey {
        let mut identity =
            format!("{}/{}/{}", self.name, self.textures[index].0, family.name()).into_bytes();
        identity.extend_from_slice(&crate::sources::TEXTURE_COOK);
        record_key(&self.source, TEXTURE_RECORD, &identity)
    }

    pub fn sources(&self) -> Result<Vec<TextureSource>, String> {
        let made = (self.make)().map_err(|error| format!("{}: {error}", self.name))?;
        if made.len() == self.textures.len() {
            Ok(made)
        } else {
            Err(format!(
                "{}: {} sources made for {} textures",
                self.name,
                made.len(),
                self.textures.len()
            ))
        }
    }
}

fn texture_cook() -> JobRef {
    JobRef::named(TEXTURE_COOK_NAME, TEXTURE_COOK_SOURCE.hash())
}

enum Make {
    Here(Option<MakeRecord>),
    Texture { batch: u64, index: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Asking,
    Fetching(JobId),
    Making,
}

struct Asking {
    interest: Interest<RecordKey>,
    handle: Option<TextureHandle>,
    make: Make,
    deliver: Option<DeliverRecord>,
    stage: Stage,
}

impl Asking {
    fn key(&self) -> RecordKey {
        *self.interest.key()
    }
}

struct AskedSet {
    set: Arc<TextureSet>,
    handles: Vec<TextureHandle>,
    asked: bool,
}

struct Batch {
    id: u64,
    set: Arc<TextureSet>,
    sources: Option<Result<Arc<Vec<TextureSource>>, String>>,
}

#[doc(hidden)]
#[derive(Resource, Default)]
pub struct RecordExchange {
    sampling: Option<TextureSampling>,
    asking: Vec<Asking>,
    sets: Vec<AskedSet>,
    batches: Vec<Batch>,
    next_batch: u64,
    records: Requests<RecordKey, ()>,
    failed: Vec<(Interest<RecordKey>, Option<TextureHandle>)>,
}

impl RecordExchange {
    pub fn set_sampling(&mut self, sampling: Option<TextureSampling>) {
        self.sampling = sampling;
    }

    pub fn sampling(&self) -> Option<TextureSampling> {
        self.sampling
    }

    pub fn request(
        &mut self,
        key: RecordKey,
        make: impl FnOnce() -> Result<Vec<u8>, String> + Send + Sync + 'static,
        deliver: impl FnOnce(&mut World, &[u8]) -> Result<(), String> + Send + Sync + 'static,
    ) {
        let interest = self.records.ask(key);
        self.asking.push(Asking {
            interest,
            handle: None,
            make: Make::Here(Some(Box::new(make))),
            deliver: Some(Box::new(deliver)),
            stage: Stage::Asking,
        });
    }

    pub fn textures(
        &mut self,
        assets: &mut AssetLibrary,
        set: TextureSet,
    ) -> Result<Vec<TextureHandle>, Error> {
        let handles = set
            .textures
            .iter()
            .map(|_| assets.reserve_texture())
            .collect::<Result<Vec<_>, _>>()?;
        self.sets.push(AskedSet {
            set: Arc::new(set),
            handles: handles.clone(),
            asked: false,
        });
        Ok(handles)
    }

    pub fn texture_sets(&self) -> impl Iterator<Item = &TextureSet> + '_ {
        self.sets.iter().map(|asked| &*asked.set)
    }

    pub fn census(&self) -> Census {
        let mut census = self.records.census();
        if self.sampling.is_some() {
            census.loading.total += self.sets.iter().filter(|asked| !asked.asked).count() as u64;
        }
        census
    }

    pub fn failures(&self) -> impl Iterator<Item = (&RecordKey, &Error)> + '_ {
        self.failed
            .iter()
            .filter_map(|(held, _)| Some((held.key(), self.failure(held)?)))
    }

    pub fn failed(&self, handle: TextureHandle) -> Option<&Error> {
        self.failed
            .iter()
            .filter(|(_, filled)| *filled == Some(handle))
            .find_map(|(held, _)| self.failure(held))
    }

    fn failure(&self, held: &Interest<RecordKey>) -> Option<&Error> {
        match self.records.of(held)? {
            crate::Request::Failed(error) => Some(error),
            crate::Request::Asked(_) | crate::Request::Ready(()) | crate::Request::Missing => None,
        }
    }

    fn end(
        &mut self,
        key: RecordKey,
        held: Option<(Interest<RecordKey>, Option<TextureHandle>)>,
        outcome: Result<(), String>,
    ) {
        let Some(ticket) = self.records.ticket(&key) else {
            return;
        };
        match outcome {
            Ok(()) => {
                self.records.ready(&key, ticket, ());
            }
            Err(error) => {
                let error = Error::new(ErrorCode::InvalidRecord, format!("$.records.{}", key.hex()))
                    .got(error);
                self.records.fail(&key, ticket, error);
                self.failed.extend(held);
            }
        }
    }

    fn ask_textures(&mut self) {
        let Some(sampling) = self.sampling else {
            return;
        };
        let shown = sampling.family();
        for asked in self.sets.iter_mut().filter(|asked| !asked.asked) {
            asked.asked = true;
            let batch = self.next_batch;
            self.next_batch += 1;
            self.batches.push(Batch {
                id: batch,
                set: Arc::clone(&asked.set),
                sources: None,
            });
            for (index, &handle) in asked.handles.iter().enumerate() {
                let key = asked.set.key(index, shown);
                self.asking.push(Asking {
                    interest: self.records.ask(key),
                    handle: Some(handle),
                    make: Make::Texture { batch, index },
                    deliver: Some(deliver_texture(handle, sampling, key)),
                    stage: Stage::Asking,
                });
            }
        }
    }

    fn plain(&mut self, id: u64, index: usize) -> Result<Vec<u8>, String> {
        let batch = self
            .batches
            .iter_mut()
            .find(|batch| batch.id == id)
            .ok_or("its sources are gone")?;
        let set = &batch.set;
        let made = batch
            .sources
            .get_or_insert_with(|| set.sources().map(Arc::new))
            .clone()?;
        engine_textures::plain(&made[index])
            .map(|texture| engine_textures::record(&texture))
            .map_err(|error| format!("{}: {error}", batch.set.name))
    }
}

fn deliver_texture(
    handle: TextureHandle,
    sampling: TextureSampling,
    key: RecordKey,
) -> DeliverRecord {
    Box::new(move |world, bytes| {
        let texture = engine_textures::decode(bytes)?;
        let texture = match sampling {
            TextureSampling::Decompressed if texture.format().is_compressed() => {
                engine_textures::decompress(&texture).map_err(|error| error.to_string())?
            }
            TextureSampling::Family(_) | TextureSampling::Decompressed => texture,
        };
        world
            .resource_mut::<AssetLibrary>()
            .fill_texture(handle, texture.with_identity(key.identity()))
            .map_err(|error| error.to_string())
    })
}

enum Step {
    Waiting(Asking),
    Arrived {
        request: Asking,
        bytes: Arc<[u8]>,
        fetched: bool,
    },
    Failed(Asking, String),
}

fn step(
    records: &mut RecordExchange,
    mut request: Asking,
    (jobs, place): (&mut JobExchange, u32),
) -> Step {
    loop {
        match request.stage {
            Stage::Asking => {
                let kind = match request.make {
                    Make::Here(_) => None,
                    Make::Texture { .. } => Some(texture_cook()),
                };
                let id = jobs.fetch(kind, request.key(), job_priority(URGENT_JOBS, place));
                request.stage = Stage::Fetching(id);
            }
            Stage::Fetching(id) => match jobs.take(id) {
                None => return Step::Waiting(request),
                Some(Ok(bytes)) => {
                    request.stage = Stage::Making;
                    return Step::Arrived {
                        request,
                        bytes,
                        fetched: true,
                    };
                }
                Some(Err(_)) => request.stage = Stage::Making,
            },
            Stage::Making => {
                let made = match &mut request.make {
                    Make::Here(make) => match make.take() {
                        Some(make) => make(),
                        None => Err("it was made already".into()),
                    },
                    &mut Make::Texture { batch, index } => records.plain(batch, index),
                };
                return match made {
                    Ok(bytes) => Step::Arrived {
                        request,
                        bytes: Arc::from(bytes),
                        fetched: false,
                    },
                    Err(error) => Step::Failed(request, error),
                };
            }
        }
    }
}

pub(crate) fn exchange(world: &mut World) {
    let now = world
        .get_resource::<crate::FrameTime>()
        .map_or(0.0, |time| time.real_elapsed);
    let asking = {
        let mut records = world.resource_mut::<RecordExchange>();
        records.ask_textures();
        records.records.settle(now);
        std::mem::take(&mut records.asking)
    };
    if asking.is_empty() {
        return;
    }
    let mut waiting = Vec::with_capacity(asking.len());
    let mut arrived = Vec::new();
    let mut ends = Vec::new();
    world.resource_scope(|world, mut jobs: Mut<JobExchange>| {
        let verifying = jobs.verifying();
        let mut records = world.resource_mut::<RecordExchange>();
        for (place, request) in asking.into_iter().enumerate() {
            match step(&mut records, request, (&mut *jobs, place as u32)) {
                Step::Waiting(request) => waiting.push(request),
                Step::Arrived {
                    mut request,
                    bytes,
                    fetched,
                } => {
                    let key = request.key();
                    let here = matches!(request.make, Make::Here(_));
                    if here && !fetched {
                        jobs.keep(key, None, bytes.to_vec());
                    }
                    let deliver = request.deliver.take();
                    let held = if here && fetched && verifying {
                        waiting.push(request);
                        None
                    } else {
                        Some((request.interest, request.handle))
                    };
                    arrived.push((key, held, deliver, bytes));
                }
                Step::Failed(request, error) => {
                    let key = request.key();
                    ends.push((key, Some((request.interest, request.handle)), Err(error)));
                }
            }
        }
    });
    for (key, held, deliver, bytes) in arrived {
        let delivered = match deliver {
            Some(deliver) => deliver(world, &bytes),
            None => Ok(()),
        };
        ends.push((key, held, delivered));
    }
    let mut records = world.resource_mut::<RecordExchange>();
    for (key, held, outcome) in ends {
        records.end(key, held, outcome);
    }
    waiting.append(&mut records.asking);
    records.asking = waiting;
    let RecordExchange { asking, batches, .. } = &mut *records;
    batches.retain(|batch| {
        asking.iter().any(
            |request| matches!(request.make, Make::Texture { batch: id, .. } if id == batch.id),
        )
    });
}
