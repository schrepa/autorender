use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{AssetLibrary, TextureHandle};
use engine_settings::{Error, ErrorCode};
pub use engine_textures::{
    Family as TextureFamily, Source as TextureSource, Usage as TextureUsage,
};

use engine_registry::{JobBatch, JobOrder, JobQueue};

use crate::jobs::record_key;
use crate::{
    Interest, JobExchange, JobKind, JobRef, List, RecordKey, Records, Requests, URGENT_JOBS,
    job_priority,
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

pub type MakeTextures = Arc<dyn Fn() -> Result<Option<Vec<TextureSource>>, String> + Send + Sync>;

pub struct TextureJobs;

impl List for TextureJobs {
    const NAME: &'static str = "texture_jobs";
    type Member = &'static JobKind;
}

#[derive(Clone)]
pub struct TextureSet {
    pub source: [u8; 32],
    pub name: String,
    pub textures: Vec<(String, TextureUsage)>,
    pub make: MakeTextures,
}

impl TextureSet {
    pub fn key(&self, job: Option<&JobKind>, index: usize, family: TextureFamily) -> RecordKey {
        let mut identity =
            format!("{}/{}/{}", self.name, self.textures[index].0, family.name()).into_bytes();
        if let Some(job) = job {
            identity.extend_from_slice(&job.source().hash());
        }
        record_key(&self.source, TEXTURE_RECORD, &identity)
    }

    pub fn sources(&self) -> Result<Option<Vec<TextureSource>>, String> {
        let Some(made) = (self.make)().map_err(|error| format!("{}: {error}", self.name))? else {
            return Ok(None);
        };
        if made.len() == self.textures.len() {
            Ok(Some(made))
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

enum Make {
    Here(Option<MakeRecord>),
    Texture { batch: u64, index: usize },
}

enum Stage {
    Asking,
    Fetching(Interest<RecordKey>),
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
    job: Option<&'static JobKind>,
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
        self.ask_textures();
        Ok(handles)
    }

    pub fn texture_sets(&self) -> impl Iterator<Item = &TextureSet> + '_ {
        self.sets.iter().map(|asked| &*asked.set)
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
                let error =
                    Error::new(ErrorCode::InvalidRecord, format!("$.records.{}", key.hex()))
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
        let job = self.job;
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
                let key = asked.set.key(job, index, shown);
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

    fn plain(&mut self, id: u64, index: usize) -> Result<Option<Vec<u8>>, String> {
        let batch = self
            .batches
            .iter_mut()
            .find(|batch| batch.id == id)
            .ok_or("its sources are gone")?;
        let made = match &batch.sources {
            Some(made) => made.clone()?,
            None => match batch.set.sources() {
                Ok(None) => return Ok(None),
                Ok(Some(made)) => {
                    let made = Arc::new(made);
                    batch.sources = Some(Ok(Arc::clone(&made)));
                    made
                }
                Err(error) => {
                    batch.sources = Some(Err(error.clone()));
                    return Err(error);
                }
            },
        };
        engine_textures::plain(&made[index])
            .map(|texture| Some(engine_textures::record(&texture)))
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
    (stored, orders, place): (&Records, &mut Vec<JobOrder>, u32),
) -> Step {
    loop {
        let found = match &request.stage {
            Stage::Asking => {
                let kind = match request.make {
                    Make::Here(_) => None,
                    Make::Texture { .. } => records.job.map(JobRef::from),
                };
                let key = request.key();
                orders.push(JobOrder::Look {
                    key: key.hash(),
                    kind,
                    priority: job_priority(URGENT_JOBS, place),
                });
                request.stage = Stage::Fetching(stored.requests.ask_later(key));
                return Step::Waiting(request);
            }
            Stage::Fetching(fetch) => stored.get(fetch),
            Stage::Making => {
                let made = match &mut request.make {
                    Make::Here(make) => match make.take() {
                        Some(make) => make().map(Some),
                        None => Err("it was made already".into()),
                    },
                    &mut Make::Texture { batch, index } => records.plain(batch, index),
                };
                return match made {
                    Ok(Some(bytes)) => Step::Arrived {
                        request,
                        bytes: Arc::from(bytes),
                        fetched: false,
                    },
                    Ok(None) => Step::Waiting(request),
                    Err(error) => Step::Failed(request, error),
                };
            }
        };
        match found {
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
        }
    }
}

pub(crate) fn install_jobs(world: &mut World) {
    let job = crate::members_in::<TextureJobs>(world).first().copied();
    world.resource_mut::<RecordExchange>().job = job;
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
    let mut orders = Vec::new();
    let mut kept = Vec::new();
    let verifying = world
        .get_resource::<JobExchange>()
        .is_some_and(JobExchange::verifying);
    world.resource_scope(|world, mut records: Mut<RecordExchange>| {
        let stored = world.resource::<Records>();
        for (place, request) in asking.into_iter().enumerate() {
            let place = u32::try_from(place).unwrap_or(u32::MAX);
            match step(&mut records, request, (stored, &mut orders, place)) {
                Step::Waiting(request) => waiting.push(request),
                Step::Arrived {
                    mut request,
                    bytes,
                    fetched,
                } => {
                    let key = request.key();
                    let here = matches!(request.make, Make::Here(_));
                    if here && !fetched {
                        kept.push((key, bytes.to_vec()));
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
    if !orders.is_empty() {
        let batch = JobBatch { owner: 0, run: 0 };
        world
            .get_resource_or_init::<JobQueue>()
            .orders
            .extend(orders.into_iter().map(|order| (batch, order)));
    }
    if let Some(mut jobs) = world.get_resource_mut::<JobExchange>() {
        for (key, bytes) in kept {
            jobs.keep(key, None, bytes);
        }
    }
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
    let RecordExchange {
        asking, batches, ..
    } = &mut *records;
    batches.retain(|batch| {
        asking.iter().any(
            |request| matches!(request.make, Make::Texture { batch: id, .. } if id == batch.id),
        )
    });
}
