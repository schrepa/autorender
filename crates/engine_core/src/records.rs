use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{AssetLibrary, TextureHandle};
use engine_settings::{Error, ErrorCode};
pub use engine_textures::{
    Family as TextureFamily, Source as TextureSource, Usage as TextureUsage,
    compress as compress_texture,
};

use crate::jobs::record_key;
use crate::{
    Census, CookMode, Interest, JobExchange, JobId, JobKind, JobRef, RecordKey, Requests,
    TEXTURE_COOK, URGENT_JOBS, VIEW_JOBS, job_priority,
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
pub enum MakeSources {
    Here(MakeTextures),
    Job {
        kind: &'static JobKind,
        input: Vec<u8>,
        size: [u32; 2],
    },
}

#[derive(Clone)]
pub struct TextureSet {
    pub source: [u8; 32],
    pub name: String,
    pub textures: Vec<(String, TextureUsage)>,
    pub make: MakeSources,
}

impl TextureSet {
    fn key(&self, index: usize, family: TextureFamily) -> RecordKey {
        let mut identity =
            format!("{}/{}/{}", self.name, self.textures[index].0, family.name()).into_bytes();
        identity.extend_from_slice(&crate::sources::TEXTURES);
        record_key(&self.source, TEXTURE_RECORD, &identity)
    }
}

enum Make {
    Here(Option<MakeRecord>),
    Texture {
        batch: u64,
        index: usize,
        family: TextureFamily,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Asking,
    Fetching(JobId),
    Making,
    Cooking(JobId),
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
    asked: [bool; 2],
}

enum Sources {
    Unmade,
    Making(JobId),
    Made(Result<Arc<Vec<TextureSource>>, String>),
}

struct Batch {
    id: u64,
    set: Arc<TextureSet>,
    sources: Sources,
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
            asked: [false; 2],
        });
        Ok(handles)
    }

    pub fn census(&self) -> Census {
        let mut census = self.records.census();
        let unasked = self.sampling.map_or(0, |sampling| {
            let shown = family_slot(sampling.family());
            self.sets.iter().filter(|asked| !asked.asked[shown]).count() as u64
        });
        census.loading.total += unasked;
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

    fn ask_textures(&mut self, cooking: bool) {
        let Some(sampling) = self.sampling else {
            return;
        };
        let shown = sampling.family();
        for asked in &mut self.sets {
            let families: Vec<(usize, TextureFamily)> = TextureFamily::ALL
                .into_iter()
                .enumerate()
                .filter(|&(slot, family)| !asked.asked[slot] && (cooking || family == shown))
                .collect();
            if families.is_empty() {
                continue;
            }
            let batch = self.next_batch;
            self.next_batch += 1;
            self.batches.push(Batch {
                id: batch,
                set: Arc::clone(&asked.set),
                sources: Sources::Unmade,
            });
            for (slot, family) in families {
                asked.asked[slot] = true;
                for (index, &handle) in asked.handles.iter().enumerate() {
                    let key = asked.set.key(index, family);
                    self.asking.push(Asking {
                        interest: self.records.ask(key),
                        handle: (family == shown).then_some(handle),
                        make: Make::Texture {
                            batch,
                            index,
                            family,
                        },
                        deliver: (family == shown).then(|| deliver_texture(handle, sampling, key)),
                        stage: Stage::Asking,
                    });
                }
            }
        }
        self.sets
            .retain(|asked| !asked.asked.iter().all(|&done| done));
    }

    fn step_sources(&mut self, id: u64, jobs: &mut JobExchange, priority: u32) {
        let Some(batch) = self.batches.iter_mut().find(|batch| batch.id == id) else {
            return;
        };
        match &batch.sources {
            Sources::Unmade => match &batch.set.make {
                MakeSources::Here(make) => {
                    let made = make().and_then(|made| {
                        if made.len() == batch.set.textures.len() {
                            Ok(Arc::new(made))
                        } else {
                            Err(format!(
                                "{} sources made for {} textures",
                                made.len(),
                                batch.set.textures.len()
                            ))
                        }
                    });
                    batch.sources =
                        Sources::Made(made.map_err(|error| format!("{}: {error}", batch.set.name)));
                }
                MakeSources::Job { kind, input, .. } => {
                    batch.sources =
                        Sources::Making(jobs.submit(JobRef::from(*kind), input.clone(), priority));
                }
            },
            &Sources::Making(job) => {
                let Some(output) = jobs.take(job) else {
                    return;
                };
                let (set, usage) = (
                    &batch.set,
                    batch.set.textures.first().map(|(_, usage)| *usage),
                );
                let made = match (&set.make, usage) {
                    (MakeSources::Job { size, .. }, Some(usage)) if set.textures.len() == 1 => {
                        output.map(|pixels| {
                            Arc::new(vec![TextureSource {
                                width: size[0],
                                height: size[1],
                                pixels: pixels.to_vec(),
                                usage,
                            }])
                        })
                    }
                    _ => Err("a job makes the sources of a set of one texture".to_owned()),
                };
                batch.sources =
                    Sources::Made(made.map_err(|error| format!("{}: {error}", set.name)));
            }
            Sources::Made(_) => {}
        }
    }
}

fn family_slot(family: TextureFamily) -> usize {
    TextureFamily::ALL
        .iter()
        .position(|&each| each == family)
        .unwrap_or(0)
}

fn deliver_texture(
    handle: TextureHandle,
    sampling: TextureSampling,
    key: RecordKey,
) -> DeliverRecord {
    Box::new(move |world, bytes| {
        let texture = engine_textures::decode(bytes)?;
        let texture = match sampling {
            TextureSampling::Family(_) => texture,
            TextureSampling::Decompressed => {
                engine_textures::decompress(&texture).map_err(|error| error.to_string())?
            }
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
        made: bool,
    },
    Failed(Asking, String),
}

fn step(
    records: &mut RecordExchange,
    mut request: Asking,
    (jobs, place): (&mut JobExchange, u32),
) -> Step {
    let make_priority = job_priority(VIEW_JOBS, place);
    loop {
        match request.stage {
            Stage::Asking => {
                let kind = match request.make {
                    Make::Here(_) => None,
                    Make::Texture { .. } => Some(JobRef::from(&TEXTURE_COOK)),
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
                        made: false,
                    };
                }
                Some(Err(_)) => request.stage = Stage::Making,
            },
            Stage::Making => match &mut request.make {
                Make::Here(make) => {
                    let Some(make) = make.take() else {
                        return Step::Failed(request, "it was made already".into());
                    };
                    return match make() {
                        Ok(bytes) => Step::Arrived {
                            request,
                            bytes: Arc::from(bytes),
                            made: true,
                        },
                        Err(error) => Step::Failed(request, error),
                    };
                }
                &mut Make::Texture {
                    batch,
                    index,
                    family,
                } => {
                    records.step_sources(batch, jobs, make_priority);
                    let sources = records
                        .batches
                        .iter()
                        .find(|each| each.id == batch)
                        .map(|batch| &batch.sources);
                    match sources {
                        Some(Sources::Made(Ok(made))) => {
                            let input = engine_textures::cook_input(&made[index], family);
                            let id = jobs.submit(JobRef::from(&TEXTURE_COOK), input, make_priority);
                            request.stage = Stage::Cooking(id);
                        }
                        Some(Sources::Made(Err(error))) => {
                            let error = error.clone();
                            return Step::Failed(request, error);
                        }
                        Some(Sources::Unmade | Sources::Making(_)) => {
                            return Step::Waiting(request);
                        }
                        None => return Step::Failed(request, "its sources are gone".into()),
                    }
                }
            },
            Stage::Cooking(id) => {
                return match jobs.take(id) {
                    None => Step::Waiting(request),
                    Some(Ok(bytes)) => Step::Arrived {
                        request,
                        bytes,
                        made: true,
                    },
                    Some(Err(error)) => Step::Failed(request, error),
                };
            }
        }
    }
}

pub(crate) fn exchange(world: &mut World) {
    let cooking = world
        .get_resource::<CookMode>()
        .is_some_and(|mode| mode.cooking);
    let now = world
        .get_resource::<crate::FrameTime>()
        .map_or(0.0, |time| time.real_elapsed);
    let asking = {
        let mut records = world.resource_mut::<RecordExchange>();
        records.ask_textures(cooking);
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
                    made,
                } => {
                    let key = request.key();
                    if made {
                        let kind = match request.make {
                            Make::Here(_) => None,
                            Make::Texture { .. } => Some(JobRef::from(&TEXTURE_COOK)),
                        };
                        jobs.keep(key, kind, bytes.to_vec());
                    }
                    let deliver = request.deliver.take();
                    let held = if !made && verifying {
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
