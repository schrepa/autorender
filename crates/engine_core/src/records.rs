use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{AssetLibrary, TextureHandle};
use engine_settings::Error;
pub use engine_textures::{
    Family as TextureFamily, Source as TextureSource, Usage as TextureUsage,
};

use crate::jobs::record_key;
use crate::{
    CookMode, JobExchange, JobFailures, JobId, JobKind, JobRef, RecordKey, TEXTURE_COOK,
    URGENT_JOBS, VIEW_JOBS, job_priority,
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

struct Request {
    key: RecordKey,
    make: Make,
    deliver: Option<DeliverRecord>,
    stage: Stage,
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
    requests: Vec<Request>,
    sets: Vec<AskedSet>,
    batches: Vec<Batch>,
    next_batch: u64,
    failures: JobFailures,
    frame: u64,
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
        self.requests.push(Request {
            key,
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

    pub fn loading(&self) -> bool {
        let unasked = self.sampling.is_some_and(|sampling| {
            let shown = family_slot(sampling.family());
            self.sets.iter().any(|asked| !asked.asked[shown])
        });
        unasked
            || self
                .requests
                .iter()
                .any(|request| request.deliver.is_some())
    }

    pub fn pending(&self) -> usize {
        self.requests.len()
    }

    pub fn failures(&self) -> (u64, Option<&str>) {
        (
            self.failures.count,
            self.failures.last.as_ref().map(|(_, error)| error.as_str()),
        )
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
                    self.requests.push(Request {
                        key,
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
    Waiting(Request),
    Arrived {
        request: Request,
        bytes: Arc<[u8]>,
        made: bool,
    },
    Failed(RecordKey, String),
}

fn step(
    records: &mut RecordExchange,
    mut request: Request,
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
                let id = jobs.fetch(kind, request.key, job_priority(URGENT_JOBS, place));
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
                        return Step::Failed(request.key, "it was made already".into());
                    };
                    return match make() {
                        Ok(bytes) => Step::Arrived {
                            request,
                            bytes: Arc::from(bytes),
                            made: true,
                        },
                        Err(error) => Step::Failed(request.key, error),
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
                            return Step::Failed(request.key, error.clone());
                        }
                        Some(Sources::Unmade | Sources::Making(_)) => {
                            return Step::Waiting(request);
                        }
                        None => return Step::Failed(request.key, "its sources are gone".into()),
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
                    Some(Err(error)) => Step::Failed(request.key, error),
                };
            }
        }
    }
}

pub(crate) fn exchange(world: &mut World) {
    let cooking = world
        .get_resource::<CookMode>()
        .is_some_and(|mode| mode.cooking);
    let requests = {
        let mut records = world.resource_mut::<RecordExchange>();
        records.frame += 1;
        records.ask_textures(cooking);
        std::mem::take(&mut records.requests)
    };
    if requests.is_empty() {
        return;
    }
    let mut waiting = Vec::with_capacity(requests.len());
    let mut arrived = Vec::new();
    let mut failed = Vec::new();
    world.resource_scope(|world, mut jobs: Mut<JobExchange>| {
        let verifying = jobs.verifying();
        let mut records = world.resource_mut::<RecordExchange>();
        for (place, request) in requests.into_iter().enumerate() {
            match step(&mut records, request, (&mut *jobs, place as u32)) {
                Step::Waiting(request) => waiting.push(request),
                Step::Arrived {
                    mut request,
                    bytes,
                    made,
                } => {
                    let key = request.key;
                    if made {
                        let kind = match request.make {
                            Make::Here(_) => None,
                            Make::Texture { .. } => Some(JobRef::from(&TEXTURE_COOK)),
                        };
                        jobs.keep(key, kind, bytes.to_vec());
                    }
                    let deliver = request.deliver.take();
                    if !made && verifying {
                        waiting.push(request);
                    }
                    arrived.push((key, deliver, bytes));
                }
                Step::Failed(key, error) => failed.push(format!("record {}: {error}", key.hex())),
            }
        }
    });
    for (key, deliver, bytes) in arrived {
        if let Some(deliver) = deliver
            && let Err(error) = deliver(world, &bytes)
        {
            failed.push(format!("record {}: {error}", key.hex()));
        }
    }
    let mut records = world.resource_mut::<RecordExchange>();
    let frame = records.frame;
    for error in failed {
        records.failures.record(frame, error);
    }
    waiting.append(&mut records.requests);
    records.requests = waiting;
    let RecordExchange {
        requests, batches, ..
    } = &mut *records;
    batches.retain(|batch| {
        requests.iter().any(
            |request| matches!(request.make, Make::Texture { batch: id, .. } if id == batch.id),
        )
    });
}
