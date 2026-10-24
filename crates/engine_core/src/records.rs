//! Records game code makes itself: content a world builds on the CPU,
//! such as the meshes and textures a scene's generators make and the
//! terrains' macro colors. Each request names its record's key, what makes
//! its bytes and what they become. Hosts hand the record back from the
//! cook's packs or what they kept; where none holds it, it is made and the
//! host keeps it, so the bytes are the record wherever they came from and a
//! cooking session's pack takes them. A host verifying its records has each
//! record it handed back made again, and compares what it keeps. Each
//! request's delivery then puts the bytes where they go: a mesh or texture
//! handle the asset library reserved for them
//! ([`AssetLibrary::reserve_mesh`], [`AssetLibrary::reserve_texture`]),
//! which names nothing until it is filled, so what reads it meanwhile finds
//! it missing.
//!
//! Textures come in sets made together ([`TextureSet`]): each texture is a
//! record of the family of compressed formats the device samples (BC7 for
//! desktops, ASTC 4 × 4 for phones and Apple's GPUs: `engine_textures`),
//! BC7's decompressed into RGBA8 levels on devices sampling neither. A
//! set's sources are made once for the records of it asked for together
//! that no host holds: by code here, or by a worker job ([`MakeSources`]).
//! Each record is then cooked from its source on a worker, as the cook does
//! (the job [`engine_textures::COOK_JOB`]), so no frame waits for an
//! encoder. An engine no renderer serves samples nothing and asks for no
//! texture. While a session cooks ([`Cooking`]) every family's record of
//! every texture is asked for, and the records still to come count as left.
//!
//! Requests are exchanged twice a frame, before its presentation and before
//! the world follows the camera, so frames stepped without drawing and
//! paused frames both take them. Requests made before a host serves the
//! engine wait for the next frame: the host answers them from then on, and
//! an engine no host serves makes every one in place at once. Hosts keep
//! preparing while a record a delivery waits for is still to come.

use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{AssetError, AssetLibrary, TextureHandle};
pub use engine_textures::{
    Family as TextureFamily, Source as TextureSource, Usage as TextureUsage,
};

use crate::{Cooking, JobFailures, JobId, Jobs, RecordKey, URGENT_JOBS, VIEW_JOBS, job_priority};

/// The kind every texture record's key names.
const TEXTURE_RECORD: &str = "texture";

/// The code making a record's bytes, or why it could not.
type MakeRecord = Box<dyn FnOnce() -> Result<Vec<u8>, String> + Send + Sync>;
/// What a record's bytes become in the world, or why they could not.
type DeliverRecord = Box<dyn FnOnce(&mut World, &[u8]) -> Result<(), String> + Send + Sync>;

/// What a device samples of cooked textures: a family's records, or BC7's
/// decompressed where it samples no compressed family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureSampling {
    Family(TextureFamily),
    Decompressed,
}

impl TextureSampling {
    /// The family whose records it reads.
    pub fn family(self) -> TextureFamily {
        match self {
            Self::Family(family) => family,
            Self::Decompressed => TextureFamily::Bc7,
        }
    }
}

/// Makes a set's sources here, one per texture in the set's order, or says
/// why it cannot.
pub type MakeTextures = Arc<dyn Fn() -> Result<Vec<TextureSource>, String> + Send + Sync>;

/// What makes a set's sources where a record of it is missing, or a
/// verifying host wants one made again: code here ([`MakeTextures`]); or a
/// worker job of `kind` on `input` whose output is the RGBA8 texels, rows
/// first, of the set's one texture, `size` texels along x and y.
#[derive(Clone)]
pub enum MakeSources {
    Here(MakeTextures),
    Job {
        kind: &'static str,
        input: Vec<u8>,
        size: [u32; 2],
    },
}

/// Textures made together: by code hashing to `source` (its crate's
/// `SOURCE`), named `name`; each texture's name and how its texels read;
/// and what makes their sources where a record is missing.
#[derive(Clone)]
pub struct TextureSet {
    pub source: [u8; 32],
    pub name: String,
    pub textures: Vec<(String, TextureUsage)>,
    pub make: MakeSources,
}

impl TextureSet {
    /// The key of texture `index`'s record of `family`: its set's code and
    /// name, its own name, the family and the cook's code
    /// (`engine_textures::SOURCE`).
    fn key(&self, index: usize, family: TextureFamily) -> RecordKey {
        let mut identity =
            format!("{}/{}/{}", self.name, self.textures[index].0, family.name()).into_bytes();
        identity.extend_from_slice(&engine_textures::SOURCE);
        RecordKey::new(&self.source, TEXTURE_RECORD, &identity)
    }
}

/// How a record no host holds is made, or made again for a verifying host:
/// by code here, at once (taken once it ran); or a texture cooked on a
/// worker from the sources of `batch`, the records of a set asked for
/// together, its `index` in the set, for `family`.
enum Make {
    Here(Option<MakeRecord>),
    Texture {
        batch: u64,
        index: usize,
        family: TextureFamily,
    },
}

/// Where a request stands: asked of the host (its lookup's job); to be made
/// (none held it, or the host verifies it); or cooking (its job).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Asking,
    Fetching(JobId),
    Making,
    Cooking(JobId),
}

/// A record asked for and not yet delivered, or being made again for a
/// verifying host: its key, what makes it, what its bytes become (`None`
/// for a record only the cook wants, another family's texture, and once
/// delivered) and where it stands.
struct Request {
    key: RecordKey,
    make: Make,
    deliver: Option<DeliverRecord>,
    stage: Stage,
}

/// A set of textures asked for: the set, the handles reserved for its
/// textures, and whether each family's records (in [`TextureFamily::ALL`]'s
/// order) were asked for.
struct AskedSet {
    set: Arc<TextureSet>,
    handles: Vec<TextureHandle>,
    asked: [bool; 2],
}

/// A set's sources for the records of it asked for together: not made yet,
/// being made by its job, or made (or why not).
enum Sources {
    Unmade,
    Making(JobId),
    Made(Result<Arc<Vec<TextureSource>>, String>),
}

/// The records of a set asked for together, which share its sources: its
/// id, the set and its sources.
struct Batch {
    id: u64,
    set: Arc<TextureSet>,
    sources: Sources,
}

/// Records game code makes itself, asked for and not yet delivered; the
/// texture sets asked for and the sources of their records being made;
/// what the device samples (`None` until a renderer serves the engine);
/// and the records that failed.
#[derive(Resource, Default)]
pub struct Records {
    sampling: Option<TextureSampling>,
    requests: Vec<Request>,
    sets: Vec<AskedSet>,
    batches: Vec<Batch>,
    next_batch: u64,
    failures: JobFailures,
    frame: u64,
}

impl Records {
    /// Hosts: what the device samples; textures load from the next frame.
    pub fn set_sampling(&mut self, sampling: Option<TextureSampling>) {
        self.sampling = sampling;
    }

    /// What the device samples, once a renderer serves the engine.
    pub fn sampling(&self) -> Option<TextureSampling> {
        self.sampling
    }

    /// Asks for the record `key`, which `make` makes where no host holds
    /// it; `deliver` takes its bytes once they are here, whichever way they
    /// came.
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

    /// Asks for `set`'s textures: their handles in `assets`, reserved until
    /// their records are here.
    pub fn textures(
        &mut self,
        assets: &mut AssetLibrary,
        set: TextureSet,
    ) -> Result<Vec<TextureHandle>, AssetError> {
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

    /// Whether a record a delivery waits for is still to come: hosts keep
    /// preparing meanwhile.
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

    /// Records asked for and not yet delivered, or being made again.
    pub fn pending(&self) -> usize {
        self.requests.len()
    }

    /// Records that failed to make or deliver, and the last one's error.
    pub fn failures(&self) -> (u64, Option<&str>) {
        (
            self.failures.count,
            self.failures.last.as_ref().map(|(_, error)| error.as_str()),
        )
    }

    /// Asks for the records of the texture sets' families not asked for
    /// yet: the family the device samples, delivered into the sets'
    /// handles, and while `cooking` every other, for the cook alone. The
    /// records of a set asked for together share its sources. Sets whose
    /// families are all asked for are done.
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
                    self.requests.push(Request {
                        key: asked.set.key(index, family),
                        make: Make::Texture {
                            batch,
                            index,
                            family,
                        },
                        deliver: (family == shown).then(|| deliver_texture(handle, sampling)),
                        stage: Stage::Asking,
                    });
                }
            }
        }
        self.sets
            .retain(|asked| !asked.asked.iter().all(|&done| done));
    }

    /// Steps the sources of batch `id`'s records: made here at once, or by
    /// its job, submitted to `jobs` at `priority` and taken once done.
    fn step_sources(&mut self, id: u64, jobs: &mut Jobs, priority: u32) {
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
                    batch.sources = Sources::Making(jobs.submit(kind, input.clone(), priority));
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
                                pixels,
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

/// The slot of `family` in [`TextureFamily::ALL`]'s order.
fn family_slot(family: TextureFamily) -> usize {
    TextureFamily::ALL
        .iter()
        .position(|&each| each == family)
        .unwrap_or(0)
}

/// What a texture record's bytes become: the texture reserved as `handle`,
/// decompressed where the device samples no compressed family.
fn deliver_texture(handle: TextureHandle, sampling: TextureSampling) -> DeliverRecord {
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
            .fill_texture(handle, texture)
            .map_err(|error| error.to_string())
    })
}

/// What became of a request this exchange: it waits on, its bytes are
/// here (to keep where it was made, to deliver), or it failed.
enum Step {
    Waiting(Request),
    Arrived {
        request: Request,
        bytes: Vec<u8>,
        made: bool,
    },
    Failed(RecordKey, String),
}

/// Steps `request` once with `jobs` (`place` its order among the
/// requests): asks the host, makes what none held (code here at once; a
/// texture's sources, then its cooking on a worker), and hands on what
/// arrived.
fn step(records: &mut Records, mut request: Request, (jobs, place): (&mut Jobs, u32)) -> Step {
    let make_priority = job_priority(VIEW_JOBS, place);
    loop {
        match request.stage {
            Stage::Asking => {
                let id = jobs.fetch(request.key, job_priority(URGENT_JOBS, place));
                request.stage = Stage::Fetching(id);
            }
            Stage::Fetching(id) => match jobs.take(id) {
                None => return Step::Waiting(request),
                // Verifying, it is made again for the host to compare,
                // once its delivery took what the host held.
                Some(Ok(bytes)) => {
                    request.stage = Stage::Making;
                    return Step::Arrived {
                        request,
                        bytes,
                        made: false,
                    };
                }
                // Where no host holds it, it is made and kept.
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
                            bytes,
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
                            let id = jobs.submit(engine_textures::COOK_JOB, input, make_priority);
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

/// Twice a frame, before its presentation and before the world follows the
/// camera ([`crate::WorldSync`]), so frames stepped without drawing and
/// paused frames drawing alone both take them: asks for the texture sets'
/// records not yet asked for, hands the host the records asked for since,
/// makes those it held none of (and, verifying, those it handed back), and
/// delivers what arrived: the host's bytes, or those made, which it keeps.
pub(crate) fn exchange(world: &mut World) {
    let cooking = world.contains_resource::<Cooking>();
    let requests = {
        let mut records = world.resource_mut::<Records>();
        records.frame += 1;
        records.ask_textures(cooking);
        std::mem::take(&mut records.requests)
    };
    if requests.is_empty() {
        if let Some(mut cooking) = world.get_resource_mut::<Cooking>() {
            cooking.left("records", 0);
        }
        return;
    }
    let mut waiting = Vec::with_capacity(requests.len());
    let mut arrived = Vec::new();
    let mut failed = Vec::new();
    world.resource_scope(|world, mut jobs: Mut<Jobs>| {
        let verifying = jobs.verifying();
        let mut records = world.resource_mut::<Records>();
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
                        jobs.keep(key, bytes.clone());
                    }
                    let deliver = request.deliver.take();
                    // Verifying, a record the host handed back is made
                    // again, its delivery done.
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
    let mut records = world.resource_mut::<Records>();
    let frame = records.frame;
    for error in failed {
        records.failures.record(frame, error);
    }
    // Requests the deliveries made wait behind those still waiting.
    waiting.append(&mut records.requests);
    records.requests = waiting;
    // Batches no request draws on any longer are done.
    let Records {
        requests, batches, ..
    } = &mut *records;
    batches.retain(|batch| {
        requests.iter().any(
            |request| matches!(request.make, Make::Texture { batch: id, .. } if id == batch.id),
        )
    });
    let left = records.requests.len() as u64;
    if let Some(mut cooking) = world.get_resource_mut::<Cooking>() {
        cooking.left("records", left);
    }
}
