//! Records game code makes itself: content a world builds on the CPU with
//! code the job workers lack, such as the meshes and textures a scene's
//! generators make. Each request names its record's key, the code making
//! its bytes and what they become. Hosts hand the record back from the
//! cook's packs or what they kept; where none holds it, its code makes it
//! here, in place, and the host keeps it, so the bytes are the record
//! wherever they came from and a cooking session's pack takes them. A host
//! verifying its records has the code make each record it handed back
//! again, and compares what it keeps. Each request's delivery then puts the
//! bytes where they go: a mesh or texture handle the asset library reserved
//! for them ([`AssetLibrary::reserve_mesh`], [`AssetLibrary::reserve_texture`]),
//! which names nothing until it is filled, so what reads it meanwhile finds
//! it missing.
//!
//! Textures come in sets made together ([`TextureSet`]): each texture is a
//! record of the family of compressed formats the device samples (BC7 for
//! desktops, ASTC 4 × 4 for phones and Apple's GPUs: `engine_textures`),
//! BC7's decompressed into RGBA8 levels on devices sampling neither. A
//! set's sources are made once for the records of it asked for together
//! that no host holds. An engine no renderer serves samples nothing and
//! asks for no texture. While a session cooks ([`Cooking`]) every family's
//! record of every texture is asked for, and the records still to come
//! count as left.
//!
//! Requests are exchanged twice a frame, before its presentation and before
//! the world follows the camera, so frames stepped without drawing and
//! paused frames both take them. Requests made before a host serves the
//! engine wait for the next frame: the host answers them from then on, and
//! an engine no host serves makes every one in place at once. Hosts keep
//! preparing while a record a delivery waits for is still to come.

use std::sync::{Arc, OnceLock};

use bevy_ecs::prelude::*;
use engine_assets::{AssetError, AssetLibrary, TextureHandle};
pub use engine_textures::{
    Family as TextureFamily, Source as TextureSource, Usage as TextureUsage,
};

use crate::{Cooking, JobFailures, JobId, Jobs, RecordKey, URGENT_JOBS, job_priority};

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

/// Makes a set's sources, one per texture in the set's order, or says why
/// it cannot.
pub type MakeTextures = Arc<dyn Fn() -> Result<Vec<TextureSource>, String> + Send + Sync>;

/// Textures made together: by code hashing to `source` (its crate's
/// `SOURCE`), named `name`; each texture's name and how its texels read;
/// and what makes their sources where a record is missing.
#[derive(Clone)]
pub struct TextureSet {
    pub source: [u8; 32],
    pub name: String,
    pub textures: Vec<(String, TextureUsage)>,
    pub make: MakeTextures,
}

impl TextureSet {
    /// The key of texture `index`'s record of `family`.
    fn key(&self, index: usize, family: TextureFamily) -> RecordKey {
        let identity = format!("{}/{}/{}", self.name, self.textures[index].0, family.name());
        RecordKey::new(&self.source, TEXTURE_RECORD, identity.as_bytes())
    }
}

/// A record asked for and not yet delivered: its key, what makes it, what
/// its bytes become (`None` for a record only the cook wants, another
/// family's texture), and the host's lookup under way.
struct Request {
    key: RecordKey,
    make: MakeRecord,
    deliver: Option<DeliverRecord>,
    job: Option<JobId>,
}

/// A set of textures asked for: the set, the handles reserved for its
/// textures, and whether each family's records (in [`TextureFamily::ALL`]'s
/// order) were asked for.
struct AskedSet {
    set: Arc<TextureSet>,
    handles: Vec<TextureHandle>,
    asked: [bool; 2],
}

/// A set's sources, made at most once for the records of it asked for
/// together.
type Sources = Arc<OnceLock<Result<Vec<TextureSource>, String>>>;

/// Records game code makes itself, asked for and not yet delivered; the
/// texture sets asked for; what the device samples (`None` until a
/// renderer serves the engine); and the records that failed.
#[derive(Resource, Default)]
pub struct Records {
    sampling: Option<TextureSampling>,
    requests: Vec<Request>,
    sets: Vec<AskedSet>,
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
            make: Box::new(make),
            deliver: Some(Box::new(deliver)),
            job: None,
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

    /// Records asked for and not yet delivered.
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
    /// handles, and while `cooking` every other, for the cook alone. Sets
    /// whose families are all asked for are done.
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
            let sources = Sources::default();
            for (slot, family) in families {
                asked.asked[slot] = true;
                for (index, &handle) in asked.handles.iter().enumerate() {
                    self.requests.push(Request {
                        key: asked.set.key(index, family),
                        make: make_texture(&asked.set, &sources, index, family),
                        deliver: (family == shown).then(|| deliver_texture(handle, sampling)),
                        job: None,
                    });
                }
            }
        }
        self.sets
            .retain(|asked| !asked.asked.iter().all(|&done| done));
    }
}

/// The slot of `family` in [`TextureFamily::ALL`]'s order.
fn family_slot(family: TextureFamily) -> usize {
    TextureFamily::ALL
        .iter()
        .position(|&each| each == family)
        .unwrap_or(0)
}

/// What makes texture `index` of `set` as a record of `family`: the set's
/// sources, made once into `sources` for the records asked for together,
/// its own cooked for the family.
fn make_texture(
    set: &Arc<TextureSet>,
    sources: &Sources,
    index: usize,
    family: TextureFamily,
) -> MakeRecord {
    let (set, sources) = (Arc::clone(set), Arc::clone(sources));
    Box::new(move || {
        let name = format!("{}/{}", set.name, set.textures[index].0);
        let source = match sources.get_or_init(|| (set.make)()) {
            Ok(made) if made.len() == set.textures.len() => &made[index],
            Ok(made) => {
                return Err(format!(
                    "{name}: {} sources made for {} textures",
                    made.len(),
                    set.textures.len()
                ));
            }
            Err(error) => return Err(format!("{name}: {error}")),
        };
        engine_textures::cook(source, family)
            .map(|texture| engine_textures::record(&texture))
            .map_err(|error| format!("{name}: {error}"))
    })
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

/// Twice a frame, before its presentation and before the world follows the
/// camera ([`crate::WorldSync`]), so frames stepped without drawing and
/// paused frames drawing alone both take them: asks for the texture sets'
/// records not yet asked for, hands the host the records asked for since,
/// and delivers those it answered: its bytes (made again and kept where
/// the host verifies them), or where it held none (or failed) the bytes
/// their code makes here, which it keeps.
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
    let verifying = {
        let mut jobs = world.resource_mut::<Jobs>();
        for (place, mut request) in requests.into_iter().enumerate() {
            let id = *request.job.get_or_insert_with(|| {
                jobs.fetch(request.key, job_priority(URGENT_JOBS, place as u32))
            });
            match jobs.take(id) {
                Some(output) => arrived.push((request, output)),
                None => waiting.push(request),
            }
        }
        jobs.verifying()
    };
    let mut failed = Vec::new();
    for (request, output) in arrived {
        let Request {
            key, make, deliver, ..
        } = request;
        let bytes = match output {
            // Verifying, its code makes it again for the host to compare.
            Ok(bytes) => {
                if verifying {
                    match make() {
                        Ok(made) => world.resource_mut::<Jobs>().keep(key, made),
                        Err(error) => failed.push(format!("record {}: {error}", key.hex())),
                    }
                }
                Ok(bytes)
            }
            // Where no host holds it, it is made here and kept.
            Err(_) => make().inspect(|bytes| {
                world.resource_mut::<Jobs>().keep(key, bytes.clone());
            }),
        };
        let delivered = bytes.and_then(|bytes| match deliver {
            Some(deliver) => deliver(world, &bytes),
            None => Ok(()),
        });
        if let Err(error) = delivered {
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
    let left = records.requests.len() as u64;
    if let Some(mut cooking) = world.get_resource_mut::<Cooking>() {
        cooking.left("records", left);
    }
}
