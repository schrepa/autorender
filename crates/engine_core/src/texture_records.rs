//! Textures that load as records: what the cook made of them for the
//! family of compressed formats the device samples (BC7 for desktops, ASTC
//! 4 × 4 for phones and Apple's GPUs: `engine_textures`), fetched from the
//! host's packs or store, and BC7's decompressed into RGBA8 levels on
//! devices sampling neither. Apps ask for sets of textures made together
//! ([`TextureSet`]): each texture has its handle at once, a stand-in until
//! its record arrives and replaces it in the asset library. A record no
//! host holds is made in place, as the cook makes it: the set's sources
//! from its maker, the app's code, each cooked for the family; the host
//! keeps it. While a session cooks ([`Cooking`]) every family's record of
//! every texture is made, and those left count. Records handed back while
//! the host verifies its records are made again and handed to it, which
//! compares. An engine no renderer serves samples nothing and asks for no
//! record. A record's key holds the hash of the code making it (the set's
//! `source`), the set's name, the texture's and the family's.

use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{AssetError, AssetLibrary, Texture, TextureHandle};
pub use engine_textures::{
    Family as TextureFamily, Source as TextureSource, Usage as TextureUsage,
};

use crate::{Cooking, JobFailures, JobId, Jobs, MISSING_RECORD, RecordKey, URGENT_JOBS};

/// The kind every texture record's key names.
const KIND: &str = "texture";

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
        RecordKey::new(&self.source, KIND, identity.as_bytes())
    }
}

/// Where a texture's record of a family stands: not asked for, asked for,
/// missing from every store (to be made), or here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Record {
    Unasked,
    Fetching(JobId),
    Missing,
    Here,
}

/// A set asked for: its handles, each texture's record of each family
/// (in [`TextureFamily::ALL`]'s order), which of them a verifying host
/// wants made again, and which textures show their record.
struct Asked {
    set: TextureSet,
    handles: Vec<TextureHandle>,
    records: [Vec<Record>; 2],
    verify: [Vec<bool>; 2],
    shown: Vec<bool>,
}

/// The textures apps asked for as records, and what the device samples
/// (`None` until a renderer serves the engine).
#[derive(Resource, Default)]
pub struct TextureRecords {
    sampling: Option<TextureSampling>,
    sets: Vec<Asked>,
    failures: JobFailures,
    frame: u64,
    made: u64,
}

impl TextureRecords {
    /// Hosts: what the device samples; textures load from the next frame.
    pub fn set_sampling(&mut self, sampling: Option<TextureSampling>) {
        self.sampling = sampling;
    }

    /// What the device samples, once a renderer serves the engine.
    pub fn sampling(&self) -> Option<TextureSampling> {
        self.sampling
    }

    /// Asks for `set`'s textures: their handles in `assets`, stand-ins
    /// until their records arrive.
    pub fn request(
        &mut self,
        assets: &mut AssetLibrary,
        set: TextureSet,
    ) -> Result<Vec<TextureHandle>, AssetError> {
        let handles = set
            .textures
            .iter()
            .map(|&(_, usage)| assets.insert_texture(stand_in(usage)?))
            .collect::<Result<Vec<_>, _>>()?;
        let count = set.textures.len();
        self.sets.push(Asked {
            set,
            handles: handles.clone(),
            records: [vec![Record::Unasked; count], vec![Record::Unasked; count]],
            verify: [vec![false; count], vec![false; count]],
            shown: vec![false; count],
        });
        Ok(handles)
    }

    /// Whether a texture the device samples still shows its stand-in:
    /// hosts keep preparing meanwhile.
    pub fn loading(&self) -> bool {
        self.sampling.is_some() && self.sets.iter().any(|asked| asked.shown.contains(&false))
    }

    /// Records made in place, the host lacking them.
    pub fn made(&self) -> u64 {
        self.made
    }

    /// How many sets failed to make or read a record, and the last one's
    /// error.
    pub fn failure(&self) -> (u64, Option<&str>) {
        (
            self.failures.count,
            self.failures.last.as_ref().map(|(_, error)| error.as_str()),
        )
    }

    /// Records a session cooking has left to make: every family's of every
    /// texture.
    fn left(&self) -> u64 {
        self.sets
            .iter()
            .flat_map(|asked| asked.records.iter().flatten())
            .filter(|record| **record != Record::Here)
            .count() as u64
    }

    /// A frame's work for the device sampling `sampling`: every family's
    /// records asked for while `cooking`, the device's otherwise; records
    /// back shown, those missing (and those a verifying host wants made
    /// again) made in place and kept.
    fn step(
        &mut self,
        sampling: TextureSampling,
        cooking: bool,
        jobs: &mut Jobs,
        assets: &mut AssetLibrary,
    ) {
        self.frame += 1;
        let frame = self.frame;
        let shown = sampling.family();
        for asked in &mut self.sets {
            for (slot, family) in TextureFamily::ALL.into_iter().enumerate() {
                if !cooking && family != shown {
                    continue;
                }
                for index in 0..asked.handles.len() {
                    let record = &mut asked.records[slot][index];
                    if *record == Record::Unasked {
                        let key = asked.set.key(index, family);
                        *record = Record::Fetching(jobs.fetch(key, crate::job_priority(URGENT_JOBS, 0)));
                    }
                    let Record::Fetching(id) = *record else {
                        continue;
                    };
                    let Some(output) = jobs.take(id) else {
                        continue;
                    };
                    let texture = match output {
                        Ok(bytes) => engine_textures::decode(&bytes),
                        Err(error) => Err(error),
                    };
                    match texture {
                        Ok(texture) => {
                            *record = Record::Here;
                            asked.verify[slot][index] = jobs.verifying();
                            if family == shown {
                                show(asked, index, texture, sampling, assets, (&mut self.failures, frame));
                            }
                        }
                        Err(error) => {
                            if error != MISSING_RECORD {
                                self.failures.record(frame, format!("{}: {error}", asked.set.name));
                            }
                            *record = Record::Missing;
                        }
                    }
                }
            }
            let wanted = |slot: usize, index: usize| {
                asked.records[slot][index] == Record::Missing || asked.verify[slot][index]
            };
            if !(0..2).any(|slot| (0..asked.handles.len()).any(|index| wanted(slot, index))) {
                continue;
            }
            self.made += make(asked, sampling, jobs, assets, (&mut self.failures, frame));
        }
    }
}

/// Makes `asked`'s missing records and those its host verifies: its
/// sources once, each cooked for its family and handed to `jobs` to keep;
/// shows those of the family `sampling` reads. The records made.
fn make(
    asked: &mut Asked,
    sampling: TextureSampling,
    jobs: &mut Jobs,
    assets: &mut AssetLibrary,
    (failures, frame): (&mut JobFailures, u64),
) -> u64 {
    let sources = match (asked.set.make)() {
        Ok(sources) if sources.len() == asked.handles.len() => sources,
        Ok(sources) => {
            failures.record(
                frame,
                format!(
                    "{}: {} sources made for {} textures",
                    asked.set.name,
                    sources.len(),
                    asked.handles.len()
                ),
            );
            give_up(asked);
            return 0;
        }
        Err(error) => {
            failures.record(frame, format!("{}: {error}", asked.set.name));
            give_up(asked);
            return 0;
        }
    };
    let mut made = 0;
    for (slot, family) in TextureFamily::ALL.into_iter().enumerate() {
        for (index, source) in sources.iter().enumerate() {
            let missing = asked.records[slot][index] == Record::Missing;
            if !missing && !asked.verify[slot][index] {
                continue;
            }
            asked.verify[slot][index] = false;
            asked.records[slot][index] = Record::Here;
            match engine_textures::cook(source, family) {
                Ok(texture) => {
                    jobs.keep(asked.set.key(index, family), engine_textures::record(&texture));
                    made += 1;
                    if missing && family == sampling.family() {
                        show(asked, index, texture, sampling, assets, (&mut *failures, frame));
                    }
                }
                Err(error) => {
                    let name = &asked.set.textures[index].0;
                    failures.record(frame, format!("{}/{name}: {error}", asked.set.name));
                }
            }
        }
    }
    made
}

/// Leaves `asked`'s missing records unmade: its textures keep their
/// stand-ins, and it no longer loads.
fn give_up(asked: &mut Asked) {
    for records in &mut asked.records {
        for record in records.iter_mut().filter(|record| **record == Record::Missing) {
            *record = Record::Here;
        }
    }
    for verify in &mut asked.verify {
        verify.fill(false);
    }
    asked.shown.fill(true);
}

/// Replaces texture `index`'s stand-in with `texture`, its record of the
/// family `sampling` reads (decompressed where it reads BC7's so).
fn show(
    asked: &mut Asked,
    index: usize,
    texture: Texture,
    sampling: TextureSampling,
    assets: &mut AssetLibrary,
    (failures, frame): (&mut JobFailures, u64),
) {
    asked.shown[index] = true;
    let texture = match sampling {
        TextureSampling::Family(_) => Ok(texture),
        TextureSampling::Decompressed => engine_textures::decompress(&texture),
    };
    if let Err(error) =
        texture.and_then(|texture| assets.replace_texture(asked.handles[index], texture))
    {
        let name = &asked.set.textures[index].0;
        failures.record(frame, format!("{}/{name}: {error}", asked.set.name));
    }
}

/// What a texture shows until its record arrives: 4 × 4 texels of white,
/// a flat normal, or full occlusion and roughness without metal.
fn stand_in(usage: TextureUsage) -> Result<Texture, AssetError> {
    let texel = match usage {
        TextureUsage::Color => [255; 4],
        TextureUsage::Normal => [128, 128, 255, 255],
        TextureUsage::Data => [255, 255, 0, 255],
    };
    Texture::new(4, 4, texel.repeat(16))
}

/// The frame's texture records ([`crate::ViewSyncSystems::Stream`]).
pub(crate) fn load_textures(world: &mut World) {
    let Some(mut records) = world.remove_resource::<TextureRecords>() else {
        return;
    };
    if let Some(sampling) = records.sampling {
        let cooking = world.contains_resource::<Cooking>();
        world.resource_scope(|world, mut jobs: Mut<Jobs>| {
            let mut assets = world.resource_mut::<AssetLibrary>();
            records.step(sampling, cooking, &mut jobs, &mut assets);
        });
        let left = records.left();
        if let Some(mut cooking) = world.get_resource_mut::<Cooking>() {
            cooking.left("textures", left);
        }
    }
    world.insert_resource(records);
}
