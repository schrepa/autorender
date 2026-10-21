//! Each terrain's macro colors ([`engine_terrain::macro_colors`]): the
//! tint its ground takes and how lush its plants grow, from what its bakes
//! found, compressed for the family of formats the device samples. A record
//! per family, kept by the content of the terrain's recipe and the code
//! making it: the host hands it back from the cook's packs or its store.
//! Where none holds it, a worker makes the colors (a job whose output is no
//! record) and then the record, cooking them for the family
//! ([`engine_textures::COOK_JOB`]), which the host keeps. While a session
//! cooks every family's record is made; a verifying host has those it
//! handed back made again, and compares. A recipe that changes asks anew;
//! the colors before stay until the new ones arrive.

use super::*;
use crate::{MISSING_RECORD, TextureFamily, TextureSampling, TextureSource, TextureUsage};
use engine_assets::Texture;
use engine_terrain::jobs::MacroJob;

/// The kind macro color records' keys name.
const KIND: &str = "terrain_macro_colors";

/// Where a family's record stands: not asked for, asked for, missing from
/// every store (to be made), being made, or here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Record {
    #[default]
    Unasked,
    Fetching(JobId),
    Missing,
    Cooking(JobId),
    Here,
}

/// A terrain's macro colors and its asking for them: the job its colors
/// come from; each family's record (in [`TextureFamily::ALL`]'s order) and
/// which of those a verifying host wants made again; the colors' own job
/// while it runs, and its texels until every record they make is asked
/// for; the map shown, in the family the device samples; and the frame a
/// job last failed at.
#[derive(Default)]
pub(super) struct MacroColors {
    job: Option<MacroJob>,
    records: [Record; 2],
    verify: [bool; 2],
    colors_job: Option<JobId>,
    texels: Option<Vec<u8>>,
    pub(super) made: Option<MacroMap>,
    failed: Option<u64>,
}

impl MacroColors {
    /// Records a session cooking has left to make: each family's not here
    /// yet, both while the colors are not asked for.
    pub(super) fn left(&self) -> u64 {
        if self.job.is_none() {
            return TextureFamily::ALL.len() as u64;
        }
        self.records
            .iter()
            .filter(|record| **record != Record::Here)
            .count() as u64
    }

    /// Whether a record of it, or its colors, is asked for and not here
    /// yet.
    pub(super) fn asking(&self) -> bool {
        self.jobs().next().is_some()
    }

    /// Its jobs in flight.
    pub(super) fn jobs(&self) -> impl Iterator<Item = JobId> + '_ {
        self.records
            .iter()
            .filter_map(|record| match *record {
                Record::Fetching(id) | Record::Cooking(id) => Some(id),
                Record::Unasked | Record::Missing | Record::Here => None,
            })
            .chain(self.colors_job)
    }

    /// Stops its jobs in flight, through `jobs`.
    fn cancel(&mut self, jobs: &mut Jobs) {
        for id in self.jobs().collect::<Vec<_>>() {
            jobs.cancel(id);
        }
        self.colors_job = None;
    }
}

/// Texels along `settings`' macro color maps' sides: their resolution, in
/// whole blocks.
fn macro_texels(settings: &TerrainSettings) -> u32 {
    let block = engine_terrain::macro_colors::MACRO_BLOCK;
    settings
        .macro_resolution
        .next_multiple_of(block)
        .min(engine_terrain::macro_colors::MAX_MACRO_TEXELS)
}

/// What `job`'s records are made from in every family: the job and the
/// cook's code (the colors' is the terrain crate's, which keys them).
fn identity(job: &MacroJob) -> Vec<u8> {
    let mut identity = job.identity();
    identity.extend_from_slice(&engine_textures::SOURCE);
    identity
}

/// The key of `job`'s record of `family`.
fn key(job: &MacroJob, family: TextureFamily) -> RecordKey {
    let mut identity = identity(job);
    identity.extend_from_slice(family.name().as_bytes());
    RecordKey::new(&engine_terrain::SOURCE, KIND, &identity)
}

/// `texture`, a record of the family `sampling` reads, as the device shows
/// it: decompressed where it samples no compressed family.
fn shown(texture: Texture, sampling: TextureSampling) -> Result<Texture, String> {
    match sampling {
        TextureSampling::Family(_) => Ok(texture),
        TextureSampling::Decompressed => {
            engine_textures::decompress(&texture).map_err(|error| error.to_string())
        }
    }
}

/// Takes `state`'s macro colors as their records arrive, once its recipe is
/// shared and `settings` draw terrain: the record of the family the device
/// samples (`sampling`, none without a renderer), every family's while
/// `cooking`. Records no host holds, and those a verifying host handed
/// back, are made: the colors by a job within `slots`, then each record by
/// a job cooking them; a while after a job failed, again.
pub(super) fn take_macro_colors(
    state: &mut TerrainState,
    settings: &TerrainSettings,
    (sampling, cooking): (Option<TextureSampling>, bool),
    slots: &mut JobBudget<'_>,
    frame: u64,
) {
    let (true, Some(sampling), Some(recipe)) = (settings.enabled, sampling, state.recipe.as_ref())
    else {
        return;
    };
    let job = MacroJob {
        recipe: recipe.key.clone(),
        level: lighting::lighting_level(&state.terrain.pyramid, settings.pinned_levels),
        size: macro_texels(settings),
    };
    let colors = &mut state.macro_colors;
    if colors.job.as_ref() != Some(&job) {
        colors.cancel(slots.jobs);
        colors.records = [Record::Unasked; 2];
        colors.verify = [false; 2];
        colors.texels = None;
        colors.failed = None;
        colors.job = Some(job.clone());
    }
    let priority = job_priority(VIEW_JOBS, 0);
    let showing = sampling.family();
    for (slot, family) in TextureFamily::ALL.into_iter().enumerate() {
        let record = colors.records[slot];
        let output = match record {
            Record::Unasked if cooking || family == showing => {
                colors.records[slot] =
                    Record::Fetching(slots.jobs.fetch(key(&job, family), priority));
                continue;
            }
            Record::Fetching(id) | Record::Cooking(id) => match slots.jobs.take(id) {
                Some(output) => output,
                None => continue,
            },
            Record::Unasked | Record::Missing | Record::Here => continue,
        };
        let texture = output.and_then(|bytes| engine_textures::decode(&bytes));
        match texture {
            Ok(texture) => {
                colors.records[slot] = Record::Here;
                // Records handed back are made again for a verifying host.
                colors.verify[slot] =
                    matches!(record, Record::Fetching(_)) && slots.jobs.verifying();
                if family == showing {
                    match shown(texture, sampling) {
                        Ok(texture) => {
                            colors.made = Some(MacroMap {
                                texture: Arc::new(texture),
                                identity: content_hash(&identity(&job)),
                            });
                        }
                        Err(error) => {
                            state
                                .failures
                                .record(frame, format!("macro colors: {error}"));
                        }
                    }
                }
            }
            Err(error) if error == MISSING_RECORD => colors.records[slot] = Record::Missing,
            Err(error) => {
                state
                    .failures
                    .record(frame, format!("macro colors: {error}"));
                colors.records[slot] = Record::Missing;
                if matches!(record, Record::Cooking(_)) {
                    colors.failed = Some(frame);
                }
            }
        }
    }
    make(
        colors,
        &job,
        (slots, priority),
        (&mut state.failures, frame),
    );
}

/// Makes `colors`' records missing and those a verifying host wants made
/// again: the colors of `job` once, by a job admitted by `slots` at
/// `priority` (none a while after one failed), then each record by a job
/// cooking them, which the host keeps.
fn make(
    colors: &mut MacroColors,
    job: &MacroJob,
    (slots, priority): (&mut JobBudget<'_>, u32),
    (failures, frame): (&mut JobFailures, u64),
) {
    let wanted = |colors: &MacroColors, slot: usize| {
        colors.records[slot] == Record::Missing || colors.verify[slot]
    };
    if !(0..TextureFamily::ALL.len()).any(|slot| wanted(colors, slot)) {
        colors.texels = None;
        return;
    }
    if let Some(id) = colors.colors_job
        && let Some(output) = slots.jobs.take(id)
    {
        colors.colors_job = None;
        match output {
            Ok(texels) => colors.texels = Some(texels),
            Err(error) => {
                failures.record(frame, format!("macro colors: {error}"));
                colors.failed = Some(frame);
            }
        }
    }
    if colors.failed.is_some_and(|at| frame < at + RETRY_AFTER) {
        return;
    }
    let Some(texels) = colors.texels.as_ref() else {
        if colors.colors_job.is_none() && slots.take(priority) {
            colors.colors_job = Some(slots.jobs.submit(
                MacroJob::KIND.name(),
                job.identity(),
                priority,
            ));
        }
        return;
    };
    for (slot, family) in TextureFamily::ALL.into_iter().enumerate() {
        if !wanted(colors, slot) || !slots.take(priority) {
            continue;
        }
        let source = TextureSource {
            width: job.size,
            height: job.size,
            pixels: texels.clone(),
            usage: TextureUsage::Data,
        };
        let input = engine_textures::cook_input(&source, family);
        colors.records[slot] = Record::Cooking(slots.jobs.submit_record(
            engine_textures::COOK_JOB,
            input,
            priority,
            key(job, family),
        ));
        colors.verify[slot] = false;
    }
    // Every record asked for: the colors are no longer needed.
    if !(0..TextureFamily::ALL.len()).any(|slot| wanted(colors, slot)) {
        colors.texels = None;
    }
}
