//! Each terrain's macro colors ([`engine_terrain::macro_colors`]): the
//! tint its ground takes and how lush its plants grow, from what its bakes
//! found. They are a texture set of one texture like every texture record
//! game code asks for ([`crate::Records`]): its source made by a worker job
//! (the colors of the terrain's recipe at its finest pinned level), its
//! records cooked for the family of compressed formats the device samples,
//! every family while a session cooks, hosts handing them back from the
//! cook's packs or their stores and, verifying, having them made again. A
//! record is kept by the content of the terrain's recipe (its key), the
//! level and the size, and the code making and cooking it. A recipe that
//! changes asks anew; the colors before stay until the new ones arrive, then
//! leave the asset library.

use super::*;
use crate::{MakeSources, Records, TextureSet, TextureUsage};
use bevy_ecs::prelude::Mut;
use engine_assets::TextureHandle;
use engine_terrain::jobs::MacroJob;

/// The colors asked of the records: their job, and the handle their records
/// fill with their identity (none where the asset library had no room).
struct Asked {
    job: MacroJob,
    held: Option<(TextureHandle, u128)>,
}

/// A terrain's macro colors: those streaming wants (none before its recipe
/// is shared, or with terrain off), those last asked for, and those shown,
/// the last filled, with their job.
#[derive(Default)]
pub(super) struct MacroColors {
    pub(super) wanted: Option<MacroJob>,
    asked: Option<Asked>,
    shown: Option<(MacroJob, TextureHandle, u128)>,
}

impl MacroColors {
    /// Whether colors are asked for and not yet shown.
    pub(super) fn asking(&self) -> bool {
        self.asked.as_ref().is_some_and(|asked| {
            asked.held.is_some()
                && self
                    .shown
                    .as_ref()
                    .is_none_or(|(shown, _, _)| *shown != asked.job)
        })
    }

    /// Whether colors are shown.
    pub(super) fn shown(&self) -> bool {
        self.shown.is_some()
    }

    /// Whether the colors wanted are not shown yet: a session cooking
    /// waits for them.
    pub(super) fn left(&self) -> bool {
        self.wanted.as_ref().is_some_and(|wanted| {
            self.shown
                .as_ref()
                .is_none_or(|(shown, _, _)| shown != wanted)
        })
    }

    /// The textures its colors hold or will, which go with the terrain.
    pub(super) fn handles(&self) -> impl Iterator<Item = TextureHandle> + '_ {
        self.asked
            .iter()
            .filter_map(|asked| asked.held.map(|(handle, _)| handle))
            .chain(self.shown.iter().map(|&(_, handle, _)| handle))
    }

    /// The colors shown as renderers take them, from `assets`.
    pub(super) fn map(&self, assets: Option<&AssetLibrary>) -> Option<MacroMap> {
        let (_, handle, identity) = self.shown.as_ref()?;
        let texture = assets?.texture(*handle)?;
        Some(MacroMap {
            texture: Arc::clone(texture),
            identity: *identity,
        })
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

/// Sets the macro colors `settings` want of `state`: those of its recipe at
/// its finest pinned level, `macro_resolution` texels a side in whole
/// blocks; none before its recipe is shared, or with terrain off, which
/// leaves those shown as they are.
pub(super) fn want_colors(state: &mut TerrainState, settings: &TerrainSettings) {
    let recipe = state.recipe.as_ref().filter(|_| settings.enabled);
    state.whole.colors.wanted = recipe.map(|recipe| MacroJob {
        recipe: recipe.key.clone(),
        level: lighting_level(&state.terrain.pyramid, settings.pinned_levels),
        size: macro_texels(settings),
    });
}

/// The texture set of `job`'s colors: named by the hash of the job, its
/// source the job's texels.
fn colors_set(job: &MacroJob) -> TextureSet {
    let identity = job.identity();
    TextureSet {
        source: engine_terrain::sources::MACRO,
        name: format!("terrain/macro/{:032x}", content_hash(&identity)),
        textures: vec![("colors".to_owned(), TextureUsage::Data)],
        make: MakeSources::Job {
            kind: MacroJob::KIND.name(),
            input: identity,
            size: [job.size, job.size],
        },
    }
}

/// What made `job`'s colors, the same in every family: the job and the
/// cook's code, which records renderers bake from them are kept by.
fn colors_identity(job: &MacroJob) -> u128 {
    let mut identity = job.identity();
    identity.extend_from_slice(&engine_textures::SOURCE);
    content_hash(&identity)
}

impl Terrains {
    /// Asks `records` for the colors each terrain wants that it has not
    /// asked for, their handles reserved in `assets`; shows those filled,
    /// and removes from `assets` the colors replaced or gone with their
    /// terrains once filled (those still to come fill first).
    pub(super) fn ask_colors(&mut self, records: &mut Records, assets: &mut AssetLibrary) {
        let frame = self.frame;
        for state in &mut self.entries {
            let colors = &mut state.whole.colors;
            if let Some(Asked {
                job,
                held: Some((handle, identity)),
            }) = &colors.asked
                && assets.texture(*handle).is_some()
                && colors
                    .shown
                    .as_ref()
                    .is_none_or(|(_, shown, _)| shown != handle)
            {
                let shown = (job.clone(), *handle, *identity);
                if let Some((_, replaced, _)) = colors.shown.replace(shown) {
                    self.colors_gone.push(replaced);
                }
            }
            let Some(job) = colors.wanted.clone() else {
                continue;
            };
            if colors.asked.as_ref().is_some_and(|asked| asked.job == job) {
                continue;
            }
            // Colors asked for and not shown are no one's once filled.
            if let Some(Asked {
                held: Some((handle, _)),
                ..
            }) = colors.asked.take()
                && colors
                    .shown
                    .as_ref()
                    .is_none_or(|(_, shown, _)| *shown != handle)
            {
                self.colors_gone.push(handle);
            }
            let held = match records.textures(assets, colors_set(&job)) {
                Ok(handles) => handles
                    .first()
                    .map(|&handle| (handle, colors_identity(&job))),
                Err(error) => {
                    state
                        .failures
                        .record(frame, format!("macro colors: {error}"));
                    None
                }
            };
            colors.asked = Some(Asked { job, held });
        }
        self.colors_gone
            .retain(|&handle| assets.texture(handle).is_none() || !assets.remove_texture(handle));
    }
}

/// Asks the records for the terrains' macro colors (`Terrains::ask_colors`),
/// where the world holds terrains, records and an asset library.
pub(super) fn ask_colors(world: &mut World) {
    if !(world.contains_resource::<Terrains>()
        && world.contains_resource::<Records>()
        && world.contains_resource::<AssetLibrary>())
    {
        return;
    }
    world.resource_scope(|world, mut terrains: Mut<Terrains>| {
        world.resource_scope(|world, mut records: Mut<Records>| {
            let mut assets = world.resource_mut::<AssetLibrary>();
            terrains.ask_colors(&mut records, &mut assets);
        });
    });
}
