use super::*;
use crate::jobs::MacroJob;
use bevy_ecs::prelude::Mut;
use engine_assets::TextureHandle;
use engine_core::{MakeSources, Records, TextureSet, TextureUsage};

struct Asked {
    job: MacroJob,
    held: Option<(TextureHandle, u128)>,
}

#[derive(Default)]
pub(super) struct MacroColors {
    pub(super) wanted: Option<MacroJob>,
    asked: Option<Asked>,
    shown: Option<(MacroJob, TextureHandle, u128)>,
}

impl MacroColors {
    pub(super) fn asking(&self) -> bool {
        self.asked.as_ref().is_some_and(|asked| {
            asked.held.is_some()
                && self
                    .shown
                    .as_ref()
                    .is_none_or(|(shown, _, _)| *shown != asked.job)
        })
    }

    pub(super) fn shown(&self) -> bool {
        self.shown.is_some()
    }

    pub(super) fn left(&self) -> bool {
        self.wanted.as_ref().is_some_and(|wanted| {
            self.shown
                .as_ref()
                .is_none_or(|(shown, _, _)| shown != wanted)
        })
    }

    pub(super) fn handles(&self) -> impl Iterator<Item = TextureHandle> + '_ {
        self.asked
            .iter()
            .filter_map(|asked| asked.held.map(|(handle, _)| handle))
            .chain(self.shown.iter().map(|&(_, handle, _)| handle))
    }

    pub(super) fn map(&self, assets: Option<&AssetLibrary>) -> Option<MacroMap> {
        let (_, handle, identity) = self.shown.as_ref()?;
        let texture = assets?.texture(*handle)?;
        Some(MacroMap {
            texture: Arc::clone(texture),
            identity: *identity,
        })
    }
}

fn macro_texels(settings: &TerrainSettings) -> u32 {
    let block = crate::macro_colors::MACRO_BLOCK;
    settings
        .macro_resolution
        .next_multiple_of(block)
        .min(crate::macro_colors::MAX_MACRO_TEXELS)
}

pub(super) fn want_colors(state: &mut TerrainState, settings: &TerrainSettings) {
    let recipe = state.recipe.as_ref().filter(|_| settings.enabled);
    state.whole.colors.wanted = recipe.map(|recipe| MacroJob {
        recipe: recipe.key.clone(),
        level: lighting_level(&state.terrain.pyramid, settings.pinned_levels),
        size: macro_texels(settings),
    });
}

fn colors_set(job: &MacroJob) -> TextureSet {
    let identity = job.identity();
    TextureSet {
        source: crate::SOURCE,
        name: format!("terrain/macro/{:032x}", content_hash(&identity)),
        textures: vec![("colors".to_owned(), TextureUsage::Data)],
        make: MakeSources::Job {
            kind: MacroJob::KIND.name(),
            input: identity,
            size: [job.size, job.size],
        },
    }
}

fn colors_identity(job: &MacroJob) -> u128 {
    let mut identity = job.identity();
    identity.extend_from_slice(&engine_textures::SOURCE);
    content_hash(&identity)
}

impl Terrains {
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
