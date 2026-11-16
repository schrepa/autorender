use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, BASE_COLOR_MAP, Background, Error, Jobs, Landed, RecordExchange, Records,
    Settling, Spawner, SurfaceSlot, TextureHandle, TextureSet, TextureSource, TextureUsage,
    VIEW_JOBS, content_hash, job_priority,
};
use engine_open_world_kit::ground::{GroundLayerAsks, GroundLayersHeld, TerrainEntry};
use engine_render::{Allocation, Bind, Buffer, Install, Lifetime, Prepare};

use crate::ScatterBrick;
use crate::gpu::ScatterGpu;
use crate::groups::Resolved;
use crate::jobs::CANOPY;
use crate::jobs::canopy::{CanopyAsk, CanopyAsset, CanopyMade, ask_bytes, read_made};
use crate::layouts::{CanopyMap, WIND_REACH};
use crate::shaders::scatter::canopy as shader;

const PLANT: &str = "foliage";
const COLOR_STEPS: f32 = 4096.0;
const METER_STEPS: f32 = 1024.0;
const CUT_SHORT: &str = "the canopy record is cut short";
const NO_ROOM: &str = "the ground holds no layer for the canopy";
const AFTER_THE_VIEW: u32 = u32::MAX;

fn steady(value: f32, steps: f32) -> f32 {
    (value * steps).round() / steps
}

fn placed_color(asset: &Resolved<'_>, library: &AssetLibrary) -> [f32; 3] {
    let material = asset.material.material;
    let factor = material.color();
    let mean = material
        .map(BASE_COLOR_MAP)
        .and_then(|map| library.texture(map.texture))
        .and_then(|texture| texture.summary())
        .map_or([1.0; 3], |summary| summary.color);
    [
        factor[0] * mean[0],
        factor[1] * mean[1],
        factor[2] * mean[2],
    ]
}

fn canopy_asset(asset: &Resolved<'_>, library: &AssetLibrary) -> CanopyAsset {
    let bounds = asset.mesh.mesh.bounds();
    let (min, max) = (bounds.min(), bounds.max());
    let (low, high) = asset
        .parts
        .iter()
        .filter_map(|part| part.meshes.first())
        .map(|mesh| mesh.mesh.bounds())
        .fold((min, max), |(low, high), held| {
            (low.min(held.min()), high.max(held.max()))
        });
    let material = asset.material.material;
    let reach = if material.slot(SurfaceSlot::Position.name()) == Some(crate::WIND) {
        WIND_REACH
    } else {
        0.0
    };
    let plant = material.slot(SurfaceSlot::BaseColor.name()) == Some(PLANT);
    let varies = |name: &str| {
        let stated = material.value(name).filter(|_| plant).unwrap_or(0.0);
        steady(engine_math::clamp(stated, 0.0, 1.0), COLOR_STEPS)
    };
    let extent = max - min;
    CanopyAsset {
        color: placed_color(asset, library).map(|channel| steady(channel, COLOR_STEPS)),
        hue: varies(crate::PLANT_LOOK[2]),
        brightness: varies(crate::PLANT_LOOK[3]),
        center: [
            steady(0.5 * (min.x + max.x), METER_STEPS),
            steady(0.5 * (min.z + max.z), METER_STEPS),
        ],
        radius: steady(0.5 * (extent.x * extent.z).sqrt(), METER_STEPS),
        top: steady(high.y, METER_STEPS),
        diameter: steady((high - low).length() + 2.0 * reach, METER_STEPS),
    }
}

#[derive(Clone, PartialEq)]
enum Want {
    Nothing,
    Later,
    Canopy(Arc<[u8]>),
}

#[derive(Clone, PartialEq)]
pub(crate) struct Wanted {
    terrain: Entity,
    rect: [f32; 4],
    want: Want,
}

fn canopy_ask(
    terrain: &TerrainEntry,
    recipe: [u8; 32],
    assets: &[CanopyAsset],
    (variation, whole): (f32, u32),
) -> Arc<[u8]> {
    let ask = CanopyAsk {
        recipe,
        size: [whole; 2],
        origin: [terrain.origin.x, terrain.origin.z],
        variation: steady(variation, COLOR_STEPS),
        assets: assets.len() as u32,
    };
    Arc::from(ask_bytes(&ask, assets))
}

pub(crate) fn wanted(
    terrains: &[TerrainEntry],
    resolved: &[(Entity, Vec<Option<Resolved<'_>>>)],
    library: &AssetLibrary,
    look: (f32, u32),
) -> Vec<Wanted> {
    terrains
        .iter()
        .zip(resolved)
        .map(|(terrain, (_, assets))| {
            let extent = terrain.pyramid.extent();
            let want = match terrain.recipe {
                Some(recipe) if !terrain.surroundings && !assets.is_empty() => assets
                    .iter()
                    .map(|asset| asset.as_ref().map(|asset| canopy_asset(asset, library)))
                    .collect::<Option<Vec<CanopyAsset>>>()
                    .filter(|_| look.1 > 0)
                    .map_or(Want::Later, |assets| {
                        Want::Canopy(canopy_ask(terrain, recipe, &assets, look))
                    }),
                Some(_) | None => Want::Nothing,
            };
            Wanted {
                terrain: terrain.terrain,
                rect: [terrain.origin.x, terrain.origin.z, extent.x, extent.y],
                want,
            }
        })
        .collect()
}

struct Shown {
    made: CanopyMade,
    handles: [TextureHandle; 2],
    layers: [u32; 2],
}

#[derive(Component)]
pub(crate) struct CanopyLayers;

enum Step {
    Bytes(Background<Arc<[u8]>>),
    Textures(CanopyMade, [TextureHandle; 2]),
}

struct Stand {
    wanted: Wanted,
    asked: Option<Arc<[u8]>>,
    step: Option<Step>,
    shown: Option<Shown>,
}

impl Stand {
    fn handles(&self) -> Vec<TextureHandle> {
        let asked = match &self.step {
            Some(Step::Textures(_, handles)) => Some(*handles),
            Some(Step::Bytes(_)) | None => None,
        };
        let shown = self.shown.as_ref().map(|shown| shown.handles);
        asked.into_iter().chain(shown).flatten().collect()
    }

    fn table(&self) -> CanopyMap {
        let [x, z, wide, deep] = self.wanted.rect;
        let made = self.shown.as_ref().map(|shown| shown.made);
        let cell = made.map_or(0.0, |made| {
            engine_math::max(wide / made.size[0] as f32, deep / made.size[1] as f32)
        });
        let layers = match self.shown.as_ref().map(|shown| shown.layers) {
            Some([colors, shapes]) if colors > 0 && shapes > 0 => [colors, shapes, 0, 0],
            Some(_) | None => [0; 4],
        };
        CanopyMap {
            rect: [x, z, 1.0 / wide, 1.0 / deep],
            shape: [
                cell,
                made.map_or(0.0, |made| made.tallest),
                made.map_or(0.0, |made| made.span),
                0.0,
            ],
            layers,
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct CanopyMaps {
    stands: Vec<Stand>,
    gone: Vec<TextureHandle>,
    settling: Settling,
    failure: Option<String>,
    whole: u32,
}

impl CanopyMaps {
    pub(crate) fn whole(&self) -> u32 {
        self.whole
    }

    pub(crate) fn want(&mut self, wanted: Vec<Wanted>, whole: u32) {
        self.whole = whole;
        let same = self.stands.len() == wanted.len()
            && self
                .stands
                .iter()
                .zip(&wanted)
                .all(|(stand, wanted)| stand.wanted == *wanted);
        if same {
            return;
        }
        let mut before = std::mem::take(&mut self.stands);
        self.stands = wanted
            .into_iter()
            .map(|wanted| {
                let held = before
                    .iter()
                    .position(|stand| stand.wanted.terrain == wanted.terrain)
                    .map(|at| before.swap_remove(at));
                match held {
                    Some(stand) => Stand { wanted, ..stand },
                    None => Stand {
                        wanted,
                        asked: None,
                        step: None,
                        shown: None,
                    },
                }
            })
            .collect();
        for stand in before {
            self.gone.extend(stand.handles());
        }
    }

    pub(crate) fn count(&self) -> u32 {
        self.stands.len() as u32
    }

    pub(crate) fn shown(&self) -> u32 {
        self.stands
            .iter()
            .filter(|stand| stand.shown.is_some())
            .count() as u32
    }

    pub(crate) fn waiting(&self) -> u32 {
        self.stands
            .iter()
            .filter(|stand| stand.step.is_some())
            .count() as u32
    }

    pub(crate) fn failure(&self) -> Option<&String> {
        self.failure.as_ref()
    }
}

fn texture_set(input: &[u8], bytes: Arc<[u8]>) -> Option<(CanopyMade, TextureSet)> {
    let (made, _, _) = read_made(&bytes)?;
    let [width, height] = made.size;
    let set = TextureSet {
        source: crate::jobs::SOURCE.hash(),
        name: format!("scatter/canopy/{:032x}", content_hash(input)),
        textures: vec![
            ("colors".to_owned(), TextureUsage::Data),
            ("shapes".to_owned(), TextureUsage::Data),
        ],
        make: Arc::new(move || {
            let (_, colors, shapes) = read_made(&bytes).ok_or(CUT_SHORT)?;
            Ok::<_, String>(vec![
                TextureSource {
                    width,
                    height,
                    pixels: colors.to_vec(),
                    usage: TextureUsage::Data,
                },
                TextureSource {
                    width,
                    height,
                    pixels: shapes.to_vec(),
                    usage: TextureUsage::Data,
                },
            ])
            .map(Some)
        }),
    };
    Some((made, set))
}

fn landed(work: &mut Background<Arc<[u8]>>, records: &Records) -> Option<Result<Arc<[u8]>, Error>> {
    match work.poll(records) {
        Landed::Waiting => None,
        Landed::Ready(bytes) => Some(Ok(Arc::clone(bytes))),
        Landed::Failed(error) => Some(Err(error.clone())),
    }
}

pub(crate) fn bake_canopy(
    mut maps: ResMut<CanopyMaps>,
    (records, mut jobs): (Res<Records>, Jobs),
    mut exchange: ResMut<RecordExchange>,
    mut assets: ResMut<AssetLibrary>,
    own: Query<(Entity, &GroundLayerAsks, &GroundLayersHeld), With<CanopyLayers>>,
    mut spawner: Spawner<ScatterBrick>,
) {
    let CanopyMaps {
        stands,
        gone,
        settling,
        failure,
        whole: _,
    } = &mut *maps;
    let own = own.single().ok();
    let layers = |handles: [TextureHandle; 2]| {
        let (_, _, held) = own.as_ref()?;
        Some([held.layer(handles[0])?, held.layer(handles[1])?])
    };
    for stand in stands.iter_mut() {
        let wanted = match &stand.wanted.want {
            Want::Later => stand.asked.clone(),
            Want::Nothing => None,
            Want::Canopy(input) => Some(Arc::clone(input)),
        };
        if stand.asked != wanted {
            if let Some(Step::Textures(_, handles)) = stand.step.take() {
                gone.extend(handles);
            }
            stand.step = wanted.as_ref().map(|input| {
                let priority = job_priority(VIEW_JOBS, AFTER_THE_VIEW);
                Step::Bytes(jobs.fetch(&CANOPY, input, priority, Ok))
            });
            if wanted.is_none()
                && let Some(shown) = stand.shown.take()
            {
                gone.extend(shown.handles);
            }
            stand.asked = wanted;
        }
        stand.step = match stand.step.take() {
            Some(Step::Bytes(mut work)) => match landed(&mut work, &records) {
                None => Some(Step::Bytes(work)),
                Some(Ok(bytes)) => {
                    let set = texture_set(stand.asked.as_deref().unwrap_or(&[]), bytes);
                    match set.map(|(made, set)| (made, exchange.textures(&mut assets, set))) {
                        Some((made, Ok(handles))) => match handles[..] {
                            [colors, shapes] => Some(Step::Textures(made, [colors, shapes])),
                            _ => None,
                        },
                        Some((_, Err(error))) => {
                            *failure = Some(error.to_string());
                            None
                        }
                        None => {
                            *failure = Some(CUT_SHORT.to_owned());
                            None
                        }
                    }
                }
                Some(Err(error)) => {
                    *failure = Some(error.to_string());
                    None
                }
            },
            Some(Step::Textures(made, handles)) => {
                let failed = handles
                    .iter()
                    .find_map(|&handle| exchange.failed(handle))
                    .map(Error::to_string);
                match (layers(handles), failed) {
                    (Some(layers), _) if layers.contains(&0) => {
                        *failure = Some(NO_ROOM.to_owned());
                        gone.extend(handles);
                        None
                    }
                    (Some(layers), _) => {
                        let shown = Shown {
                            made,
                            handles,
                            layers,
                        };
                        if let Some(before) = stand.shown.replace(shown) {
                            gone.extend(before.handles);
                        }
                        None
                    }
                    (None, Some(error)) => {
                        *failure = Some(error);
                        gone.extend(handles);
                        None
                    }
                    (None, None) => Some(Step::Textures(made, handles)),
                }
            }
            None => None,
        };
        if let Some(shown) = &mut stand.shown {
            shown.layers = layers(shown.handles).unwrap_or([0; 2]);
        }
    }
    let asks = GroundLayerAsks {
        textures: stands.iter().flat_map(Stand::handles).collect(),
    };
    match &own {
        Some((entity, asked, _)) => {
            if **asked != asks {
                spawner.insert(*entity, asks);
            }
        }
        None => {
            if !asks.textures.is_empty() {
                spawner.spawn((CanopyLayers, asks, GroundLayersHeld::default()));
            }
        }
    }
    if gone
        .iter()
        .any(|&handle| assets.texture(handle).is_some())
    {
        let library = &mut *assets;
        gone.retain(|&handle| library.texture(handle).is_none() || !library.remove_texture(handle));
    }
    let landing = |stand: &&Stand| matches!(stand.step, Some(Step::Textures(..)));
    settling.left(stands.iter().filter(landing).count() as u32);
}

pub(crate) struct CanopyGpu {
    maps: Buffer<ScatterGpu>,
    table: Vec<CanopyMap>,
}

impl CanopyGpu {
    pub(crate) fn install(cx: &mut Install<'_, ScatterGpu>) -> Result<Self, Error> {
        let maps = cx.buffer("canopy_maps", Lifetime::Kept);
        cx.frame(shader::CANOPY_MAPS, Bind::Buffer(maps))?;
        Ok(Self {
            maps,
            table: Vec::new(),
        })
    }

    pub(crate) fn prepare(&mut self, cx: &mut Prepare<'_, ScatterGpu>, maps: &CanopyMaps) {
        let table: Vec<CanopyMap> = maps.stands.iter().map(Stand::table).collect();
        let bytes = (table.len().max(1) * size_of::<CanopyMap>()) as u64;
        let fresh = cx.allocate_buffer(self.maps, bytes) == Allocation::Fresh;
        if fresh || table != self.table {
            if !table.is_empty() {
                cx.write(self.maps, 0, &table);
            }
            self.table = table;
        }
    }
}
