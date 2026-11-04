use super::operations::{coordinate, reply, within};
use super::*;
use crate::layer::maps;
use crate::{HydrologyBake, TerrainLayer};
use engine_core::{Operation, SceneIds, bounds, stated};
use serde_json::Value;

pub const FALLS_SHOWN: usize = 16;
pub const RIVERS_SHOWN: usize = 16;

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TerrainInspect {
    #[serde(default)]
    pub terrain: Option<String>,
}

#[derive(Serialize)]
pub struct TerrainView {
    pub terrain: crate::scene::SceneTerrain,
    pub levels: u32,
    pub extent: [f32; 2],
    pub resident_tiles_by_level: Vec<u32>,
    pub level_errors: Vec<f32>,
    pub level_ranges: Vec<f32>,
    pub level_spacings: Vec<f32>,
    pub hydrology: Option<HydrologyView>,
    pub erosion: Option<ErosionView>,
    pub paths: Vec<PathView>,
    pub edits: Vec<TerrainEditView>,
    pub rivers: Vec<TerrainRiverView>,
    pub surroundings: Option<SurroundingsStatus>,
    pub baking: bool,
}

#[derive(Serialize)]
pub struct TerrainEditView {
    pub id: String,
    pub stamp: Stamp,
}

#[derive(Serialize)]
pub struct TerrainRiverView {
    pub id: String,
    pub river: GivenRiver,
}

#[derive(Serialize)]
pub struct PathView {
    pub layer: usize,
    pub baked: bool,
    pub length: Option<f32>,
    pub samples: Option<usize>,
    pub steepest: Option<f32>,
    pub deepest_cut: Option<f32>,
    pub highest_fill: Option<f32>,
    pub bridges: Option<usize>,
    pub bridged: Option<f32>,
    pub tunnels: Option<usize>,
    pub tunneled: Option<f32>,
}

#[derive(Serialize)]
pub struct ErosionView {
    pub baking: bool,
    pub samples: Option<usize>,
    pub rounds: Option<u32>,
    pub deepest: Option<f32>,
    pub highest: Option<f32>,
}

#[derive(Serialize)]
pub struct HydrologyView {
    pub baking: bool,
    pub samples: Option<usize>,
    pub channel_samples: Option<usize>,
    pub lake_samples: Option<usize>,
    pub max_area: Option<f32>,
    pub falls: Option<usize>,
    pub given: Option<usize>,
    pub uncut: Option<usize>,
    pub flow: Option<RiverFlowView>,
    pub fall_list: Vec<FallView>,
    pub rivers: Vec<RiverView>,
}

#[derive(Serialize)]
pub struct RiverFlowView {
    pub rivers: usize,
    pub lakes: usize,
    pub points: usize,
    pub max_discharge: f32,
    pub fastest: f32,
    pub rapid_share: f32,
}

#[derive(Serialize)]
pub struct FallView {
    pub top: [f32; 3],
    pub bottom: [f32; 3],
    pub width: f32,
    pub speed: f32,
    pub discharge: f32,
}

#[derive(Serialize)]
pub struct RiverView {
    pub name: Option<String>,
    pub given: Option<[u32; 2]>,
    pub courses: usize,
    pub length: f32,
    pub first: [f32; 2],
    pub last: [f32; 2],
    pub width: [f32; 2],
    pub depth: [f32; 2],
    pub flood: [f32; 2],
    pub given_width: bool,
    pub given_depth: bool,
    pub given_flood: bool,
    pub uncut: f32,
}

impl Operation for TerrainInspect {
    const NAME: &'static str = "terrain_view";
    type Params = (
        Res<'static, Terrains>,
        Query<'static, 'static, (Entity, &'static TerrainScene)>,
        SceneIds<'static, 'static>,
    );
    type Request = Self;

    fn run(
        (terrains, scenes, ids): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let built = |entity: Entity| {
            scenes
                .get(entity)
                .ok()
                .and_then(|(_, scene)| Some((scene, scene.built.as_ref()?)))
        };
        let found = match request.terrain.as_deref() {
            Some(name) => ids.entity(name).and_then(|entity| Some((entity, built(entity)?))),
            None => scenes
                .iter()
                .find_map(|(entity, _)| Some((entity, built(entity)?))),
        };
        let Some((entity, (scene, terrain))) = found else {
            let error = Error::new(ErrorCode::UnknownTerrain, "$.terrain");
            return Err(match request.terrain {
                Some(name) => error.got(name),
                None => error,
            });
        };
        Ok(reply(terrain_view(&terrains, entity, scene, terrain)))
    }
}

fn terrain_view(
    terrains: &Terrains,
    entity: Entity,
    scene: &TerrainScene,
    terrain: &Terrain,
) -> TerrainView {
    let mut levels = vec![0_u32; terrain.pyramid.levels() as usize];
    let (mut level_errors, mut level_ranges) = (Vec::new(), Vec::new());
    let (mut hydrology, mut erosion, mut paths) = (None, None, Vec::new());
    for tile in terrains.tiles(entity) {
        levels[usize::from(tile.key().level)] += 1;
    }
    if let Some((errors, ranges)) = terrains.level_detail(entity) {
        level_errors = errors.to_vec();
        level_ranges = ranges.to_vec();
    }
    let holds = |wanted: fn(&dyn TerrainLayer) -> bool| {
        terrain
            .layers
            .layers()
            .iter()
            .any(|layer| wanted(layer.as_ref()))
    };
    let bakes = terrains.bakes(entity);
    if holds(|layer| layer.makes_water()) {
        hydrology = Some(hydrology_view(terrains, entity));
    }
    if let Some(layers) = terrains.layers(entity) {
        for (index, layer) in layers.layers().iter().enumerate() {
            if publishes(layer.as_ref(), maps::PATH) {
                let stats = layer.stats();
                let stats = stats.as_ref();
                paths.push(PathView {
                    layer: index,
                    baked: stats.is_some(),
                    length: field(stats, "length"),
                    samples: field(stats, "samples"),
                    steepest: field(stats, "steepest"),
                    deepest_cut: field(stats, "deepest_cut"),
                    highest_fill: field(stats, "highest_fill"),
                    bridges: field(stats, "bridges"),
                    bridged: field(stats, "bridged"),
                    tunnels: field(stats, "tunnels"),
                    tunneled: field(stats, "tunneled"),
                });
            }
        }
    }
    if holds(|layer| publishes(layer, maps::WORN)) {
        let stats = terrains.layers(entity).and_then(|layers| {
            layers
                .layers()
                .iter()
                .find(|layer| publishes(layer.as_ref(), maps::WORN))?
                .stats()
        });
        let stats = stats.as_ref();
        erosion = Some(ErosionView {
            baking: bakes.baking,
            samples: field(stats, "samples"),
            rounds: field(stats, "rounds"),
            deepest: field(stats, "deepest"),
            highest: field(stats, "highest"),
        });
    }
    TerrainView {
        terrain: scene.authored.clone(),
        levels: terrain.pyramid.levels(),
        extent: terrain.pyramid.extent().to_array(),
        resident_tiles_by_level: levels,
        level_errors,
        level_ranges,
        level_spacings: (0..terrain.pyramid.levels())
            .map(|level| terrain.pyramid.spacing(level))
            .collect(),
        hydrology,
        erosion,
        paths,
        edits: terrain
            .edits
            .iter()
            .map(|edit| TerrainEditView {
                id: edit.id.to_string(),
                stamp: edit.stamp,
            })
            .collect(),
        rivers: terrain
            .rivers
            .iter()
            .map(|drawn| TerrainRiverView {
                id: drawn.id.to_string(),
                river: drawn.river.clone(),
            })
            .collect(),
        surroundings: terrains.surroundings(entity),
        baking: bakes.baking,
    }
}

fn publishes(layer: &dyn TerrainLayer, name: &str) -> bool {
    layer.outputs().iter().any(|output| output == name)
}

fn field<T: serde::de::DeserializeOwned>(stats: Option<&Value>, name: &str) -> Option<T> {
    serde_json::from_value(stats?.get(name)?.clone()).ok()
}

fn flow_view(found: &HydrologyBake) -> RiverFlowView {
    let flow = found.flow_summary();
    RiverFlowView {
        rivers: flow.rivers,
        lakes: flow.lakes,
        points: flow.points,
        max_discharge: flow.max_discharge,
        fastest: flow.fastest,
        rapid_share: flow.rapid_share,
    }
}

fn hydrology_view(terrains: &Terrains, entity: Entity) -> HydrologyView {
    let bakes = terrains.bakes(entity);
    let stats = bakes.hydrology;
    let baked = terrains.layers(entity);
    let found = baked.as_deref().and_then(Layers::hydrology);
    HydrologyView {
        baking: bakes.baking,
        samples: stats.map(|stats| stats.samples),
        channel_samples: stats.map(|stats| stats.channel_samples),
        lake_samples: stats.map(|stats| stats.lake_samples),
        max_area: stats.map(|stats| stats.max_area),
        falls: stats.map(|stats| stats.falls),
        given: stats.map(|stats| stats.given),
        uncut: stats.map(|stats| stats.uncut),
        rivers: baked
            .as_deref()
            .map(|layers| layers.river_listing(RIVERS_SHOWN))
            .unwrap_or_default()
            .into_iter()
            .map(|river| RiverView {
                name: river.name,
                given: river.given,
                courses: river.courses,
                length: river.length,
                first: river.first,
                last: river.last,
                width: river.width,
                depth: river.depth,
                flood: river.flood,
                given_width: river.given_width,
                given_depth: river.given_depth,
                given_flood: river.given_flood,
                uncut: river.uncut,
            })
            .collect(),
        flow: found.map(flow_view),
        fall_list: found
            .map(|found| {
                found
                    .falls
                    .iter()
                    .take(FALLS_SHOWN)
                    .map(|fall| FallView {
                        top: fall.top,
                        bottom: fall.bottom,
                        width: fall.width,
                        speed: fall.speed,
                        discharge: fall.discharge,
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TerrainLayerSweep {
    pub center: [f64; 2],
    #[schemars(transform = stated(bounds::SWEEP_EXTENT))]
    pub half_extent: f64,
    #[schemars(transform = stated(bounds::SWEEP_SAMPLES))]
    pub samples: u32,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct LayerSweep {
    pub compared: usize,
    pub worst: SweepWorst,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct SweepWorst {
    pub max_difference: f32,
    pub at: Option<[f32; 2]>,
}

impl Operation for TerrainLayerSweep {
    const NAME: &'static str = "terrain_layer_sweep";
    type Params = Res<'static, Terrains>;
    type Request = Self;

    fn run(
        terrains: SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let center = Vec2::new(
            coordinate(request.center[0], "$.center[0]")?,
            coordinate(request.center[1], "$.center[1]")?,
        );
        let half_extent = within(request.half_extent, bounds::SWEEP_EXTENT, "$.half_extent")?;
        within(
            f64::from(request.samples),
            bounds::SWEEP_SAMPLES,
            "$.samples",
        )?;
        Ok(reply(layer_sweep(
            &terrains,
            center,
            half_extent as f32,
            request.samples,
        )))
    }
}

fn layer_sweep(
    terrains: &Terrains,
    center: Vec2,
    half_extent: f32,
    samples: u32,
) -> LayerSweep {
    let (mut compared, mut worst, mut at) = (0_usize, 0.0_f32, None);
    for row in 0..samples {
        for column in 0..samples {
            let unit = Vec2::new(column as f32, row as f32) / (samples - 1) as f32 * 2.0 - 1.0;
            let point = center + unit * half_extent;
            let Some(sample) = terrains.sample(point.x, point.y) else {
                continue;
            };
            if sample.hole || sample.level != 0 {
                continue;
            }
            let (Some(terrain), Some(stack)) = (
                terrains.terrain(sample.terrain),
                terrains.layers(sample.terrain),
            ) else {
                continue;
            };
            let local = point - Vec2::new(terrain.origin.x, terrain.origin.z);
            let height = terrain.origin.y + stack.height_at(&terrain.pyramid, local);
            let difference = (height - sample.height).abs();
            compared += 1;
            if difference > worst || at.is_none() {
                worst = engine_math::max(worst, difference);
                at = Some(point.to_array());
            }
        }
    }
    LayerSweep {
        compared,
        worst: SweepWorst {
            max_difference: worst,
            at,
        },
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TerrainListing {}

#[derive(Serialize, schemars::JsonSchema)]
pub struct TerrainListed {
    pub terrains: Vec<ListedTerrain>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct ListedTerrain {
    pub surroundings: bool,
    pub tiles: Vec<[u16; 3]>,
    pub drawn: bool,
}

impl Operation for TerrainListing {
    const NAME: &'static str = "terrain_listing";
    type Params = (Res<'static, Terrains>, Res<'static, TerrainSettings>);
    type Request = Self;

    fn run(
        (terrains, settings): SystemParamItem<'_, '_, Self::Params>,
        _: Self,
    ) -> Result<Value, Error> {
        Ok(reply(TerrainListed {
            terrains: terrains
                .entries
                .iter()
                .map(|state| ListedTerrain {
                    surroundings: state.around.is_some(),
                    tiles: state
                        .store
                        .tiles()
                        .map(|tile| {
                            let key = tile.key();
                            [u16::from(key.level), key.x, key.z]
                        })
                        .collect(),
                    drawn: settings.enabled && state.drawn,
                })
                .collect(),
        }))
    }
}
