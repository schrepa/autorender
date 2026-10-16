//! What frames hand renderers of each terrain.

use super::*;

/// A surface material's engine material: its factors, and
/// the base color, normal and metallic-roughness textures it names, those
/// loaded. Terrain reads height from the metallic-roughness map's alpha,
/// and occlusion from its red where the material's occlusion map is the
/// same texture.
#[derive(Clone, Debug)]
pub struct RenderSurfaceMaps {
    pub material: Material,
    pub base_color: Option<RenderTexture>,
    pub normal: Option<RenderTexture>,
    pub metallic_roughness: Option<RenderTexture>,
}

impl RenderSurfaceMaps {
    /// `material`'s maps from `assets`; each one named but missing counts
    /// in `missing`.
    fn of(material: Material, assets: &AssetLibrary, missing: &mut usize) -> Self {
        let mut texture = |map: Option<engine_assets::TextureRef>| {
            let handle = map?.texture;
            let found = assets.texture(handle).map(|texture| RenderTexture {
                handle,
                texture: Arc::clone(texture),
            });
            *missing += usize::from(found.is_none());
            found
        };
        Self {
            base_color: texture(material.base_color_texture()),
            normal: texture(material.normal_texture()),
            metallic_roughness: texture(material.metallic_roughness_texture()),
            material,
        }
    }
}

/// A placement asset of a frame: its mesh, its material
/// and the material's textures, its coarser meshes with their errors, and
/// its sway.
#[derive(Clone, Debug)]
pub struct RenderPlacementAsset {
    pub mesh: RenderMesh,
    pub material: RenderMaterial,
    pub textures: Vec<RenderTexture>,
    pub lods: Vec<(RenderMesh, f32)>,
    pub sway: f32,
}

impl RenderPlacementAsset {
    /// `asset` from `assets`, or `None` with what is missing counted in
    /// `missing`.
    fn of(asset: &PlacementAsset, assets: &AssetLibrary, missing: &mut usize) -> Option<Self> {
        let found = (|| {
            let mesh = assets.mesh(asset.mesh)?;
            let material = assets.material(asset.material)?;
            let lods = asset
                .lods
                .iter()
                .map(|lod| {
                    assets.mesh(lod.mesh).map(|mesh| {
                        (
                            RenderMesh {
                                handle: lod.mesh,
                                mesh: Arc::clone(mesh),
                            },
                            lod.error,
                        )
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            let textures = material
                .textures()
                .map(|handle| {
                    assets.texture(handle).map(|texture| RenderTexture {
                        handle,
                        texture: Arc::clone(texture),
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Self {
                mesh: RenderMesh {
                    handle: asset.mesh,
                    mesh: Arc::clone(mesh),
                },
                material: RenderMaterial {
                    handle: asset.material,
                    material: *material,
                    version: assets.material_version(asset.material).unwrap_or(0),
                },
                textures,
                lods,
                sway: asset.sway,
            })
        })();
        *missing += usize::from(found.is_none());
        found
    }
}

/// One terrain of a frame.
#[derive(Clone, Debug)]
pub struct RenderTerrain {
    /// The terrain's entity bits, which renderers key its tiles by.
    pub id: u64,
    pub origin: Vec3,
    pub pyramid: Pyramid,
    /// How each of its surface's materials looks, by the index tiles'
    /// control values hold.
    pub surface: Vec<SurfaceMaterial>,
    /// Each surface material's engine material, where it names one that
    /// exists.
    pub maps: Vec<Option<RenderSurfaceMaps>>,
    /// What its placement layers stand on it, by the asset
    /// indices its tiles' placed objects hold; `None` where the mesh or
    /// material is missing.
    pub assets: Vec<Option<RenderPlacementAsset>>,
    /// How each placement layer's objects thin with distance,
    /// by the layer indices its tiles' placed objects hold: the level
    /// of the tiles holding them, the level they also stand far away at and
    /// the share standing there (their own level and 0 without).
    pub thinning: Vec<(u8, u8, f32)>,
    /// Its resident tiles, in key order: renderers upload those they lack
    /// and release the rest. The list changes only when `tiles_revision`
    /// advances.
    pub tiles: Arc<[Arc<Tile>]>,
    pub tiles_revision: u64,
    /// The patches the camera draws.
    pub patches: Vec<Patch>,
    /// Meters from the eye within which each level's tiles load, finest
    /// first: where renderers can shade with them.
    pub reach: Vec<f32>,
    /// Meters from the eye up to which each level draws, finest first; each
    /// but the coarsest morphs into the next over its last quarter
    /// ([`LodRanges`]): where renderers find the ground as drawn.
    pub ranges: Vec<f32>,
    /// Where its hydrology bake found lakes standing, and
    /// its rivers.
    pub water: Option<Arc<engine_terrain::WaterGrid>>,
    pub rivers: Option<Arc<engine_terrain::Rivers>>,
    /// Whether it is a terrain's surroundings, which grow
    /// no blades: their grassy ground shows as canopy at every distance.
    pub surroundings: bool,
    /// Whether its tiles may hold holes and whether a hydrology bake will
    /// give it water: renderers create what draws them before it shows,
    /// while it loads.
    pub may_hole: bool,
    pub hydrology: bool,
    /// Its surface for the GPU to give its tiles control values,
    /// which they then hold none of; `None` where they hold them.
    pub gpu_surface: Option<Arc<engine_terrain::gpu::SurfaceData>>,
}

/// The frame's terrains.
#[derive(Clone, Debug, Default)]
pub struct RenderTerrains {
    /// Immutable optional basin-fog maps, current to terrain geometry.
    pub basin_fog: Vec<crate::RenderBasinFog>,

    pub terrains: Vec<RenderTerrain>,
    /// The camera's view and projection without its translation: terrain
    /// draws camera-relative, from positions less the camera's.
    pub relative_view_projection: Mat4,
    pub stats: TerrainStats,
    /// What terrain shows ([`TerrainDebug`]).
    pub view: TerrainView,
}

/// Lists every terrain's resident tiles and the camera's patches.
pub(crate) fn extract_terrains(world: &World, aspect: f32, snapshot: &mut RenderSnapshot) {
    snapshot.terrain.basin_fog.clear();
    if let Some(fog) = world.get_resource::<crate::BasinFogs>() {
        fog.render(&mut snapshot.terrain.basin_fog);
    }
    let listing = &mut snapshot.terrain;
    let missing = &mut snapshot.missing_assets;
    let assets = world.get_resource::<AssetLibrary>();
    let Some(terrains) = world.get_resource::<Terrains>() else {
        listing.terrains.clear();
        listing.stats = TerrainStats::default();
        return;
    };
    let camera = world.resource::<Camera>();
    listing.relative_view_projection = camera.relative_view_projection(aspect);
    listing.stats = terrains.stats;
    listing.view = world
        .get_resource::<TerrainDebug>()
        .map_or(TerrainView::Shaded, |debug| debug.view);
    let enabled = world
        .get_resource::<TerrainSettings>()
        .is_some_and(|settings| settings.enabled);
    listing.terrains.truncate(terrains.entries.len());
    for (index, state) in terrains.entries.iter().enumerate() {
        if index == listing.terrains.len() {
            listing.terrains.push(RenderTerrain {
                id: 0,
                origin: Vec3::ZERO,
                pyramid: state.terrain.pyramid,
                surface: Vec::new(),
                maps: Vec::new(),
                assets: Vec::new(),
                thinning: Vec::new(),
                tiles: Arc::from(Vec::new()),
                tiles_revision: 0,
                patches: Vec::new(),
                reach: Vec::new(),
                ranges: Vec::new(),
                water: None,
                rivers: None,
                surroundings: false,
                may_hole: false,
                hydrology: false,
                gpu_surface: None,
            });
        }
        let render = &mut listing.terrains[index];
        render.id = state.render_id();
        render.surroundings = state.around.is_some();
        (render.may_hole, render.hydrology) = match &state.terrain.source {
            TerrainSource::Generated(layers) => (layers.may_hole(), layers.has_hydrology()),
            TerrainSource::File { .. } => (false, false),
        };
        render.origin = state.terrain.origin;
        render.pyramid = state.terrain.pyramid;
        render.surface.clone_from(&state.terrain.surface.materials);
        render.assets.clear();
        for asset in &state.terrain.surface.assets {
            render
                .assets
                .push(assets.and_then(|assets| RenderPlacementAsset::of(asset, assets, missing)));
        }
        render.thinning.clear();
        render.thinning.extend(
            state
                .terrain
                .surface
                .rules
                .authored()
                .placements
                .iter()
                .map(|layer| match layer.far {
                    Some(far) => (layer.level, far.level, far.share),
                    None => (layer.level, layer.level, 0.0),
                }),
        );
        render.maps.clear();
        for look in &state.terrain.surface.materials {
            let material = look.material.and_then(|handle| {
                let material = assets.and_then(|assets| Some((assets, *assets.material(handle)?)));
                *missing += usize::from(material.is_none());
                material
            });
            render.maps.push(
                material.map(|(assets, material)| RenderSurfaceMaps::of(material, assets, missing)),
            );
        }
        if render.tiles_revision != state.revision() || render.id != state.render_id() {
            let tiles: Vec<Arc<Tile>> = state.store.tiles().cloned().collect();
            render.tiles = Arc::from(tiles);
            render.tiles_revision = state.revision();
        }
        render.patches.clear();
        if enabled {
            render.patches.extend_from_slice(&state.selection.patches);
        }
        let hydrology = state.baked.as_ref().and_then(|layers| layers.hydrology());
        render.water = hydrology.map(|hydrology| Arc::clone(&hydrology.water));
        render.rivers = hydrology.map(|hydrology| Arc::clone(&hydrology.rivers));
        render.gpu_surface = if state.materials {
            None
        } else {
            state.surface.as_ref().and_then(|(_, data)| data.clone())
        };
        render.reach.clear();
        render
            .reach
            .extend(state.ranges.iter().map(|range| range * LOAD_REACH));
        render.ranges.clone_from(&state.ranges);
    }
}
