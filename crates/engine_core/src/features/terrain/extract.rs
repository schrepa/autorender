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
/// and the textures of every material it draws with, its coarser meshes
/// with their errors, its parts, its shadow proxy
/// ([`PlacementAsset::shadow`]) and its sway.
#[derive(Clone, Debug)]
pub struct RenderPlacementAsset {
    pub mesh: RenderMesh,
    pub material: RenderMaterial,
    pub textures: Vec<RenderTexture>,
    pub lods: Vec<(RenderMesh, f32)>,
    pub parts: Vec<RenderPlacementPart>,
    pub shadow: Option<RenderMesh>,
    pub sway: f32,
}

/// A part of a placement asset of a frame ([`PlacementPart`]): its
/// material, and its mesh at each of the asset's levels it reaches.
#[derive(Clone, Debug)]
pub struct RenderPlacementPart {
    pub material: RenderMaterial,
    pub meshes: Vec<RenderMesh>,
}

impl RenderPlacementAsset {
    /// `asset` from `assets`, or `None` with what is missing counted in
    /// `missing`.
    fn of(asset: &PlacementAsset, assets: &AssetLibrary, missing: &mut usize) -> Option<Self> {
        let found = (|| {
            let mesh = RenderMesh::of(assets, asset.mesh)?;
            let material = |handle: MaterialHandle| {
                assets.material(handle).map(|material| RenderMaterial {
                    handle,
                    material: *material,
                    version: assets.material_version(handle).unwrap_or(0),
                })
            };
            let lods = asset
                .lods
                .iter()
                .map(|lod| RenderMesh::of(assets, lod.mesh).map(|mesh| (mesh, lod.error)))
                .collect::<Option<Vec<_>>>()?;
            let parts = asset
                .parts
                .iter()
                .map(|part| {
                    Some(RenderPlacementPart {
                        material: material(part.material)?,
                        meshes: part
                            .meshes
                            .iter()
                            .map(|&mesh| RenderMesh::of(assets, mesh))
                            .collect::<Option<Vec<_>>>()?,
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            let shadow = match asset.shadow {
                Some(handle) => Some(RenderMesh::of(assets, handle)?),
                None => None,
            };
            let material = material(asset.material)?;
            let textures = std::iter::once(&material.material)
                .chain(parts.iter().map(|part| &part.material.material))
                .flat_map(Material::textures)
                .map(|handle| {
                    assets.texture(handle).map(|texture| RenderTexture {
                        handle,
                        texture: Arc::clone(texture),
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Self {
                mesh,
                material,
                textures,
                lods,
                parts,
                shadow,
                sway: asset.sway,
            })
        })();
        *missing += usize::from(found.is_none());
        found
    }

    /// Every mesh it draws: its own, its coarser ones, its parts' and its
    /// shadow proxy.
    pub fn meshes(&self) -> impl Iterator<Item = &RenderMesh> {
        std::iter::once(&self.mesh)
            .chain(self.lods.iter().map(|(lod, _)| lod))
            .chain(self.parts.iter().flat_map(|part| &part.meshes))
            .chain(&self.shadow)
    }

    /// Every material it draws with: its own, then its parts'.
    pub fn materials(&self) -> impl Iterator<Item = &RenderMaterial> {
        std::iter::once(&self.material).chain(self.parts.iter().map(|part| &part.material))
    }
}

/// A placement layer of a frame: how its objects thin with distance, the
/// level of the tiles holding them, and the level they also stand far
/// away at with the share standing there (their own level and 0 without);
/// and what the ground under them takes on where they retire: its objects
/// per square meter where everything allows, their scale range, the
/// surface materials it stands on (a bit per material; `None`: any) and
/// the assets it chooses from, by index, each with its share of them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderPlacementLayer {
    pub level: u8,
    pub far: (u8, f32),
    pub density: f32,
    pub scale: [f32; 2],
    pub materials: Option<[u64; 4]>,
    pub assets: Vec<(u16, f32)>,
}

impl RenderPlacementLayer {
    /// Whether its objects stand on surface material `material`.
    pub fn stands_on(&self, material: usize) -> bool {
        self.materials.is_none_or(|set| {
            material < 256 && set[material / 64] & (1 << (material % 64)) != 0
        })
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
    /// Its surface's steep layer, which renderers decide again at every
    /// point of the ground its level-0 samples shade (every level's, on
    /// surroundings), coarser levels holding it as their shading averaged
    /// it; `None` without one.
    pub steep: Option<engine_terrain::Steep>,
    /// Each surface material's engine material, where it names one that
    /// exists.
    pub maps: Vec<Option<RenderSurfaceMaps>>,
    /// What its placement layers stand on it, by the asset
    /// indices its tiles' placed objects hold; `None` where the mesh or
    /// material is missing.
    pub assets: Vec<Option<RenderPlacementAsset>>,
    /// Its placement layers, by the layer indices its tiles' placed objects
    /// hold: how their objects thin with distance, and what the ground
    /// takes on where they retire.
    pub layers: Vec<RenderPlacementLayer>,
    /// Its resident tiles, in key order: renderers upload those they lack
    /// and release the rest, and upload a tile again as its shading arrives
    /// ([`Tile::shaded`]). The list changes only when `tiles_revision`
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
    /// The largest height difference measured between each level's tiles
    /// and the next coarser level's, finest first, as tiles arrive (zero
    /// until measured): how far the ground one level draws may stand off
    /// another's.
    pub errors: Vec<f32>,
    /// Whether it is a terrain's surroundings, which grow
    /// no blades: their grassy ground shows as canopy at every distance.
    pub surroundings: bool,
    /// For a terrain's surroundings, the ground the terrain they surround
    /// covers (terrain-local, least corner then greatest): their patches
    /// never draw there.
    pub cutout: Option<(Vec2, Vec2)>,
    /// Whether its tiles may hold holes and whether a hydrology bake will
    /// give it water: renderers create what draws them before it shows,
    /// while it loads.
    pub may_hole: bool,
    pub hydrology: bool,
    /// Its surface for the GPU to give its tiles shading with their own
    /// samples (level-0 tiles, and every tile of surroundings) control
    /// values, which they then hold none of; `None` where they hold them.
    /// Its coarser tiles take theirs, and their normal maps, from their
    /// shading ([`Tile::shading`]).
    pub gpu_surface: Option<Arc<engine_terrain::gpu::SurfaceData>>,
    /// Where its rivers leave their beds (terrain-local): renderers draw
    /// their jets where water draws.
    pub falls: Arc<[Fall]>,
    /// The square around the camera its water is simulated over: renderers
    /// run the simulation there and draw its water instead of the tiles'.
    pub water_window: Option<Arc<WaterWindow>>,
    /// Its lighting, once here: the sky its ground sees and its horizons,
    /// which renderers upload as it changes.
    pub lighting: Option<Arc<TerrainLighting>>,
    /// Its bounce transfer, once here: what probes over its ground see of
    /// it, which renderers upload as it changes and relight every frame.
    pub bounce: Option<Arc<TerrainBounce>>,
    /// Its macro colors ([`engine_terrain::macro_colors`]), once here: the
    /// tint and growth its bakes give its ground over its extent, which
    /// renderers upload as they change.
    pub macro_map: Option<MacroMap>,
}

/// A terrain's macro colors as renderers take them: the map, in the family
/// of formats the device samples, with its mips; and the hash of what made
/// it, the same in every family, which records renderers bake from it are
/// kept by.
#[derive(Clone, Debug)]
pub struct MacroMap {
    pub texture: Arc<engine_assets::Texture>,
    pub identity: u128,
}

/// The frame's terrains.
#[derive(Clone, Debug, Default)]
pub struct RenderTerrains {
    pub terrains: Vec<RenderTerrain>,
    /// The camera's view and projection without its translation: terrain
    /// draws camera-relative, from positions less the camera's.
    pub relative_view_projection: Mat4,
    pub stats: TerrainStats,
}

/// Lists every terrain's resident tiles and the camera's patches.
pub(crate) fn extract_terrains(world: &World, aspect: f32, snapshot: &mut RenderSnapshot) {
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
                steep: None,
                maps: Vec::new(),
                assets: Vec::new(),
                layers: Vec::new(),
                tiles: Arc::from(Vec::new()),
                tiles_revision: 0,
                patches: Vec::new(),
                reach: Vec::new(),
                ranges: Vec::new(),
                errors: Vec::new(),
                surroundings: false,
                cutout: None,
                may_hole: false,
                hydrology: false,
                gpu_surface: None,
                falls: Arc::from(Vec::new()),
                water_window: None,
                lighting: None,
                bounce: None,
                macro_map: None,
            });
        }
        let render = &mut listing.terrains[index];
        render.id = state.render_id();
        render.surroundings = state.around.is_some();
        render.cutout = state.around.as_ref().map(|around| around.cutout);
        let layers = &state.terrain.layers;
        (render.may_hole, render.hydrology) = (layers.may_hole(), layers.has_hydrology());
        render.origin = state.terrain.origin;
        render.pyramid = state.terrain.pyramid;
        render.surface.clone_from(&state.terrain.surface.materials);
        render.steep = state.terrain.surface.rules.steep();
        render.assets.clear();
        for asset in &state.terrain.surface.assets {
            render
                .assets
                .push(assets.and_then(|assets| RenderPlacementAsset::of(asset, assets, missing)));
        }
        extract_layers(state.terrain.surface.rules.authored(), &mut render.layers);
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
        render.errors.clear();
        render.errors.extend_from_slice(state.store.errors());
        render.falls = state.falls.clone();
        render.water_window = state.hydrology.window.clone();
        render.lighting.clone_from(&state.whole.lighting.made);
        render.bounce.clone_from(&state.whole.bounce.made);
        render.macro_map = state.whole.colors.map(assets);
    }
}

/// `rules`' placement layers as frames hand them to renderers, into
/// `layers`, whose vectors it keeps from frame to frame.
fn extract_layers(rules: &Biomes, layers: &mut Vec<RenderPlacementLayer>) {
    layers.truncate(rules.placements.len());
    for (index, layer) in rules.placements.iter().enumerate() {
        if index == layers.len() {
            layers.push(RenderPlacementLayer::default());
        }
        let held = &mut layers[index];
        held.level = layer.level;
        held.far = layer.far.map_or((layer.level, 0.0), |far| (far.level, far.share));
        // Layers count their objects per hectare.
        held.density = layer.density / 10_000.0;
        held.scale = layer.scale;
        held.materials = (!layer.materials.is_empty()).then(|| {
            let mut set = [0_u64; 4];
            for id in &layer.materials {
                if let Some(material) = rules.materials.iter().position(|known| known == id)
                    && material < 256
                {
                    set[material / 64] |= 1 << (material % 64);
                }
            }
            set
        });
        let total: f32 = layer.assets.iter().map(|choice| choice.weight).sum();
        held.assets.clear();
        held.assets.extend(layer.assets.iter().filter_map(|choice| {
            let asset = rules.assets.iter().position(|known| *known == choice.asset)?;
            Some((asset as u16, choice.weight / total.max(f32::MIN_POSITIVE)))
        }));
    }
}
