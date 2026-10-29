use super::*;
use crate::Reach;
use engine_core::{EnvironmentSection, WaterReach};
use std::sync::LazyLock;

#[derive(Clone, Debug)]
pub struct RenderSurfaceMaps {
    pub material: Material,
    pub base_color: Option<RenderTexture>,
    pub normal: Option<RenderTexture>,
    pub metallic_roughness: Option<RenderTexture>,
}

impl RenderSurfaceMaps {
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

#[derive(Clone, Debug)]
pub struct RenderPlacementPart {
    pub material: RenderMaterial,
    pub meshes: Vec<RenderMesh>,
}

impl RenderPlacementAsset {
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

    pub fn meshes(&self) -> impl Iterator<Item = &RenderMesh> {
        std::iter::once(&self.mesh)
            .chain(self.lods.iter().map(|(lod, _)| lod))
            .chain(self.parts.iter().flat_map(|part| &part.meshes))
            .chain(&self.shadow)
    }

    pub fn materials(&self) -> impl Iterator<Item = &RenderMaterial> {
        std::iter::once(&self.material).chain(self.parts.iter().map(|part| &part.material))
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderPlacementLayer {
    pub density: f32,
    pub scale: [f32; 2],
    pub materials: Option<[u64; 4]>,
    pub assets: Vec<(u16, f32)>,
}

impl RenderPlacementLayer {
    pub fn stands_on(&self, material: usize) -> bool {
        self.materials
            .is_none_or(|set| material < 256 && set[material / 64] & (1 << (material % 64)) != 0)
    }
}

#[derive(Clone, Debug)]
pub struct RenderTerrain {
    pub id: u64,
    pub origin: Vec3,
    pub pyramid: Pyramid,
    pub surface: Vec<SurfaceMaterial>,
    pub steep: Option<crate::Steep>,
    pub maps: Vec<Option<RenderSurfaceMaps>>,
    pub assets: Vec<Option<RenderPlacementAsset>>,
    pub layers: Vec<RenderPlacementLayer>,
    pub tiles: Arc<[Arc<Tile>]>,
    pub tiles_revision: u64,
    pub drawn: bool,
    pub reach: Vec<f32>,
    pub errors: Vec<f32>,
    pub surroundings: bool,
    pub cutout: Option<(Vec2, Vec2)>,
    pub may_hole: bool,
    pub hydrology: bool,
    pub gpu_surface: Option<Arc<crate::gpu::SurfaceData>>,
    pub falls: Arc<[Fall]>,
    pub water_window: Option<Arc<WaterWindow>>,
    pub water_touches: Vec<engine_core::WaterObstacle>,
    pub water_touches_beyond: u32,
    pub lighting: Option<Arc<TerrainLighting>>,
    pub bounce: Option<Arc<TerrainBounce>>,
    pub macro_map: Option<MacroMap>,
}

#[derive(Clone, Debug)]
pub struct MacroMap {
    pub texture: Arc<engine_assets::Texture>,
    pub identity: u128,
}

#[derive(Clone, Debug, Default)]
pub struct RenderTerrains {
    pub terrains: Vec<RenderTerrain>,
    pub relative_view_projection: Mat4,
    pub stats: TerrainStats,
}

impl RenderTerrains {
    pub fn of(snapshot: &RenderSnapshot) -> &Self {
        static NONE: LazyLock<RenderTerrains> = LazyLock::new(RenderTerrains::default);
        snapshot.extracted.get::<Self>().unwrap_or(&*NONE)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainEnvironment {
    pub terrain: TerrainSettings,
    pub ground_cover: GroundCover,
    pub wind: Wind,
    pub occupancy: Occupancy,
    pub water: Water,
}

impl TerrainEnvironment {
    pub fn current(world: &World) -> Self {
        fn section<T: EnvironmentSection>(world: &World) -> T {
            world
                .get_resource::<T>()
                .copied()
                .unwrap_or_else(T::section_off)
                .section_sanitized()
        }
        Self {
            terrain: section(world),
            ground_cover: section(world),
            wind: section(world),
            occupancy: section(world),
            water: section(world),
        }
    }

    pub fn off() -> Self {
        fn off<T: EnvironmentSection>() -> T {
            T::section_off().section_sanitized()
        }
        Self {
            terrain: off(),
            ground_cover: off(),
            wind: off(),
            occupancy: off(),
            water: off(),
        }
    }

    pub fn of(snapshot: &RenderSnapshot) -> Self {
        snapshot
            .extracted
            .get::<Self>()
            .copied()
            .unwrap_or_else(Self::off)
    }
}

impl From<Reach> for WaterReach {
    fn from(reach: Reach) -> Self {
        match reach {
            Reach::Pool => Self::Pool,
            Reach::Plane => Self::Plane,
            Reach::Steps => Self::Steps,
            Reach::Cascade => Self::Cascade,
            Reach::Falls => Self::Falls,
        }
    }
}

pub(crate) fn extract_terrains(world: &World, snapshot: &mut RenderSnapshot) {
    snapshot.extracted.insert(TerrainEnvironment::current(world));
    if snapshot.extracted.get::<RenderTerrains>().is_none() {
        snapshot.extracted.insert(RenderTerrains::default());
    }
    let Some(listing) = snapshot.extracted.get_mut::<RenderTerrains>() else {
        return;
    };
    let missing = &mut snapshot.missing_assets;
    let assets = world.get_resource::<AssetLibrary>();
    let Some(terrains) = world.get_resource::<Terrains>() else {
        listing.terrains.clear();
        listing.stats = TerrainStats::default();
        return;
    };
    let camera = world.resource::<Camera>();
    let viewport = world
        .get_resource::<Viewport>()
        .copied()
        .unwrap_or_default();
    let aspect = viewport.width.max(1) as f32 / viewport.height.max(1) as f32;
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
                drawn: false,
                reach: Vec::new(),
                errors: Vec::new(),
                surroundings: false,
                cutout: None,
                may_hole: false,
                hydrology: false,
                gpu_surface: None,
                falls: Arc::from(Vec::new()),
                water_window: None,
                water_touches: Vec::new(),
                water_touches_beyond: 0,
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
        render.drawn = enabled && state.drawn;
        render.gpu_surface = if state.materials {
            None
        } else {
            state.surface.as_ref().and_then(|(_, data)| data.clone())
        };
        render.reach.clear();
        render
            .reach
            .extend(state.ranges.iter().map(|range| range * LOAD_REACH));
        render.errors.clear();
        render.errors.extend_from_slice(state.store.errors());
        render.falls = state.falls.clone();
        render.water_window = state.hydrology.window.clone();
        render.water_touches.clear();
        render.water_touches_beyond = 0;
        if let (Some(window), Some(obstacles)) = (
            state.hydrology.window.as_deref(),
            world.get_resource::<engine_core::WaterObstacles>(),
        ) {
            render.water_touches_beyond = window.touching(
                &obstacles.obstacles,
                camera.position,
                &mut render.water_touches,
            );
        }
        render.lighting.clone_from(&state.whole.lighting.made);
        render.bounce.clone_from(&state.whole.bounce.made);
        render.macro_map = state.whole.colors.map(assets);
    }
}

fn extract_layers(rules: &Biomes, layers: &mut Vec<RenderPlacementLayer>) {
    layers.truncate(rules.placements.len());
    for (index, layer) in rules.placements.iter().enumerate() {
        if index == layers.len() {
            layers.push(RenderPlacementLayer::default());
        }
        let held = &mut layers[index];
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
            let asset = rules
                .assets
                .iter()
                .position(|known| *known == choice.asset)?;
            Some((
                asset as u16,
                choice.weight / engine_math::max(total, f32::MIN_POSITIVE),
            ))
        }));
    }
}
