//! Scene version 9 terrain (decision 051): an entity field describing a
//! heightfield terrain, generated from a layer stack or read from an `ETR1`
//! file beside the scene, and its surface (decision 054): the materials its
//! ground is made of and the biomes deciding where each lies.

use std::sync::Arc;

use engine_core::{
    Biome, Biomes, Blend, HeightImage, HeightmapLayer, Layer, Layers, MaterialHandle, NoiseLayer,
    Pyramid, Stamp, SurfaceMaterial, SurfaceProjection, Terrain, TerrainSource, TerrainSurface,
    glam::{Vec2, Vec3},
};
use serde::{Deserialize, Serialize};

use crate::{SceneError, SceneFiles, SceneMaterial, files::valid_uri};

/// A heightfield terrain on this entity (decision 051). Requires version 9.
///
/// Its first corner (least x and z) lies at `origin` in world space,
/// whatever the entity's transform; heights rise from the origin's y. Its
/// heights come from `layers`, applied in order to every sample (later
/// layers win), or from an `ETR1` `file` beside the scene, whose tiles
/// stream by byte range. Level-0 tiles span 256 samples `spacing` meters
/// apart; coarser levels halve the tiles up to one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrain {
    pub origin: [f32; 3],
    /// Level-0 tiles along x and z, 1 to 4096 each. Required with layers;
    /// a file's own when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiles: Option<[u32; 2]>,
    /// Meters between level-0 samples, 1/64 to 1024. Required with layers;
    /// a file's own when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spacing: Option<f32>,
    /// The layers generating its heights, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<SceneTerrainLayer>,
    /// An `ETR1` terrain file beside the scene, instead of layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Its materials and biomes. Omitted, the engine's default: grass,
    /// bare dirt, rock on steep ground and snow above 650 meters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<SceneTerrainSurface>,
    /// Whether it collides where collision is installed. Omitted, true.
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub collision: bool,
}

fn yes() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

/// A terrain's surface (decision 054): up to 255 materials, which biome
/// layers name by ID, and up to 255 biomes. Later biomes paint over earlier
/// ones where they fit; the first covers wherever none of the others do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainSurface {
    pub materials: Vec<SceneTerrainMaterial>,
    pub biomes: Vec<Biome>,
}

/// How one surface material looks: a `material` of this document whose
/// maps and factors shade it, or a linear `color` and `roughness`; the
/// meters its textures repeat over (0.01 to 1000); how sharply its height
/// map wins where it blends with another (0 to 1); and whether steep ground
/// projects it from the side.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainMaterial {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default = "gray")]
    pub color: [f32; 3],
    #[serde(default = "rough")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.03, max = 1.0)))]
    pub roughness: f32,
    #[serde(default = "four_meters")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.01, max = 1000.0)))]
    pub tiling: f32,
    #[serde(default = "half")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub height_contrast: f32,
    #[serde(default)]
    pub projection: SurfaceProjection,
}

fn gray() -> [f32; 3] {
    [0.5; 3]
}

fn rough() -> f32 {
    0.9
}

fn four_meters() -> f32 {
    4.0
}

fn half() -> f32 {
    0.5
}

/// A validated terrain whose surface materials name this document's
/// materials by index, which instantiation turns into handles.
#[derive(Clone, Debug)]
pub(crate) struct PreparedTerrain {
    pub terrain: Terrain,
    /// Each surface material's document material, if it names one.
    pub materials: Vec<Option<usize>>,
}

impl PreparedTerrain {
    /// The component, its surface materials resolved through the
    /// document's `materials` handles.
    pub(crate) fn runtime(&self, materials: &[MaterialHandle]) -> Terrain {
        let mut terrain = self.terrain.clone();
        for (look, index) in terrain.surface.materials.iter_mut().zip(&self.materials) {
            look.material = index.map(|index| materials[index]);
        }
        terrain
    }
}

/// One layer of a terrain's stack: procedural noise with erosion, a stamp
/// (raise, lower, flatten or cut holes), or an imported heightmap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneTerrainLayer {
    Noise(NoiseLayer),
    Stamp(Stamp),
    Heightmap(SceneHeightmap),
}

/// An imported heightmap: a file of little-endian 16-bit samples, row by row
/// along z (`.r16`), `width` by `height` of them, spanning `size` meters
/// from terrain-local `min` (x and z). Samples 0 to 65535 map to heights
/// `low` to `high`; the layer fades in over `falloff` meters inside its
/// edges and combines by `blend`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneHeightmap {
    pub uri: String,
    #[cfg_attr(feature = "schema", schemars(range(min = 2, max = 16384)))]
    pub width: u32,
    #[cfg_attr(feature = "schema", schemars(range(min = 2, max = 16384)))]
    pub height: u32,
    pub min: [f32; 2],
    pub size: [f32; 2],
    pub low: f32,
    pub high: f32,
    #[serde(default)]
    pub falloff: f32,
    #[serde(default)]
    pub blend: Blend,
}

/// The largest heightmap side, in samples.
const MAX_HEIGHTMAP: u32 = 16_384;

impl SceneTerrain {
    /// The files it references: its heightmaps (buffers) and its terrain
    /// file.
    pub(crate) fn files(&self) -> impl Iterator<Item = (&String, bool)> {
        self.layers
            .iter()
            .filter_map(|layer| match layer {
                SceneTerrainLayer::Heightmap(map) => Some((&map.uri, false)),
                _ => None,
            })
            .chain(self.file.iter().map(|uri| (uri, true)))
    }

    /// The surface at authored path `path`, its materials naming the
    /// document's `materials`: the component's surface and each surface
    /// material's document material index.
    fn surface(
        &self,
        path: &str,
        materials: &[SceneMaterial],
    ) -> Result<(TerrainSurface, Vec<Option<usize>>), SceneError> {
        let invalid = |field: String, rule: String| {
            SceneError::new("invalid_terrain", format!("{path}.surface.{field}"), rule)
        };
        let Some(surface) = &self.surface else {
            let looks = TerrainSurface::default();
            let count = looks.materials.len();
            return Ok((looks, vec![None; count]));
        };
        let mut looks = Vec::with_capacity(surface.materials.len());
        let mut indices = Vec::with_capacity(surface.materials.len());
        for (index, material) in surface.materials.iter().enumerate() {
            let at = format!("materials[{index}]");
            let look = SurfaceMaterial {
                material: None,
                color: material.color,
                roughness: material.roughness,
                tiling: material.tiling,
                height_contrast: material.height_contrast,
                projection: material.projection,
            };
            if !look.is_usable() {
                return Err(invalid(
                    at,
                    "color must be in [0, 1], roughness in [0.03, 1], tiling in [0.01, 1000] and height_contrast in [0, 1]".into(),
                ));
            }
            let document = material
                .material
                .as_ref()
                .map(|id| {
                    materials
                        .iter()
                        .position(|candidate| candidate.id == *id)
                        .ok_or_else(|| {
                            invalid(
                                format!("{at}.material"),
                                "must name a material of this document".into(),
                            )
                        })
                })
                .transpose()?;
            looks.push(look);
            indices.push(document);
        }
        let biomes = Biomes {
            materials: surface
                .materials
                .iter()
                .map(|material| material.id.clone())
                .collect(),
            biomes: surface.biomes.clone(),
        };
        let surface =
            TerrainSurface::new(&biomes, looks).map_err(|error| invalid(error.path, error.rule))?;
        Ok((surface, indices))
    }

    /// The component at authored path `path`, validated; `files` supplies
    /// heightmaps and terrain files, or with `None` they read as
    /// placeholders, and its surface materials may name the document's
    /// `materials`.
    pub(crate) fn prepare(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
        materials: &[SceneMaterial],
    ) -> Result<PreparedTerrain, SceneError> {
        let (surface, indices) = self.surface(path, materials)?;
        let terrain = self.terrain(path, files, surface)?;
        Ok(PreparedTerrain {
            terrain,
            materials: indices,
        })
    }

    /// The component at authored path `path`, validated, for tools reading
    /// a document without instantiating it: its surface materials keep no
    /// engine material.
    pub fn prepare_component(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
        materials: &[SceneMaterial],
    ) -> Result<Terrain, SceneError> {
        Ok(self.prepare(path, files, materials)?.terrain)
    }

    fn terrain(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
        surface: TerrainSurface,
    ) -> Result<Terrain, SceneError> {
        let invalid = |field: &str, rule: String| {
            SceneError::new("invalid_terrain", format!("{path}.{field}"), rule)
        };
        if !self.origin.iter().all(|value| value.is_finite()) {
            return Err(invalid("origin", "must be finite".into()));
        }
        let origin = Vec3::from_array(self.origin);
        let source = match (&self.file, self.layers.is_empty()) {
            (Some(_), false) => {
                return Err(invalid(
                    "file",
                    "a terrain has layers or a file, not both".into(),
                ));
            }
            (None, true) => {
                return Err(invalid("layers", "a terrain needs layers or a file".into()));
            }
            (Some(uri), true) => {
                valid_uri(uri, &format!("{path}.file"))?;
                let Some(files) = files else {
                    // Structure only: the file's own dimensions are unknown.
                    let pyramid =
                        Pyramid::new(self.tiles.unwrap_or([1, 1]), self.spacing.unwrap_or(1.0))
                            .map_err(|error| invalid("tiles", error.to_string()))?;
                    return Ok(Terrain {
                        origin,
                        pyramid,
                        source: TerrainSource::Generated(Arc::new(
                            Layers::new(Vec::new()).expect("no layers are valid"),
                        )),
                        surface,
                        collision: self.collision,
                    });
                };
                let (header, url, data) = files.terrain(uri, &format!("{path}.file"))?;
                let pyramid = *header.pyramid();
                if self
                    .tiles
                    .is_some_and(|tiles| tiles != pyramid.base_tiles())
                    || self
                        .spacing
                        .is_some_and(|spacing| spacing != pyramid.spacing(0))
                {
                    return Err(invalid(
                        "tiles",
                        "tiles and spacing, when given, must match the file's".into(),
                    ));
                }
                return Ok(Terrain {
                    origin,
                    pyramid,
                    source: TerrainSource::File {
                        header: Arc::new(header),
                        url: url.unwrap_or_default().into(),
                        data: data.map(Arc::from),
                    },
                    surface,
                    collision: self.collision,
                });
            }
            (None, false) => self.layers(path, files)?,
        };
        let (Some(tiles), Some(spacing)) = (self.tiles, self.spacing) else {
            return Err(invalid(
                "tiles",
                "terrains generated from layers need tiles and spacing".into(),
            ));
        };
        let pyramid =
            Pyramid::new(tiles, spacing).map_err(|error| invalid("tiles", error.to_string()))?;
        Ok(Terrain {
            origin,
            pyramid,
            source: TerrainSource::Generated(Arc::new(source)),
            surface,
            collision: self.collision,
        })
    }

    fn layers(&self, path: &str, files: Option<&SceneFiles>) -> Result<Layers, SceneError> {
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let at = format!("{path}.layers[{index}]");
                Ok(match layer {
                    SceneTerrainLayer::Noise(noise) => Layer::Noise(*noise),
                    SceneTerrainLayer::Stamp(stamp) => Layer::Stamp(*stamp),
                    SceneTerrainLayer::Heightmap(map) => {
                        Layer::Heightmap(map.prepare(&format!("{at}.heightmap"), files)?)
                    }
                })
            })
            .collect::<Result<Vec<_>, SceneError>>()?;
        Layers::new(layers).map_err(|error| {
            SceneError::new(
                "invalid_terrain",
                format!("{path}.layers[{}].{}", error.layer, error.field),
                format!("must be finite and in [{}, {}]", error.low, error.high),
            )
        })
    }

    /// The scene form of `terrain`: its file by `file`'s name when read from
    /// one, heightmaps by the names their layers keep, and its surface
    /// (omitted when the default) by `materials`' names for the materials
    /// it uses.
    pub fn from_runtime(
        terrain: &Terrain,
        file: Option<&str>,
        materials: impl Fn(MaterialHandle) -> Option<String>,
    ) -> Self {
        let surface = (terrain.surface != TerrainSurface::default()).then(|| {
            let authored = terrain.surface.rules.authored();
            SceneTerrainSurface {
                materials: authored
                    .materials
                    .iter()
                    .zip(&terrain.surface.materials)
                    .map(|(id, look)| SceneTerrainMaterial {
                        id: id.clone(),
                        material: look.material.and_then(&materials),
                        color: look.color,
                        roughness: look.roughness,
                        tiling: look.tiling,
                        height_contrast: look.height_contrast,
                        projection: look.projection,
                    })
                    .collect(),
                biomes: authored.biomes.clone(),
            }
        });
        let layers = match &terrain.source {
            TerrainSource::Generated(layers) => layers
                .layers()
                .iter()
                .map(|layer| match layer {
                    Layer::Noise(noise) => SceneTerrainLayer::Noise(*noise),
                    Layer::Stamp(stamp) => SceneTerrainLayer::Stamp(*stamp),
                    Layer::Heightmap(map) => SceneTerrainLayer::Heightmap(SceneHeightmap {
                        uri: map.uri.clone().unwrap_or_default(),
                        width: map.image.width(),
                        height: map.image.height(),
                        min: map.min.to_array(),
                        size: map.size.to_array(),
                        low: map.low,
                        high: map.high,
                        falloff: map.falloff,
                        blend: map.blend,
                    }),
                })
                .collect(),
            TerrainSource::File { .. } => Vec::new(),
        };
        let generated = matches!(terrain.source, TerrainSource::Generated(_));
        Self {
            origin: terrain.origin.to_array(),
            tiles: generated.then(|| terrain.pyramid.base_tiles()),
            spacing: generated.then(|| terrain.pyramid.spacing(0)),
            layers,
            file: match &terrain.source {
                TerrainSource::File { url, .. } => Some(
                    file.map(str::to_owned)
                        .unwrap_or_else(|| url.rsplit('/').next().unwrap_or_default().to_owned()),
                ),
                TerrainSource::Generated(_) => None,
            },
            surface,
            collision: terrain.collision,
        }
    }
}

impl SceneHeightmap {
    fn prepare(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
    ) -> Result<HeightmapLayer, SceneError> {
        valid_uri(&self.uri, &format!("{path}.uri"))?;
        if !(2..=MAX_HEIGHTMAP).contains(&self.width) || !(2..=MAX_HEIGHTMAP).contains(&self.height)
        {
            return Err(SceneError::new(
                "invalid_terrain",
                format!("{path}.width"),
                format!("width and height must be 2 to {MAX_HEIGHTMAP} samples"),
            ));
        }
        let image = match files {
            Some(files) => {
                let bytes = files.buffer(&self.uri, &format!("{path}.uri"))?;
                HeightImage::from_le_bytes(self.width, self.height, bytes).ok_or_else(|| {
                    SceneError::new(
                        "invalid_terrain",
                        format!("{path}.uri"),
                        format!(
                            "a {} by {} heightmap holds {} bytes of 16-bit samples",
                            self.width,
                            self.height,
                            self.width as usize * self.height as usize * 2
                        ),
                    )
                })?
            }
            // Structure only: a flat stand-in of the declared size.
            None => HeightImage::new(
                self.width,
                self.height,
                vec![0; self.width as usize * self.height as usize],
            )
            .expect("declared dimensions checked above"),
        };
        Ok(HeightmapLayer {
            image: Arc::new(image),
            min: Vec2::from_array(self.min),
            size: Vec2::from_array(self.size),
            low: self.low,
            high: self.high,
            falloff: self.falloff,
            blend: self.blend,
            uri: Some(self.uri.clone()),
        })
    }
}
