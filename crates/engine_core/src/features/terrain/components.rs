//! A terrain as entities author it: where its tiles come from, its surface
//! and the objects placed on it, its edits, and what development shows of
//! it.

use super::*;

/// Where a terrain's tiles come from.
#[derive(Clone, Debug, PartialEq)]
pub enum TerrainSource {
    /// Generated in the engine from a layer stack, a tile at a time.
    Generated(Arc<Layers>),
    /// Read from an `ETR1` file: its header and directory, and the URL
    /// hosts fetch its tiles' byte ranges from, or the whole file, which
    /// tiles then decode from without fetching.
    File {
        header: Arc<FileHeader>,
        url: Arc<str>,
        data: Option<Arc<[u8]>>,
    },
}

/// How a terrain's ground looks: the rules choosing each
/// sample's materials (its biomes), and how each material looks, in the
/// rules' order.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainSurface {
    pub rules: Arc<BiomeRules>,
    pub materials: Vec<SurfaceMaterial>,
    /// What the rules' placement layers stand on the terrain,
    /// in the rules' order of asset IDs.
    pub assets: Vec<PlacementAsset>,
    /// The material its roads' bridges draw with; without
    /// one, a weathered stone.
    pub bridge: Option<MaterialHandle>,
}

/// An object placement layers stand on a terrain: the mesh
/// it draws and the material it draws with, the coarser meshes it draws at
/// a distance, finest first (without them, the mesh's automatic levels of
/// detail), how far it sways in the wind:
/// meters at ten meters up in full wind, its bend growing with the square
/// of the height, 0 (rigid) to 10, and its objects' collider,
/// if they collide.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacementAsset {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub lods: Vec<PlacementLod>,
    pub sway: f32,
    pub collider: Option<PlacedCollider>,
}

impl PlacementAsset {
    /// The rule of [`Self::sway`], which terrains check and authored
    /// placement assets' schemas state.
    pub const SWAY: Rule = Rule::range(0.0, 10.0);
}

/// The collider of a placement asset's objects, in the
/// asset's mesh space at scale 1: its primitive about `center`, and the
/// collision layers the objects belong to and collide with. Objects within
/// terrains' collision radius collide; those standing far away
/// never do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedCollider {
    pub shape: PlacedShape,
    pub center: [f32; 3],
    pub memberships: u32,
    pub filter: u32,
}

/// Local dimensions admitted by collision shapes, independent of render
/// meshes: placed colliders', scenes' and the collision world's.
pub const MIN_COLLIDER_EXTENT: f32 = 1.0e-4;

pub const MAX_COLLIDER_EXTENT: f32 = 1.0e4;

/// A placed collider's primitive: a box's half extents or a sphere's
/// radius, each from [`MIN_COLLIDER_EXTENT`] to [`MAX_COLLIDER_EXTENT`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlacedShape {
    Cuboid { half_extents: [f32; 3] },
    Sphere { radius: f32 },
}

impl PlacedCollider {
    /// A collider of `shape` about `center` in the default layers: a
    /// member of the first, colliding with every one.
    pub fn new(shape: PlacedShape, center: [f32; 3]) -> Self {
        Self {
            shape,
            center,
            memberships: 1,
            filter: u32::MAX,
        }
    }

    /// Whether its dimensions are in range and its center finite.
    fn is_usable(&self) -> bool {
        let usable = |value: f32| (MIN_COLLIDER_EXTENT..=MAX_COLLIDER_EXTENT).contains(&value);
        let dimensions = match self.shape {
            PlacedShape::Cuboid { half_extents } => half_extents.iter().all(|&half| usable(half)),
            PlacedShape::Sphere { radius } => usable(radius),
        };
        dimensions && self.center.iter().all(|value| value.is_finite())
    }
}

/// A placed collider keeps the collision extents, about a finite center.
impl Setting for PlacedCollider {
    fn check(&self, field: &'static str, _: Rule) -> Result<(), InvalidSetting> {
        if self.is_usable() {
            Ok(())
        } else {
            Err(InvalidSetting::new(
                field,
                format!(
                    "needs dimensions in [{MIN_COLLIDER_EXTENT}, {MAX_COLLIDER_EXTENT}] and a finite center"
                ),
            ))
        }
    }
}

/// A coarser mesh of a placement asset and the largest deviation from the
/// asset's mesh it shows, in mesh units: it draws where that projects
/// within the pixel error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacementLod {
    pub mesh: MeshHandle,
    pub error: f32,
}

/// Coarser meshes a placement asset lists at most: renderers list each
/// object's levels in a fixed number of planes.
pub const MAX_PLACEMENT_LODS: usize = 4;

impl PlacementLod {
    /// Checks a placement asset's coarser meshes' `errors`, in order: at
    /// most [`MAX_PLACEMENT_LODS`], each finite, positive and greater than
    /// the last. Errors name the offending index (none for the count) and
    /// the rule.
    pub fn check(errors: &[f32]) -> Result<(), (Option<usize>, &'static str)> {
        if errors.len() > MAX_PLACEMENT_LODS {
            return Err((None, "lists at most 4 meshes"));
        }
        let mut last = 0.0;
        for (level, &error) in errors.iter().enumerate() {
            if !(error.is_finite() && error > last) {
                return Err((
                    Some(level),
                    "must be finite, positive and grow with each mesh",
                ));
            }
            last = error;
        }
        Ok(())
    }
}

settings! {
    unauthored;
    /// How a surface material looks: an engine material's maps and factors,
    /// or without one a linear `color` and `roughness`; how its textures
    /// repeat and blend; and the cover growing where it lies.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct SurfaceMaterial {
        /// The engine material shading it, if any.
        pub material: Option<MaterialHandle> = None => any,
        /// Linear color without a material.
        pub color: [f32; 3] = [0.5; 3] => range(0.0, 1.0),
        /// Roughness without a material.
        pub roughness: f32 = 0.9 => range(0.03, 1.0),
        /// The meters its textures repeat over.
        pub tiling: f32 = 4.0 => range(0.01, 1000.0),
        /// How sharply its height map wins where it blends with another: 0
        /// even, 1 sharpest.
        pub height_contrast: f32 = 0.5 => range(0.0, 1.0),
        /// Whether steep ground projects it from the side.
        pub projection: SurfaceProjection = SurfaceProjection::Top => any,
        /// The cover growing where it lies.
        pub cover: Option<crate::Cover> = None => any,
        /// The meters its maps' heights span where relief sinks the ground
        /// between its stones and tufts: 0 flat. Top projections only.
        pub relief: f32 = 0.0 => range(0.0, 1.0),
        /// The meters tracks sink it at most: 0 rigid.
        pub deform: f32 = 0.0 => range(0.0, 2.0),
    }
}

/// How a surface material's textures lie on the ground: from above, or
/// on steep ground also from the side, so cliffs don't stretch them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SurfaceProjection {
    #[default]
    Top,
    Biplanar,
}

impl Setting for SurfaceProjection {}

impl SurfaceMaterial {
    /// A material of plain linear `color` and `roughness`, its textures (if
    /// it gains any) repeating as the defaults do.
    pub fn color(color: [f32; 3], roughness: f32) -> Self {
        Self {
            color,
            roughness,
            ..Self::default()
        }
    }
}

impl TerrainSurface {
    /// The surface of `surface`'s rules with `materials`, one per
    /// material it names, in order.
    pub fn new(surface: &Biomes, materials: Vec<SurfaceMaterial>) -> Result<Self, BiomeError> {
        let rules = surface.compile()?;
        if materials.len() != surface.materials.len() {
            return Err(BiomeError {
                path: "materials".into(),
                rule: "needs one look per material".into(),
            });
        }
        Ok(Self {
            rules: Arc::new(rules),
            materials,
            assets: Vec::new(),
            bridge: None,
        })
    }

    /// The surface with `assets`, one per asset its rules name, in order.
    pub fn with_assets(mut self, assets: Vec<PlacementAsset>) -> Result<Self, BiomeError> {
        let fail = |path: String, rule: &str| {
            Err(BiomeError {
                path,
                rule: rule.into(),
            })
        };
        if assets.len() != self.rules.authored().assets.len() {
            return fail("assets".into(), "needs one mesh and material per asset");
        }
        for (index, asset) in assets.iter().enumerate() {
            if let Err(invalid) = asset.sway.check("sway", PlacementAsset::SWAY) {
                return fail(format!("assets[{index}].sway"), &invalid.rule);
            }
            if let Some(Err(invalid)) = asset
                .collider
                .map(|collider| collider.check("collider", Rule::Any))
            {
                return fail(format!("assets[{index}].collider"), &invalid.rule);
            }
            let errors: Vec<f32> = asset.lods.iter().map(|lod| lod.error).collect();
            if let Err((level, rule)) = PlacementLod::check(&errors) {
                let path = match level {
                    Some(level) => format!("assets[{index}].lods[{level}].error"),
                    None => format!("assets[{index}].lods"),
                };
                return fail(path, rule);
            }
        }
        self.assets = assets;
        Ok(self)
    }

    /// The first value breaking its rule, if any: one material and asset
    /// for each the rules name, every material valid.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        if self.materials.len() != self.rules.authored().materials.len() {
            return Err(InvalidSetting::new(
                "materials",
                "must hold one material for each the rules name",
            ));
        }
        if self.assets.len() != self.rules.authored().assets.len() {
            return Err(InvalidSetting::new(
                "assets",
                "must hold one asset for each the rules name",
            ));
        }
        for material in &self.materials {
            material
                .validate()
                .map_err(|invalid| invalid.within("materials"))?;
        }
        Ok(())
    }

    /// The default surface's rules as authored: grass, bare dirt where noise
    /// and slope open it up, rock on steep ground and snow above
    /// `snow_height` meters on gentler slopes.
    pub fn default_rules(snow_height: f32) -> Biomes {
        let noise = |wavelength: f32, seed: u32| {
            Mask::Noise(NoiseMask {
                wavelength,
                seed,
                octaves: 2,
            })
        };
        let scaled = |mask: Mask, by: f32| Mask::Multiply(vec![mask, Mask::Constant(by)]);
        let remap = |mask: Mask, from: f32, to: f32| {
            Mask::Remap(Remap {
                mask: Box::new(mask),
                from,
                to,
            })
        };
        let layer = |material: &str, mask: Mask| BiomeLayer {
            material: material.into(),
            mask,
            opacity: 1.0,
        };
        Biomes {
            assets: Vec::new(),
            placements: Vec::new(),
            placed: Vec::new(),
            materials: ["grass", "dirt", "rock", "snow"].map(String::from).to_vec(),
            biomes: vec![Biome {
                id: "ground".into(),
                fitness: Mask::one(),
                layers: vec![
                    layer("grass", Mask::one()),
                    layer(
                        "dirt",
                        remap(
                            Mask::Add(vec![
                                scaled(noise(60.0, 1), 0.7),
                                scaled(noise(9.0, 2), 0.3),
                                scaled(Mask::Slope([10.0, 40.0]), 0.5),
                            ]),
                            0.55,
                            0.75,
                        ),
                    ),
                    layer(
                        "rock",
                        remap(
                            Mask::Add(vec![Mask::Slope([25.0, 45.0]), scaled(noise(60.0, 1), 0.3)]),
                            0.55,
                            0.75,
                        ),
                    ),
                    layer(
                        "snow",
                        Mask::Multiply(vec![
                            remap(
                                Mask::Add(vec![
                                    Mask::Height([snow_height - 100.0, snow_height + 100.0]),
                                    scaled(noise(60.0, 3), 0.4),
                                    Mask::Constant(-0.2),
                                ]),
                                0.4,
                                0.6,
                            ),
                            Mask::Invert(Box::new(Mask::Slope([35.0, 50.0]))),
                        ]),
                    ),
                ],
            }],
        }
    }

    /// The default materials' looks, matching [`Self::default_rules`].
    pub fn default_materials() -> Vec<SurfaceMaterial> {
        vec![
            SurfaceMaterial::color([0.105, 0.135, 0.045], 0.92),
            SurfaceMaterial::color([0.21, 0.165, 0.11], 0.92),
            SurfaceMaterial::color([0.25, 0.235, 0.22], 0.82),
            SurfaceMaterial::color([0.82, 0.84, 0.88], 0.55),
        ]
    }
}

impl Default for TerrainSurface {
    fn default() -> Self {
        Self::new(&Self::default_rules(650.0), Self::default_materials())
            .expect("the default surface is valid")
    }
}

/// A heightfield terrain whose first corner (least x and z)
/// lies at `origin`, heights measured up from it. Its pyramid sets its
/// level-0 tiles and spacing; its source gives its tiles.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Terrain {
    pub origin: Vec3,
    pub pyramid: Pyramid,
    pub source: TerrainSource,
    pub surface: TerrainSurface,
    /// Whether it collides where collision is installed.
    pub collision: bool,
    /// Edits over its generated layers and their bakes, in
    /// order: replacing them regenerates only the tiles under the edits
    /// added or taken away.
    pub edits: Arc<[TerrainEdit]>,
    /// Ground generated around it to the horizon.
    pub surroundings: Option<Surroundings>,
}

/// One edit of a generated terrain: a stamp over its layers
/// and bakes (terrain-local), under an `id` that repeating it with leaves
/// alone.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainEdit {
    pub id: Arc<str>,
    pub stamp: Stamp,
}

impl Terrain {
    /// A terrain generated from `layers`.
    pub fn generated(origin: Vec3, pyramid: Pyramid, layers: Layers) -> Self {
        Self {
            origin,
            pyramid,
            source: TerrainSource::Generated(Arc::new(layers)),
            surface: TerrainSurface::default(),
            collision: true,
            edits: Arc::from(Vec::new()),
            surroundings: None,
        }
    }

    #[must_use]
    pub fn with_surface(mut self, surface: TerrainSurface) -> Self {
        self.surface = surface;
        self
    }

    /// The terrain with ground generated around it to the horizon.
    #[must_use]
    pub fn with_surroundings(mut self, surroundings: Surroundings) -> Self {
        self.surroundings = Some(surroundings);
        self
    }

    /// The first value breaking its rule, if any: streaming uses only
    /// valid terrains.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        if !self.origin.is_finite() {
            return Err(InvalidSetting::new("origin", "must be finite"));
        }
        self.surface
            .validate()
            .map_err(|invalid| invalid.within("surface"))
    }
}

/// What terrain shows: its material, or for development its
/// levels of detail, normals or tiles and patches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TerrainView {
    #[default]
    Shaded,
    Levels,
    Normals,
    Tiles,
    /// Each surface material in its own color, blended as the ground is.
    Materials,
    /// The share of the sky the ground sees, unlit, from
    /// black (none) to white (all of it).
    Sky,
    /// How much detail the ground shows: green where all
    /// its maps show, yellow where their color alone does, red where their
    /// means stand in, blue where it shows as its cover's canopy.
    Detail,
}

/// Temporary terrain state for development: an eye that
/// streaming and level of detail follow instead of the camera's, so agents
/// can look at the patches it selects from elsewhere, and what terrain
/// shows. Absent, terrain follows the camera, shaded.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct TerrainDebug {
    pub freeze: Option<Vec3>,
    pub view: TerrainView,
}

settings! {
    /// Names a place the camera may be sent to, such as a spawn point:
    /// while the view wants nothing, workers generate the terrain tiles a camera
    /// there wants into memory, as they do ahead of the camera, and those tiles
    /// stay there while it is named, so the camera sent there finds them at once.
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct TerrainPrefetch {
        /// Each level's tiles within this many times its range; 2 is what a
        /// camera there loads.
        pub reach: f32 = LOAD_REACH => range(0.0, 8.0),
    }
}
