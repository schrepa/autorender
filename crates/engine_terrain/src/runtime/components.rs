use super::*;
use crate::surface::{setting_error, terrain_error};
use engine_places::solids::SolidSource;

#[derive(Clone, Debug, PartialEq)]
pub struct TerrainSurface {
    pub rules: Arc<BiomeRules>,
    pub materials: Vec<SurfaceMaterial>,
    pub assets: Vec<PlacementAsset>,
    pub bridge: Option<MaterialHandle>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlacementAsset {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub collider: Option<PlacedCollider>,
}

impl PlacementAsset {
    pub fn solid(&self, assets: &AssetLibrary) -> Option<Solid> {
        if let Some(collider) = self.collider {
            let (half, round) = match collider.shape {
                PlacedShape::Cuboid { half_extents } => (half_extents, false),
                PlacedShape::Sphere { radius } => ([radius; 3], true),
            };
            return Some(Solid {
                center: collider.center,
                half,
                round,
            });
        }
        let bounds = assets.mesh(self.mesh)?.bounds();
        let half = 0.5 * (bounds.max() - bounds.min());
        (half.min_element() >= MIN_COLLIDER_EXTENT).then(|| Solid {
            center: (0.5 * (bounds.max() + bounds.min())).to_array(),
            half: half.to_array(),
            round: false,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedCollider {
    pub shape: PlacedShape,
    pub center: [f32; 3],
    pub memberships: u32,
    pub filter: u32,
}

pub const MIN_COLLIDER_EXTENT: f32 = 1.0e-4;

pub const MAX_COLLIDER_EXTENT: f32 = 1.0e4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlacedShape {
    Cuboid { half_extents: [f32; 3] },
    Sphere { radius: f32 },
}

impl PlacedCollider {
    pub fn new(shape: PlacedShape, center: [f32; 3]) -> Self {
        Self {
            shape,
            center,
            memberships: 1,
            filter: u32::MAX,
        }
    }

    fn is_usable(&self) -> bool {
        let usable = |value: f32| (MIN_COLLIDER_EXTENT..=MAX_COLLIDER_EXTENT).contains(&value);
        let dimensions = match self.shape {
            PlacedShape::Cuboid { half_extents } => half_extents.iter().all(|&half| usable(half)),
            PlacedShape::Sphere { radius } => usable(radius),
        };
        dimensions && self.center.iter().all(|value| value.is_finite())
    }
}

impl Setting for PlacedCollider {
    fn check(&self, field: &'static str, _: Rule) -> Result<(), InvalidSetting> {
        if self.is_usable() {
            Ok(())
        } else {
            Err(
                InvalidSetting::new(field, engine_core::Expected::Finite).allowed(
                    engine_core::Allowed::Range {
                        min: MIN_COLLIDER_EXTENT as f64,
                        max: MAX_COLLIDER_EXTENT as f64,
                    },
                ),
            )
        }
    }
}

pub fn surface_color(color: [f32; 3], roughness: f32) -> SurfaceMaterial {
    SurfaceMaterial {
        color,
        roughness,
        ..SurfaceMaterial::default()
    }
}

pub fn check_material(material: &SurfaceMaterial) -> Result<(), InvalidSetting> {
    material.color.check("color", Rule::Range(0.0, 1.0))?;
    material.roughness.check("roughness", Rule::Range(0.03, 1.0))?;
    material.tiling.check("tiling", Rule::Range(0.01, 1000.0))?;
    material
        .height_contrast
        .check("height_contrast", Rule::Range(0.0, 1.0))?;
    material.depth.check("depth", Rule::Range(0.0, 2.0))
}

impl TerrainSurface {
    pub fn new(surface: &Biomes, materials: Vec<SurfaceMaterial>) -> Result<Self, Error> {
        let rules = surface.compile()?;
        let named = surface.materials.len();
        if materials.len() != named {
            return Err(terrain_error("$.materials")
                .range(named as f64, named as f64)
                .got(materials.len()));
        }
        Ok(Self {
            rules: Arc::new(rules),
            materials,
            assets: Vec::new(),
            bridge: None,
        })
    }

    pub fn with_assets(mut self, assets: Vec<PlacementAsset>) -> Result<Self, Error> {
        let named = self.rules.authored().assets.len();
        if assets.len() != named {
            return Err(terrain_error("$.assets")
                .range(named as f64, named as f64)
                .got(assets.len()));
        }
        for (index, asset) in assets.iter().enumerate() {
            let at = format!("$.assets[{index}]");
            if let Some(Err(invalid)) = asset
                .collider
                .map(|collider| collider.check("collider", Rule::Any))
            {
                return Err(setting_error(invalid, &at));
            }
        }
        self.assets = assets;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), InvalidSetting> {
        if self.materials.len() != self.rules.authored().materials.len() {
            return Err(InvalidSetting::new(
                "materials",
                engine_core::Expected::Matching,
            ));
        }
        if self.assets.len() != self.rules.authored().assets.len() {
            return Err(InvalidSetting::new(
                "assets",
                engine_core::Expected::Matching,
            ));
        }
        for material in &self.materials {
            check_material(material).map_err(|invalid| invalid.within("materials"))?;
        }
        Ok(())
    }

    pub fn default_rules(snow_height: f32) -> Biomes {
        let noise = |wavelength: f32, seed: u32| {
            Expr::Noise(NoiseMask {
                wavelength,
                seed,
                octaves: 2,
            })
        };
        let scaled = |mask: Expr, by: f32| Expr::Mul(vec![mask, Expr::Constant(by)]);
        let remap = |mask: Expr, from: f32, to: f32| {
            Expr::Remap(crate::expr::Remap {
                of: Box::new(mask),
                from,
                to,
            })
        };
        let layer = |material: &str, mask: Expr| BiomeLayer {
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
                fitness: Expr::one(),
                layers: vec![
                    layer("grass", Expr::one()),
                    layer(
                        "dirt",
                        remap(
                            Expr::Add(vec![
                                scaled(noise(60.0, 1), 0.7),
                                scaled(noise(9.0, 2), 0.3),
                            ]),
                            0.58,
                            0.72,
                        ),
                    ),
                    layer(
                        "snow",
                        remap(
                            Expr::Add(vec![
                                Expr::Ramp(crate::expr::Ramp {
                                    map: crate::layer::maps::HEIGHT.to_owned(),
                                    from: snow_height - 100.0,
                                    to: snow_height + 100.0,
                                }),
                                scaled(noise(60.0, 3), 0.4),
                                Expr::Constant(-0.2),
                            ]),
                            0.4,
                            0.6,
                        ),
                    ),
                ],
                wetness: None,
            }],
            steep: Some(crate::SteepLayer {
                material: "rock".into(),
                slope: [28.0, 35.0],
            }),
            macro_colors: crate::MacroCauses::default(),
        }
    }

    pub fn default_materials() -> Vec<SurfaceMaterial> {
        vec![
            surface_color([0.105, 0.135, 0.045], 0.92),
            surface_color([0.21, 0.165, 0.11], 0.92),
            surface_color([0.25, 0.235, 0.22], 0.82),
            surface_color([0.82, 0.84, 0.88], 0.55),
        ]
    }
}

impl Default for TerrainSurface {
    fn default() -> Self {
        Self::new(&Self::default_rules(650.0), Self::default_materials())
            .expect("default_surface_valid")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Terrain {
    pub origin: Vec3,
    pub pyramid: Pyramid,
    pub layers: Arc<Layers>,
    pub surface: TerrainSurface,
    pub collision: bool,
    pub edits: Arc<[TerrainEdit]>,
    pub rivers: Arc<[TerrainRiver]>,
    pub surroundings: Option<Surroundings>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TerrainEdit {
    pub id: Arc<str>,
    pub stamp: Stamp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TerrainRiver {
    pub id: Arc<str>,
    pub river: GivenRiver,
}

impl Terrain {
    pub fn generated(origin: Vec3, pyramid: Pyramid, layers: Layers) -> Self {
        Self {
            origin,
            pyramid,
            layers: Arc::new(layers),
            surface: TerrainSurface::default(),
            collision: true,
            edits: Arc::from(Vec::new()),
            rivers: Arc::from(Vec::new()),
            surroundings: None,
        }
    }

    #[must_use]
    pub fn with_surface(mut self, surface: TerrainSurface) -> Self {
        self.surface = surface;
        self
    }

    #[must_use]
    pub fn with_surroundings(mut self, surroundings: Surroundings) -> Self {
        self.surroundings = Some(surroundings);
        self
    }

    pub fn validate(&self) -> Result<(), InvalidSetting> {
        if !self.origin.is_finite() {
            return Err(InvalidSetting::new(
                "origin",
                engine_core::Expected::Finite,
            ));
        }
        self.surface
            .validate()
            .map_err(|invalid| invalid.within("surface"))
    }
}

#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct TerrainBuilt(pub Option<Terrain>);

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[require(TerrainBuilt, SolidSource)]
#[serde(transparent)]
pub struct TerrainScene(pub crate::scene::SceneTerrain);

fields! {
    #[derive(
        Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct TerrainPrefetch {
        pub reach: f32 = LOAD_REACH => range(0.0, 8.0),
    }
}
