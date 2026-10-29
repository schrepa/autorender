use super::*;

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
    pub lods: Vec<PlacementLod>,
    pub parts: Vec<PlacementPart>,
    pub shadow: Option<MeshHandle>,
    pub sway: f32,
    pub collider: Option<PlacedCollider>,
}

impl PlacementAsset {
    pub const SWAY: Rule = Rule::range(0.0, 10.0);

    pub fn solid(&self, assets: Option<&AssetLibrary>) -> Option<Solid> {
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
        let bounds = assets?.mesh(self.mesh)?.bounds();
        let half = 0.5 * (bounds.max() - bounds.min());
        (half.min_element() >= MIN_COLLIDER_EXTENT).then(|| Solid {
            center: (0.5 * (bounds.max() + bounds.min())).to_array(),
            half: half.to_array(),
            round: false,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlacementPart {
    pub material: MaterialHandle,
    pub meshes: Vec<MeshHandle>,
}

pub const MAX_PLACEMENT_PARTS: usize = 2;

impl PlacementPart {
    pub fn check(parts: &[usize], levels: usize) -> Result<(), (Option<usize>, &'static str)> {
        if parts.len() > MAX_PLACEMENT_PARTS {
            return Err((None, "lists at most 2 parts"));
        }
        match parts
            .iter()
            .position(|&meshes| meshes == 0 || meshes > levels)
        {
            Some(index) => Err((
                Some(index),
                "needs a mesh, and one for each of the asset's levels at most",
            )),
            None => Ok(()),
        }
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
            Err(InvalidSetting::new(
                field,
                format!(
                    "needs dimensions in [{MIN_COLLIDER_EXTENT}, {MAX_COLLIDER_EXTENT}] and a finite center"
                ),
            ))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacementLod {
    pub mesh: MeshHandle,
    pub error: f32,
}

pub const MAX_PLACEMENT_LODS: usize = 4;

impl PlacementLod {
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
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct SurfaceMaterial {
        pub material: Option<MaterialHandle> = None => any,
        pub color: [f32; 3] = [0.5; 3] => range(0.0, 1.0),
        pub roughness: f32 = 0.9 => range(0.03, 1.0),
        pub tiling: f32 = 4.0 => range(0.01, 1000.0),
        pub height_contrast: f32 = 0.5 => range(0.0, 1.0),
        pub projection: SurfaceProjection = SurfaceProjection::Top => any,
        pub cover: Option<Cover> = None => any,
        pub depth: f32 = 0.0 => range(0.0, 2.0),
    }
}

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
    pub fn color(color: [f32; 3], roughness: f32) -> Self {
        Self {
            color,
            roughness,
            ..Self::default()
        }
    }
}

impl TerrainSurface {
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
            let parts: Vec<usize> = asset.parts.iter().map(|part| part.meshes.len()).collect();
            if let Err((part, rule)) = PlacementPart::check(&parts, 1 + asset.lods.len()) {
                let path = match part {
                    Some(part) => format!("assets[{index}].parts[{part}].meshes"),
                    None => format!("assets[{index}].parts"),
                };
                return fail(path, rule);
            }
        }
        self.assets = assets;
        Ok(self)
    }

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
                            ]),
                            0.58,
                            0.72,
                        ),
                    ),
                    layer(
                        "snow",
                        remap(
                            Mask::Add(vec![
                                Mask::Height([snow_height - 100.0, snow_height + 100.0]),
                                scaled(noise(60.0, 3), 0.4),
                                Mask::Constant(-0.2),
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

#[derive(Component, Clone, Debug, PartialEq)]
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
            return Err(InvalidSetting::new("origin", "must be finite"));
        }
        self.surface
            .validate()
            .map_err(|invalid| invalid.within("surface"))
    }
}

settings! {
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct TerrainPrefetch {
        pub reach: f32 = LOAD_REACH => range(0.0, 8.0),
    }
}
