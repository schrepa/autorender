use std::{collections::BTreeMap, io::Write};

use engine_collision::{Collider, CollisionWorld, refresh_collision_world};
use engine_core::{
    AnimationPlayer, AssetLibrary, Atmosphere, Camera, Decal, Engine, FixedTime, FrameTime,
    GlobalTransforms, InterpolateTransform, LocalLight, Mesh3d, Parent, ParticleEmitter,
    PresentationMode, PresentationState, PresentedTransforms, ReflectionProbe, Skin, Tint,
    Transform,
    bevy_ecs::prelude::{Component, Entity, Resource, World},
};
use engine_scene::{
    EnvironmentSection, SceneAtmosphere, SceneDecal, SceneEmitter, SceneEntity, SceneEnvironment,
    SceneInstance, SceneLight, SceneReflectionProbe, SectionVisitor,
};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{
    DevError, DevRequest, MAX_INSPECTOR_BYTES, MAX_INSPECTORS, MAX_PAGE_SIZE, MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES, MAX_SCAN_ENTITIES, PROTOCOL_VERSION,
    views::{
        AnimationPlayerView, CameraView, DecalView, GlobalTransformView, HierarchyView,
        InterpolateTransformView, Mesh3dView, MeshBoundsView, PresentationModeView,
        PresentationStateView, PresentedTransformView, SkinView, SurfaceView, TransformView,
    },
};

/// Names address deliberate SDK projections, never Rust type names or pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectorKind {
    Component,
    Resource,
}

/// Structural metadata for a registered projection. Built-in schemas are
/// generated during host compilation from the same DTOs used for responses.
#[derive(Debug, Clone, Serialize)]
pub struct InspectorDescriptor {
    pub name: String,
    pub description: String,
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
}

impl InspectorDescriptor {
    pub fn new(name: impl Into<String>, description: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
            default: None,
        }
    }

    pub fn with_default(mut self, value: Value) -> Self {
        self.default = Some(value);
        self
    }
}

type ReadInspector =
    Box<dyn Fn(&World, Option<Entity>, Option<&SceneInstance>) -> Option<Value> + Send + Sync>;
type HasInspector = Box<dyn Fn(&World, Option<Entity>) -> bool + Send + Sync>;

struct Inspector {
    descriptor: InspectorDescriptor,
    kind: InspectorKind,
    read: ReadInspector,
    has: HasInspector,
}

struct InspectSelector<'a> {
    entity: Option<&'a str>,
    parent: Option<&'a str>,
    component: Option<&'a str>,
    offset: usize,
    limit: usize,
}

/// Explicitly opted-in inspection registry. Registrations belong to the game
/// session and cannot be replaced silently. Extension callbacks are trusted
/// game code: keep their work and allocations bounded and never scan the world.
pub struct DevTools {
    inspectors: BTreeMap<String, Inspector>,
}

impl Default for DevTools {
    fn default() -> Self {
        let schemas: Value =
            serde_json::from_str(include_str!(concat!(env!("OUT_DIR"), "/inspectors.json")))
                .expect("build-generated schemas are valid JSON");
        let mut tools = Self {
            inspectors: BTreeMap::new(),
        };
        tools.register_poses(&schemas);
        tools.register_components(&schemas);
        tools.register_animation(&schemas);
        tools.register_environment(&schemas);
        tools
    }
}

impl DevTools {
    /// Local, world and presented poses, and the hierarchy.
    fn register_poses(&mut self, schemas: &Value) {
        self.register_component::<Transform>(
            InspectorDescriptor::new("Transform", "Local transform relative to Parent, or world transform for a root. Quaternion is normalized XYZW; authored scale is positive and nonsingular.", schemas["Transform"].clone())
                .with_default(transform_value(&Transform::default())),
            transform_value,
        ).expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("GlobalTransform", "Derived world transform, refreshed before inspection. Column-major matrix preserves shear; invalid hierarchies return null matrix and translation and skip rendering.", schemas["GlobalTransform"].clone()),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| entity.is_some_and(|entity| world.get::<Transform>(entity).is_some())),
            read: Box::new(|world, entity, _scene| {
                let entity = entity?;
                world.get::<Transform>(entity)?;
                let value = world.get_resource::<GlobalTransforms>().and_then(|globals| globals.get(entity)).unwrap_or_default();
                Some(json!(GlobalTransformView {
                    valid: value.matrix().is_some(),
                    matrix: value.matrix().map(|matrix| matrix.to_cols_array()),
                    translation: value.translation().map(|translation| translation.to_array()),
                }))
            }),
        }).expect("built-in descriptor is bounded");
        self.register_component::<InterpolateTransform>(
            InspectorDescriptor::new("InterpolateTransform", "Opt-in marker for fixed-tick interpolation of this local transform. Descendants inherit the presented parent matrix. Gameplay and authored transforms remain authoritative.", schemas["InterpolateTransform"].clone()),
            |_| json!(InterpolateTransformView { enabled: true }),
        ).expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("PresentedTransform", "Last extracted world matrix, including interpolated ancestors; column-major and preserves shear. Inspection does not recompute it, except to materialize the exact CPU reference of a GPU-presented frame while its inputs are unchanged. Null until extracted or when the presented branch is invalid. See PresentationState for actual mode, alpha and gpu_pending.", schemas["PresentedTransform"].clone()),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| entity.is_some_and(|entity| world.get::<Transform>(entity).is_some())),
            read: Box::new(|world, entity, _scene| {
                let entity = entity?;
                world.get::<Transform>(entity)?;
                let value = world.get_resource::<PresentedTransforms>().and_then(|poses| poses.get(entity)).unwrap_or_default();
                Some(json!(PresentedTransformView {
                    valid: value.matrix().is_some(),
                    matrix: value.matrix().map(|matrix| matrix.to_cols_array()),
                    translation: value.matrix().map(|matrix| matrix.w_axis.truncate().to_array()),
                }))
            }),
        }).expect("built-in descriptor is bounded");
        self.register_resource::<PresentationState>(
            InspectorDescriptor::new("PresentationState", "Last prepared extraction's frame/tick and actual mode/fraction; null frame/tick before extraction. Current mode uses alpha one. History counters measure bounded storage. gpu_pending marks a GPU-presented extraction whose CPU poses are not materialized. Inspection prepares presentation only to materialize such a frame; this is not GPU submission/completion or total process memory.", schemas["PresentationState"].clone()),
            presentation_value,
        ).expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("Hierarchy", "Current parent and source prefab metadata. IDs are authored IDs, never runtime handles. Use inspect with parent to page direct children.", schemas["Hierarchy"].clone()),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| entity.is_some_and(|entity| world.get::<Transform>(entity).is_some())),
            read: Box::new(|world, entity, scene| {
                let entity = entity?;
                world.get::<Transform>(entity)?;
                let parent = world.get::<Parent>(entity).map(Parent::parent);
                let parent_metadata = parent.and_then(|parent| world.get::<SceneEntity>(parent));
                let metadata = world.get::<SceneEntity>(entity);
                Some(json!(HierarchyView {
                    parent: parent_metadata.map(|metadata| metadata.authored_id.clone()),
                    parent_scene: parent_metadata.map(|metadata| metadata.scene_id.clone()),
                    parent_alive: parent.map(|parent| world.get_entity(parent).is_ok()),
                    parent_in_active_scene: parent.map(|parent| parent_metadata.is_some_and(|metadata| scene.and_then(|scene| scene.entity(&metadata.authored_id)) == Some(parent))),
                    authored_parent: metadata.and_then(|metadata| metadata.authored_parent.clone()),
                    prefab_id: metadata.and_then(|metadata| metadata.prefab_id.clone()),
                    prefab_instance: metadata.and_then(|metadata| metadata.prefab_instance.clone()),
                    prefab_local_id: metadata.and_then(|metadata| metadata.prefab_local_id.clone()),
                }))
            }),
        }).expect("built-in descriptor is bounded");
    }

    /// Collision, the camera, meshes and the components entities carry.
    fn register_components(&mut self, schemas: &Value) {
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("Collider", "Local cuboid/sphere, symmetric layer filters and sensor flag. Current authoritative world admission and refreshed collision-cache status are independent of visibility and interpolation.", schemas["Collider"].clone()),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| entity.is_some_and(|entity| world.get::<Collider>(entity).is_some())),
            read: Box::new(|world, entity, _scene| crate::collision::collider_value(world, entity?)),
        }).expect("built-in descriptor is bounded");
        self.register_resource::<CollisionWorld>(
            InspectorDescriptor::new("CollisionWorld", "Authoritative spatial cache refreshed for inspection without advancing clocks or committing sensor events. Includes source frame/tick, cache revision, completeness, invalid/overflow counts and query budgets. Memory counts are cache payload, not process totals.", schemas["CollisionWorld"].clone()),
            crate::collision::world_value,
        ).expect("built-in descriptor is bounded");
        self.register_resource::<Camera>(
            InspectorDescriptor::new("Camera", "Right-handed Y-up perspective. Lens distances use world units; 0 < near < far; FOV is in radians.", schemas["Camera"].clone())
                .with_default(camera_value(&Camera::default())),
            camera_value,
        ).expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("Mesh3d", "Resolved immutable mesh/material readiness. Missing assets skip rendering. This view contains no persistent runtime handles.", schemas["Mesh3d"].clone()),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| entity.is_some_and(|entity| world.get::<Mesh3d>(entity).is_some())),
            read: Box::new(|world, entity, _scene| {
                let mesh3d = world.get::<Mesh3d>(entity?)?;
                let assets = world.get_resource::<AssetLibrary>();
                let mesh = assets.and_then(|assets| assets.mesh(mesh3d.mesh));
                let material = assets.and_then(|assets| assets.material(mesh3d.material));
                let texture_handle = material.and_then(|material| material.texture());
                let texture = texture_handle.and_then(|handle| assets.and_then(|assets| assets.texture(handle)));
                Some(json!(Mesh3dView {
                    mesh_alive: mesh.is_some(), material_alive: material.is_some(),
                    vertices: mesh.map(|mesh| mesh.vertices().len()),
                    indices: mesh.map(|mesh| mesh.indices().len()),
                    local_bounds: mesh.map(|mesh| MeshBoundsView { min: mesh.bounds().min().to_array(), max: mesh.bounds().max().to_array() }),
                    color: material.map(|material| material.color()),
                    opacity: material.map(|material| material.opacity()),
                    surface: material.map(|material| {
                        let surface = material.surface();
                        SurfaceView { roughness: surface.roughness, metalness: surface.metalness, emission: surface.emission, ripples: surface.ripples }
                    }),
                    texture_alive: texture_handle.map(|_| texture.is_some()),
                    texture_dimensions: texture.map(|texture| [texture.width(), texture.height()]),
                    texture_emission_mask: texture.map(|texture| texture.has_emission_mask()),
                }))
            }),
        }).expect("built-in descriptor is bounded");
        self.register_component::<LocalLight>(
            InspectorDescriptor::new("LocalLight", "A point or spot light on this entity (decision 023) as a scene version 6 `light` field; on an entity with a mesh it follows the displayed pose.", schemas["LocalLight"].clone())
                .with_default(json!(SceneLight::default())),
            |light| json!(SceneLight::from(light)),
        ).expect("built-in descriptor is bounded");
        self.register_component::<Tint>(
            InspectorDescriptor::new("Tint", "Per-object color factors (decision 033) as a scene version 8 `tint` field: red, green, blue and opacity multiply this object's material or cube color, each in [0, 1]. Changing it uploads only this object's color.", schemas["Tint"].clone())
                .with_default(json!(Tint::WHITE.0)),
            |tint| json!(tint.0),
        ).expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("Decal", "A decal on this entity (decision 039) as a scene version 8 `decal` field: its material projected down the entity's local y onto the opaque surfaces inside its unit box, fading on surfaces turning away beyond `fade_degrees`.", schemas["Decal"].clone()),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| entity.is_some_and(|entity| world.get::<Decal>(entity).is_some())),
            read: Box::new(|world, entity, scene| {
                let decal = world.get::<Decal>(entity?)?;
                let assets = world.get_resource::<AssetLibrary>();
                let material = scene.and_then(|scene| scene.materials().find(|(_, handle)| *handle == decal.material).map(|(id, _)| id.to_owned()));
                let field = SceneDecal::from_runtime(decal, material.clone().unwrap_or_default());
                Some(json!(DecalView {
                    material,
                    material_alive: assets.is_some_and(|assets| assets.material(decal.material).is_some()),
                    blend: field.blend,
                    opacity: field.opacity,
                    fade_degrees: field.fade_degrees,
                }))
            }),
        }).expect("built-in descriptor is bounded");
        self.register_component::<ReflectionProbe>(
            InspectorDescriptor::new("ReflectionProbe", "A reflection probe on this entity (decision 043) as a scene version 8 `reflection_probe` field: surfaces inside the world-axis box reaching `extent` from the entity's world position reflect the still objects and lights it captured from there, fading out over `blend` inside its faces.", schemas["ReflectionProbe"].clone()),
            |probe| json!(SceneReflectionProbe::from(probe)),
        ).expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("Terrain", "A heightfield terrain on this entity (decision 051) as a scene version 9 `terrain` field: its origin, tiles, spacing, layers or file, material and collision, with its streaming state: resident tiles by level.", schemas["Terrain"].clone()),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| entity.is_some_and(|entity| world.get::<engine_core::Terrain>(entity).is_some())),
            read: Box::new(|world, entity, scene| {
                let entity = entity?;
                let terrain = world.get::<engine_core::Terrain>(entity)?;
                let mut levels = vec![0_u32; terrain.pyramid.levels() as usize];
                let (mut level_errors, mut level_ranges) = (Vec::new(), Vec::new());
                if let Some(terrains) = world.get_resource::<engine_core::Terrains>() {
                    for tile in terrains.tiles(entity) {
                        levels[usize::from(tile.key().level)] += 1;
                    }
                    if let Some((errors, ranges)) = terrains.level_detail(entity) {
                        level_errors = errors.to_vec();
                        level_ranges = ranges.to_vec();
                    }
                }
                Some(json!(crate::views::TerrainView {
                    terrain: engine_scene::SceneTerrain::from_runtime(terrain, None, |handle| {
                        scene.and_then(|scene| scene.materials().find(|(_, candidate)| *candidate == handle).map(|(id, _)| id.to_owned()))
                    }),
                    levels: terrain.pyramid.levels(),
                    extent: terrain.pyramid.extent().to_array(),
                    resident_tiles_by_level: levels,
                    level_errors,
                    level_ranges,
                }))
            }),
        }).expect("built-in descriptor is bounded");
        self.register_resource::<engine_core::Terrains>(
            InspectorDescriptor::new("Terrains", "Terrain streaming (decision 051) at the last extraction: terrains, resident and loading tiles, tiles dropped by the budget or the device, patches drawn and patches standing in for finer levels not resident yet. Query heights with terrain_height and compare queries, collision and layers with terrain_sweep.", schemas["Terrains"].clone()),
            |terrains| {
                let stats = terrains.stats();
                json!(crate::views::TerrainsView {
                    terrains: stats.terrains,
                    resident_tiles: stats.resident_tiles,
                    loading_tiles: stats.loading_tiles,
                    dropped_tiles: stats.dropped_tiles,
                    patches: stats.patches,
                    fallback_patches: stats.fallback_patches,
                })
            },
        ).expect("built-in descriptor is bounded");
        self.register_component::<ParticleEmitter>(
            InspectorDescriptor::new("ParticleEmitter", "A particle emitter on this entity (decision 026) as a scene version 6 `particles` field; particles are a function of presentation time.", schemas["ParticleEmitter"].clone())
                .with_default(json!(SceneEmitter::default())),
            |emitter| json!(SceneEmitter::from(emitter)),
        ).expect("built-in descriptor is bounded");
    }

    /// Animation players and skins (decisions 035 and 036), naming clips and
    /// entities by their authored IDs.
    fn register_animation(&mut self, schemas: &Value) {
        // Animation (decisions 035 and 036): players and skins name clips
        // and entities by their authored IDs.
        let authored = |world: &World, entity: Option<Entity>| {
            entity
                .and_then(|entity| world.get::<SceneEntity>(entity))
                .map(|source| source.authored_id.clone())
        };
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("AnimationPlayer", "An animation player (decision 035) as a scene version 8 `animation` field with its clip's duration and its targets: every fixed tick it advances `time` by the tick times `speed`, wrapping when `loop` is set, and poses its targets' transforms.", schemas["AnimationPlayer"].clone()),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| entity.is_some_and(|entity| world.get::<AnimationPlayer>(entity).is_some())),
            read: Box::new(move |world, entity, scene| {
                let player = world.get::<AnimationPlayer>(entity?)?;
                let clip = world.get_resource::<AssetLibrary>().and_then(|assets| assets.clip(player.clip));
                Some(json!(AnimationPlayerView {
                    clip: scene.and_then(|scene| scene.clips().find(|(_, handle)| *handle == player.clip).map(|(id, _)| id.to_owned())),
                    clip_alive: clip.is_some(),
                    duration: clip.map(|clip| clip.duration()),
                    time: player.time,
                    speed: player.speed,
                    looping: player.looping,
                    targets: player.targets.iter().map(|&target| authored(world, target)).collect(),
                }))
            }),
        }).expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("Skin", "The joints deforming this entity's mesh (decision 036), as a scene version 8 `skin` names them: each vertex moves by its weighted joints' displayed world matrices times their inverse bind matrices.", schemas["Skin"].clone()),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| entity.is_some_and(|entity| world.get::<Skin>(entity).is_some())),
            read: Box::new(move |world, entity, _scene| {
                let skin = world.get::<Skin>(entity?)?;
                Some(json!(SkinView {
                    joints: skin.joints.iter().map(|&joint| authored(world, Some(joint))).collect(),
                }))
            }),
        }).expect("built-in descriptor is bounded");
    }

    /// The look (decision 022): the atmosphere, every effect section and the
    /// whole environment.
    fn register_environment(&mut self, schemas: &Value) {
        // The environment (decision 022): each resource serializes as its
        // scene section, so values copy into controls and v5 scenes as is.
        self.register_resource::<Atmosphere>(
            InspectorDescriptor::new("Atmosphere", "Sun, sky colors, fog and grade of every frame (decision 018), every field set over the golden_hour preset. Change it with the environment control or a v5 scene environment.", schemas["Atmosphere"].clone())
                .with_default(json!(SceneAtmosphere::from(&Atmosphere::default()))),
            |atmosphere| json!(SceneAtmosphere::from(atmosphere)),
        ).expect("built-in descriptor is bounded");
        // Every effect section of a scene environment as its resource, in
        // the section's shape (decision 022).
        struct Sections<'t> {
            tools: &'t mut DevTools,
            schemas: &'t Value,
        }
        impl SectionVisitor for Sections<'_> {
            fn section<T: EnvironmentSection>(&mut self, name: &'static str, _: Option<T>) {
                let schema = self.schemas[T::NAME].clone();
                self.tools
                    .register_resource::<T>(
                        InspectorDescriptor::new(T::NAME, section_description(name), schema)
                            .with_default(json!(T::default())),
                        |settings| json!(settings),
                    )
                    .expect("built-in descriptor is bounded");
            }
        }
        SceneEnvironment::default().visit(&mut Sections {
            tools: self,
            schemas,
        });
        // The whole environment's schema without field descriptions, which
        // each section's own inspector carries, so it stays within bounds
        // as sections grow.
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("Environment", "The whole current look as a scene environment (decisions 022 to 026): paste it into a scene to persist it, or send sections to the environment control. Missing effect resources read as off. Each section's inspector describes its fields.", undescribed(schemas["Environment"].clone())),
            kind: InspectorKind::Resource,
            has: Box::new(|_, _| true),
            read: Box::new(|world, _, _| Some(json!(SceneEnvironment::current(world)))),
        }).expect("built-in descriptor is bounded");
    }
}

/// `schema` without its descriptions, at any depth.
fn undescribed(schema: Value) -> Value {
    match schema {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                // Annotations are strings; a field named so is a schema.
                .filter(|(key, value)| !(key == "description" && value.is_string()))
                .map(|(key, value)| (key, undescribed(value)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(undescribed).collect()),
        other => other,
    }
}

/// What the inspector of environment section `name` reports.
fn section_description(name: &str) -> &'static str {
    match name {
        "shadows" => {
            "Sun shadows (decision 020): kept-map bounds, fade distance, near-map distance and resolution. Missing means no shadows. The rendering observation reports per-map casters and kept-map redraws."
        }
        "ambient_occlusion" => {
            "Screen-space ambient occlusion (decision 021); intensity 0 or a missing resource turns it off."
        }
        "bloom" => {
            "Glow beyond bright pixels (decision 021); intensity 0 or a missing resource turns it off."
        }
        "local_lights" => {
            "Local lighting settings (decision 023): whether point and spot lights shade, the per-frame light budget (nearest to the camera win) and an intensity scale. Missing means no local lights. The rendering observation reports lights shaded, visible and cluster overflow."
        }
        "sun_shafts" => {
            "Sunlight scattered by the fog between shadow casters (decision 024). Opt-in: missing or intensity 0 means off; it needs shadows."
        }
        "reflections" => {
            "Screen-space reflections in glossy surfaces (decision 025), falling back to the sky off screen. Opt-in: missing or intensity 0 means off."
        }
        "particles" => {
            "Particle settings (decision 026): whether emitters draw and the per-frame particle budget. Missing means no particles."
        }
        "refraction" => {
            "Refraction through glass (decision 038): transmissive materials show the opaque scene behind them, bent by their index of refraction through their thickness and blurred by their roughness. Opt-in: missing or strength 0 means glass blends as plain transparency."
        }
        "decals" => {
            "Decal settings (decision 039): whether decals draw and the per-frame decal budget (nearest to the camera win). Missing means no decals. The rendering observation reports decals drawn and dropped."
        }
        "reflection_probes" => {
            "Reflection probe settings (decision 043): whether probes capture and reflect, the texels across each environment map face, the budget of probes kept captured (nearest to the camera win) and the cube faces captured per frame. Missing means no probes. The rendering observation reports probes reflected, faces captured and probes dropped."
        }
        "virtual_geometry" => {
            "Virtual geometry settings (decision 044): whether meshes built as cluster hierarchies draw the clusters each view needs (otherwise their fallback cut), and the largest simplification error a drawn cluster may show in pixels of the viewport height. Missing means fallback cuts. The rendering observation reports clusters and triangles drawn for the camera and shadow maps, occluded clusters and dropped selections."
        }
        "terrain" => {
            "Terrain settings (decision 051): whether terrains draw, their level of detail's pixel error, the coarsest levels kept resident everywhere, an optional tile budget, per-frame tile work, and the collision radius around the camera and focus entities. Missing means no terrain draws. The rendering observation reports terrain patches, triangles and shadow patches, and resident, loading and dropped tiles."
        }
        "level_of_detail" => {
            "Level of detail settings (decisions 047 and 048): GPU-presented frames draw each instance at the coarsest level of its mesh whose error stays within the pixel error of the viewport height, the coarsest being its draw group's impostor unless impostors are off; max_impostors budgets baked impostors. Missing means every instance draws its mesh. The rendering observation reports instances per level, impostors baked and those left out."
        }
        _ => "An environment section (decision 022).",
    }
}

impl DevTools {
    /// Registers a typed, read-only component projection without reflection.
    pub fn register_component<T: Component>(
        &mut self,
        descriptor: InspectorDescriptor,
        inspect: impl Fn(&T) -> Value + Send + Sync + 'static,
    ) -> Result<(), DevError> {
        self.insert(Inspector {
            descriptor,
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<T>(entity).is_some())
            }),
            read: Box::new(move |world, entity, _scene| world.get::<T>(entity?).map(&inspect)),
        })
    }

    /// Registers a typed world resource, queried with `component` and no entity.
    pub fn register_resource<T: Resource>(
        &mut self,
        descriptor: InspectorDescriptor,
        inspect: impl Fn(&T) -> Value + Send + Sync + 'static,
    ) -> Result<(), DevError> {
        self.insert(Inspector {
            descriptor,
            kind: InspectorKind::Resource,
            has: Box::new(|world, _| world.get_resource::<T>().is_some()),
            read: Box::new(move |world, _, _scene| world.get_resource::<T>().map(&inspect)),
        })
    }

    fn insert(&mut self, inspector: Inspector) -> Result<(), DevError> {
        let name = &inspector.descriptor.name;
        validate_name(name, "$.name")?;
        if ["SceneDocument", "protocol", "BrowserControl"].contains(&name.as_str())
            || self.inspectors.contains_key(name)
        {
            return Err(DevError::new(
                "duplicate_inspector",
                "$.name",
                "inspector name is already registered or reserved",
            ));
        }
        if self.inspectors.len() == MAX_INSPECTORS {
            return Err(DevError::new(
                "inspector_limit",
                "$.name",
                "at most 64 inspectors may be registered",
            ));
        }
        if inspector.descriptor.description.len() > 512
            || encode(&inspector.descriptor, MAX_INSPECTOR_BYTES).is_err()
        {
            return Err(DevError::new(
                "inspector_too_large",
                "$.schema",
                "descriptor must fit 16384 bytes and description 512 bytes",
            ));
        }
        self.inspectors.insert(name.clone(), inspector);
        Ok(())
    }

    /// Processes one request outside ordinary frames. At most 10,000 authored
    /// records are scanned, 100 rows returned, and 64 KiB encoded. Counters and
    /// revisions are decimal strings to retain u64 precision across JavaScript.
    /// Refreshes derived transforms without simulation advance, extraction, or GPU work.
    pub fn execute(
        &self,
        engine: &mut Engine,
        scene: Option<&SceneInstance>,
        source: &str,
    ) -> String {
        let world = engine.app.world();
        let observation = json!({
            "frame": world.get_resource::<FrameTime>().map(|time| time.frame.to_string()),
            "tick": world.get_resource::<FixedTime>().map(|time| time.tick.to_string()),
            "scene": scene.map(|scene| json!({"id": scene.id(), "revision": scene.revision().to_string()})),
            "presentation": presentation_value(engine.presentation()),
        });
        let result = self.parse_and_execute(engine, scene, source);
        let response = match result {
            Ok(result) => {
                json!({"version": PROTOCOL_VERSION, "ok": true, "observation": observation, "result": result})
            }
            Err(error) => {
                json!({"version": PROTOCOL_VERSION, "ok": false, "observation": observation, "error": error})
            }
        };
        encode(&response, MAX_RESPONSE_BYTES).unwrap_or_else(|_| {
            json!({"version": PROTOCOL_VERSION, "ok": false, "observation": observation,
                "error": DevError::new("response_too_large", "$", "response exceeds 65536 bytes; reduce limit or select one component")}).to_string()
        })
    }

    fn parse_and_execute(
        &self,
        engine: &mut Engine,
        scene: Option<&SceneInstance>,
        source: &str,
    ) -> Result<Value, DevError> {
        if source.len() > MAX_REQUEST_BYTES {
            return Err(DevError::new(
                "request_too_large",
                "$",
                "request must fit 16384 UTF-8 bytes",
            ));
        }
        let request: DevRequest = serde_json::from_str(source)
            .map_err(|error| DevError::new("invalid_request", "$", error.to_string()))?;
        if request.version() != PROTOCOL_VERSION {
            return Err(DevError::new(
                "unsupported_version",
                "$.version",
                "accepted protocol version: 1",
            ));
        }
        if scene.is_some_and(|scene| !scene.belongs_to(engine.app.world())) {
            return Err(DevError::new(
                "wrong_world",
                "$.scene",
                "scene instance belongs to a different engine world",
            ));
        }
        if crate::collision::spatial_request(&request) {
            return crate::collision::execute(engine, scene, request);
        }
        if crate::terrain::terrain_request(&request) {
            return crate::terrain::execute(engine, scene, request);
        }
        if let DevRequest::Inspect {
            component: Some(component),
            entity,
            parent,
            offset,
            limit,
            ..
        } = &request
            && matches!(component.as_str(), "Collider" | "CollisionWorld")
        {
            // Validate selectors and pagination before refreshing any derived cache.
            self.inspect(
                engine.app.world(),
                scene,
                InspectSelector {
                    entity: entity.as_deref(),
                    parent: parent.as_deref(),
                    component: Some(component),
                    offset: *offset,
                    limit: *limit,
                },
            )?;
            if engine.app.world().contains_resource::<CollisionWorld>() {
                refresh_collision_world(engine.app.world_mut())
                    .map_err(crate::collision::native_error)?;
            }
        }
        if matches!(request, DevRequest::Inspect { .. }) {
            // Materialize a GPU-presented frame before a refresh changes its inputs.
            engine.prepare_presented_transforms();
            engine.refresh_transforms();
        }
        let world = engine.app.world();
        match request {
            DevRequest::Status { .. } => Ok(json!({
                "world_entities": world.entities().len(),
                "authored_entities": scene.map(SceneInstance::entity_count),
                "inspectors": self.inspectors.len(),
                "culling_enabled": engine.frustum_culling(),
                "presentation": presentation_value(engine.presentation()),
                "collision": world.get_resource::<CollisionWorld>().map(crate::collision::world_value),
                "fixed_delta_seconds": world.get_resource::<FixedTime>().map(|time| time.delta_seconds),
                "limits": {"request_bytes": MAX_REQUEST_BYTES, "response_bytes": MAX_RESPONSE_BYTES, "page_size": MAX_PAGE_SIZE, "scan_entities": MAX_SCAN_ENTITIES},
            })),
            DevRequest::Assets { .. } => {
                let assets = world.get_resource::<AssetLibrary>().ok_or_else(|| {
                    DevError::new(
                        "missing_resource",
                        "$.assets",
                        "AssetLibrary is not installed",
                    )
                })?;
                let textures: Vec<_> = scene.into_iter().flat_map(SceneInstance::textures).map(|(id, handle)| {
                    let texture = assets.texture(handle);
                    json!({"id":id,"state":if texture.is_some() {"cpu_ready"} else {"missing"},
                        "width":texture.map(|texture| texture.width()), "height":texture.map(|texture| texture.height()),
                        "bytes":texture.map(|texture| texture.byte_len())})
                }).collect();
                Ok(
                    json!({"meshes": assets.mesh_count(), "materials": assets.material_count(),
                    "textures": assets.texture_count(), "texture_cpu_bytes":assets.texture_bytes(),
                    "texture_assets":textures, "resident_cpu_bytes": assets.resident_bytes(),
                    "readiness": "synchronous", "ownership": "library payload excludes retained snapshots and allocator metadata"}),
                )
            }
            DevRequest::Describe { name, .. } => self.describe(name.as_deref()),
            DevRequest::Inspect {
                entity,
                parent,
                component,
                offset,
                limit,
                ..
            } => self.inspect(
                world,
                scene,
                InspectSelector {
                    entity: entity.as_deref(),
                    parent: parent.as_deref(),
                    component: component.as_deref(),
                    offset,
                    limit,
                },
            ),
            DevRequest::Raycast { .. }
            | DevRequest::Overlap { .. }
            | DevRequest::ShapeCast { .. }
            | DevRequest::CollisionEvents { .. }
            | DevRequest::TerrainHeight { .. }
            | DevRequest::TerrainSweep { .. } => unreachable!("spatial requests return above"),
        }
    }

    fn describe(&self, name: Option<&str>) -> Result<Value, DevError> {
        let Some(name) = name else {
            let mut types: Vec<Value> = self.inspectors.values().map(|inspector| json!({"name": inspector.descriptor.name, "kind": inspector.kind, "description": inspector.descriptor.description})).collect();
            types.push(json!({"name": "SceneDocument", "kind": "authoring", "source": engine_scene::schema_path()}));
            types.push(json!({"name": "protocol", "kind": "protocol"}));
            types.push(json!({"name": "BrowserControl", "kind": "protocol"}));
            return Ok(json!({"types": types}));
        };
        match name {
            "SceneDocument" => Ok(
                json!({"name": name, "kind": "authoring", "source": engine_scene::schema_path(), "schema": serde_json::from_str::<Value>(include_str!(concat!(env!("OUT_DIR"), "/scene.schema.json"))).expect("generated scene schema is JSON"), "constraints": format!("Rust validation checks references, unique expanded IDs, cycles, depth, finite world matrices, collision admission, environment values and asset budgets. Fields newer than version 1 require the version that introduced them: {}. Every older version stays accepted with its own semantics. Prefabs cannot nest. An environment sets every section on instantiation, omitted ones to the engine default state (sun shafts, reflections and refraction off; decals and reflection probes on).", engine_scene::version_features())}),
            ),
            "protocol" | "BrowserControl" => {
                let schemas: Value = serde_json::from_str(include_str!(concat!(
                    env!("OUT_DIR"),
                    "/inspectors.json"
                )))
                .expect("generated schemas are JSON");
                Ok(
                    json!({"name": name, "kind": "protocol", "schema": schemas[name], "constraints": "version=1; mutations, capture and benchmark require a paused host; reload revisions must fit u64. Spatial queries need CollisionPlugin but no pause: positions/endpoints within +/-1e6; nonzero finite directions; distance[0,2e6]; shape dimensions[1e-4,1e4]; normalized XYZW rotation. Symmetric layer filtering; sensors include/exclude/only. Raycast/cast return closest complete hit, terrain heightfields included; casts translate with fixed orientation. Terrain height takes 1..256 [x,z] points (null where no tile is resident); terrain sweep compares queries, collision rays and layers over samples2..64 per side within half_extent0..4096 (decision 051). Overlap limit1..100; budget failures never return partial hits. Events are non-consuming latest fixed-tick sensor transitions, paged1..100, with attempted and committed ticks. Present/capture alpha finite[0,1]; preview preserves clocks/input. The terrain control sets pixel_error1..64, freezes streaming at the camera's position, and shows shaded, levels, normals or tiles. Ordinary render/step/capture show current poses. Benchmark warmup0..120, samples1..240; step advances one tick, present samples alpha0.5 without ticks; clears input. Controls do not save source."}),
                )
            }
            _ => {
                let inspector = self.inspector(name)?;
                Ok(json!({"kind": inspector.kind, "descriptor": inspector.descriptor}))
            }
        }
    }

    fn inspector(&self, name: &str) -> Result<&Inspector, DevError> {
        validate_name(name, "$.component")?;
        self.inspectors.get(name).ok_or_else(|| {
            DevError::new(
                "unknown_inspector",
                "$.component",
                "use describe to discover registered inspectors",
            )
        })
    }

    fn inspect(
        &self,
        world: &World,
        scene: Option<&SceneInstance>,
        selector: InspectSelector<'_>,
    ) -> Result<Value, DevError> {
        let InspectSelector {
            entity,
            parent,
            component,
            offset,
            limit,
        } = selector;
        if entity.is_some() && parent.is_some() {
            return Err(DevError::new(
                "invalid_selector",
                "$.parent",
                "entity and parent selectors are mutually exclusive",
            ));
        }
        if !(1..=MAX_PAGE_SIZE).contains(&limit) || offset > MAX_SCAN_ENTITIES {
            return Err(DevError::new(
                "invalid_page",
                "$.limit",
                "limit must be 1..100 and offset 0..10000",
            ));
        }
        let inspector = component.map(|name| self.inspector(name)).transpose()?;
        if let Some(inspector) =
            inspector.filter(|inspector| inspector.kind == InspectorKind::Resource)
        {
            if entity.is_some() || parent.is_some() || offset != 0 {
                return Err(DevError::new(
                    "invalid_selector",
                    "$.entity",
                    "resources have no entity or parent selector or pagination offset",
                ));
            }
            let value = read_value(inspector, world, None, scene)?;
            return Ok(json!({"resource": inspector.descriptor.name, "value": value}));
        }
        let scene = scene.ok_or_else(|| {
            DevError::new(
                "no_scene",
                "$.entity",
                "authored entity inspection requires an active scene instance",
            )
        })?;
        if scene.entity_count() > MAX_SCAN_ENTITIES {
            return Err(DevError::new(
                "scan_limit",
                "$.scene",
                "scene exceeds the 10000 authored-record scan budget",
            ));
        }
        if let Some(id) = entity {
            validate_name(id, "$.entity")?;
            if offset != 0 {
                return Err(DevError::new(
                    "invalid_selector",
                    "$.offset",
                    "a selected entity has no pagination offset",
                ));
            }
            let entity = scene.entity(id).ok_or_else(|| {
                DevError::new(
                    "unknown_entity",
                    "$.entity",
                    "authored ID does not exist in the active scene",
                )
            })?;
            if world.get_entity(entity).is_err() {
                return Err(DevError::new(
                    "entity_removed",
                    "$.entity",
                    "gameplay removed this authored entity",
                ));
            }
            return Ok(json!({"entity": self.entity_row(world, scene, id, entity, inspector)?}));
        }
        let parent = parent
            .map(|id| {
                validate_name(id, "$.parent")?;
                let entity = scene.entity(id).ok_or_else(|| {
                    DevError::new(
                        "unknown_entity",
                        "$.parent",
                        "parent authored ID does not exist in the active scene",
                    )
                })?;
                if world.get_entity(entity).is_err() {
                    return Err(DevError::new(
                        "entity_removed",
                        "$.parent",
                        "gameplay removed this authored parent",
                    ));
                }
                Ok(entity)
            })
            .transpose()?;
        let mut rows = Vec::with_capacity(limit);
        let mut total = 0;
        for (id, entity) in scene.entities() {
            if world.get_entity(entity).is_err()
                || parent.is_some_and(|parent| {
                    world.get::<Parent>(entity).map(Parent::parent) != Some(parent)
                })
                || inspector.is_some_and(|inspector| !(inspector.has)(world, Some(entity)))
            {
                continue;
            }
            if total >= offset && rows.len() < limit {
                rows.push(self.entity_row(world, scene, id, entity, inspector)?);
            }
            total += 1;
        }
        let next = offset.saturating_add(rows.len());
        Ok(
            json!({"entities": rows, "offset": offset, "limit": limit, "total": total, "next_offset": (next < total).then_some(next)}),
        )
    }

    fn entity_row(
        &self,
        world: &World,
        scene: &SceneInstance,
        id: &str,
        entity: Entity,
        inspector: Option<&Inspector>,
    ) -> Result<Value, DevError> {
        let metadata = world.get::<SceneEntity>(entity);
        let mut row = json!({"id": id, "name": metadata.map(|metadata| metadata.name.as_str()).unwrap_or(""), "tags": metadata.map(|metadata| &metadata.tags)});
        if let Some(inspector) = inspector {
            row["components"] = json!({inspector.descriptor.name.clone(): read_value(inspector, world, Some(entity), Some(scene))?});
        } else {
            row["components"] = json!(
                self.inspectors
                    .values()
                    .filter(|inspector| inspector.kind == InspectorKind::Component
                        && (inspector.has)(world, Some(entity)))
                    .map(|inspector| &inspector.descriptor.name)
                    .collect::<Vec<_>>()
            );
        }
        Ok(row)
    }
}

fn read_value(
    inspector: &Inspector,
    world: &World,
    entity: Option<Entity>,
    scene: Option<&SceneInstance>,
) -> Result<Value, DevError> {
    let value = (inspector.read)(world, entity, scene).ok_or_else(|| {
        DevError::new(
            "missing_component",
            "$.component",
            "selected component or resource is absent",
        )
    })?;
    encode(&value, MAX_INSPECTOR_BYTES).map_err(|_| {
        DevError::new(
            "inspector_too_large",
            "$.component",
            "inspector projection exceeds 16384 encoded bytes",
        )
    })?;
    Ok(value)
}

fn validate_name(name: &str, path: &str) -> Result<(), DevError> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    {
        return Err(DevError::new(
            "invalid_name",
            path,
            "names require 1..64 ASCII letters, digits, underscores, dots, or hyphens",
        ));
    }
    Ok(())
}

fn transform_value(value: &Transform) -> Value {
    json!(TransformView {
        translation: value.translation.to_array(),
        rotation: value.rotation.to_array(),
        scale: value.scale.to_array()
    })
}

fn presentation_value(value: &PresentationState) -> Value {
    let (mode, alpha) = match value.mode() {
        PresentationMode::Current => (PresentationModeView::Current, 1.0),
        PresentationMode::Interpolated { alpha } => (PresentationModeView::Interpolated, alpha),
    };
    json!(PresentationStateView {
        frame: value.frame().map(|frame| frame.to_string()),
        tick: value.tick().map(|tick| tick.to_string()),
        mode,
        alpha,
        gpu_pending: value.gpu_pending(),
        history_nodes: value.history_nodes(),
        history_capacity: value.history_capacity(),
        history_bytes: value.history_bytes(),
        presentation_scratch_bytes: value.presentation_scratch_bytes(),
        presented_nodes: value.presented_nodes(),
        presented_store_bytes: value.presented_store_bytes(),
    })
}

fn camera_value(value: &Camera) -> Value {
    json!(CameraView {
        position: value.position.to_array(),
        target: value.target.to_array(),
        vertical_fov_radians: value.vertical_fov_radians,
        near: value.near,
        far: value.far
    })
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other("JSON byte budget exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn encode(value: &impl Serialize, limit: usize) -> Result<String, serde_json::Error> {
    let mut output = BoundedWriter {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut output, value)?;
    Ok(String::from_utf8(output.bytes).expect("JSON is UTF-8"))
}
