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
    DevError, DevRequest, MAX_DESCRIPTION_BYTES, MAX_INSPECTOR_BYTES, MAX_INSPECTORS,
    MAX_PAGE_SIZE, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, MAX_SCAN_ENTITIES, PROTOCOL_VERSION,
    views::{
        AnimationPlayerView, CameraView, DecalView, GlobalTransformView, HierarchyView,
        InterpolateTransformView, Mesh3dView, MeshBoundsView, PresentationModeView,
        PresentationStateView, PresentedTransformView, SkinView, SplineMeshView, SurfaceView,
        TransformView,
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

    /// A built-in inspector's descriptor with the schema build.rs generated
    /// under its name: a name without one is a programming error.
    /// The descriptor of the schema build.rs generated for `name`, described
    /// by its type's doc comment (the build checks every one has one).
    pub(crate) fn generated(schemas: &Value, name: &str) -> Self {
        let schema = schemas
            .get(name)
            .cloned()
            .unwrap_or_else(|| panic!("build.rs generates no schema named {name}"));
        let description = schema["description"]
            .as_str()
            .unwrap_or_else(|| panic!("the schema of {name} has no description"))
            .replace('\n', " ");
        Self::new(name, description, schema)
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
    // The engine's own inspectors, once registered: games register up to
    // MAX_INSPECTORS beyond them.
    builtins: Option<usize>,
}

impl Default for DevTools {
    fn default() -> Self {
        let schemas: Value =
            serde_json::from_str(include_str!(concat!(env!("OUT_DIR"), "/inspectors.json")))
                .expect("build-generated schemas are valid JSON");
        let mut tools = Self {
            inspectors: BTreeMap::new(),
            builtins: None,
        };
        tools.register_poses(&schemas);
        tools.register_components(&schemas);
        tools.register_animation(&schemas);
        tools.register_environment(&schemas);
        tools.builtins = Some(tools.inspectors.len());
        tools
    }
}

impl DevTools {
    /// Local, world and presented poses, and the hierarchy.
    fn register_poses(&mut self, schemas: &Value) {
        self.register_component::<Transform>(
            InspectorDescriptor::generated(schemas, "Transform")
                .with_default(transform_value(&Transform::default())),
            transform_value,
        )
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "GlobalTransform"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Transform>(entity).is_some())
            }),
            read: Box::new(|world, entity, _scene| {
                let entity = entity?;
                world.get::<Transform>(entity)?;
                let value = world
                    .get_resource::<GlobalTransforms>()
                    .and_then(|globals| globals.get(entity))
                    .unwrap_or_default();
                Some(json!(GlobalTransformView {
                    valid: value.matrix().is_some(),
                    matrix: value.matrix().map(|matrix| matrix.to_cols_array()),
                    translation: value
                        .translation()
                        .map(|translation| translation.to_array()),
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
        self.register_component::<InterpolateTransform>(
            InspectorDescriptor::generated(schemas, "InterpolateTransform"),
            |_| json!(InterpolateTransformView { enabled: true }),
        )
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "PresentedTransform"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Transform>(entity).is_some())
            }),
            read: Box::new(|world, entity, _scene| {
                let entity = entity?;
                world.get::<Transform>(entity)?;
                let value = world
                    .get_resource::<PresentedTransforms>()
                    .and_then(|poses| poses.get(entity))
                    .unwrap_or_default();
                Some(json!(PresentedTransformView {
                    valid: value.matrix().is_some(),
                    matrix: value.matrix().map(|matrix| matrix.to_cols_array()),
                    translation: value
                        .matrix()
                        .map(|matrix| matrix.w_axis.truncate().to_array()),
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
        self.register_resource::<PresentationState>(
            InspectorDescriptor::generated(schemas, "PresentationState"),
            presentation_value,
        )
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Hierarchy"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Transform>(entity).is_some())
            }),
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
                    parent_in_active_scene: parent
                        .map(|parent| parent_metadata.is_some_and(|metadata| scene
                            .and_then(|scene| scene.entity(&metadata.authored_id))
                            == Some(parent))),
                    authored_parent: metadata.and_then(|metadata| metadata.authored_parent.clone()),
                    prefab_id: metadata.and_then(|metadata| metadata.prefab_id.clone()),
                    prefab_instance: metadata.and_then(|metadata| metadata.prefab_instance.clone()),
                    prefab_local_id: metadata.and_then(|metadata| metadata.prefab_local_id.clone()),
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
    }

    /// Collision, the camera, meshes and the components entities carry.
    fn register_components(&mut self, schemas: &Value) {
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Collider"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Collider>(entity).is_some())
            }),
            read: Box::new(|world, entity, _scene| {
                crate::collision::collider_value(world, entity?)
            }),
        })
        .expect("built-in descriptor is bounded");
        self.register_resource::<CollisionWorld>(
            InspectorDescriptor::generated(schemas, "CollisionWorld"),
            crate::collision::world_value,
        )
        .expect("built-in descriptor is bounded");
        self.register_resource::<Camera>(
            InspectorDescriptor::generated(schemas, "Camera")
                .with_default(camera_value(&Camera::default())),
            camera_value,
        )
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Mesh3d"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Mesh3d>(entity).is_some())
            }),
            read: Box::new(|world, entity, _scene| {
                let mesh3d = world.get::<Mesh3d>(entity?)?;
                let assets = world.get_resource::<AssetLibrary>();
                let mesh = assets.and_then(|assets| assets.mesh(mesh3d.mesh));
                let material = assets.and_then(|assets| assets.material(mesh3d.material));
                let texture_handle = material.and_then(|material| material.texture());
                let texture = texture_handle
                    .and_then(|handle| assets.and_then(|assets| assets.texture(handle)));
                Some(json!(Mesh3dView {
                    mesh_alive: mesh.is_some(),
                    material_alive: material.is_some(),
                    vertices: mesh.map(|mesh| mesh.vertices().len()),
                    indices: mesh.map(|mesh| mesh.indices().len()),
                    local_bounds: mesh.map(|mesh| MeshBoundsView {
                        min: mesh.bounds().min().to_array(),
                        max: mesh.bounds().max().to_array()
                    }),
                    color: material.map(|material| material.color()),
                    opacity: material.map(|material| material.opacity()),
                    surface: material.map(|material| {
                        let surface = material.surface();
                        SurfaceView {
                            roughness: surface.roughness,
                            metalness: surface.metalness,
                            emission: surface.emission,
                            ripples: surface.ripples,
                        }
                    }),
                    texture_alive: texture_handle.map(|_| texture.is_some()),
                    texture_dimensions: texture.map(|texture| [texture.width(), texture.height()]),
                    texture_emission_mask: texture.map(|texture| texture.has_emission_mask()),
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
        self.register_component::<LocalLight>(
            InspectorDescriptor::generated(schemas, "LocalLight")
                .with_default(json!(SceneLight::default())),
            |light| json!(SceneLight::from(light)),
        )
        .expect("built-in descriptor is bounded");
        self.register_component::<engine_core::Floating>(
            InspectorDescriptor::generated(schemas, "Floating")
                .with_default(json!(engine_core::Floating::default())),
            |floating| json!(floating),
        )
        .expect("built-in descriptor is bounded");
        self.register_component::<engine_core::TerrainPrefetch>(
            InspectorDescriptor::generated(schemas, "TerrainPrefetch")
                .with_default(json!(engine_core::TerrainPrefetch::default())),
            |prefetch| json!(prefetch),
        )
        .expect("built-in descriptor is bounded");
        self.register_component::<engine_core::Deformer>(
            InspectorDescriptor::generated(schemas, "Deformer")
                .with_default(json!(engine_core::Deformer::default())),
            |deformer| json!(deformer),
        )
        .expect("built-in descriptor is bounded");
        self.register_resource::<engine_core::Tracks>(
            InspectorDescriptor::generated(schemas, "Tracks"),
            |tracks| {
                json!(crate::views::TracksView {
                    stamps: tracks.len(),
                    pressed: tracks.pressed(),
                    dropped: tracks.dropped(),
                    latest: tracks.latest().copied(),
                })
            },
        )
        .expect("built-in descriptor is bounded");
        self.register_resource::<engine_core::WorldCells>(
            InspectorDescriptor::generated(schemas, "WorldCells"),
            |cells| {
                let stats = cells.stats();
                let grids = cells.grids();
                json!(crate::views::WorldCellsView {
                    cells: stats.cells,
                    loaded: stats.loaded,
                    pending: stats.pending,
                    items: stats.items,
                    entities: stats.entities,
                    spawned: stats.spawned,
                    despawned: stats.despawned,
                    velocity: cells.velocity().to_array(),
                    grids: engine_core::STREAMING_GRIDS
                        .iter()
                        .zip(grids)
                        .map(|(grid, [count, loaded])| crate::views::StreamingGridView {
                            cell: grid.cell,
                            range: grid.range,
                            reach: grid.reach.is_finite().then_some(grid.reach),
                            cells: count,
                            loaded,
                        })
                        .collect(),
                    nearest: cells
                        .nearest(32)
                        .into_iter()
                        .map(|cell| crate::views::StreamedCellView {
                            grid: cell.grid,
                            key: cell.key,
                            items: cell.items,
                            entities: cell.entities,
                            loaded: cell.loaded,
                            distance: cell.distance,
                        })
                        .collect(),
                })
            },
        )
        .expect("built-in descriptor is bounded");
        self.register_resource::<engine_core::WorldChanges>(
            InspectorDescriptor::generated(schemas, "WorldChanges"),
            |changes| {
                json!(crate::views::WorldChangesView {
                    removed: changes.removed().len(),
                    ids: changes.removed().take(64).map(str::to_owned).collect(),
                })
            },
        )
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "NavAgent"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<engine_core::NavAgent>(entity).is_some())
            }),
            read: Box::new(|world, entity, _scene| {
                let entity = entity?;
                let agent = *world.get::<engine_core::NavAgent>(entity)?;
                let walk = world.get::<engine_core::NavWalk>(entity)?;
                let position = world
                    .get::<Transform>(entity)
                    .map(|transform| transform.translation);
                let left = position.map_or(0.0, |position| {
                    let mut from = position;
                    let mut left = 0.0;
                    for point in walk.points.iter().skip(walk.next) {
                        left += from.distance(*point);
                        from = *point;
                    }
                    left
                });
                Some(json!(crate::views::NavAgentView {
                    agent,
                    status: walk.status,
                    searched_for: walk.destination,
                    length: walk.length,
                    points: walk.points.len(),
                    next: walk.next,
                    left,
                    searched: walk.searched,
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "NavAgents"),
            kind: InspectorKind::Resource,
            has: Box::new(|world, _| world.get_resource::<engine_core::NavAgents>().is_some()),
            read: Box::new(|world, _, _| {
                let stats = world.get_resource::<engine_core::NavAgents>()?.stats();
                let mut query = world.try_query::<(
                    Entity,
                    &engine_core::NavAgent,
                    &engine_core::NavWalk,
                    &Transform,
                )>()?;
                let first = query
                    .iter(world)
                    .take(16)
                    .map(|(entity, agent, walk, transform)| {
                        let mut from = transform.translation;
                        let mut left = 0.0;
                        for point in walk.points.iter().skip(walk.next) {
                            left += from.distance(*point);
                            from = *point;
                        }
                        crate::views::NavAgentRow {
                            id: world
                                .get::<SceneEntity>(entity)
                                .map(|metadata| metadata.authored_id.clone()),
                            status: walk.status,
                            position: transform.translation.to_array(),
                            destination: agent.destination,
                            left,
                        }
                    })
                    .collect();
                Some(json!(crate::views::NavAgentsView {
                    agents: stats.agents,
                    searching: stats.searching,
                    walking: stats.walking,
                    arrived: stats.arrived,
                    unreachable: stats.unreachable,
                    work: stats.work,
                    first,
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
        self.register_component::<engine_core::StreamedId>(
            InspectorDescriptor::generated(schemas, "StreamedId"),
            |id| {
                json!(crate::views::StreamedIdView {
                    id: id.0.to_string()
                })
            },
        )
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Grounded"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<engine_core::Grounded>(entity).is_some())
            }),
            read: Box::new(|world, entity, _scene| {
                let entity = entity?;
                let grounded = world.get::<engine_core::Grounded>(entity)?;
                let at = world.get::<engine_core::GroundedAt>(entity);
                Some(json!(crate::views::GroundedView {
                    offset: grounded.offset,
                    level: at.and_then(|at| at.level),
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
        self.register_component::<Tint>(
            InspectorDescriptor::generated(schemas, "Tint")
                .with_default(json!(crate::views::TintView(Tint::WHITE.0))),
            |tint| json!(crate::views::TintView(tint.0)),
        )
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Decal"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Decal>(entity).is_some())
            }),
            read: Box::new(|world, entity, scene| {
                let decal = world.get::<Decal>(entity?)?;
                let assets = world.get_resource::<AssetLibrary>();
                let material = scene.and_then(|scene| {
                    scene
                        .materials()
                        .find(|(_, handle)| *handle == decal.material)
                        .map(|(id, _)| id.to_owned())
                });
                let field = SceneDecal::from_runtime(decal, material.clone().unwrap_or_default());
                Some(json!(DecalView {
                    material,
                    material_alive: assets
                        .is_some_and(|assets| assets.material(decal.material).is_some()),
                    blend: field.blend,
                    opacity: field.opacity,
                    fade_degrees: field.fade_degrees,
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "SplineMesh"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<engine_core::SplineMesh>(entity).is_some())
            }),
            read: Box::new(|world, entity, scene| {
                let entity = entity?;
                let spline = world.get::<engine_core::SplineMesh>(entity)?;
                let chunks = world.get::<engine_core::SplineChunks>(entity);
                let mesh = scene.and_then(|scene| {
                    scene
                        .meshes()
                        .find(|(_, handle)| *handle == spline.mesh)
                        .map(|(id, _)| id.to_owned())
                });
                let material = scene.and_then(|scene| {
                    scene
                        .materials()
                        .find(|(_, handle)| *handle == spline.material)
                        .map(|(id, _)| id.to_owned())
                });
                Some(json!(SplineMeshView {
                    points: spline.points.iter().map(|point| point.to_array()).collect(),
                    mesh,
                    material,
                    length: chunks.map(|chunks| chunks.length),
                    copies: chunks.map(|chunks| chunks.copies),
                    chunks: chunks.map(|chunks| chunks.entities.len()),
                    vertices: chunks.map(|chunks| chunks.vertices),
                    error: chunks.and_then(|chunks| chunks.error.clone()),
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
        self.register_component::<ReflectionProbe>(
            InspectorDescriptor::generated(schemas, "ReflectionProbe"),
            |probe| json!(SceneReflectionProbe::from(probe)),
        )
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Terrain"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<engine_core::Terrain>(entity).is_some())
            }),
            read: Box::new(|world, entity, scene| {
                let entity = entity?;
                let terrain = world.get::<engine_core::Terrain>(entity)?;
                let mut levels = vec![0_u32; terrain.pyramid.levels() as usize];
                let (mut level_errors, mut level_ranges) = (Vec::new(), Vec::new());
                let (mut hydrology, mut erosion, mut paths) = (None, None, Vec::new());
                if let Some(terrains) = world.get_resource::<engine_core::Terrains>() {
                    for tile in terrains.tiles(entity) {
                        levels[usize::from(tile.key().level)] += 1;
                    }
                    if let Some((errors, ranges)) = terrains.level_detail(entity) {
                        level_errors = errors.to_vec();
                        level_ranges = ranges.to_vec();
                    }
                    let holds = |wanted: fn(&engine_core::Layer) -> bool| {
                        matches!(&terrain.source, engine_core::TerrainSource::Generated(layers)
                            if layers.layers().iter().any(wanted))
                    };
                    let bakes = terrains.bakes(entity);
                    if holds(|layer| matches!(layer, engine_core::Layer::Hydrology(_))) {
                        let stats = bakes.hydrology;
                        hydrology = Some(crate::views::HydrologyView {
                            baking: bakes.baking,
                            samples: stats.map(|stats| stats.samples),
                            channel_samples: stats.map(|stats| stats.channel_samples),
                            lake_samples: stats.map(|stats| stats.lake_samples),
                            max_area: stats.map(|stats| stats.max_area),
                            falls: stats.map(|stats| stats.falls),
                        });
                    }
                    // Each path layer's profile, once baked.
                    if let Some(layers) = terrains.layers(entity) {
                        for (index, layer) in layers.layers().iter().enumerate() {
                            if let engine_core::Layer::Path(stage) = layer {
                                let stats = stage.baked.as_ref().map(|profile| profile.stats);
                                paths.push(crate::views::PathView {
                                    layer: index,
                                    baked: stats.is_some(),
                                    length: stats.map(|stats| stats.length),
                                    samples: stats.map(|stats| stats.samples),
                                    steepest: stats.map(|stats| stats.steepest),
                                    deepest_cut: stats.map(|stats| stats.deepest_cut),
                                    highest_fill: stats.map(|stats| stats.highest_fill),
                                    bridges: stats.map(|stats| stats.bridges),
                                    bridged: stats.map(|stats| stats.bridged),
                                    tunnels: stats.map(|stats| stats.tunnels),
                                    tunneled: stats.map(|stats| stats.tunneled),
                                });
                            }
                        }
                    }
                    if holds(|layer| matches!(layer, engine_core::Layer::Erosion(_))) {
                        let stats = bakes.erosion;
                        erosion = Some(crate::views::ErosionView {
                            baking: bakes.baking,
                            samples: stats.map(|stats| stats.samples),
                            rounds: stats.map(|stats| stats.rounds),
                            deepest: stats.map(|stats| stats.deepest),
                            highest: stats.map(|stats| stats.highest),
                        });
                    }
                }
                Some(json!(crate::views::TerrainView {
                    terrain: engine_scene::SceneTerrain::from_runtime(
                        terrain,
                        None,
                        |handle| {
                            scene.and_then(|scene| {
                                scene
                                    .materials()
                                    .find(|(_, candidate)| *candidate == handle)
                                    .map(|(id, _)| id.to_owned())
                            })
                        },
                        |handle| {
                            scene.and_then(|scene| {
                                scene
                                    .meshes()
                                    .find(|(_, candidate)| *candidate == handle)
                                    .map(|(id, _)| id.to_owned())
                            })
                        }
                    ),
                    levels: terrain.pyramid.levels(),
                    extent: terrain.pyramid.extent().to_array(),
                    resident_tiles_by_level: levels,
                    level_errors,
                    level_ranges,
                    hydrology,
                    erosion,
                    paths,
                    edits: terrain
                        .edits
                        .iter()
                        .map(|edit| crate::views::TerrainEditView {
                            id: edit.id.to_string(),
                            stamp: edit.stamp,
                        })
                        .collect(),
                    surroundings: world
                        .get_resource::<engine_core::Terrains>()
                        .and_then(|terrains| terrains.surroundings(entity))
                        .map(|status| crate::views::SurroundingsView {
                            reach: status.reach,
                            tiles: status.tiles,
                            spacing: status.spacing,
                            resident_tiles: status.resident_tiles,
                            patches: status.patches,
                        }),
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
        self.register_resource::<engine_core::Terrains>(
            InspectorDescriptor::generated(schemas, "Terrains"),
            |terrains| {
                let stats = terrains.stats();
                json!(crate::views::TerrainsView {
                    terrains: stats.terrains,
                    resident_tiles: stats.resident_tiles,
                    loading_tiles: stats.loading_tiles,
                    shaped_tiles: stats.shaped_tiles,
                    waiting_tiles: stats.waiting_tiles,
                    cached_tiles: stats.cached_tiles,
                    cached_bytes: stats.cached_bytes,
                    restored_tiles: stats.restored_tiles,
                    idle_tiles: stats.idle_tiles,
                    banded_tiles: stats.banded_tiles,
                    dropped_tiles: stats.dropped_tiles,
                    patches: stats.patches,
                    fallback_patches: stats.fallback_patches,
                    baking: stats.baking,
                    bridges: stats.bridges,
                    tunnels: stats.tunnels,
                    falls: stats.falls,
                    surroundings_tiles: stats.surroundings_tiles,
                    surroundings_patches: stats.surroundings_patches,
                })
            },
        )
        .expect("built-in descriptor is bounded");
        self.register_resource::<engine_core::Jobs>(
            InspectorDescriptor::generated(schemas, "Jobs"),
            |jobs| {
                let stats = jobs.stats();
                json!(crate::views::JobsView {
                    workers: stats.workers,
                    running: stats.running,
                    submitted: stats.submitted,
                    completed: stats.completed,
                    failed: stats.failed,
                    cancelled: stats.cancelled,
                    parts: stats.parts,
                    shares: stats.shares,
                    shared_bytes: stats.shared_bytes,
                    cache_hits: stats.cache_hits,
                    cache_misses: stats.cache_misses,
                    cache_bytes: stats.cache_bytes,
                    cache_mismatches: stats.cache_mismatches,
                })
            },
        )
        .expect("built-in descriptor is bounded");
        self.register_component::<ParticleEmitter>(
            InspectorDescriptor::generated(schemas, "ParticleEmitter")
                .with_default(json!(SceneEmitter::default())),
            |emitter| json!(SceneEmitter::from(emitter)),
        )
        .expect("built-in descriptor is bounded");
    }

    /// Animation players and skins, naming clips and
    /// entities by their authored IDs.
    fn register_animation(&mut self, schemas: &Value) {
        // Animation: players and skins name clips
        // and entities by their authored IDs.
        let authored = |world: &World, entity: Option<Entity>| {
            entity
                .and_then(|entity| world.get::<SceneEntity>(entity))
                .map(|source| source.authored_id.clone())
        };
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "AnimationPlayer"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<AnimationPlayer>(entity).is_some())
            }),
            read: Box::new(move |world, entity, scene| {
                let player = world.get::<AnimationPlayer>(entity?)?;
                let clip = world
                    .get_resource::<AssetLibrary>()
                    .and_then(|assets| assets.clip(player.clip));
                Some(json!(AnimationPlayerView {
                    clip: scene.and_then(|scene| scene
                        .clips()
                        .find(|(_, handle)| *handle == player.clip)
                        .map(|(id, _)| id.to_owned())),
                    clip_alive: clip.is_some(),
                    duration: clip.map(|clip| clip.duration()),
                    time: player.time,
                    speed: player.speed,
                    looping: player.looping,
                    targets: player
                        .targets
                        .iter()
                        .map(|&target| authored(world, target))
                        .collect(),
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Skin"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Skin>(entity).is_some())
            }),
            read: Box::new(move |world, entity, _scene| {
                let skin = world.get::<Skin>(entity?)?;
                Some(json!(SkinView {
                    joints: skin
                        .joints
                        .iter()
                        .map(|&joint| authored(world, Some(joint)))
                        .collect(),
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
    }

    /// The look: the atmosphere, every effect section and the
    /// whole environment.
    fn register_environment(&mut self, schemas: &Value) {
        // The environment: each resource serializes as its
        // scene section, so values copy into controls and v5 scenes as is.
        self.register_resource::<Atmosphere>(
            InspectorDescriptor::generated(schemas, "Atmosphere")
                .with_default(json!(SceneAtmosphere::from(&Atmosphere::default()))),
            |atmosphere| json!(SceneAtmosphere::from(atmosphere)),
        )
        .expect("built-in descriptor is bounded");
        // Every effect section of a scene environment as its resource, in
        // the section's shape.
        struct Sections<'t> {
            tools: &'t mut DevTools,
            schemas: &'t Value,
        }
        impl SectionVisitor for Sections<'_> {
            fn section<T: EnvironmentSection>(&mut self, _: &'static str, _: Option<T>) {
                let schema =
                    self.schemas.get(T::NAME).cloned().unwrap_or_else(|| {
                        panic!("build.rs generates no schema named {}", T::NAME)
                    });
                // The settings type's doc comment, whole in the schema.
                let description = summary(
                    schema["description"]
                        .as_str()
                        .expect("every environment section's settings type documents itself"),
                );
                self.tools
                    .register_resource::<T>(
                        InspectorDescriptor::new(T::NAME, description, schema)
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
        // The whole environment's schema naming each section's inspector,
        // which carries its fields, so it stays within bounds as sections
        // grow.
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("Environment", "The whole current look as a scene environment: paste it into a scene to persist it, or send sections to the environment control. Missing effect resources read as off. Each section's inspector describes its fields.", sections_schema()),
            kind: InspectorKind::Resource,
            has: Box::new(|_, _| true),
            read: Box::new(|world, _, _| Some(json!(SceneEnvironment::current(world)))),
        }).expect("built-in descriptor is bounded");
    }
}

/// The scene environment's schema: each section an object (or null) whose
/// fields the inspector it names describes.
fn sections_schema() -> Value {
    struct Names(Vec<(&'static str, &'static str)>);
    impl SectionVisitor for Names {
        fn section<T: EnvironmentSection>(&mut self, name: &'static str, _: Option<T>) {
            self.0.push((name, T::NAME));
        }
    }
    let mut names = Names(vec![("atmosphere", "Atmosphere")]);
    SceneEnvironment::default().visit(&mut names);
    let properties: serde_json::Map<String, Value> = names
        .0
        .into_iter()
        .map(|(field, inspector)| {
            (
                field.to_owned(),
                json!({
                    "type": ["object", "null"],
                    "description": format!("The {inspector} inspector describes its fields."),
                }),
            )
        })
        .collect();
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SceneEnvironment",
        "type": "object",
        "additionalProperties": false,
        "properties": properties,
    })
}

/// The scene schema, which exceeds one response: with `definition`, that
/// definition; without, the top level with its definitions' names in place
/// of their bodies.
fn scene_schema(definition: Option<&str>) -> Result<Value, DevError> {
    let mut schema: Value =
        serde_json::from_str(include_str!(concat!(env!("OUT_DIR"), "/scene.schema.json")))
            .expect("generated scene schema is JSON");
    let definitions = schema
        .as_object_mut()
        .and_then(|schema| schema.remove("$defs"))
        .unwrap_or_default();
    match definition {
        Some(name) => definitions.get(name).cloned().ok_or_else(|| {
            DevError::new(
                "unknown_definition",
                "$.definition",
                "not a definition of the scene schema; describe SceneDocument lists them",
            )
        }),
        None => {
            let names: Vec<&String> = definitions
                .as_object()
                .map(|definitions| definitions.keys().collect())
                .unwrap_or_default();
            schema["definitions"] = json!(names);
            Ok(schema)
        }
    }
}

/// The first sentence of `doc`, within an inspector description's bytes.
fn summary(doc: &str) -> String {
    let sentence = doc
        .find(". ")
        .map_or(doc, |end| &doc[..=end])
        .replace('\n', " ");
    if sentence.len() <= MAX_DESCRIPTION_BYTES {
        return sentence;
    }
    let mut end = MAX_DESCRIPTION_BYTES - 3;
    while !sentence.is_char_boundary(end) {
        end -= 1;
    }
    let cut = sentence[..end].rfind(' ').unwrap_or(end);
    format!("{}...", &sentence[..cut])
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
        if let Some(builtins) = self.builtins
            && self.inspectors.len() >= builtins + MAX_INSPECTORS
        {
            return Err(DevError::new(
                "inspector_limit",
                "$.name",
                format!("a game may register at most {MAX_INSPECTORS} inspectors"),
            ));
        }
        if inspector.descriptor.description.len() > MAX_DESCRIPTION_BYTES
            || encode(&inspector.descriptor, MAX_INSPECTOR_BYTES).is_err()
        {
            return Err(DevError::new(
                "inspector_too_large",
                "$.schema",
                format!(
                    "descriptor must fit {MAX_INSPECTOR_BYTES} bytes and description {MAX_DESCRIPTION_BYTES} bytes"
                ),
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
                format!("request must fit {MAX_REQUEST_BYTES} UTF-8 bytes"),
            ));
        }
        let request: DevRequest = serde_json::from_str(source)
            .map_err(|error| DevError::new("invalid_request", "$", error.to_string()))?;
        if request.version() != PROTOCOL_VERSION {
            return Err(DevError::new(
                "unsupported_version",
                "$.version",
                format!("accepted protocol version: {PROTOCOL_VERSION}"),
            ));
        }
        if scene.is_some_and(|scene| !scene.belongs_to(engine.app.world())) {
            return Err(DevError::new(
                "wrong_world",
                "$.scene",
                "scene instance belongs to a different engine world",
            ));
        }
        match request.handler() {
            crate::Handler::Collision => return crate::collision::execute(engine, scene, request),
            crate::Handler::Terrain => return crate::terrain::execute(engine, scene, request),
            crate::Handler::Tools => {}
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
            DevRequest::Describe {
                name, definition, ..
            } => self.describe(name.as_deref(), definition.as_deref()),
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
            | DevRequest::TerrainSweep { .. }
            | DevRequest::TerrainRoute { .. }
            | DevRequest::WaterQuery { .. }
            | DevRequest::DeformationQuery { .. }
            | DevRequest::NavPath { .. } => unreachable!("only tools queries reach the tools"),
        }
    }

    fn describe(&self, name: Option<&str>, definition: Option<&str>) -> Result<Value, DevError> {
        let Some(name) = name else {
            let mut types: Vec<Value> = self.inspectors.values().map(|inspector| json!({"name": inspector.descriptor.name, "kind": inspector.kind, "description": inspector.descriptor.description})).collect();
            types.push(json!({"name": "SceneDocument", "kind": "authoring", "source": engine_scene::schema_path()}));
            types.push(json!({"name": "protocol", "kind": "protocol"}));
            types.push(json!({"name": "BrowserControl", "kind": "protocol"}));
            types.push(json!({"name": "FrameStats", "kind": "observation"}));
            return Ok(json!({"types": types}));
        };
        match name {
            "SceneDocument" => Ok(
                json!({"name": name, "kind": "authoring", "source": engine_scene::schema_path(), "schema": scene_schema(definition)?, "constraints": format!("Rust validation checks references, unique expanded IDs, cycles, depth, finite world matrices, collision admission, environment values and asset budgets. Fields newer than version 1 require the version that introduced them: {}. Every older version stays accepted with its own semantics. Prefabs cannot nest. An environment sets every section on instantiation, omitted ones to the engine default state; the opt-in ones ({}) are then off.", engine_scene::version_features(), SceneEnvironment::opt_in_sections().join(", "))}),
            ),
            "protocol" | "BrowserControl" => {
                let schemas: Value = serde_json::from_str(include_str!(concat!(
                    env!("OUT_DIR"),
                    "/inspectors.json"
                )))
                .expect("generated schemas are JSON");
                Ok(
                    json!({"name": name, "kind": "protocol", "schema": schemas[name], "constraints": format!("version {PROTOCOL_VERSION}; every control but pause and resume requires a paused host; reload revisions must fit u64. Numeric bounds, lengths and defaults are in the schema. Spatial queries need CollisionPlugin but no pause; directions are nonzero and finite; rotations are normalized XYZW. Layer filtering is symmetric; sensors are included, excluded or the only ones. Raycasts and casts return the closest complete hit, terrain heightfields included; casts translate with a fixed orientation. Terrain heights read null where no tile is resident; terrain sweeps compare queries, collision rays and layers. Budget failures never return partial hits. Events are non-consuming latest fixed-tick sensor transitions, with attempted and committed ticks. Present and capture previews preserve clocks and input. The terrain control freezes streaming at the camera's position. Ordinary render, step and capture show current poses. Benchmarks in step mode advance one tick a sample, in present mode sample alpha 0.5 without ticks, and clear input. Controls do not save source.")}),
                )
            }
            "FrameStats" => {
                let schema: Value = serde_json::from_str(crate::FRAME_STATS_SCHEMA)
                    .expect("generated schema is JSON");
                Ok(
                    json!({"name": name, "kind": "observation", "schema": schema, "source": "every successful reply's result.rendering: the last frame's stats"}),
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
            let entity = authored_entity(world, scene, id).ok_or_else(|| {
                DevError::new(
                    "unknown_entity",
                    "$.entity",
                    "authored ID does not exist in the active scene (streamed entities only while their cell is loaded)",
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
                let entity = authored_entity(world, scene, id).ok_or_else(|| {
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

/// The entity of authored ID `id` in `scene`: one it spawned, or else a
/// loaded streamed entity it owns.
pub(crate) fn authored_entity(world: &World, scene: &SceneInstance, id: &str) -> Option<Entity> {
    scene.entity(id).or_else(|| {
        scene.streamed()?;
        world
            .try_query::<(Entity, &SceneEntity)>()?
            .iter(world)
            .find(|(_, metadata)| metadata.authored_id == id && metadata.scene_id == scene.id())
            .map(|(entity, _)| entity)
    })
}
