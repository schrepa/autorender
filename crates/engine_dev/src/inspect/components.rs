//! Inspectors of the core's components and resources: the camera, meshes,
//! lights, grounding, tints, decals, spline meshes, probes, input, jobs and
//! particle emitters.

use super::*;

impl DevTools {
    /// The camera, meshes and the core's components entities carry.
    pub(super) fn register_components(&mut self, schemas: &Value) {
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
            read: Box::new(|world, entity, context| {
                let mesh3d = world.get::<Mesh3d>(entity?)?;
                let assets = world.get_resource::<AssetLibrary>()?;
                Some(json!(mesh_view(assets, (entity?, mesh3d), context)))
            }),
        })
        .expect("built-in descriptor is bounded");
        coverage::register(self, schemas);
        self.register_component::<LocalLight>(
            InspectorDescriptor::generated(schemas, "LocalLight")
                .with_default(json!(SceneLight::default())),
            |light| json!(SceneLight::from(light)),
        )
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Grounded"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<engine_core::Grounded>(entity).is_some())
            }),
            read: Box::new(grounded_view),
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
            read: Box::new(decal_view),
        })
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "SplineMesh"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<engine_core::SplineMesh>(entity).is_some())
            }),
            read: Box::new(spline_mesh_view),
        })
        .expect("built-in descriptor is bounded");
        self.register_component::<ReflectionProbe>(
            InspectorDescriptor::generated(schemas, "ReflectionProbe"),
            |probe| json!(SceneReflectionProbe::from(probe)),
        )
        .expect("built-in descriptor is bounded");
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Input"),
            kind: InspectorKind::Resource,
            has: Box::new(|world, _| world.get_resource::<InputState>().is_some()),
            read: Box::new(input_view),
        })
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
                    last_failure: jobs.last_failure().map(str::to_owned),
                    cancelled: stats.cancelled,
                    parts: stats.parts,
                    shares: stats.shares,
                    shared_bytes: stats.shared_bytes,
                    cache_hits: stats.cache_hits,
                    cache_misses: stats.cache_misses,
                    cache_bytes: stats.cache_bytes,
                    cache_mismatches: stats.cache_mismatches,
                    held: stats.held,
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
}

/// How far above the ground the entity stands, and the level of the tile under it.
fn grounded_view(world: &World, entity: Option<Entity>, _context: &Context<'_>) -> Option<Value> {
    let entity = entity?;
    let grounded = world.get::<engine_core::Grounded>(entity)?;
    let at = world.get::<engine_core::GroundedAt>(entity);
    Some(json!(crate::views::GroundedView {
        offset: grounded.offset,
        level: at.and_then(|at| at.level),
    }))
}

/// The entity's decal with its material's authored ID.
fn decal_view(world: &World, entity: Option<Entity>, context: &Context<'_>) -> Option<Value> {
    let scene = context.scene;
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
        material_alive: assets.is_some_and(|assets| assets.material(decal.material).is_some()),
        blend: field.blend,
        opacity: field.opacity,
        fade_degrees: field.fade_degrees,
    }))
}

/// The entity's spline mesh with its mesh's and material's authored IDs and its chunks.
fn spline_mesh_view(world: &World, entity: Option<Entity>, context: &Context<'_>) -> Option<Value> {
    let scene = context.scene;
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
}

/// The input state with the actions held.
fn input_view(world: &World, _: Option<Entity>, _: &Context<'_>) -> Option<Value> {
    let input = world.get_resource::<InputState>()?;
    let actions = world.get_resource::<ActionInput>();
    Some(json!(InputView {
        move_forward: input.move_forward,
        move_right: input.move_right,
        move_up: input.move_up,
        look_yaw: input.look_yaw,
        look_pitch: input.look_pitch,
        held: (0..32)
            .filter(|&index| {
                ActionId::new(index as u8)
                    .is_some_and(|action| actions.is_some_and(|actions| actions.pressed(action)))
            })
            .collect(),
    }))
}

/// What the Mesh3d inspector reads of `mesh3d`: its mesh, its material as
/// a scene authors it and the textures its maps read, with what the renderer
/// made of them.
pub(super) fn mesh_view(
    assets: &AssetLibrary,
    (entity, mesh3d): (Entity, &Mesh3d),
    context: &Context<'_>,
) -> Mesh3dView {
    let Context {
        scene,
        renderer,
        render_slot,
    } = *context;
    let mesh = assets.mesh(mesh3d.mesh).map(|mesh| {
        let origin = scene.and_then(|scene| scene.mesh_origin(mesh3d.mesh));
        let bounds = mesh.bounds();
        MeshView {
            id: origin.map(|(id, _)| id.to_owned()),
            source: origin.map(|(_, source)| source.clone()),
            vertices: mesh.vertices().len(),
            indices: mesh.indices().len(),
            local_bounds: MeshBoundsView {
                min: bounds.min().to_array(),
                max: bounds.max().to_array(),
            },
            colors: mesh.colors().is_some(),
            uv1: mesh.uv1().is_some(),
            joints: mesh.joint_count(),
            virtual_geometry: mesh.virtual_geometry().map(|geometry| VirtualMeshView {
                levels: geometry.levels(),
                clusters: geometry.cluster_count() as u32,
                pages: geometry.page_count() as u32,
                triangles: u64::from(geometry.triangle_count()),
            }),
            rendered: renderer.and_then(|renderer| {
                let rendered = renderer.mesh(mesh3d.mesh)?;
                Some(MeshRenderView {
                    skinned: rendered.skinned,
                    levels: rendered
                        .levels
                        .iter()
                        .map(|&(indices, error)| LodLevelView { indices, error })
                        .collect(),
                    impostor: renderer.impostor(mesh3d.mesh, mesh3d.material),
                })
            }),
        }
    });
    let material = assets.material(mesh3d.material);
    let textures = material
        .map(|material| {
            material
                .maps()
                .into_iter()
                .flatten()
                .map(|map| {
                    let origin = scene.and_then(|scene| scene.texture_origin(map.texture));
                    let texture = assets.texture(map.texture);
                    TextureView {
                        id: origin.map(|(id, _)| id.to_owned()),
                        uri: origin.and_then(|(_, uri)| uri.map(str::to_owned)),
                        inline: origin.is_some_and(|(_, uri)| uri.is_none()),
                        alive: texture.is_some(),
                        width: texture.map(|texture| texture.width()),
                        height: texture.map(|texture| texture.height()),
                        emission_mask: texture.map(|texture| texture.has_emission_mask()),
                        gpu_resident: renderer
                            .map(|renderer| renderer.texture_resident(map.texture)),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let material = material.map(|material| {
        let id = scene
            .and_then(|scene| {
                scene
                    .materials()
                    .find(|&(_, handle)| handle == mesh3d.material)
            })
            .map(|(id, _)| id.to_owned());
        let texture = |handle| {
            scene
                .and_then(|scene| scene.texture_origin(handle))
                .map(|(id, _)| id.to_owned())
                .unwrap_or_default()
        };
        SceneMaterial::from_runtime(id.unwrap_or_default(), material, texture)
    });
    let drawn_level = renderer
        .zip(render_slot.and_then(|slot| slot(entity)))
        .and_then(|(renderer, slot)| renderer.drawn_level(slot));
    Mesh3dView {
        mesh,
        material,
        textures,
        drawn_level,
    }
}

pub(super) fn camera_value(value: &Camera) -> Value {
    json!(CameraView {
        position: value.position.to_array(),
        target: value.target.to_array(),
        vertical_fov_radians: value.vertical_fov_radians,
        near: value.near,
        far: value.far
    })
}
