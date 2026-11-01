use super::*;

impl DevTools {
    pub(super) fn register_components(&mut self, schemas: &Value) {
        self.register_resource::<Camera>(
            InspectorDescriptor::generated(schemas, "Camera")
                .with_default(camera_value(&Camera::default())),
            camera_value,
        )
        .expect("descriptor_bounded");
        self.insert(Inspector {
            brick: None,
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
        .expect("descriptor_bounded");
        self.insert(Inspector {
            brick: None,
            descriptor: InspectorDescriptor::generated(schemas, "Input"),
            kind: InspectorKind::Resource,
            has: Box::new(|world, _| world.get_resource::<InputState>().is_some()),
            read: Box::new(input_view),
        })
        .expect("descriptor_bounded");
    }
}

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
            rendered: renderer.and_then(|renderer| {
                let rendered = renderer.mesh(mesh3d.mesh)?;
                Some(MeshRenderView {
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
                    let (uri, inline, provided) = match origin.map(|(_, source)| source) {
                        Some(TextureOrigin::File { uri }) => (Some(uri.clone()), false, None),
                        Some(TextureOrigin::Inline) => (None, true, None),
                        Some(TextureOrigin::Provided { name }) => (None, false, Some(name.clone())),
                        None => (None, false, None),
                    };
                    TextureView {
                        id: origin.map(|(id, _)| id.to_owned()),
                        uri,
                        inline,
                        provided,
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
    let (vertical_fov_radians, height) = match value.projection {
        Projection::Perspective {
            vertical_fov_radians,
        } => (Some(vertical_fov_radians), None),
        Projection::Orthographic { height } => (None, Some(height)),
    };
    json!(CameraView {
        position: value.position.to_array(),
        target: value.target.to_array(),
        vertical_fov_radians,
        height,
        near: value.near,
        far: value.far
    })
}
