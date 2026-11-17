use super::*;

type StateRegister = fn(&mut DevTools, &Value, &str) -> Result<(), Error>;

impl DevTools {
    pub(super) fn register_poses(&mut self, schemas: &Value) {
        self.register_component::<Transform>(
            InspectorDescriptor::generated(schemas, "Transform")
                .with_default(transform_value(&Transform::default())),
            transform_value,
        )
        .expect("descriptor_bounded");
        self.insert(Inspector {
            brick: None,
            descriptor: InspectorDescriptor::generated(schemas, "GlobalTransform"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Transform>(entity).is_some())
            }),
            read: Box::new(global_transform_view),
        })
        .expect("descriptor_bounded");
        self.insert(Inspector {
            brick: None,
            descriptor: InspectorDescriptor::generated(schemas, "NodeWord"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Transform>(entity).is_some())
            }),
            read: Box::new(node_word_view),
        })
        .expect("descriptor_bounded");
        let states: [(&str, StateRegister); 5] = [
            (
                "PresentationState",
                Self::register_state::<PresentationState>,
            ),
            ("Readiness", Self::register_state::<engine_core::Readiness>),
            ("Records", Self::register_state::<engine_core::Records>),
            (
                "MeshLevels",
                Self::register_state::<engine_core::MeshLevels>,
            ),
            ("Devices", Self::register_state::<engine_core::Devices>),
        ];
        for (name, register) in states {
            register(self, schemas, name).expect("descriptor_bounded");
        }
        self.insert(Inspector {
            brick: None,
            descriptor: InspectorDescriptor::generated(schemas, "Hierarchy"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Transform>(entity).is_some())
            }),
            read: Box::new(hierarchy_view),
        })
        .expect("descriptor_bounded");
    }
}

fn global_transform_view(
    world: &World,
    entity: Option<Entity>,
    _context: &Context<'_>,
) -> Option<Value> {
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
}

fn node_word_view(world: &World, entity: Option<Entity>, _context: &Context<'_>) -> Option<Value> {
    let word = world.get_resource::<GlobalTransforms>()?.word(entity?)?;
    Some(json!(word))
}

fn hierarchy_view(world: &World, entity: Option<Entity>, context: &Context<'_>) -> Option<Value> {
    let scene = context.scene;
    let entity = entity?;
    world.get::<Transform>(entity)?;
    let parent = world.get::<Parent>(entity).map(Parent::parent);
    let parent_metadata = parent.and_then(|parent| world.get::<SceneEntity>(parent));
    let metadata = world.get::<SceneEntity>(entity);
    Some(json!(HierarchyView {
        parent: parent_metadata.map(|metadata| metadata.authored_id.clone()),
        parent_scene: parent_metadata.map(|metadata| metadata.scene_id.clone()),
        parent_alive: parent.map(|parent| world.get_entity(parent).is_ok()),
        parent_in_active_scene: parent.map(|parent| parent_metadata.is_some_and(|metadata| {
            active_entity(world, scene, &metadata.authored_id) == Some(parent)
        })),
        authored_parent: metadata.and_then(|metadata| metadata.authored_parent.clone()),
        prefab_id: metadata.and_then(|metadata| metadata.prefab_id.clone()),
        prefab_instance: metadata.and_then(|metadata| metadata.prefab_instance.clone()),
        prefab_local_id: metadata.and_then(|metadata| metadata.prefab_local_id.clone()),
    }))
}

fn active_entity(world: &World, scene: Option<&SceneInstance>, id: &str) -> Option<Entity> {
    match scene {
        Some(scene) => scene.entity(id),
        None => world
            .get_resource::<OperationScene>()
            .and_then(|active| active.entities.get(id).copied()),
    }
}

pub(super) fn transform_value(value: &Transform) -> Value {
    json!(TransformView {
        translation: value.translation.to_array(),
        rotation: value.rotation.to_array(),
        scale: value.scale.to_array()
    })
}
