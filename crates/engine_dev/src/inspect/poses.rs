use super::*;

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
        self.register_component::<InterpolateTransform>(
            InspectorDescriptor::generated(schemas, "InterpolateTransform"),
            |_| json!(InterpolateTransformView { enabled: true }),
        )
        .expect("descriptor_bounded");
        self.register_resource::<PresentationState>(
            InspectorDescriptor::generated(schemas, "PresentationState"),
            presentation_value,
        )
        .expect("descriptor_bounded");
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

pub(super) fn presentation_value(value: &PresentationState) -> Value {
    json!(presentation_view(value))
}

pub(super) fn presentation_view(value: &PresentationState) -> PresentationStateView {
    PresentationStateView {
        frame: value.frame().map(|frame| frame.to_string()),
        tick: value.tick().map(|tick| tick.to_string()),
        alpha: value.alpha(),
        history_nodes: value.history_nodes(),
        history_capacity: value.history_capacity(),
        history_bytes: value.history_bytes(),
        presentation_scratch_bytes: value.presentation_scratch_bytes(),
        blended_nodes: value.blended_nodes(),
        largest_blend_distance: value.largest_blend_distance(),
    }
}
