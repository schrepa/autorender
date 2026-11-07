use std::collections::BTreeMap;
use std::sync::LazyLock;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_registry::{CORE_ROW, MembersOf, OperationScene, SceneEntity};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    Camera, Clear, ComponentKind, Drawn, DrawnProjection, DrawnTable, Error, ErrorCode, Expected,
    GlobalTransforms, InstanceValues, Layers, List, MaterialHandle, Mesh3d, MeshHandle, Mover,
    Node, Parent, PassMask, Placed, Placement, Rect, SceneComponents, SceneContext, SceneCx,
    SceneForm, ScenePasses, Target, TextureHandle, View, scene_component, scene_components,
};

pub(crate) const COMPONENT: &str = "component";

static CORE: LazyLock<[ComponentKind; 5]> = LazyLock::new(|| {
    let node = ComponentKind::of::<Node>("node");
    let mover = ComponentKind::of::<Mover>("mover");
    let camera = ComponentKind::of::<Camera>("camera").check(camera_checked);
    let view = ComponentKind::form::<ViewForm>("view")
        .check(view_checked)
        .prepare(view_camera);
    let drawn = ComponentKind::form::<DrawnForm>("drawn");
    #[cfg(feature = "schema")]
    let (node, mover, camera, view, drawn) = (
        node.schema(schema_of::<Node>),
        mover.schema(schema_of::<Mover>),
        camera.schema(schema_of::<Camera>),
        view.schema(schema_of::<ViewForm>),
        drawn.schema(schema_of::<DrawnForm>),
    );
    [
        node.into(),
        mover.into(),
        camera.into(),
        view.into(),
        drawn.into(),
    ]
});

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DrawnForm {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<MeshHandle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialHandle>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub passes: BTreeMap<String, bool>,
    #[serde(default)]
    pub layers: Layers,
    #[serde(default)]
    pub order: i16,
    #[serde(default = "receives_shadows")]
    pub receives_shadows: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<DrawnProjection>,
    #[serde(default)]
    pub values: InstanceValues,
}

fn receives_shadows() -> bool {
    true
}

impl SceneForm for DrawnForm {
    type Component = Drawn;

    fn insert(self, world: &mut World, entity: Entity) -> Result<(), Error> {
        let table = world.get_resource::<DrawnTable>();
        let kind = table
            .and_then(|table| {
                table
                    .kinds()
                    .iter()
                    .find(|declared| declared.kind.name() == self.kind)
            })
            .map(|declared| declared.kind)
            .ok_or_else(|| {
                Error::new(ErrorCode::UnknownVariant, "$.kind")
                    .got(&self.kind)
                    .variants(
                        table
                            .into_iter()
                            .flat_map(DrawnTable::kinds)
                            .map(|declared| declared.kind.name()),
                    )
            })?;
        let passes = if self.passes.is_empty() {
            None
        } else {
            let unlisted = ScenePasses::default();
            let listed = world.get_resource::<ScenePasses>().unwrap_or(&unlisted);
            let named = serde_json::to_value(&self.passes).unwrap_or(Value::Null);
            Some(listed.encode(&named, listed.defaults(kind), "$.passes")?)
        };
        let shown = world.get::<Mesh3d>(entity).copied().unwrap_or_default();
        world.entity_mut(entity).insert(Drawn {
            kind,
            geometry: self.mesh.unwrap_or(shown.mesh),
            material: self.material.unwrap_or(shown.material),
            node: entity,
            passes,
            layers: self.layers,
            order: self.order,
            receives_shadows: self.receives_shadows,
            projection: self.projection,
            values: self.values,
        });
        Ok(())
    }

    fn read(world: &World, entity: Entity) -> Option<Self> {
        let drawn = world.get::<Drawn>(entity)?;
        let shown = world.get::<Mesh3d>(entity).copied().unwrap_or_default();
        let passes = drawn
            .passes
            .zip(world.get_resource::<ScenePasses>())
            .map(|(mask, listed)| switched(listed, mask, listed.defaults(drawn.kind)))
            .unwrap_or_default();
        Some(Self {
            kind: drawn.kind.name().to_owned(),
            mesh: (drawn.geometry != shown.mesh).then_some(drawn.geometry),
            material: (drawn.material != shown.material).then_some(drawn.material),
            passes,
            layers: drawn.layers,
            order: drawn.order,
            receives_shadows: drawn.receives_shadows,
            projection: drawn.projection,
            values: drawn.values.clone(),
        })
    }
}

fn switched(listed: &ScenePasses, mask: PassMask, base: PassMask) -> BTreeMap<String, bool> {
    listed
        .listed()
        .iter()
        .filter_map(|pass| pass.bit.map(|bit| (pass.name, bit)))
        .filter(|&(_, bit)| mask.has(bit) != base.has(bit))
        .map(|(name, bit)| (name.to_owned(), mask.has(bit)))
        .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ViewForm {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<TextureHandle>,
    #[serde(default)]
    pub viewport: Rect,
    #[serde(default = "every_layer")]
    pub layers: Layers,
    #[serde(default)]
    pub clear: Clear,
    #[serde(default)]
    pub order: i32,
    #[serde(skip)]
    found: Option<Entity>,
}

fn every_layer() -> Layers {
    Layers::ALL
}

fn view_checked(form: &ViewForm, path: &str) -> Result<(), Error> {
    form.viewport.check(&format!("{path}.viewport"))
}

fn view_camera(mut form: ViewForm, cx: &SceneCx<'_>, path: &str) -> Result<ViewForm, Error> {
    if let (Some(id), Some(scene)) = (form.camera.as_deref(), cx.scene()) {
        let found = scene.entity(id).ok_or_else(|| {
            Error::new(ErrorCode::UnknownReference, format!("{path}.camera")).got(id)
        })?;
        form.found = Some(found);
    }
    Ok(form)
}

impl SceneForm for ViewForm {
    type Component = View;

    fn insert(self, world: &mut World, entity: Entity) -> Result<(), Error> {
        let camera = match &self.camera {
            None => entity,
            Some(id) => self
                .found
                .filter(|found| *found != Entity::PLACEHOLDER)
                .ok_or_else(|| Error::new(ErrorCode::UnknownReference, "$.camera").got(id))?,
        };
        world.entity_mut(entity).insert(View {
            camera,
            target: self.texture.map_or(Target::Surface, Target::Texture),
            viewport: self.viewport,
            layers: self.layers,
            clear: self.clear,
            order: self.order,
        });
        Ok(())
    }

    fn read(world: &World, entity: Entity) -> Option<Self> {
        let view = world.get::<View>(entity)?;
        let camera = if view.camera == entity {
            None
        } else {
            Some(world.get::<SceneEntity>(view.camera)?.authored_id.clone())
        };
        let texture = match view.target {
            Target::Surface => None,
            Target::Texture(texture) => Some(texture),
        };
        Some(Self {
            camera,
            texture,
            viewport: view.viewport,
            layers: view.layers,
            clear: view.clear,
            order: view.order,
            found: None,
        })
    }
}

fn camera_checked(camera: &Camera, path: &str) -> Result<(), Error> {
    camera.check().map_err(|error| error.within(path))
}

#[cfg(feature = "schema")]
fn schema_of<C: schemars::JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(C)).unwrap_or(Value::Null)
}

#[doc(hidden)]
pub fn core_components() -> &'static [ComponentKind] {
    &*CORE
}

pub(crate) fn keys_taken(world: &World) -> Vec<Error> {
    let Some(members) = world.get_resource::<MembersOf<SceneComponents>>() else {
        return Vec::new();
    };
    members
        .all()
        .iter()
        .zip(members.owners())
        .filter(|(kind, _)| core_components().iter().any(|core| core.key() == kind.key()))
        .map(|(kind, &owner)| {
            Error::new(
                ErrorCode::NameTwice,
                format!("$.places.{}.{}", SceneComponents::NAME, kind.key()),
            )
            .got(CORE_ROW)
            .nearest(owner)
        })
        .collect()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentControl {
    entity: String,
    key: String,
    #[serde(default)]
    value: Value,
}

pub(crate) fn run_component(world: &mut World, request: Value) -> Result<Value, Error> {
    let ComponentControl { entity, key, value } = engine_settings::parse_value(request)?;
    let node = world
        .get_resource::<OperationScene>()
        .and_then(|scene| scene.entity(world, &entity))
        .ok_or_else(|| Error::new(ErrorCode::UnknownEntity, "$.entity").got(&entity))?;
    let kind = scene_component(world, &key).ok_or_else(|| {
        Error::new(ErrorCode::UnknownVariant, "$.key")
            .got(&key)
            .variants(scene_components(world).map(ComponentKind::key))
    })?;
    let context = world.get_resource::<SceneContext>().cloned();
    let set = || author(world, &kind, node, &value);
    match &context {
        Some(context) => context.scope(set),
        None => set(),
    }?;
    let value = kind.value(world, node).unwrap_or(Value::Null);
    Ok(json!({ "entity": entity, "key": key, "value": value }))
}

fn author(
    world: &mut World,
    kind: &ComponentKind,
    node: Entity,
    value: &Value,
) -> Result<(), Error> {
    if value.is_null() {
        kind.remove(world, node);
        return Ok(());
    }
    kind.validate(value, "$.value")?;
    match kind.placement() {
        Placement::Roots(code) if world.get::<Parent>(node).is_some() => {
            return Err(Error::new(code, "$.entity").expected(Expected::Absent));
        }
        Placement::Roots(_) | Placement::Anywhere | Placement::Entities(_) => {}
    }
    let placed = world
        .get_resource::<GlobalTransforms>()
        .and_then(|globals| globals.get(node))
        .and_then(|global| global.matrix());
    if let Some(matrix) = placed {
        kind.validate_placed(value, Placed { world: matrix }, "$.value")?;
    }
    kind.read(world, node, value)
        .map_err(|error| error.within("$.value"))
}
