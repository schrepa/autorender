use std::collections::BTreeMap;
use std::sync::LazyLock;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_registry::{CORE_ROW, MembersOf};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    Callable, Camera, Clear, ComponentKind, Drawn, DrawnProjection, DrawnTable, Error, ErrorCode,
    Expected, GeometryHandle, GlobalTransforms, InstanceValues, Layers, List, MaterialHandle, Mover,
    Node, Parent, PassMask, Placed, Placement, Rect, Rows, Rule, SceneComponents, SceneContext,
    SceneCx, SceneEntity, SceneForm, ScenePasses, Setting, Target, TextureHandle, View,
    entity_named, scene_component, scene_components,
};

pub struct ChangeOperation;

impl Callable for ChangeOperation {
    const NAME: &'static str = "change";
    type Request = ChangeRequest;
    type Reply = Rows<ComponentChanged>;

    fn answer(world: &mut World, request: ChangeRequest) -> Result<Self::Reply, Error> {
        let context = world.get_resource::<SceneContext>().cloned();
        Ok(Rows::each("changes", request.changes, |change| match &context {
            Some(context) => context.scope(|| changed(world, change)),
            None => changed(world, change),
        }))
    }

    fn request_schema(world: &World) -> Value {
        change_request(world)
    }
}

static CORE: LazyLock<[ComponentKind; 5]> = LazyLock::new(|| {
    let node = ComponentKind::of::<Node>("node");
    let mover = ComponentKind::of::<Mover>("mover");
    let camera = ComponentKind::of::<Camera>("camera").check(camera_checked);
    let view = ComponentKind::form::<ViewForm>("view")
        .check(view_checked)
        .prepare(view_camera);
    let drawn = ComponentKind::form::<DrawnForm>("drawn").check(drawn_checked);
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<GeometryHandle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialHandle>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub passes: BTreeMap<String, bool>,
    #[serde(default)]
    pub layers: Layers,
    #[serde(default)]
    pub order: i16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<DrawnProjection>,
    #[serde(default)]
    pub values: InstanceValues,
}

const TINT_RULE: Rule = Rule::range(0.0, 1.0);

fn drawn_checked(form: &DrawnForm, path: &str) -> Result<(), Error> {
    Setting::check(&form.values.tint, "tint", TINT_RULE).map_err(|invalid| Error {
        code: ErrorCode::InvalidValue,
        path: format!(
            "{path}.values.tint{}",
            invalid.field.strip_prefix("tint").unwrap_or_default()
        ),
        detail: Box::new(invalid.detail),
    })
}

impl SceneForm for DrawnForm {
    type Component = Drawn;

    fn insert(self, world: &mut World, entity: Entity) -> Result<(), Error> {
        let held = world.get::<Drawn>(entity).cloned();
        let table = world.get_resource::<DrawnTable>();
        let kind = match &self.kind {
            None => match held.as_ref().map(|drawn| drawn.kind) {
                Some(kind) => kind,
                None => {
                    return Err(Error::new(ErrorCode::MissingField, "$.kind").variants(
                        table
                            .into_iter()
                            .flat_map(DrawnTable::kinds)
                            .map(|declared| declared.kind.name()),
                    ));
                }
            },
            Some(name) => table
                .and_then(|table| {
                    table
                        .kinds()
                        .iter()
                        .find(|declared| declared.kind.name() == name)
                })
                .map(|declared| declared.kind)
                .ok_or_else(|| {
                    Error::new(ErrorCode::UnknownVariant, "$.kind")
                        .got(name)
                        .variants(
                            table
                                .into_iter()
                                .flat_map(DrawnTable::kinds)
                                .map(|declared| declared.kind.name()),
                        )
                })?,
        };
        let passes = if self.passes.is_empty() {
            None
        } else {
            let unlisted = ScenePasses::default();
            let listed = world.get_resource::<ScenePasses>().unwrap_or(&unlisted);
            let named = serde_json::to_value(&self.passes).unwrap_or(Value::Null);
            Some(listed.encode(&named, listed.defaults(kind), "$.passes")?)
        };
        world.entity_mut(entity).insert(Drawn {
            kind,
            geometry: self
                .mesh
                .or_else(|| held.as_ref().and_then(|drawn| drawn.geometry)),
            material: self
                .material
                .or_else(|| held.as_ref().and_then(|drawn| drawn.material)),
            passes,
            layers: self.layers,
            order: self.order,
            projection: self.projection,
            values: self.values,
        });
        Ok(())
    }

    fn read(world: &World, entity: Entity) -> Option<Self> {
        let drawn = world.get::<Drawn>(entity)?;
        let passes = drawn
            .passes
            .zip(world.get_resource::<ScenePasses>())
            .map(|(mask, listed)| switched(listed, mask, listed.defaults(drawn.kind)))
            .unwrap_or_default();
        Some(Self {
            kind: Some(drawn.kind.name().to_owned()),
            mesh: drawn.geometry,
            material: drawn.material,
            passes,
            layers: drawn.layers,
            order: drawn.order,
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
    #[serde(default = "black")]
    pub clear_color: [f32; 4],
    #[serde(default)]
    pub order: i32,
    #[serde(default)]
    pub steady: bool,
    #[serde(skip)]
    found: Option<Entity>,
}

fn every_layer() -> Layers {
    Layers::ALL
}

fn black() -> [f32; 4] {
    View::BLACK
}

fn view_checked(form: &ViewForm, path: &str) -> Result<(), Error> {
    if let Some(at) = form.clear_color.iter().position(|value| !value.is_finite()) {
        return Err(
            Error::new(ErrorCode::InvalidValue, format!("{path}.clear_color[{at}]"))
                .got(form.clear_color[at]),
        );
    }
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
            clear_color: self.clear_color,
            order: self.order,
            steady: self.steady,
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
        Some(Self::of_view(view, camera))
    }

    fn held(world: &World, entity: Entity) -> Option<Self> {
        world.get::<View>(entity).map(|view| Self::of_view(view, None))
    }
}

impl ViewForm {
    fn of_view(view: &View, camera: Option<String>) -> Self {
        let texture = match view.target {
            Target::Surface => None,
            Target::Texture(texture) => Some(texture),
        };
        Self {
            camera,
            texture,
            viewport: view.viewport,
            layers: view.layers,
            clear: view.clear,
            clear_color: view.clear_color,
            order: view.order,
            steady: view.steady,
            found: None,
        }
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
        .filter(|(kind, _)| {
            core_components()
                .iter()
                .any(|core| core.key() == kind.key())
        })
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

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComponentChange {
    pub entity: String,
    pub component: String,
    #[serde(default)]
    pub data: Value,
}

fn change_request(world: &World) -> Value {
    let changes: Vec<Value> = scene_components(world)
        .map(|kind| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "entity": { "type": "string" },
                    "component": { "const": kind.key(), "type": "string" },
                    "data": crate::authored_component(kind.key()),
                },
                "required": ["entity", "component"],
                "additionalProperties": false,
            })
        })
        .collect();
    let mut schema = crate::schema_of::<ChangeRequest>();
    if let Some(held) = schema.get_mut("$defs").and_then(Value::as_object_mut) {
        held.insert(
            "ComponentChange".to_owned(),
            serde_json::json!({ "oneOf": changes }),
        );
    }
    schema
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChangeRequest {
    pub changes: Vec<ComponentChange>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct ComponentChanged {
    pub entity: String,
    pub component: String,
    pub removed: bool,
}

fn changed(world: &mut World, change: ComponentChange) -> Result<ComponentChanged, Error> {
    let ComponentChange {
        entity,
        component,
        data,
    } = change;
    let node = entity_named(world, &entity, "$.entity")?;
    let kind = scene_component(world, &component).ok_or_else(|| {
        Error::new(ErrorCode::UnknownVariant, "$.component")
            .got(&component)
            .variants(scene_components(world).map(ComponentKind::key))
    })?;
    author(world, &kind, node, &data)?;
    Ok(ComponentChanged {
        entity,
        component,
        removed: data.is_null(),
    })
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
    kind.validate(value, "$.data")?;
    match kind.placement() {
        Placement::Roots(code) if world.get::<Parent>(node).is_some() => {
            return Err(Error::new(code, "$.entity").expected(Expected::Absent));
        }
        Placement::Roots(_) | Placement::Anywhere => {}
    }
    let placed = world
        .get_resource::<GlobalTransforms>()
        .and_then(|globals| globals.get(node))
        .and_then(|global| global.matrix());
    if let Some(matrix) = placed {
        kind.validate_placed(value, Placed { world: matrix }, "$.data")?;
    }
    kind.read(world, node, value)
        .map_err(|error| error.within("$.data"))
}
