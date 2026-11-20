use std::collections::BTreeMap;
use std::sync::LazyLock;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_registry::{CORE_ROW, MembersOf};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    Asset, Camera, Clear, ComponentKind, Drawn, DrawnTable, Error, ErrorCode, InstanceValues,
    Layers, List, Material, MeshFile, Mover, Node, PassMask, Rect, Rule, SceneComponents, SceneCx,
    SceneEntity, SceneForm, ScenePasses, Setting, Target, Texture, View,
};

static CORE: LazyLock<[ComponentKind; 5]> = LazyLock::new(|| {
    [
        ComponentKind::of::<Node>("node")
            .schema(crate::schema_of::<Node>)
            .into(),
        ComponentKind::of::<Mover>("mover")
            .schema(crate::schema_of::<Mover>)
            .into(),
        ComponentKind::of::<Camera>("camera")
            .check(camera_checked)
            .schema(crate::schema_of::<Camera>)
            .into(),
        ComponentKind::form::<ViewForm>("view")
            .check(view_checked)
            .prepare(view_camera)
            .schema(crate::schema_of::<ViewForm>)
            .into(),
        ComponentKind::form::<DrawnForm>("drawn")
            .check(drawn_checked)
            .schema(crate::schema_of::<DrawnForm>)
            .into(),
    ]
});

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct DrawnForm {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<Asset<MeshFile>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<Asset<Material>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub passes: BTreeMap<String, bool>,
    #[serde(default)]
    pub layers: Layers,
    #[serde(default)]
    pub order: i16,
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
#[derive(schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ViewForm {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<Asset<Texture>>,
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
            clear_color: view.clear_color,
            order: view.order,
            found: None,
        })
    }
}

fn camera_checked(camera: &Camera, path: &str) -> Result<(), Error> {
    camera.check().map_err(|error| error.within(path))
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
