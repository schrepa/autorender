use std::collections::BTreeSet;

use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Res, ResMut, SystemParamItem};
use bevy_ecs::world::World;
use engine_core::{Camera, Error, Operation};
use engine_registry::Registry;
use engine_settings::{ErrorCode, Expected};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::validation::identifier;
use crate::{SceneCamera, SceneDocument, SceneEnvironment};

pub const SCENE_LOOK: &str = "scene";

fn view_position() -> [f32; 3] {
    SceneCamera::default().position
}

fn view_target() -> [f32; 3] {
    SceneCamera::default().target
}

fn view_fov_degrees() -> f32 {
    SceneCamera::default().fov_degrees
}

fn view_near() -> f32 {
    SceneCamera::default().near
}

fn view_far() -> f32 {
    SceneCamera::default().far
}

engine_settings::fields! {
    extra_rules;
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct SceneView {
        pub name: String = String::new() => id,
        #[serde(default = "view_position")]
        pub position: [f32; 3] = view_position() => same(SceneCamera::rule_of("position")),
        #[serde(default = "view_target")]
        pub target: [f32; 3] = view_target() => same(SceneCamera::rule_of("target")),
        #[serde(default = "view_fov_degrees")]
        pub fov_degrees: f32 = view_fov_degrees() => same(SceneCamera::rule_of("fov_degrees")),
        #[serde(default = "view_near")]
        pub near: f32 = view_near() => same(SceneCamera::rule_of("near")),
        #[serde(default = "view_far")]
        pub far: f32 = view_far() => same(SceneCamera::rule_of("far")),
    }
}

impl engine_settings::ExtraRules for SceneView {
    fn check_extra(&self) -> Result<(), engine_settings::InvalidSetting> {
        engine_settings::ExtraRules::check_extra(&self.scene_camera())
    }
}

impl SceneView {
    pub fn camera(&self) -> Camera {
        self.scene_camera().camera()
    }

    fn scene_camera(&self) -> SceneCamera {
        SceneCamera {
            position: self.position,
            target: self.target,
            fov_degrees: self.fov_degrees,
            near: self.near,
            far: self.far,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneLook {
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::fields::id()))
    )]
    pub name: String,
    pub environment: SceneEnvironment,
}

#[derive(Resource, Clone, Debug, Default)]
pub struct SceneViews {
    views: Vec<SceneView>,
    looks: Vec<SceneLook>,
    environment: Option<SceneEnvironment>,
}

impl SceneViews {
    pub fn of(document: &SceneDocument) -> Self {
        Self {
            views: document.views.clone(),
            looks: document.looks.clone(),
            environment: document.environment.clone(),
        }
    }

    pub fn views(&self) -> &[SceneView] {
        &self.views
    }

    pub fn looks(&self) -> &[SceneLook] {
        &self.looks
    }

    pub fn view_names(&self) -> impl Iterator<Item = &str> {
        self.views.iter().map(|view| view.name.as_str())
    }

    pub fn look_names(&self) -> impl Iterator<Item = &str> {
        std::iter::once(SCENE_LOOK).chain(self.looks.iter().map(|look| look.name.as_str()))
    }

    pub fn camera(&self, name: &str, path: &str) -> Result<Camera, Error> {
        self.views
            .iter()
            .find(|view| view.name == name)
            .map(SceneView::camera)
            .ok_or_else(|| {
                Error::new(ErrorCode::UnknownVariant, path)
                    .got(name)
                    .variants(self.view_names())
            })
    }

    pub fn look_keys(&self) -> BTreeSet<String> {
        self.looks
            .iter()
            .flat_map(|look| look.environment.sections.keys().cloned())
            .collect()
    }

    pub fn environment(&self, name: &str, path: &str) -> Result<(SceneEnvironment, String), Error> {
        let mut environment = self.environment.clone().unwrap_or_default();
        if name == SCENE_LOOK {
            return Ok((environment, "$.environment".to_owned()));
        }
        let Some((index, look)) = self
            .looks
            .iter()
            .enumerate()
            .find(|(_, look)| look.name == name)
        else {
            return Err(Error::new(ErrorCode::UnknownVariant, path)
                .got(name)
                .variants(self.look_names()));
        };
        environment.sections.extend(
            look.environment
                .sections
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        Ok((environment, format!("$.looks[{index}].environment")))
    }
}

pub fn show_view(world: &mut World, name: &str, path: &str) -> Result<Camera, Error> {
    let camera = view_camera(world.get_resource::<SceneViews>(), name, path)?;
    world.insert_resource(camera);
    Ok(camera)
}

pub fn show_look(world: &mut World, name: &str, path: &str) -> Result<(), Error> {
    let Some(look) = look_named(world.get_resource::<SceneViews>(), name, path)? else {
        return Ok(());
    };
    look.environment.check(world, &look.at)?;
    look.apply(world)
}

fn view_camera(views: Option<&SceneViews>, name: &str, path: &str) -> Result<Camera, Error> {
    match views {
        Some(views) => views.camera(name, path),
        None => SceneViews::default().camera(name, path),
    }
}

struct Look {
    environment: SceneEnvironment,
    at: String,
    looked: BTreeSet<String>,
}

impl Look {
    fn apply(&self, world: &mut World) -> Result<(), Error> {
        self.environment
            .apply_where(world, &self.at, |key| self.looked.contains(key))
    }
}

fn look_named(views: Option<&SceneViews>, name: &str, path: &str) -> Result<Option<Look>, Error> {
    let (environment, at) = match views {
        Some(views) => views.environment(name, path)?,
        None if name == SCENE_LOOK => return Ok(None),
        None => SceneViews::default().environment(name, path)?,
    };
    let looked = views.map(SceneViews::look_keys).unwrap_or_default();
    Ok(Some(Look {
        environment,
        at,
        looked,
    }))
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneViewControl {
    pub name: String,
}

impl Operation for SceneViewControl {
    const NAME: &'static str = "scene_view";
    type Params = (Option<Res<'static, SceneViews>>, ResMut<'static, Camera>);
    type Request = Self;

    fn run(
        (views, mut camera): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        *camera = view_camera(views.as_deref(), &request.name, "$.name")?;
        Ok(json!({ "temporary": true, "view": request.name }))
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneLookControl {
    pub name: String,
}

impl Operation for SceneLookControl {
    const NAME: &'static str = "scene_look";
    type Params = (
        Option<Res<'static, SceneViews>>,
        Option<Res<'static, Registry>>,
        Commands<'static, 'static>,
    );
    type Request = Self;

    fn run(
        (views, registry, mut commands): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        if let Some(look) = look_named(views.as_deref(), &request.name, "$.name")? {
            look.environment.check_in(registry.as_deref(), &look.at)?;
            commands.queue(move |world: &mut World| {
                let _ = look.apply(world);
            });
        }
        Ok(json!({ "temporary": true, "look": request.name }))
    }
}

impl SceneDocument {
    pub(crate) fn validate_views(&self) -> Result<(), Error> {
        let mut names = BTreeSet::new();
        for (index, view) in self.views.iter().enumerate() {
            let path = format!("$.views[{index}]");
            unique_name(&view.name, &format!("{path}.name"), &mut names)?;
            view.validate()
                .map_err(|invalid| invalid.error(ErrorCode::InvalidCamera, &path))?;
        }
        let mut names = BTreeSet::from([SCENE_LOOK]);
        for (index, look) in self.looks.iter().enumerate() {
            let path = format!("$.looks[{index}]");
            unique_name(&look.name, &format!("{path}.name"), &mut names)?;
        }
        Ok(())
    }
}

fn unique_name<'a>(name: &'a str, path: &str, names: &mut BTreeSet<&'a str>) -> Result<(), Error> {
    identifier(name, path)?;
    if !names.insert(name) {
        return Err(Error::new(ErrorCode::DuplicateId, path)
            .expected(Expected::Unique)
            .got(name));
    }
    Ok(())
}
