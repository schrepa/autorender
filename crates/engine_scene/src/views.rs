use std::collections::BTreeSet;

use engine_core::bevy_ecs::{
    self,
    prelude::{Resource, World},
};
use engine_core::{Camera, Error, Expected};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};

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

engine_settings::settings! {
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
        engine_settings::ExtraRules::check_extra(&self.camera())
    }
}

impl SceneView {
    pub fn camera(&self) -> Camera {
        SceneCamera {
            position: self.position,
            target: self.target,
            fov_degrees: self.fov_degrees,
            near: self.near,
            far: self.far,
        }
        .camera()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneLook {
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::rule::id()))
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
    let camera = match world.get_resource::<SceneViews>() {
        Some(views) => views.camera(name, path)?,
        None => SceneViews::default().camera(name, path)?,
    };
    world.insert_resource(camera);
    Ok(camera)
}

pub fn show_look(world: &mut World, name: &str, path: &str) -> Result<(), Error> {
    let (environment, at) = match world.get_resource::<SceneViews>() {
        Some(views) => views.environment(name, path)?,
        None if name == SCENE_LOOK => return Ok(()),
        None => SceneViews::default().environment(name, path)?,
    };
    let looked: BTreeSet<String> = world
        .get_resource::<SceneViews>()
        .map(SceneViews::look_keys)
        .unwrap_or_default();
    environment.check(world, &at)?;
    environment.apply_where(world, &at, |key| looked.contains(key))
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
            look.environment.validate(&format!("{path}.environment"))?;
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
