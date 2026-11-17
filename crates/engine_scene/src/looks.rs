use std::collections::BTreeSet;

use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Res, SystemParamItem};
use bevy_ecs::world::World;
use engine_core::{Error, Operation, SceneContext};
use engine_registry::Registry;
use engine_settings::{ErrorCode, Expected};
use serde::{Deserialize, Serialize};

use crate::validation::identifier;
use crate::{SceneDocument, SceneEnvironment};

pub const SCENE_LOOK: &str = "scene";

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
pub struct SceneLooks {
    looks: Vec<SceneLook>,
    environment: Option<SceneEnvironment>,
}

impl SceneLooks {
    pub fn of(document: &SceneDocument) -> Self {
        Self {
            looks: document.looks.clone(),
            environment: document.environment.clone(),
        }
    }

    pub fn looks(&self) -> &[SceneLook] {
        &self.looks
    }

    pub fn look_names(&self) -> impl Iterator<Item = &str> {
        std::iter::once(SCENE_LOOK).chain(self.looks.iter().map(|look| look.name.as_str()))
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

fn look_named(looks: Option<&SceneLooks>, name: &str, path: &str) -> Result<Option<Look>, Error> {
    let (environment, at) = match looks {
        Some(looks) => looks.environment(name, path)?,
        None if name == SCENE_LOOK => return Ok(None),
        None => SceneLooks::default().environment(name, path)?,
    };
    let looked = looks.map(SceneLooks::look_keys).unwrap_or_default();
    Ok(Some(Look {
        environment,
        at,
        looked,
    }))
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneLookControl {
    pub name: String,
}

#[cfg(not(feature = "schema"))]
impl schemars::JsonSchema for SceneLookControl {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SceneLookControl".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
        })
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SceneLookReply {
    pub temporary: bool,
    pub look: String,
}

impl Operation for SceneLookControl {
    const NAME: &'static str = "scene_look";
    type Params = (
        Option<Res<'static, SceneLooks>>,
        Option<Res<'static, Registry>>,
        Option<Res<'static, SceneContext>>,
        Commands<'static, 'static>,
    );
    type Request = Self;
    type Reply = SceneLookReply;

    fn run(
        (looks, registry, context, mut commands): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<SceneLookReply, Error> {
        if let Some(look) = look_named(looks.as_deref(), &request.name, "$.name")? {
            let context = context.as_deref().cloned();
            crate::loading::within(context.as_ref(), || {
                look.environment.check_in(registry.as_deref(), &look.at)
            })?;
            commands.queue(move |world: &mut World| {
                let _ = crate::loading::within(context.as_ref(), || look.apply(world));
            });
        }
        Ok(SceneLookReply {
            temporary: true,
            look: request.name,
        })
    }
}

impl SceneDocument {
    pub(crate) fn validate_looks(&self) -> Result<(), Error> {
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
