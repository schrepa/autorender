//! A small authored game demonstrating the public engine SDK.
#[cfg(target_arch = "wasm32")]
mod browser;
mod gameplay;

use engine::prelude::*;
use engine::scene::SceneError;
pub use gameplay::{GameProgress, PlaygroundPlugin};

/// Retains explicit ownership of the active scene across reloads.
pub struct Playground {
    instance: Option<SceneInstance>,
    player_entity: Option<Entity>,
}

impl Playground {
    /// Register gameplay once, then load an authored world at the frame boundary.
    pub fn new(engine: &mut Engine, source: &str) -> Result<Self, SceneError> {
        let mut game = Self {
            instance: None,
            player_entity: None,
        };
        game.reload(engine, source)?;
        engine.app.add_plugins(PlaygroundPlugin);
        Ok(game)
    }

    /// A failed parse, validation, or asset allocation preserves the current game.
    pub fn reload(&mut self, engine: &mut Engine, source: &str) -> Result<(), SceneError> {
        let document = SceneDocument::from_json(source)?;
        let expanded = document.expanded_entities()?;
        for object in &expanded {
            if object
                .tags
                .iter()
                .any(|tag| ["player", "pickup"].contains(&tag.as_str()))
                && object.parent.is_some()
            {
                return Err(SceneError {
                    code: "gameplay_transform",
                    path: "$.entities".into(),
                    message: format!(
                        "expanded entity {}: playground player and pickup motion systems require root transforms",
                        object.id
                    ),
                });
            }
            if object
                .tags
                .iter()
                .any(|tag| ["player", "solid", "pickup"].contains(&tag.as_str()))
                && object.collider.is_none()
            {
                return Err(SceneError {
                    code: "gameplay_collider",
                    path: "$.entities".into(),
                    message: format!(
                        "expanded entity {}: player, solid and pickup tags require an authored v4 collider",
                        object.id
                    ),
                });
            }
        }
        let players = expanded
            .iter()
            .filter(|entity| entity.tags.iter().any(|tag| tag == "player"))
            .count();
        if players != 1 {
            return Err(SceneError {
                code: "player_count",
                path: "$.entities".into(),
                message: "the playground requires exactly one entity tagged player".into(),
            });
        }
        let instance = document.instantiate(engine.app.world_mut())?;
        let player_id = &expanded
            .iter()
            .find(|entity| entity.tags.iter().any(|tag| tag == "player"))
            .expect("one player validated")
            .id;
        self.player_entity = instance.entity(player_id);
        if let Some(previous) = self.instance.take() {
            previous.despawn(engine.app.world_mut());
        }
        gameplay::attach(engine.app.world_mut(), &instance);
        self.instance = Some(instance);
        engine.reset_clock();
        Ok(())
    }

    pub fn revision(&self) -> u64 {
        self.instance.as_ref().map_or(0, SceneInstance::revision)
    }
    pub fn scene_id(&self) -> &str {
        self.instance.as_ref().map_or("", SceneInstance::id)
    }
    pub fn player(&self) -> Option<Entity> {
        self.player_entity
    }

    /// Active scene ownership for optional development tools.
    pub fn scene_instance(&self) -> Option<&SceneInstance> {
        self.instance.as_ref()
    }

    pub fn reset_player(&self, engine: &mut Engine) {
        if let Some(entity) = self.player() {
            gameplay::reset_player(engine.app.world_mut(), entity);
        }
        engine.reset_clock();
    }

    pub fn unload(mut self, engine: &mut Engine) {
        if let Some(instance) = self.instance.take() {
            instance.despawn(engine.app.world_mut());
        }
    }
}

#[cfg(test)]
mod tests;
