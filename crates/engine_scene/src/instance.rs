use engine_settings::{ErrorCode, Expected};
use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::{World, WorldId};
use engine_core::{
    Assets, DrawnTable, Error, OperationScene, SceneContext, propagate_transforms, set_parents,
};

use crate::validation::{Landing, Pick};
use crate::{SceneDocument, SceneEnvironment, SceneKinds};

#[must_use = "retain the instance and explicitly despawn it when the scene is disposed"]
pub struct SceneInstance {
    id: String,
    revision: u64,
    world_id: WorldId,
    entities: Arc<BTreeMap<String, Entity>>,
    environment: SceneEnvironment,
    groups: Vec<Vec<Landing>>,
    landed: usize,
}

impl SceneInstance {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn entity(&self, authored_id: &str) -> Option<Entity> {
        self.entities.get(authored_id).copied()
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn landed(&self) -> bool {
        self.landed == self.groups.len()
    }

    pub fn land(&mut self, world: &mut World) -> Result<bool, Error> {
        let first = self.landed == 0;
        while !self.landed()
            && world
                .get_resource::<Assets>()
                .is_some_and(|assets| assets.arrived(self.landed))
        {
            let group = self.landed;
            let landing = std::mem::take(&mut self.groups[group]);
            self.landed += 1;
            self.land_group(world, group, landing)?;
        }
        Ok(first && self.landed > 0)
    }

    fn land_group(
        &self,
        world: &mut World,
        group: usize,
        landing: Vec<Landing>,
    ) -> Result<(), Error> {
        if group == 0 {
            world.insert_resource(SceneContext {
                entities: Arc::clone(&self.entities),
            });
            world.insert_resource(OperationScene::new(
                self.id.clone(),
                Arc::clone(&self.entities),
            ));
        }
        let mut parents = Vec::new();
        for landed in landing {
            match world.get_entity_mut(landed.entity) {
                Ok(mut target) => {
                    target.insert((landed.transform, landed.scene));
                }
                Err(_) => {
                    return Err(Error::new(ErrorCode::UnknownEntity, "$.entities")
                        .got(&landed.scene.authored_id));
                }
            }
            for component in landed.components {
                component.insert(world, landed.entity)?;
            }
            if let Some(mut table) = world.get_resource_mut::<DrawnTable>() {
                table.hold(landed.entity);
            }
            if let Some(parent) = landed.parent {
                parents.push((landed.entity, Some(parent)));
            }
        }
        set_parents(world, &parents).map_err(|error| {
            error
                .coded(ErrorCode::HierarchyInstallFailed)
                .within("$.entities")
        })?;
        propagate_transforms(world);
        if group == 0 {
            crate::named(world, |world| {
                self.environment.apply_all(world, "$.environment")
            })?;
        }
        if let Some(mut assets) = world.get_resource_mut::<Assets>() {
            assets.landed(group);
        }
        Ok(())
    }

    pub fn despawn(self, world: &mut World) {
        assert_eq!(self.world_id, world.id(), "scene_world_matches");
        let installed: Option<Vec<Entity>> = world
            .get_resource::<OperationScene>()
            .filter(|scene| Arc::ptr_eq(&scene.entities, &self.entities))
            .map(|scene| scene.installed.values().copied().collect());
        if installed.is_some() {
            world.remove_resource::<OperationScene>();
        }
        if world
            .get_resource::<SceneContext>()
            .is_some_and(|context| Arc::ptr_eq(&context.entities, &self.entities))
        {
            world.remove_resource::<SceneContext>();
        }
        for entity in installed.into_iter().flatten() {
            let _ = world.try_despawn(entity);
        }
        for &entity in self.entities.values() {
            let _ = world.try_despawn(entity);
        }
    }
}

impl SceneDocument {
    pub fn instantiate(
        &self,
        world: &mut World,
        variant: Option<&str>,
    ) -> Result<SceneInstance, Error> {
        let kinds = SceneKinds::of(world);
        if let Some(first) = self.unlisted(world, &kinds).into_iter().next() {
            return Err(first);
        }
        let expanded = self.expand()?;
        let mut assets = world.get_resource::<Assets>().map(Assets::fresh).ok_or_else(|| {
            Error::new(ErrorCode::MissingResource, "$")
                .expected(Expected::Present)
                .got("Assets")
        })?;
        let entities: Arc<BTreeMap<String, Entity>> = Arc::new(
            expanded
                .entities
                .iter()
                .map(|object| (object.source.id.clone(), world.spawn_empty().id()))
                .collect(),
        );
        let prepared = self.prepare(&kinds, &mut assets, expanded, &entities, Pick::Variant(variant));
        let groups = match prepared {
            Ok(groups) => groups,
            Err(error) => {
                for &entity in entities.values() {
                    let _ = world.try_despawn(entity);
                }
                return Err(error);
            }
        };
        world.insert_resource(assets);
        Ok(SceneInstance {
            id: self.id.clone(),
            revision: self.revision,
            world_id: world.id(),
            entities,
            environment: self.environment.clone().unwrap_or_default(),
            groups,
            landed: 0,
        })
    }
}
