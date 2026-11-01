use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::math::max;
use engine_core::{Authored, Mesh3d, Reach, SceneContext, Spawner, Transform};
use glam::{Mat4, Vec3};
use serde_json::Value;

use crate::WorldStreamingBrick;
use crate::scene::{ExpandedList, ExpandedObject, StreamedScene, kind};

struct Prepared {
    transform: Transform,
    mesh: Option<Mesh3d>,
    authored: Authored,
    components: Vec<(String, Value)>,
}

pub(crate) struct TreeItem {
    pub(crate) id: Arc<str>,
    pub(crate) entities: Vec<usize>,
    pub(crate) position: Vec3,
    pub(crate) reach: f32,
}

#[derive(Default)]
pub(crate) struct Tree {
    entities: Vec<Prepared>,
    parents: Vec<Option<usize>>,
    pub(crate) items: Vec<TreeItem>,
}

#[derive(Default)]
pub(crate) struct Content {
    pub(crate) streamed: Tree,
    pub(crate) kept: Tree,
}

pub(crate) fn content(scene: &StreamedScene, context: &SceneContext) -> Content {
    let Ok(expanded) = scene.expand(context) else {
        return Content::default();
    };
    Content {
        streamed: tree(expanded.streamed, context),
        kept: tree(expanded.kept, context),
    }
}

fn tree(list: ExpandedList, context: &SceneContext) -> Tree {
    let ExpandedList {
        entities,
        transforms,
    } = list;
    let indices: BTreeMap<&str, usize> = entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (entity.source.id.as_str(), index))
        .collect();
    let parents: Vec<Option<usize>> = entities
        .iter()
        .map(|entity| {
            entity
                .source
                .parent
                .as_ref()
                .map(|parent| indices[parent.as_str()])
        })
        .collect();
    let roots = item_roots(&parents);
    let mut worlds: Vec<Option<Mat4>> = vec![None; entities.len()];
    let mut chain = Vec::new();
    for start in 0..worlds.len() {
        let mut cursor = Some(start);
        while let Some(index) = cursor.filter(|&index| worlds[index].is_none()) {
            chain.push(index);
            cursor = parents[index];
        }
        while let Some(index) = chain.pop() {
            let local = transforms[index];
            let local = Mat4::from_scale_rotation_translation(
                local.scale,
                local.rotation,
                local.translation,
            );
            worlds[index] = Some(parents[index].map_or(local, |parent| {
                worlds[parent].expect("parent_composed_first") * local
            }));
        }
    }
    let sizes = |id: &str| {
        context
            .bounds
            .get(id)
            .map_or(0.0, |bounds| (bounds.max() - bounds.min()).length())
    };
    let mut items: Vec<TreeItem> = Vec::new();
    let mut item_of = HashMap::default();
    for (index, &root) in roots.iter().enumerate() {
        let item = *item_of.entry(root).or_insert_with(|| {
            items.push(TreeItem {
                id: Arc::from(format!("{}/{}", context.id, entities[root].source.id)),
                entities: Vec::new(),
                position: worlds[root]
                    .expect("world_matrix_composed")
                    .w_axis
                    .truncate(),
                reach: 0.0,
            });
            items.len() - 1
        });
        items[item].entities.push(index);
        let position = items[item].position;
        let world = worlds[index].expect("world_matrix_composed");
        let from = world.w_axis.truncate().distance(position);
        let standing = Reach {
            world,
            position,
            from,
            mesh_sizes: &sizes,
        };
        let source = &entities[index].source;
        let mut reach = from;
        let mut extend = |value: f32| reach = max(reach, value);
        if let Some(bounds) = source.mesh.as_ref().and_then(|id| context.bounds.get(id)) {
            extend(standing.of_box(bounds.min(), bounds.max()));
        }
        for (name, value) in source.authored() {
            if let Some(kind) = kind(&context.kinds, name) {
                kind.extend_reach(value, &standing, &mut extend);
            }
        }
        if reach.is_finite() {
            items[item].reach = max(items[item].reach, reach);
        }
    }
    for item in &mut items {
        item.entities
            .sort_by_key(|&index| (parents[index].is_some(), index));
        if let Some(reach) = item
            .entities
            .iter()
            .find_map(|&index| entities[index].reach)
        {
            item.reach = reach;
        }
    }
    Tree {
        entities: entities
            .into_iter()
            .zip(transforms)
            .map(|(entity, transform)| prepared(entity, transform, context))
            .collect(),
        parents,
        items,
    }
}

fn prepared(entity: ExpandedObject, transform: Transform, context: &SceneContext) -> Prepared {
    let source = entity.source;
    let mesh = match (&source.mesh, &source.material) {
        (Some(mesh), Some(material)) => context
            .assets
            .meshes
            .get(mesh)
            .zip(context.assets.materials.get(material))
            .map(|(&mesh, &material)| Mesh3d { mesh, material }),
        (Some(_), None) | (None, Some(_)) | (None, None) => None,
    };
    let components = source
        .authored()
        .filter(|&(name, _)| kind(&context.kinds, name).is_some())
        .map(|(name, value)| (name.to_owned(), value.clone()))
        .collect();
    Prepared {
        transform,
        mesh,
        authored: Authored {
            id: source.id,
            name: source.name,
            tags: source.tags,
            parent: source.parent,
            prefab_id: entity.prefab_id,
            prefab_instance: entity.prefab_instance,
            prefab_local_id: entity.prefab_local_id,
        },
        components,
    }
}

fn item_roots(parents: &[Option<usize>]) -> Vec<usize> {
    (0..parents.len())
        .map(|mut index| {
            while let Some(next) = parents[index] {
                index = next;
            }
            index
        })
        .collect()
}

impl Tree {
    pub(crate) fn entities_of(&self, item: usize) -> u32 {
        self.items[item].entities.len() as u32
    }

    pub(crate) fn spawn(
        &self,
        item: usize,
        spawner: &mut Spawner<WorldStreamingBrick>,
        spawned: &mut Vec<Entity>,
    ) {
        let item = &self.items[item];
        let mine: Vec<Entity> = item
            .entities
            .iter()
            .map(|&index| {
                let prepared = &self.entities[index];
                let entity = spawner.spawn(prepared.transform);
                spawner.author(entity, prepared.authored.clone());
                if let Some(mesh) = prepared.mesh {
                    spawner.insert(entity, mesh);
                }
                for (key, value) in &prepared.components {
                    spawner.insert_scene(entity, key, value.clone());
                }
                entity
            })
            .collect();
        let entity_of = |index: usize| {
            mine[item
                .entities
                .iter()
                .position(|&own| own == index)
                .expect("reference_within_item")]
        };
        let parents: Vec<_> = item
            .entities
            .iter()
            .zip(&mine)
            .filter_map(|(&index, &entity)| {
                self.parents[index].map(|parent| (entity, Some(entity_of(parent))))
            })
            .collect();
        spawner.set_parents(parents);
        spawned.extend(mine);
    }
}
