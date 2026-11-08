use std::collections::BTreeMap;
use std::sync::Arc;

use engine_core::collections::HashMap;
use engine_core::math::max;
use engine_core::{
    Aabb, AssetLibrary, Authored, MeshHandle, SceneContext, SubScene, SubSceneEntity, Transform,
};
use engine_places_world::streaming::{StreamedAt, StreamedKey, StreamedKeys, Streams};
use glam::{Mat4, Vec3};
use serde_json::Value;

use crate::scene::{ExpandedList, ExpandedObject, StreamedScene, kind};

const DRAWN: &str = "drawn";

pub(crate) struct TreeItem {
    pub(crate) id: Arc<str>,
    pub(crate) entities: Vec<usize>,
    pub(crate) position: Vec3,
    pub(crate) reach: f32,
}

#[derive(Default)]
pub(crate) struct Tree {
    entities: Vec<SubSceneEntity>,
    parents: Vec<Option<usize>>,
    pub(crate) items: Vec<TreeItem>,
}

#[derive(Default)]
pub(crate) struct Content {
    pub(crate) streamed: Tree,
    pub(crate) kept: Tree,
}

pub(crate) fn content(
    scene: &StreamedScene,
    context: &SceneContext,
    assets: &AssetLibrary,
) -> Content {
    let Ok(expanded) = scene.expand(context) else {
        return Content::default();
    };
    let keys: Vec<StreamedKey> = context.members::<StreamedKeys>().copied().collect();
    let bounds = |mesh: MeshHandle| assets.mesh(mesh).map(|mesh| mesh.bounds());
    let sizes = |mesh: MeshHandle| bounds(mesh).map_or(0.0, |bounds| size(&bounds));
    let reaching = Reaching {
        keys: &keys,
        bounds: &bounds,
        sizes: &sizes,
    };
    Content {
        streamed: tree(expanded.streamed, context, &reaching),
        kept: tree(expanded.kept, context, &reaching),
    }
}

struct Reaching<'a> {
    keys: &'a [StreamedKey],
    bounds: &'a dyn Fn(MeshHandle) -> Option<Aabb>,
    sizes: &'a dyn Fn(MeshHandle) -> f32,
}

fn size(bounds: &Aabb) -> f32 {
    (bounds.max() - bounds.min()).length()
}

fn of_box(at: &StreamedAt<'_>, bounds: &Aabb) -> f32 {
    let (low, high) = (bounds.min(), bounds.max());
    (0..8)
        .map(|corner| {
            let pick = |bit: u32, low: f32, high: f32| {
                if corner & (1 << bit) == 0 { low } else { high }
            };
            let point = Vec3::new(
                pick(0, low.x, high.x),
                pick(1, low.y, high.y),
                pick(2, low.z, high.z),
            );
            at.world.transform_point3(point).distance(at.position)
        })
        .fold(0.0, max)
}

fn tree(list: ExpandedList, context: &SceneContext, reaching: &Reaching<'_>) -> Tree {
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
        let standing = StreamedAt {
            world,
            position,
            from,
            mesh_sizes: reaching.sizes,
        };
        let source = &entities[index].source;
        let mut reach = from;
        let mut extend = |value: f32| reach = max(reach, value);
        let mesh = source
            .components
            .get(DRAWN)
            .and_then(|drawn| drawn.get("mesh"))
            .and_then(Value::as_str)
            .and_then(|id| context.assets.meshes.get(id))
            .and_then(|&mesh| (reaching.bounds)(mesh));
        if let Some(bounds) = mesh {
            extend(of_box(&standing, &bounds));
        }
        for (name, value) in source.authored() {
            for key in reaching.keys.iter().filter(|key| key.key == name) {
                match key.streams {
                    Streams::Reaching(of) => extend(of(value, &standing)),
                    Streams::Never(_) => {}
                }
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

fn prepared(entity: ExpandedObject, transform: Transform, context: &SceneContext) -> SubSceneEntity {
    let source = entity.source;
    let components = source
        .authored()
        .filter(|&(name, _)| kind(&context.kinds, name).is_some())
        .map(|(name, value)| (name.to_owned(), value.clone()))
        .collect();
    SubSceneEntity {
        transform,
        authored: Authored {
            id: source.id,
            name: source.name,
            tags: source.tags,
            parent: source.parent,
            prefab_id: entity.prefab_id,
            prefab_instance: entity.prefab_instance,
            prefab_local_id: entity.prefab_local_id,
        },
        parent: None,
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

    pub(crate) fn sub_scene(&self, item: usize) -> SubScene {
        let item = &self.items[item];
        let local = |index: usize| {
            item.entities
                .iter()
                .position(|&own| own == index)
                .expect("reference_within_item")
        };
        SubScene {
            entities: item
                .entities
                .iter()
                .map(|&index| SubSceneEntity {
                    parent: self.parents[index].map(local),
                    ..self.entities[index].clone()
                })
                .collect(),
        }
    }
}
