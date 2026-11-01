use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::math::max;
use engine_core::{AssetLibrary, Mesh3d, Spawner, Transform};
use engine_places::ground::Grounded;
use engine_places::streaming::StreamingPaths;
use glam::{Mat4, Vec3};
use serde_json::Value;

use crate::scene::{ExpandedObject, Placed, StreamedScene};
use crate::{WorldCells, WorldStreamingBrick};

struct Prepared {
    transform: Transform,
    mesh: Option<Mesh3d>,
    grounded: Option<Grounded>,
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

pub(crate) fn prepare_streamed(
    scene: Res<StreamedScene>,
    paths: Res<StreamingPaths>,
    assets: Res<AssetLibrary>,
    mut cells: ResMut<WorldCells>,
) {
    if !scene.is_changed() && !paths.is_changed() && !cells.waiting {
        return;
    }
    let placed: Vec<Placed> = scene
        .along
        .iter()
        .enumerate()
        .filter_map(|(index, along)| {
            let path = paths.paths.iter().find(|path| path.layer == along.layer)?;
            let spots = along
                .spots(&path.points, &format!("streamed.along[{index}]"))
                .ok()?;
            Some(Placed {
                along: index,
                origin: path.origin,
                spots,
            })
        })
        .collect();
    if !scene.is_changed() && !cells.waiting && placed == cells.placed {
        return;
    }
    cells.placed = placed;
    let Ok(expanded) = scene.expand(&cells.placed) else {
        cells.waiting = false;
        cells.incoming = Some(Arc::new(Content::default()));
        return;
    };
    let resident = |objects: &[ExpandedObject]| {
        objects.iter().all(|object| {
            object
                .source
                .mesh
                .is_none_or(|mesh| assets.mesh(mesh).is_some())
        })
    };
    if !resident(&expanded.streamed) || !resident(&expanded.kept) {
        cells.waiting = true;
        return;
    }
    cells.waiting = false;
    cells.incoming = Some(Arc::new(Content {
        streamed: tree(expanded.streamed, &assets),
        kept: tree(expanded.kept, &assets),
    }));
}

fn tree(entities: Vec<ExpandedObject>, assets: &AssetLibrary) -> Tree {
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
            let local = entities[index].transform;
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
                id: Arc::from(entities[root].source.id.as_str()),
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
        let mut reach = world.w_axis.truncate().distance(position);
        if let Some(mesh) = entities[index]
            .source
            .mesh
            .and_then(|mesh| assets.mesh(mesh))
        {
            let bounds = mesh.bounds();
            reach = max(reach, of_box(world, position, bounds.min(), bounds.max()));
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
            .map(|entity| Prepared {
                transform: entity.transform,
                mesh: entity
                    .source
                    .mesh
                    .zip(entity.source.material)
                    .map(|(mesh, material)| Mesh3d { mesh, material }),
                grounded: entity.grounded,
                components: entity.source.components.into_iter().collect(),
            })
            .collect(),
        parents,
        items,
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

fn of_box(world: Mat4, position: Vec3, min: Vec3, max_corner: Vec3) -> f32 {
    (0..8)
        .map(|corner| {
            let pick = |bit: u32, low: f32, high: f32| {
                if corner & (1 << bit) == 0 { low } else { high }
            };
            let point = Vec3::new(
                pick(0, min.x, max_corner.x),
                pick(1, min.y, max_corner.y),
                pick(2, min.z, max_corner.z),
            );
            world.transform_point3(point).distance(position)
        })
        .fold(0.0, max)
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
                if let Some(mesh) = prepared.mesh {
                    spawner.insert(entity, mesh);
                }
                if let Some(grounded) = prepared.grounded {
                    spawner.insert(entity, grounded);
                }
                for (key, value) in &prepared.components {
                    spawner.insert_authored(entity, key, value);
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
        for (&index, &entity) in item.entities.iter().zip(&mine) {
            if let Some(parent) = self.parents[index] {
                spawner.set_parent(entity, Some(entity_of(parent)));
            }
        }
        spawned.extend(mine);
    }
}
