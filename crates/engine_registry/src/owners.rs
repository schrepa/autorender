use std::any::TypeId;

use bevy_ecs::component::ComponentId;
use bevy_ecs::query::{ComponentAccessKind, FilteredAccessSet};
use bevy_ecs::resource::IsResource;
use bevy_ecs::world::World;
use serde::Serialize;

use crate::BrickId;
use crate::access::core_type;
use crate::record::{BrickRecord, crate_segment};
use crate::state::Reset;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueKind {
    Resource,
    Component,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Shared {
    pub value: String,
    pub kind: ValueKind,
    pub bricks: Vec<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Used {
    pub(crate) brick: BrickId,
    pub(crate) id: ComponentId,
    pub(crate) write: bool,
}

#[derive(Default)]
pub(crate) struct Ownership {
    pub(crate) shared: Vec<Shared>,
    pub(crate) private: Vec<Vec<(ComponentId, Reset)>>,
    pub(crate) needs: Vec<Vec<BrickId>>,
}

type ResetFn = fn(&mut World);

pub(crate) struct Values<'a> {
    pub(crate) bricks: &'a [BrickRecord],
    pub(crate) catalogued: &'a dyn Fn(TypeId) -> Option<Option<ResetFn>>,
    pub(crate) dropped: &'a [Vec<TypeId>],
}

pub(crate) fn ownership(world: &World, values: &Values, uses: &[Used]) -> Ownership {
    let count = values.bricks.len();
    let mut uses = uses.to_vec();
    uses.sort_unstable_by_key(|used| (used.id, used.brick, used.write));
    uses.dedup();
    let is_resource = world.component_id::<IsResource>();
    let mut held = Ownership {
        shared: Vec::new(),
        private: vec![Vec::new(); count],
        needs: vec![Vec::new(); count],
    };
    for group in uses.chunk_by(|a, b| a.id == b.id) {
        let id = group[0].id;
        let Some(info) = world.components().get_info(id) else {
            continue;
        };
        let name = info.name().to_string();
        let resource = is_resource.is_some_and(|marker| {
            info.required_components()
                .iter_ids()
                .any(|required| required == marker)
        });
        let mut writers: Vec<BrickId> = group
            .iter()
            .filter(|used| used.write)
            .map(|used| used.brick)
            .collect();
        writers.dedup();
        let catalogued = info
            .type_id()
            .and_then(|type_id| (values.catalogued)(type_id));
        match (catalogued, writers.as_slice()) {
            (Some(Some(reset)), &[only]) if !resource => {
                held.private[only.index()].push((id, Reset::Typed(reset)));
            }
            (Some(_), _) => {}
            (None, &[only]) if !resource => {
                let own = values
                    .bricks
                    .get(only.index())
                    .and_then(BrickRecord::owner)
                    .is_some_and(|owner| crate_segment(&name) == owner);
                if own && !core_type(&name) && !required(world, id) {
                    held.private[only.index()].push((id, Reset::Remove(id)));
                }
            }
            (None, [_, _, ..]) if resource || !core_type(&name) => {
                held.shared.push(Shared {
                    value: name.clone(),
                    kind: if resource {
                        ValueKind::Resource
                    } else {
                        ValueKind::Component
                    },
                    bricks: writers
                        .iter()
                        .filter_map(|brick| values.bricks.get(brick.index()))
                        .map(|record| record.package)
                        .collect(),
                });
            }
            (None, _) => {}
        }
        let Some(type_id) = info.type_id() else {
            continue;
        };
        let Some(holder) = values
            .dropped
            .iter()
            .position(|dropped| dropped.contains(&type_id))
        else {
            continue;
        };
        for used in group {
            let needs = &mut held.needs[used.brick.index()];
            let need = BrickId(holder as u32);
            if used.brick != need && !needs.contains(&need) {
                needs.push(need);
            }
        }
    }
    held
}

pub(crate) fn note(uses: &mut Vec<Used>, brick: BrickId, access: &FilteredAccessSet) {
    let Ok(accesses) = access.combined_access().try_iter_access() else {
        return;
    };
    for used in accesses {
        let write = match used {
            ComponentAccessKind::Archetypal(_) => continue,
            ComponentAccessKind::Shared(_) => false,
            ComponentAccessKind::Exclusive(_) => true,
        };
        uses.push(Used {
            brick,
            id: *used.index(),
            write,
        });
    }
}

fn required(world: &World, id: ComponentId) -> bool {
    world.components().iter_registered().any(|info| {
        info.id() != id
            && info
                .required_components()
                .iter_ids()
                .any(|required| required == id)
    })
}
