use std::any::TypeId;

use bevy_ecs::query::{ComponentAccessKind, FilteredAccessSet};
use bevy_ecs::world::World;
use engine_math::collections::HashMap;
use engine_place::{Error, ErrorCode, PlaceKind};

use crate::BrickId;
use crate::record::{crate_segment, within};

const CORE_CRATES: &[&str] = &[
    "bevy_ecs",
    "engine_ecs",
    "engine_place",
    "engine_core",
    "engine_render",
    "engine_assets",
    "engine_settings",
    "engine_scene",
    "engine_profile",
    "engine_panic",
    "engine_registry",
    "engine_textures",
    "core",
    "alloc",
    "std",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Catalogued {
    Slot(Option<BrickId>),
    Shared,
}

#[derive(Clone, Debug, Default)]
pub struct AccessRule {
    packages: Vec<(&'static str, Option<&'static str>)>,
    places: HashMap<TypeId, Catalogued>,
}

impl AccessRule {
    pub(crate) fn new(packages: Vec<(&'static str, Option<&'static str>)>) -> Self {
        Self {
            packages,
            places: HashMap::default(),
        }
    }

    pub(crate) fn place(&mut self, type_id: TypeId, kind: PlaceKind, filler: Option<BrickId>) {
        let catalogued = match kind {
            PlaceKind::Slot => Catalogued::Slot(filler),
            PlaceKind::List | PlaceKind::Component => Catalogued::Shared,
        };
        self.places.insert(type_id, catalogued);
    }

    pub fn catalogued(&self, type_id: TypeId) -> bool {
        self.places.contains_key(&type_id)
    }

    pub fn filler(&self, slot: TypeId) -> Option<BrickId> {
        match self.places.get(&slot) {
            Some(Catalogued::Slot(filler)) => *filler,
            Some(Catalogued::Shared) | None => None,
        }
    }

    pub fn check_into(
        &self,
        brick: BrickId,
        system: &str,
        access: &FilteredAccessSet,
        world: &World,
        problems: &mut Vec<Error>,
    ) {
        let Some(&(package, owner)) = self.packages.get(brick.index()) else {
            return;
        };
        let Ok(accesses) = access.combined_access().try_iter_access() else {
            problems.push(foreign(package, system, "*"));
            return;
        };
        for used in accesses {
            let write = match used {
                ComponentAccessKind::Archetypal(_) => continue,
                ComponentAccessKind::Shared(_) => false,
                ComponentAccessKind::Exclusive(_) => true,
            };
            let Some(info) = world.components().get_info(*used.index()) else {
                continue;
            };
            let name = info.name();
            if let Some(place) = info.type_id().and_then(|type_id| self.places.get(&type_id)) {
                if let Catalogued::Slot(filler) = place
                    && write
                    && *filler != Some(brick)
                {
                    let path = format!("$.bricks.{package}.{system}.{}", &*name);
                    problems.push(Error::new(ErrorCode::PlaceNotFilled, path).got(package));
                }
                continue;
            }
            if owner.is_some_and(|owner| within(&name, owner))
                || CORE_CRATES.contains(&crate_segment(&name))
            {
                continue;
            }
            problems.push(foreign(package, system, &name));
        }
    }

    pub fn check_type(
        &self,
        brick: BrickId,
        type_id: TypeId,
        type_name: &str,
        filled_slot: bool,
        problems: &mut Vec<Error>,
    ) {
        let Some(&(package, owner)) = self.packages.get(brick.index()) else {
            return;
        };
        if owner.is_some_and(|owner| within(type_name, owner))
            || (filled_slot && self.filler(type_id) == Some(brick))
        {
            return;
        }
        problems.push(foreign(package, "registers", type_name));
    }
}

fn foreign(package: &str, system: &str, type_name: &str) -> Error {
    Error::new(
        ErrorCode::ForeignAccess,
        format!("$.bricks.{package}.{system}.{type_name}"),
    )
    .got(type_name)
}
