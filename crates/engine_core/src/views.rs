use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::{Mut, QueryState, Resource, World};
use glam::{Mat4, UVec2, Vec3};

use crate::{
    AssetLibrary, Camera, Clear, ColorSpace, DrawnPoses, Error, ErrorCode, FrameFacts,
    GlobalTransforms, Layers, Target, View,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewFacts {
    pub view: Entity,
    pub slot: u32,
    pub camera: Entity,
    pub target: Target,
    pub pixels: [u32; 4],
    pub layers: Layers,
    pub clear: Clear,
    pub clear_color: [f32; 4],
    pub order: i32,
    pub lens: Camera,
    pub eye: Vec3,
    pub forward: Vec3,
    pub view_projection: Mat4,
}

impl ViewFacts {
    pub fn size(&self) -> UVec2 {
        UVec2::new(self.pixels[2], self.pixels[3])
    }

    pub fn whole(&self, target: UVec2) -> bool {
        self.pixels == [0, 0, target.x, target.y]
    }

    pub fn aspect(&self) -> f32 {
        self.pixels[2].max(1) as f32 / self.pixels[3].max(1) as f32
    }
}

#[derive(Resource, Default)]
pub struct Views {
    listed: Vec<ViewFacts>,
    refused: Vec<(Entity, Error)>,
    found: Vec<(Entity, View)>,
    slotted: Vec<Option<Entity>>,
    query: Option<QueryState<(Entity, &'static View)>>,
    frame: FrameFacts,
}

impl Views {
    pub fn listed(&self) -> &[ViewFacts] {
        &self.listed
    }

    pub fn slots(&self) -> usize {
        self.slotted.len()
    }

    pub fn frame(&self) -> FrameFacts {
        self.frame
    }

    pub fn main(&self) -> Option<&ViewFacts> {
        self.listed.first()
    }

    pub fn get(&self, view: Entity) -> Option<&ViewFacts> {
        self.listed.iter().find(|facts| facts.view == view)
    }

    pub fn refused(&self) -> &[(Entity, Error)] {
        &self.refused
    }
}

pub fn main_camera(world: &World) -> Option<Entity> {
    let mut views = world.try_query::<(Entity, &View)>()?;
    views
        .iter(world)
        .min_by_key(|(entity, view)| (!view.target.surface(), view.order, *entity))
        .map(|(_, view)| view.camera)
}

fn refuse(view: Entity, field: &str, code: ErrorCode) -> Error {
    Error::new(code, format!("$.views.{}.{field}", view.to_bits()))
}

fn target_size(assets: Option<&AssetLibrary>, target: Target, surface: UVec2) -> Option<UVec2> {
    match target {
        Target::Surface => Some(surface.max(UVec2::ONE)),
        Target::Texture(texture) => assets?
            .texture(texture)
            .filter(|texture| texture.is_target() && texture.color_space() == ColorSpace::Srgb)
            .map(|texture| UVec2::new(texture.width(), texture.height()).max(UVec2::ONE)),
    }
}

fn eye_of(pose: Mat4) -> Option<(Vec3, Vec3, Vec3)> {
    let eye = pose.w_axis.truncate();
    let forward = pose.transform_vector3(Vec3::NEG_Z).try_normalize()?;
    let up = pose.transform_vector3(Vec3::Y).try_normalize()?;
    eye.is_finite().then_some((eye, forward, up))
}

impl Views {
    fn find(&mut self, world: &World) {
        self.found.clear();
        let Some(query) = self.query.as_mut() else {
            return;
        };
        query.update_archetypes(world);
        self.found.extend(
            query
                .iter_manual(world)
                .map(|(entity, view)| (entity, *view)),
        );
        self.found
            .sort_by_key(|(entity, view)| (!view.target.surface(), view.order, *entity));
    }

    fn slot_views(&mut self) {
        let found = &self.found;
        for slot in &mut self.slotted {
            if slot.is_some_and(|view| !found.iter().any(|(entity, _)| *entity == view)) {
                *slot = None;
            }
        }
        for &(entity, _) in found {
            if self.slotted.contains(&Some(entity)) {
                continue;
            }
            match self.slotted.iter_mut().find(|slot| slot.is_none()) {
                Some(free) => *free = Some(entity),
                None => self.slotted.push(Some(entity)),
            }
        }
        while self.slotted.last().is_some_and(Option::is_none) {
            self.slotted.pop();
        }
    }

    fn slot_of(&self, view: Entity) -> u32 {
        self.slotted
            .iter()
            .position(|slot| *slot == Some(view))
            .map_or(u32::MAX, |slot| slot as u32)
    }
}

type Placed = (Camera, (Vec3, Vec3, Vec3), UVec2);

fn placed(world: &World, entity: Entity, view: &View, facts: FrameFacts) -> Result<Placed, Error> {
    let lens = world
        .get::<Camera>(view.camera)
        .copied()
        .ok_or_else(|| refuse(entity, "camera", ErrorCode::MissingComponent))?;
    lens.check()
        .map_err(|error| error.within(&format!("$.views.{}.lens", entity.to_bits())))?;
    view.viewport
        .check("$")
        .map_err(|error| error.within(&format!("$.views.{}.viewport", entity.to_bits())))?;
    let pose = world
        .get_resource::<GlobalTransforms>()
        .zip(world.get_resource::<DrawnPoses>())
        .and_then(|(globals, poses)| poses.drawn_pose(globals, view.camera));
    let placed = pose
        .and_then(eye_of)
        .ok_or_else(|| refuse(entity, "camera", ErrorCode::InvalidCamera))?;
    let size = target_size(
        world.get_resource::<AssetLibrary>(),
        view.target,
        facts.display_size,
    )
    .ok_or_else(|| refuse(entity, "target", ErrorCode::InvalidTexture))?;
    Ok((lens, placed, size))
}

pub(crate) fn settle(world: &mut World, facts: FrameFacts) {
    world.resource_scope(|world, mut views: Mut<Views>| {
        let views = &mut *views;
        if views.query.is_none() {
            views.query = Some(world.query());
        }
        let world: &World = world;
        views.find(world);
        views.slot_views();
        views.listed.clear();
        views.refused.clear();
        views.frame = facts;
        for at in 0..views.found.len() {
            let (entity, view) = views.found[at];
            let (lens, (eye, forward, up), size) = match placed(world, entity, &view, facts) {
                Ok(placed) => placed,
                Err(error) => {
                    views.refused.push((entity, error));
                    continue;
                }
            };
            let pixels = view.viewport.pixels((size.x, size.y));
            let aspect = pixels[2].max(1) as f32 / pixels[3].max(1) as f32;
            let view_projection = lens.clip(aspect) * Mat4::look_to_rh(eye, forward, up);
            if !view_projection.is_finite() {
                views
                    .refused
                    .push((entity, refuse(entity, "camera", ErrorCode::InvalidCamera)));
                continue;
            }
            let slot = views.slot_of(entity);
            views.listed.push(ViewFacts {
                view: entity,
                slot,
                camera: view.camera,
                target: view.target,
                pixels,
                layers: view.layers,
                clear: view.clear,
                clear_color: view.clear_color,
                order: view.order,
                lens,
                eye,
                forward,
                view_projection,
            });
        }
    });
}
