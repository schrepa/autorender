use std::any::Any;
use std::sync::Arc;

use bevy_ecs::prelude::{Entity, Resource, World};
use engine_core::smoke::{
    Checked, SmokeFailure, SmokeResult, SmokeSession, above, control, near, number, query, render,
};
use engine_core::{BrickRegistry, List, Members, Slot, TILE_QUADS, TileKey, smoke_check, smoke_eq};
use glam::{Mat4, Vec3};
use serde_json::{Value, json};

use crate::{Ground, GroundSlot, WorldSolidsList};

const SETTLE_FRAMES: u32 = 2400;
const STEADY_FRAMES: u32 = 16;
const HEIGHT_WITHIN: f64 = 0.05;

pub trait FieldSamples: Send + Sync + 'static {
    fn height(&self, x: i32, z: i32) -> f32;

    fn hole(&self, x: u32, z: u32) -> bool;

    fn same(&self, other: &dyn FieldSamples) -> bool;

    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone)]
pub struct SolidField {
    pub owner: Entity,
    pub key: TileKey,
    pub origin: Vec3,
    pub spacing: f32,
    pub bounds: [f32; 2],
    pub samples: Arc<dyn FieldSamples>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SolidShape {
    Cuboid { half_extents: Vec3 },
    Sphere { radius: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldSolid {
    pub owner: Entity,
    pub layer: u16,
    pub asset: u16,
    pub shape: SolidShape,
    pub model: Mat4,
    pub memberships: u32,
    pub filter: u32,
}

pub trait SolidsFace: Send + Sync + 'static {
    fn revision(&self, world: &World) -> u64;

    fn fields(&self, _world: &World) -> Vec<SolidField> {
        Vec::new()
    }

    fn solids(&self, _world: &World) -> Vec<WorldSolid> {
        Vec::new()
    }
}

pub struct SolidsMember {
    pub brick: &'static str,
    pub name: &'static str,
    face: Box<dyn SolidsFace>,
}

impl SolidsMember {
    pub fn face(&self) -> &dyn SolidsFace {
        &*self.face
    }
}

#[derive(Resource, Default)]
pub struct WorldSolids {
    members: Vec<SolidsMember>,
}

impl Members for WorldSolids {
    type Member = (&'static str, Box<dyn SolidsFace>);

    fn add(&mut self, brick: &'static str, (name, face): Self::Member) {
        match self
            .members
            .iter_mut()
            .find(|member| member.brick == brick && member.name == name)
        {
            Some(member) => member.face = face,
            None => self.members.push(SolidsMember { brick, name, face }),
        }
    }
}

impl WorldSolids {
    pub fn members(&self) -> &[SolidsMember] {
        &self.members
    }

    pub fn revision(&self, world: &World) -> u64 {
        self.members
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |folded, member| {
                (folded ^ member.face.revision(world)).wrapping_mul(0x0100_0000_01b3)
            })
    }

    pub fn fields(&self, world: &World) -> Vec<SolidField> {
        self.members
            .iter()
            .flat_map(|member| member.face.fields(world))
            .collect()
    }

    pub fn solids(&self, world: &World) -> Vec<WorldSolid> {
        self.members
            .iter()
            .flat_map(|member| member.face.solids(world))
            .collect()
    }
}

pub(crate) fn world_solids(session: &mut dyn SmokeSession) -> SmokeResult {
    let members = place_members(session, WorldSolidsList::NAME);
    if members.is_empty() {
        return Ok(());
    }
    answered(session, &members)?;
    let grounded = !place_members(session, GroundSlot::NAME).is_empty();
    if grounded {
        session.source(&scene().to_string());
        session.restart(session.bricks());
    }
    control(session, "pause", json!({}))?;
    let resident = grounded && session.gpu();
    if resident {
        settled(session)?;
    }
    let before = seen(session).0;
    session.step(2);
    smoke_eq!(seen(session).0, before, "revision_holds_without_edits");
    if resident && raycasts(session) {
        meets_ground(session)?;
    }
    Ok(())
}

struct Probe {
    x: f32,
    z: f32,
    top: f32,
    reach: f32,
    height: f32,
}

fn scene() -> Value {
    json!({
        "id": "world-solids", "revision": 1,
        "camera": { "position": [0, 40, 0], "target": [60, 0, 60], "fov_degrees": 60, "near": 0.1, "far": 1000 },
        "textures": [], "materials": [], "meshes": [],
        "entities": [{
            "id": "ground",
            "terrain": { "origin": [-128, -20, -128], "tiles": [2, 2], "spacing": 1, "layers": [{ "type": "noise", "seed": 3, "height": 8, "wavelength": 120 }] },
        }],
    })
}

fn place_members(session: &mut dyn SmokeSession, place: &str) -> Vec<&'static str> {
    session
        .app()
        .world()
        .get_resource::<BrickRegistry>()
        .map_or_else(Vec::new, |registry| registry.members(place))
}

fn raycasts(session: &mut dyn SmokeSession) -> bool {
    session
        .app()
        .world()
        .get_resource::<BrickRegistry>()
        .is_some_and(|registry| registry.operation("raycast").is_some())
}

fn answered(session: &mut dyn SmokeSession, members: &[&'static str]) -> SmokeResult {
    let world = session.app().world();
    let joined: Vec<&'static str> = world
        .get_resource::<WorldSolids>()
        .map_or_else(Vec::new, |solids| {
            solids.members().iter().map(|member| member.brick).collect()
        });
    let missing: Vec<&'static str> = members
        .iter()
        .copied()
        .filter(|brick| !joined.contains(brick))
        .collect();
    smoke_check!(
        missing.is_empty(),
        "members_answer_world_solids",
        missing,
        json!([])
    );
    Ok(())
}

fn seen(session: &mut dyn SmokeSession) -> (u64, usize) {
    let world = session.app().world();
    world
        .get_resource::<WorldSolids>()
        .map_or((0, 0), |solids| {
            (solids.revision(world), solids.fields(world).len())
        })
}

fn settled(session: &mut dyn SmokeSession) -> SmokeResult {
    let (mut last, mut held) = (None, 0);
    for _ in 0..SETTLE_FRAMES {
        render(session)?;
        let (revision, fields) = seen(session);
        held = if fields > 0 && last == Some(revision) {
            held + 1
        } else {
            0
        };
        last = Some(revision);
        if held >= STEADY_FRAMES {
            return Ok(());
        }
    }
    smoke_check!(
        false,
        "world_solids_settle",
        json!({ "fields": seen(session).1, "steady_frames": held }),
        json!({ "fields": above(0), "steady_frames": STEADY_FRAMES })
    );
    Ok(())
}

fn probe(session: &mut dyn SmokeSession) -> Checked<Probe> {
    let world = session.app().world();
    let fields = world
        .get_resource::<WorldSolids>()
        .map_or_else(Vec::new, |solids| solids.fields(world));
    fields
        .iter()
        .find_map(|field| {
            let size = TILE_QUADS as f32 * field.spacing;
            let x = field.origin.x + f32::from(field.key.x) * size + size * 0.5;
            let z = field.origin.z + f32::from(field.key.z) * size + size * 0.5;
            let [low, high] = field.bounds;
            Ground::sample_in(world, x, z).map(|ground| Probe {
                x,
                z,
                top: field.origin.y + high + 1.0,
                reach: high - low + 2.0,
                height: ground.height,
            })
        })
        .ok_or_else(|| {
            SmokeFailure::new(
                "ground_answers_a_field_centre",
                &fields.len(),
                &json!({ "fields": above(0) }),
                concat!(file!(), ":", line!()),
            )
        })
}

fn meets_ground(session: &mut dyn SmokeSession) -> SmokeResult {
    let probe = probe(session)?;
    let found = query(
        session,
        "raycast",
        json!({
            "origin": [probe.x, probe.top, probe.z],
            "direction": [0.0, -1.0, 0.0],
            "max_distance": probe.reach,
        }),
    )?;
    let hit = number(&found["hit"]["point"][1]);
    let met = (hit - f64::from(probe.height)).abs() <= HEIGHT_WITHIN;
    smoke_check!(
        met,
        "raycast_meets_ground",
        found["hit"],
        near(probe.height, HEIGHT_WITHIN)
    );
    Ok(())
}
