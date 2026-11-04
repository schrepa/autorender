use bevy_ecs::entity::Entity;
use glam::{Affine3A, Mat4, UVec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameFacts {
    pub temporal: bool,
    pub capture: bool,
    pub debug: bool,
    pub display_size: UVec2,
    pub render_size: UVec2,
    pub pixel_ratio: f32,
    pub submitted: bool,
}

impl Default for FrameFacts {
    fn default() -> Self {
        Self {
            temporal: false,
            capture: false,
            debug: false,
            display_size: UVec2::new(1920, 1080),
            render_size: UVec2::new(1920, 1080),
            pixel_ratio: 1.0,
            submitted: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewState {
    facts: FrameFacts,
    view_projection: Mat4,
    eye: Vec3,
    frozen: Option<Vec3>,
    scene: u64,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            facts: FrameFacts::default(),
            view_projection: Mat4::IDENTITY,
            eye: Vec3::ZERO,
            frozen: None,
            scene: 0,
        }
    }
}

impl ViewState {
    pub fn identified(scene: u64) -> Self {
        Self {
            scene,
            ..Self::default()
        }
    }

    pub fn facts(&self) -> FrameFacts {
        self.facts
    }

    pub fn render_size(&self) -> UVec2 {
        self.facts.render_size
    }

    pub fn display_size(&self) -> UVec2 {
        self.facts.display_size
    }

    pub fn scene(&self) -> u64 {
        self.scene
    }

    pub fn view_projection(&self) -> Mat4 {
        self.view_projection
    }

    pub fn eye(&self) -> Vec3 {
        self.eye
    }

    pub fn detail_eye(&self) -> Vec3 {
        self.frozen.unwrap_or(self.eye)
    }

    pub fn frozen(&self) -> Option<Vec3> {
        self.frozen
    }

    pub fn set(&mut self, facts: FrameFacts, view_projection: Mat4, eye: Vec3) {
        self.facts = facts;
        self.view_projection = view_projection;
        self.eye = eye;
    }

    pub fn freeze(&mut self, at: Option<Vec3>) {
        self.frozen = at;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputState {
    held: u32,
    pending_pressed: u32,
    pending_released: u32,
    tick_pressed: u32,
    tick_released: u32,
}

impl InputState {
    pub fn ingest(&mut self, held: u32, pressed: u32, released: u32) {
        self.pending_pressed |= pressed | (held & !self.held);
        self.pending_released |= released | (self.held & !held);
        self.held = held;
    }

    pub fn pressed(&self, action: u8) -> bool {
        self.held & mask(action) != 0
    }

    pub fn just_pressed(&self, action: u8) -> bool {
        self.tick_pressed & mask(action) != 0
    }

    pub fn just_released(&self, action: u8) -> bool {
        self.tick_released & mask(action) != 0
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn begin_tick(&mut self) {
        self.tick_pressed = std::mem::take(&mut self.pending_pressed);
        self.tick_released = std::mem::take(&mut self.pending_released);
    }

    pub fn end_tick(&mut self) {
        self.tick_pressed = 0;
        self.tick_released = 0;
    }
}

fn mask(action: u8) -> u32 {
    1_u32.checked_shl(u32::from(action)).unwrap_or(0)
}

#[derive(Clone, Copy, Debug)]
struct Indexed {
    entity: Entity,
    value: usize,
}

const EMPTY: Indexed = Indexed {
    entity: Entity::PLACEHOLDER,
    value: usize::MAX,
};

#[derive(Clone, Debug, Default)]
struct EntityIndices {
    entries: Vec<Indexed>,
    occupied: Vec<u32>,
}

impl EntityIndices {
    fn get(&self, entity: Entity) -> Option<usize> {
        let entry = self.entries.get(entity.index_u32() as usize)?;
        (entry.entity == entity).then_some(entry.value)
    }

    fn insert(&mut self, entity: Entity, value: usize) {
        let slot = entity.index_u32() as usize;
        if slot >= self.entries.len() {
            self.entries.resize(slot + 1, EMPTY);
        }
        let entry = &mut self.entries[slot];
        if entry.entity == Entity::PLACEHOLDER {
            self.occupied.push(slot as u32);
        }
        *entry = Indexed { entity, value };
    }

    fn clear(&mut self) {
        for slot in self.occupied.drain(..) {
            self.entries[slot as usize] = EMPTY;
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PoseTable {
    entities: Vec<Entity>,
    poses: Vec<Option<Affine3A>>,
    indices: EntityIndices,
}

impl PoseTable {
    pub fn of(&self, entity: Entity) -> Option<Affine3A> {
        let index = self.indices.get(entity)?;
        self.poses.get(index).copied().flatten()
    }

    pub fn set(&mut self, entity: Entity, pose: Option<Affine3A>) {
        match self.indices.get(entity) {
            Some(index) => self.poses[index] = pose,
            None => {
                self.indices.insert(entity, self.entities.len());
                self.entities.push(entity);
                self.poses.push(pose);
            }
        }
    }

    pub fn clear(&mut self) {
        self.entities.clear();
        self.poses.clear();
        self.indices.clear();
    }

    pub fn reserve(&mut self, additional: usize) {
        self.entities.reserve(additional);
        self.poses.reserve(additional);
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Entity, Option<Affine3A>)> + '_ {
        self.entities
            .iter()
            .copied()
            .zip(self.poses.iter().copied())
    }
}
