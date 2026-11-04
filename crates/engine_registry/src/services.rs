use glam::{Mat4, UVec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameFacts {
    pub temporal: bool,
    pub capture: bool,
    pub debug: bool,
    pub display_size: UVec2,
    pub render_size: UVec2,
    pub pixel_ratio: f32,
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
struct Edges {
    pressed: u32,
    released: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputState {
    held: u32,
    next_tick: Edges,
    next_frame: Edges,
    shown: Edges,
}

impl InputState {
    pub fn ingest(&mut self, held: u32, pressed: u32, released: u32) {
        let pressed = pressed | (held & !self.held);
        let released = released | (self.held & !held);
        for edges in [&mut self.next_tick, &mut self.next_frame] {
            edges.pressed |= pressed;
            edges.released |= released;
        }
        self.held = held;
    }

    pub fn pressed(&self, action: u8) -> bool {
        self.held & mask(action) != 0
    }

    pub fn just_pressed(&self, action: u8) -> bool {
        self.shown.pressed & mask(action) != 0
    }

    pub fn just_released(&self, action: u8) -> bool {
        self.shown.released & mask(action) != 0
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn begin_tick(&mut self) {
        self.shown = std::mem::take(&mut self.next_tick);
    }

    pub fn begin_frame(&mut self) {
        self.shown = std::mem::take(&mut self.next_frame);
    }

    pub fn hide_edges(&mut self) {
        self.shown = Edges::default();
    }
}

fn mask(action: u8) -> u32 {
    1_u32.checked_shl(u32::from(action)).unwrap_or(0)
}
