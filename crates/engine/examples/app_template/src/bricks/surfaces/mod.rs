engine::brick!(surfaces);

mod forms;
mod layout;
mod records;

use engine_core::{Brick, Bricks, Source};
use engine_render::RenderBricks;

pub use forms::{
    Form, FormDraw, FormSide, FragmentUse, KindForm, PassFragment, PassRole, SurfaceForms,
    Varyings, role,
};
pub use layout::{LightingModule, ModelForm, SlotModule};
pub use records::FaceCull;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PointInput {
    pub name: &'static str,
    pub ty: &'static str,
    pub standard: &'static str,
}

impl PointInput {
    pub const fn new(name: &'static str, ty: &'static str, standard: &'static str) -> Self {
        Self { name, ty, standard }
    }
}

pub trait SurfaceBricks {
    fn material_slot(&mut self, module: SlotModule) -> &mut Self;
    fn point_input(&mut self, input: PointInput) -> &mut Self;
    fn lighting_model(&mut self, module: LightingModule) -> &mut Self;
}

impl SurfaceBricks for Bricks<'_> {
    fn material_slot(&mut self, module: SlotModule) -> &mut Self {
        self.entry(module)
    }

    fn point_input(&mut self, input: PointInput) -> &mut Self {
        self.entry(input)
    }

    fn lighting_model(&mut self, module: LightingModule) -> &mut Self {
        self.entry(module)
    }
}

pub struct SurfacesBrick;

impl Brick for SurfacesBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .modules(layout::made)
            .part::<records::SurfaceRecords>();
    }
}
