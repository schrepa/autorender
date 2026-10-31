use engine_core::{BrickRegistry, Bricks};

use crate::gpu_layouts::{Constant, Layout};
use crate::linker::ShaderModule;
use crate::parts::PartEntry;
use crate::pipeline_layouts::PipelineLayout;

#[derive(Default)]
pub struct RenderBrick {
    pub parts: Vec<&'static PartEntry>,
    pub modules: Vec<&'static [ShaderModule]>,
    pub defaults: Vec<&'static [ShaderModule]>,
    pub layouts: Vec<&'static [(&'static str, &'static PipelineLayout)]>,
    pub mirrors: Vec<&'static [Layout]>,
    pub constants: Vec<&'static [Constant]>,
}

impl RenderBrick {
    pub fn modules(&self) -> impl Iterator<Item = &'static ShaderModule> + '_ {
        self.modules.iter().flat_map(|modules| modules.iter())
    }

    pub fn defaults(&self) -> impl Iterator<Item = &'static ShaderModule> + '_ {
        self.defaults.iter().flat_map(|modules| modules.iter())
    }

    pub fn faces(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.parts
            .iter()
            .flat_map(|part| part.decl.faces.iter().map(engine_core::FaceOffer::name))
            .chain(
                self.modules()
                    .filter_map(|module| crate::faces::face_of(module.path)),
            )
    }
}

pub trait AddRenderParts {
    fn render_part(&mut self, part: &'static PartEntry) -> &mut Self;

    fn shaders(&mut self, modules: &'static [ShaderModule]) -> &mut Self;

    fn face_defaults(&mut self, modules: &'static [ShaderModule]) -> &mut Self;

    fn pipeline_layouts(
        &mut self,
        layouts: &'static [(&'static str, &'static PipelineLayout)],
    ) -> &mut Self;

    fn mirrors(&mut self, layouts: &'static [Layout], constants: &'static [Constant]) -> &mut Self;
}

impl AddRenderParts for Bricks<'_> {
    fn render_part(&mut self, part: &'static PartEntry) -> &mut Self {
        let passes: Vec<&'static str> = part.decl.passes.iter().map(|pass| pass.name).collect();
        self.passes(&passes);
        if let Some(render) = self.extension::<RenderBrick>() {
            render.parts.push(part);
        }
        self
    }

    fn shaders(&mut self, modules: &'static [ShaderModule]) -> &mut Self {
        if let Some(render) = self.extension::<RenderBrick>() {
            render.modules.push(modules);
        }
        self
    }

    fn face_defaults(&mut self, modules: &'static [ShaderModule]) -> &mut Self {
        if let Some(render) = self.extension::<RenderBrick>()
            && !render
                .defaults
                .iter()
                .any(|known| std::ptr::eq(*known, modules))
        {
            render.defaults.push(modules);
        }
        self
    }

    fn pipeline_layouts(
        &mut self,
        layouts: &'static [(&'static str, &'static PipelineLayout)],
    ) -> &mut Self {
        if let Some(render) = self.extension::<RenderBrick>() {
            render.layouts.push(layouts);
        }
        self
    }

    fn mirrors(&mut self, layouts: &'static [Layout], constants: &'static [Constant]) -> &mut Self {
        if let Some(render) = self.extension::<RenderBrick>() {
            render.mirrors.push(layouts);
            render.constants.push(constants);
        }
        self
    }
}

pub fn render_bricks(
    registry: &BrickRegistry,
) -> impl Iterator<Item = (&'static str, &RenderBrick)> + '_ {
    registry
        .bricks()
        .iter()
        .filter_map(|brick| Some((brick.name, brick.extension::<RenderBrick>()?)))
}
