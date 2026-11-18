use engine_core::{Error, ShaderText, Source, TargetName};

use crate::install::Work;
use crate::{GpuPart, Install, Pass, Prepare};

pub(crate) const POST: &str = "post";

pub(crate) struct ScenePart {
    passes: Vec<Pass<Self>>,
}

impl GpuPart for ScenePart {
    type Reads = ();
    const NAME: &'static str = "scene";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let listed: Vec<engine_core::ScenePass> = cx
            .core
            .passes
            .listed()
            .iter()
            .filter(|listed| listed.bit.is_some())
            .filter(|listed| {
                cx.core
                    .draws
                    .iter()
                    .any(|member| member.kind.joins(listed.name))
            })
            .copied()
            .collect();
        let mut passes = Vec::with_capacity(listed.len());
        for listed in listed {
            let pass = cx.core.pass(listed.name, Work::Render);
            cx.core.draws_scene(pass, listed.name, listed.pass);
            if listed.pass.writes.contains(TargetName::CANVAS)
                && let Some(declared) = cx.core.pass_mut(pass)
            {
                declared.targets.push((TargetName::CANVAS, crate::Access::Clear));
            }
            passes.push(Pass(pass, std::marker::PhantomData));
        }
        Ok(Self { passes })
    }

    fn prepare(&mut self, (): (), cx: &mut Prepare<'_, Self>) {
        for pass in &self.passes {
            cx.run(*pass);
        }
    }

    fn encode(&mut self, _pass: Pass<Self>, cx: &mut crate::Encode<'_, Self>) {
        cx.render(crate::Attach::Declared).scene(self);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PostUniform {
    pub(crate) size: [f32; 4],
    pub(crate) kept: [u32; 4],
}

pub(crate) fn post_uniform(render: (u32, u32)) -> PostUniform {
    let (width, height) = (render.0 as f32, render.1 as f32);
    PostUniform {
        size: [width, height, 1.0 / width, 1.0 / height],
        kept: [0; 4],
    }
}

pub(crate) fn core_source() -> &'static Source {
    &crate::shaders::CORE
}

pub(crate) const fn generated(path: &'static str) -> ShaderText {
    ShaderText::generated(path, "", "")
}
