use engine_core::{Error, ShaderText, Source, TargetName};

use crate::install::Work;
use crate::{GpuPart, Install, Pass, Prepare, SharedTarget};

pub(crate) const POST: &str = "post";
pub(crate) const WRITES_MODULE: &str = "generated::writes";
pub(crate) const PASSES_MODULE: &str = "generated::passes";
pub(crate) const KINDS_MODULE: &str = "generated::kinds";

pub(crate) struct ScenePart {
    passes: Vec<Pass<Self>>,
}

impl GpuPart for ScenePart {
    type Reads = ();
    const NAME: &'static str = "scene";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        cx.shared_target(SharedTarget {
            name: TargetName::COLOR,
            format: wgpu::TextureFormat::Rgba16Float,
            samples: 1,
        });
        cx.shared_target(SharedTarget {
            name: TargetName::DEPTH,
            format: wgpu::TextureFormat::Depth32Float,
            samples: 1,
        });
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
            passes.push(Pass(pass, std::marker::PhantomData));
        }
        Ok(Self { passes })
    }

    fn prepare(&mut self, (): (), cx: &mut Prepare<'_, Self>) {
        for pass in &self.passes {
            cx.run(*pass);
        }
    }

    fn encode(&mut self, _pass: Pass<Self>, cx: &mut crate::Encode<'_, Self>) -> u32 {
        cx.render(crate::Attach::Declared).scene(self)
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

const SHADED: &str = "
fn point_shaded(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {
    return shade(point, pixel);
}

fn slot_shaded(input: SurfaceInput, point: SurfacePoint, pixel: vec2<f32>) -> Shaded {
    return slot_output(input, point, shade(point, pixel));
}
";

pub(crate) fn pass_module() -> (String, String) {
    let text = format!(
        "import package::engine::shading::{{Shaded, SurfaceInput, SurfacePoint}};\nimport package::engine::slots::{{shade, slot_output}};\n\nstruct Outputs {{\n    @location(0) color: vec4<f32>,\n}}\n\nfn outputs(point: SurfacePoint, shaded: Shaded) -> Outputs {{\n    var written: Outputs;\n    written.color = vec4<f32>(shaded.linear, 1.0);\n    return written;\n}}\n\nfn blended_outputs(point: SurfacePoint, shaded: Shaded) -> Outputs {{\n    var written = outputs(point, shaded);\n    written.color = vec4<f32>(shaded.linear * shaded.opacity, shaded.opacity);\n    return written;\n}}\n{SHADED}"
    );
    (engine_core::TargetName::COLOR.name().to_owned(), text)
}

pub(crate) fn core_source() -> &'static Source {
    &crate::shaders::CORE
}

pub(crate) const fn generated(path: &'static str) -> ShaderText {
    ShaderText::generated(path, "", "")
}
