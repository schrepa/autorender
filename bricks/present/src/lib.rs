#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

engine::brick!();

use std::marker::PhantomData;

use bevy_ecs::system::SystemParamItem;
use engine::prelude::*;
use engine_core::{Error, ErrorCode, Filter, Sampler, ViewFacts, ViewSkip, Views, Wrap};
use engine_render::{Access, Encode, FULLSCREEN, Own, PerBatch, SharedTarget, TargetClear};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const PRESENTED: TargetName = TargetName("presented");

pub trait Scale: Resource {
    fn fraction(&self) -> f32;

    fn direct_supported() -> bool {
        false
    }
}

const PRESENT_PASS: &str = "present";

const COVER_PASS: &str = "cover";

const LINEAR_CLAMP: Sampler = Sampler {
    wrap_u: Wrap::ClampToEdge,
    wrap_v: Wrap::ClampToEdge,
    wrap_w: Wrap::ClampToEdge,
    mag_filter: Filter::Linear,
    min_filter: Filter::Linear,
    mipmap_filter: Some(Filter::Linear),
    compare: None,
    anisotropy: 1,
};

fn default_antialias_max_pixel_ratio() -> Option<f32> {
    Some(1.5)
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Present {
    pub antialias: bool,
    #[serde(default = "default_antialias_max_pixel_ratio")]
    pub antialias_max_pixel_ratio: Option<f32>,
}

impl Default for Present {
    fn default() -> Self {
        Self {
            antialias: false,
            antialias_max_pixel_ratio: default_antialias_max_pixel_ratio(),
        }
    }
}

impl Present {
    pub fn antialias_active(self, pixel_ratio: f32) -> bool {
        self.antialias
            && self
                .antialias_max_pixel_ratio
                .is_none_or(|limit| pixel_ratio <= limit)
    }
}

pub fn needs_intermediate(present: Option<Present>, scale: f32, pixel_ratio: f32) -> bool {
    scale < 1.0 || present.is_some_and(|present| present.antialias_active(pixel_ratio))
}

pub struct PresentBrick<S>(PhantomData<fn() -> S>);

impl<S: Scale> Brick for PresentBrick<S> {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Present>("present")
                    .schema(engine_core::schema_of::<Present>),
            )
            .part::<Presenting<S>>();
    }
}

struct Presenting<S> {
    pass: engine_render::Pass<Self>,
    blit: Pipeline<Self, Own>,
    smooth: Pipeline<Self, Own>,
    cover: engine_render::Pass<Self>,
    antialias: bool,
    direct: bool,
    several: bool,
    skipped: Vec<Entity>,
    scale: PhantomData<fn() -> S>,
}

fn fullscreen(
    fragment: engine_core::Entry,
    constants: &'static [(&'static str, f64)],
) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: FULLSCREEN,
        fragment: Some(fragment),
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets: &[(None, wgpu::ColorWrites::ALL)],
        constants,
        per_batch: PerBatch::NONE,
    }
}

impl<S: Scale> GpuPart for Presenting<S> {
    type Reads = (
        Query<'static, 'static, &'static Present>,
        Query<'static, 'static, Entity, With<ViewSkip>>,
        Option<Res<'static, S>>,
        Option<Res<'static, Views>>,
    );
    const NAME: &'static str = "present";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::present::canvas;
        cx.shared_target(SharedTarget {
            name: PRESENTED,
            format: wgpu::TextureFormat::Rgba8Unorm,
            samples: 1,
            clear: TargetClear::Color([0.0, 0.0, 0.0, 1.0]),
        });
        cx.frame(canvas::PRESENTED, Bind::Target(PRESENTED))?;
        let sampler = cx.sampler(LINEAR_CLAMP);
        let pass = cx
            .pass(PRESENT_PASS)
            .label("present")
            .span("present")
            .target(TargetName::CANVAS, Access::Clear)
            .id();
        let bound = [(canvas::PRESENTED_SAMPLER, Bind::Sampler(sampler))];
        let blit = cx.render_pipeline(pass, &fullscreen(canvas::FS_PRESENT, &[]), &bound)?;
        let smooth = cx.render_pipeline(
            pass,
            &fullscreen(canvas::FS_PRESENT, &[("PRESENT_ANTIALIAS", 1.0)]),
            &bound,
        )?;
        let cover = cx
            .pass(COVER_PASS)
            .label("cover")
            .span("cover")
            .target(TargetName::CANVAS, Access::Clear)
            .id();
        Ok(Self {
            pass,
            blit,
            smooth,
            cover,
            antialias: false,
            direct: false,
            several: false,
            skipped: Vec::new(),
            scale: PhantomData,
        })
    }

    fn prepare(
        &mut self,
        (presents, skipped, scale, views): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.skipped.clear();
        self.skipped.extend(&skipped);
        let fraction = scale.map_or(1.0, |scale| scale.fraction());
        cx.scale_target(PRESENTED, fraction);
        let mut found = presents.iter();
        let present = found.next().copied();
        let several = found.next().is_some();
        if several && !self.several {
            cx.report(
                Error::new(ErrorCode::NameTwice, "$.entities.present").got(presents.iter().count()),
            );
        }
        self.several = several;
        let pixel_ratio = views.map_or(1.0, |views| views.frame().pixel_ratio);
        self.antialias = present.is_some_and(|present| present.antialias_active(pixel_ratio));
        self.direct =
            S::direct_supported() && !needs_intermediate(present, fraction, pixel_ratio);
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.skipped.contains(&view.view) {
            cx.run(self.cover);
        } else if !self.direct {
            cx.run(self.pass);
        }
    }

    fn encode(&mut self, pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        if pass == self.cover {
            let _ = cx.render();
            return;
        }
        let pipeline = if self.antialias {
            &self.smooth
        } else {
            &self.blit
        };
        cx.render().draw(
            pipeline,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
                key: None,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn antialias_and_intermediate_respect_pixel_ratio_limit_and_scale() {
        let always = Present {
            antialias: true,
            antialias_max_pixel_ratio: None,
        };
        assert!(always.antialias_active(1.0));
        assert!(always.antialias_active(2.0));
        assert!(needs_intermediate(Some(always), 1.0, 2.0));

        let capped = Present {
            antialias: true,
            antialias_max_pixel_ratio: Some(1.5),
        };
        assert!(capped.antialias_active(1.0));
        assert!(capped.antialias_active(1.5));
        assert!(!capped.antialias_active(2.0));
        assert!(needs_intermediate(Some(capped), 1.0, 1.0));
        assert!(!needs_intermediate(Some(capped), 1.0, 2.0));
        assert!(needs_intermediate(Some(capped), 0.75, 2.0));
    }
}
