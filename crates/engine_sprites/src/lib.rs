include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod flat;
mod smokes;

use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, ComponentKind, Error, Poses, Runs, SceneComponents, Smoke, Source, TextureHandle, gpu_struct};
use engine_render::{
    Bind, Buffer, DepthUse, Draw, GpuPart, Install, Layer, Lifetime, MainView, Pipeline, Prepare, RenderBricks, RenderPipeline, SamplerDesc,
    Shape, wgpu,
};

#[derive(Component, Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Sprite {
    pub sheet: TextureHandle,
    pub rect: [f32; 4],
    pub tint: [f32; 4],
}

gpu_struct! {
    pub struct SpriteInstance {
        pub place: [f32; 4],
        pub rect: [f32; 4],
        pub tint: [f32; 4],
    }
}

pub struct SpritesBrick;

impl Brick for SpritesBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[Smoke { name: "sprites_drawn", runs: Runs::Gpu, run: smokes::drawn }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(ComponentKind::of::<Sprite>("sprite"), None)
            .draws::<SpritesGpu, MainView>(Layer::Opaque, SpritesGpu::draw);
    }
}

struct SpritesGpu {
    pipeline: Pipeline<Self, MainView>,
    instances: Buffer<Self>,
    staged: Vec<SpriteInstance>,
    runs: Vec<(TextureHandle, Range<u32>)>,
    drawn: Vec<(Pipeline<Self, MainView>, Range<u32>)>,
}

impl GpuPart for SpritesGpu {
    type Reads = (Query<'static, 'static, (Entity, Ref<'static, Sprite>)>, Res<'static, Poses>);

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::sprites::draw;
        let instances = cx.buffer("instances", Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc::NEAREST);
        let pipeline = cx.render_pipeline(
            MainView,
            &RenderPipeline {
                vertex: draw::VS_SPRITE,
                fragment: Some(draw::FS_SPRITE),
                debug: None,
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::None,
                blend: None,
                writes: wgpu::ColorWrites::ALL,
            },
            &[(draw::INSTANCES, Bind::Buffer(instances)), (draw::SHEET, Bind::Asset), (draw::SHEET_SAMPLER, Bind::Sampler(sampler))],
        )?;
        Ok(Self { pipeline, instances, staged: Vec::new(), runs: Vec::new(), drawn: Vec::new() })
    }

    fn prepare(&mut self, (sprites, poses): SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>) {
        if poses.is_changed() || sprites.iter().count() != self.staged.len() || sprites.iter().any(|(_, sprite)| sprite.is_changed()) {
            stage(&sprites, &poses, &mut self.staged, &mut self.runs);
            cx.allocate_buffer(self.instances, std::mem::size_of_val(self.staged.as_slice()) as u64);
            cx.write(self.instances, 0, &self.staged);
        }
        self.drawn.clear();
        for (sheet, range) in &self.runs {
            self.drawn.push((cx.fill(&self.pipeline, &[*sheet]), range.clone()));
        }
    }
}

impl SpritesGpu {
    fn draw(&self, _: &(), draw: &mut Draw<'_, Self, MainView>) -> u32 {
        for (pipeline, range) in &self.drawn {
            draw.draw(pipeline, Shape::Pulled { vertices: 0..6, instances: range.clone() });
        }
        self.drawn.len() as u32
    }
}

fn stage(
    sprites: &Query<(Entity, Ref<'static, Sprite>)>,
    poses: &Poses,
    staged: &mut Vec<SpriteInstance>,
    runs: &mut Vec<(TextureHandle, Range<u32>)>,
) {
    staged.clear();
    runs.clear();
    let mut sorted: Vec<(TextureHandle, SpriteInstance)> = sprites
        .iter()
        .filter_map(|(entity, sprite)| {
            let pose = poses.of(entity)?;
            let place = [pose.translation.x, pose.translation.y, pose.translation.z, pose.matrix3.x_axis.length()];
            Some((sprite.sheet, SpriteInstance { place, rect: sprite.rect, tint: sprite.tint }))
        })
        .collect();
    sorted.sort_by_key(|(sheet, _)| *sheet);
    for (sheet, instance) in sorted {
        let at = staged.len() as u32;
        staged.push(instance);
        let extends = runs.last().is_some_and(|(last, _)| *last == sheet);
        if extends {
            if let Some((_, range)) = runs.last_mut() {
                range.end = at + 1;
            }
        } else {
            runs.push((sheet, at..at + 1));
        }
    }
}
