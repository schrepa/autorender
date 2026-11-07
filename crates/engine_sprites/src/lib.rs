include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod flat;
mod smokes;

use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::{
    Asset, AssetResidency, AssetUses, Attributes, Brick, Bricks, ColorPass, ComponentKind,
    DrawKind, DrawnPoses, Error, GlobalTransforms, KindDecl, PrimaryView, Runs, SceneComponents,
    Schedule, Smoke, Source, Spawner, TextureHandle, gpu_struct,
};
use engine_render::{
    Bind, Buffer, DepthUse, Draw, GpuPart, Install, Lifetime, Pipeline, Prepare, RenderBricks,
    RenderPipeline, SamplerDesc, Scene, SceneDraw, Shape, wgpu,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub use flat::{DARK, LIGHT};

pub const SPRITES: DrawKind = DrawKind("sprites");

const DRAWN_IN: Scene = Scene::of::<ColorPass>();

const GENERATED: &[&str] = &[<SpriteInstance as engine_core::GpuStruct>::WGSL];

#[derive(
    Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema,
)]
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

fn sprite_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(Sprite)).unwrap_or(Value::Null)
}

fn sprite_example() -> Value {
    json!({ "sheet": "sheet", "rect": [0.0, 0.0, 1.0, 1.0], "tint": [1.0, 1.0, 1.0, 1.0] })
}

pub struct SpritesBrick;

impl Brick for SpritesBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "sprites_drawn",
        runs: Runs::Gpu,
        run: smokes::drawn,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Sprite>("sprite")
                    .schema(sprite_schema)
                    .example(sprite_example),
                None,
            )
            .systems(Schedule::ViewSync, publish_sheets)
            .generated(GENERATED)
            .draw_kind(KindDecl {
                kind: SPRITES,
                vertex: "vs_sprite",
                needs: Attributes::NONE,
            })
            .draws::<SpritesGpu>(SPRITES, SpritesGpu::draw);
    }
}

#[derive(Component)]
struct Sheets;

fn publish_sheets(
    sprites: Query<(Entity, &Sprite)>,
    globals: Res<GlobalTransforms>,
    residency: Res<AssetResidency>,
    own: Query<(Entity, &AssetUses), With<Sheets>>,
    mut spawner: Spawner<SpritesBrick>,
) {
    let mut wanted: Vec<Asset> = sprites
        .iter()
        .filter(|(entity, _)| globals.pose(*entity).is_some())
        .map(|(_, sprite)| Asset::Texture(sprite.sheet))
        .collect();
    wanted.sort_unstable();
    wanted.dedup();
    match own.single() {
        Ok((entity, held)) => {
            if held.assets() != wanted.as_slice() {
                spawner.insert(entity, AssetUses::new(&residency, wanted));
            }
        }
        Err(_) => {
            spawner.spawn((Sheets, AssetUses::new(&residency, wanted)));
        }
    }
}

struct SpritesGpu {
    flat: flat::Flat,
    pipeline: Pipeline<Self, Scene>,
    instances: Buffer<Self>,
    staged: Vec<SpriteInstance>,
    runs: Vec<(TextureHandle, Range<u32>)>,
    drawn: Vec<(Pipeline<Self, Scene>, Range<u32>)>,
}

impl GpuPart for SpritesGpu {
    type Reads = (
        Query<'static, 'static, (Entity, &'static Sprite)>,
        (Res<'static, GlobalTransforms>, Res<'static, DrawnPoses>),
        Res<'static, PrimaryView>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::sprites::draw;
        let flat = flat::Flat::install(cx)?;
        let instances = cx.buffer("instances", Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc::NEAREST);
        let pipeline = cx.render_pipeline(
            DRAWN_IN,
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
            &[
                (draw::INSTANCES, Bind::Buffer(instances)),
                (draw::SHEET, Bind::Asset),
                (draw::SHEET_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;
        Ok(Self {
            flat,
            pipeline,
            instances,
            staged: Vec::new(),
            runs: Vec::new(),
            drawn: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (sprites, (globals, poses), view): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.flat.prepare(&view, cx);
        stage(
            &sprites,
            (&globals, &poses),
            &mut self.staged,
            &mut self.runs,
        );
        cx.allocate_buffer(
            self.instances,
            std::mem::size_of_val(self.staged.as_slice()) as u64,
        );
        cx.write(self.instances, 0, &self.staged);
        self.drawn.clear();
        for (sheet, range) in &self.runs {
            self.drawn
                .push((cx.fill(&self.pipeline, &[*sheet]), range.clone()));
        }
    }
}

impl SpritesGpu {
    fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if scene.pass != DRAWN_IN {
            return 0;
        }
        for (pipeline, range) in &self.drawn {
            draw.draw(
                pipeline,
                Shape::Pulled {
                    vertices: 0..6,
                    instances: range.clone(),
                },
            );
        }
        self.drawn.len() as u32 + self.flat.draw(draw)
    }
}

fn stage(
    sprites: &Query<(Entity, &'static Sprite)>,
    (globals, poses): (&GlobalTransforms, &DrawnPoses),
    staged: &mut Vec<SpriteInstance>,
    runs: &mut Vec<(TextureHandle, Range<u32>)>,
) {
    staged.clear();
    runs.clear();
    let mut sorted: Vec<(TextureHandle, SpriteInstance)> = sprites
        .iter()
        .filter(|&(entity, _)| globals.drawn(entity))
        .filter_map(|(entity, sprite)| {
            let pose = poses.drawn_pose(globals, entity)?;
            let place = [
                pose.w_axis.x,
                pose.w_axis.y,
                pose.w_axis.z,
                pose.x_axis.truncate().length(),
            ];
            Some((
                sprite.sheet,
                SpriteInstance {
                    place,
                    rect: sprite.rect,
                    tint: sprite.tint,
                },
            ))
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
