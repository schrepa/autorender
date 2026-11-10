include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod flat;

use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::{
    Asset, AssetResidency, AssetUses, AttachPoint, Attributes, Brick, Bricks, ComponentKind,
    DrawKind, Drawn, DrawnPoses, DrawnTable, Error, ErrorCode, GlobalTransforms, KindDecl, KindSkip,
    PassViews, Refusals,
    SceneComponents, ScenePass, Schedule, Source, Spawner, TargetName, TextureHandle, ViewFacts,
    gpu_struct,
};
use engine_render::{
    Allocation, Bind, Buffer, DepthUse, Draw, GpuPart, Install, Lifetime, Pipeline, Prepare,
    RenderBricks, RenderPipeline, Scene, SceneDraw, Shape, wgpu,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use flat::{DARK, LIGHT};

pub const SPRITES: DrawKind = DrawKind("sprites");

const GENERATED: &[&str] = &[<SpriteInstance as engine_core::GpuStruct>::WGSL];

#[derive(
    Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[require(Drawn = Drawn::procedural(SPRITES))]
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
        pub drawn: [u32; 4],
    }
}

fn sprite_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(Sprite)).unwrap_or(Value::Null)
}

pub struct SpritesBrick;

impl Brick for SpritesBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Sprite>("sprite")
                    .schema(sprite_schema),
                None,
            )
            .systems(Schedule::ViewSync, (judge_sprites, publish_sheets))
            .on_remove::<Sprite, _>(forget_sprite)
            .generated(GENERATED)
            .draw_kind(KindDecl {
                kind: SPRITES,
                needs: Attributes::NONE,
                skips: &[
                    KindSkip {
                        pass: "shadow",
                        reason: "sprites cast no shadows",
                    },
                    KindSkip {
                        pass: "probe",
                        reason: "sprites stay out of probe captures",
                    },
                ],
            })
            .draws::<SpritesGpu>(SPRITES, SpritesGpu::draw);
    }
}

#[derive(Component)]
struct Sheets;

pub const SPRITE_PART: &str = "sprite";

type ChangedSprites = (With<Sprite>, Or<(Added<Sprite>, Changed<Drawn>)>);

fn judge_sprites(
    sprites: Query<(Entity, &Drawn), ChangedSprites>,
    mut refusals: ResMut<Refusals>,
) {
    for (entity, drawn) in &sprites {
        let verdict = if drawn.kind == SPRITES {
            Ok(())
        } else {
            Err(Error::new(ErrorCode::UsageConflict, "$.sprite")
                .got(drawn.kind.name())
                .nearest(SPRITES.name()))
        };
        refusals.judge(entity, SPRITE_PART, verdict);
    }
}

fn forget_sprite(In(entity): In<Entity>, mut refusals: ResMut<Refusals>) {
    refusals.accept(entity, SPRITE_PART);
}

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
    pipelines: Vec<(&'static str, Pipeline<Self, Scene>)>,
    instances: Buffer<Self>,
    staged: Vec<SpriteInstance>,
    runs: Vec<(TextureHandle, Range<u32>)>,
    drawn: Vec<(&'static str, Pipeline<Self, Scene>, Range<u32>)>,
}

fn drawn_in(cx: &Install<'_, SpritesGpu>) -> Vec<ScenePass> {
    cx.passes()
        .listed()
        .iter()
        .filter(|listed| {
            listed.bit.is_some()
                && cx.passes().takes(listed, SPRITES)
                && listed.pass.views != PassViews::Owned
                && listed.pass.at != AttachPoint::Transparent
        })
        .copied()
        .collect()
}

impl GpuPart for SpritesGpu {
    type Reads = (
        (Query<'static, 'static, &'static Sprite>, Res<'static, DrawnTable>),
        (Res<'static, GlobalTransforms>, Res<'static, DrawnPoses>),
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::sprites::draw;
        let listed = drawn_in(cx);
        let flat = flat::Flat::install(cx, &listed)?;
        let instances = cx.buffer("instances", Lifetime::Kept);
        let sampler = cx.sampler(engine_core::Sampler::NEAREST);
        let mut pipelines = Vec::with_capacity(listed.len());
        for listed in listed
            .iter()
            .filter(|listed| listed.pass.writes.contains(TargetName::COLOR))
        {
            let pipeline = cx.render_pipeline(
                Scene::named(listed.name),
                &RenderPipeline {
                    vertex: draw::VS_SPRITE,
                    fragment: Some(draw::FS_SPRITE),
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
            pipelines.push((listed.name, pipeline));
        }
        Ok(Self {
            flat,
            pipelines,
            instances,
            staged: Vec::new(),
            runs: Vec::new(),
            drawn: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        ((sprites, table), (globals, poses)): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.flat.prepare(cx);
        stage(
            (&sprites, &table),
            (&globals, &poses),
            &mut self.staged,
            &mut self.runs,
        );
        let bytes = std::mem::size_of_val(self.staged.as_slice()) as u64;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.instances, bytes);
        cx.write(self.instances, 0, &self.staged);
        self.drawn.clear();
        for (name, pipeline) in &self.pipelines {
            for (sheet, range) in &self.runs {
                self.drawn
                    .push((*name, cx.fill(pipeline, &[*sheet]), range.clone()));
            }
        }
    }

    fn view(&mut self, view: &ViewFacts, _cx: &mut Prepare<'_, Self>) {
        self.flat.view(view);
    }
}

impl SpritesGpu {
    fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        let name = scene.pass.name();
        let mut drawn = 0;
        for (_, pipeline, range) in self.drawn.iter().filter(|(pass, _, _)| *pass == name) {
            draw.draw(
                pipeline,
                Shape::Pulled {
                    vertices: 0..6,
                    instances: range.clone(),
                    material: None,
                },
            );
            drawn += 1;
        }
        drawn + self.flat.draw(name, draw)
    }
}

fn stage(
    (sprites, table): (&Query<&'static Sprite>, &DrawnTable),
    (globals, poses): (&GlobalTransforms, &DrawnPoses),
    staged: &mut Vec<SpriteInstance>,
    runs: &mut Vec<(TextureHandle, Range<u32>)>,
) {
    staged.clear();
    runs.clear();
    let mut sorted: Vec<(TextureHandle, SpriteInstance)> = table
        .rows(SPRITES)
        .filter_map(|(slot, entity, _)| {
            let sprite = sprites.get(entity).ok()?;
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
                    drawn: [slot, 0, 0, 0],
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
