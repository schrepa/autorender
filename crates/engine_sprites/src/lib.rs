include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use std::collections::BTreeMap;
use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::{
    AlphaMode, Asset, AssetLibrary, AssetResidency, AssetUses, Attributes, BASE_COLOR_MAP, Brick,
    Bricks, ComponentKind, Cull, DrawKind, Drawn, DrawnPoses, DrawnTable, Error, ErrorCode,
    GlobalTransforms, KindDecl, KindSkip, LightingModel, MAPPED, Material, MaterialHandle,
    PassRole, Refusals, Sampler, SceneComponents, Schedule, Source, Spawner, SurfaceSlot,
    TextureHandle, TextureRef, gpu_struct,
};
use engine_render::{
    Allocation, Bind, Buffer, Draw, Form, FormSide, GpuPart, Install, KindForm, Lifetime, Prepare,
    RenderBricks, Scene, SceneDraw, Shape, Varyings,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SPRITES: DrawKind = DrawKind("sprites");

pub const SPRITE_PART: &str = "sprite";

const SHEET_CUTOFF: f32 = 0.5;

const GENERATED: &[&str] = &[<SpriteInstance as engine_core::GpuStruct>::WGSL];

const FORM: KindForm = KindForm {
    kind: SPRITES,
    placed: FormSide {
        vertex: shaders::sprites::draw::VS_SPRITE,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    },
    owned: Some(FormSide {
        vertex: shaders::sprites::draw::VS_SPRITE_VIEW,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    }),
    varyings: Varyings::SurfaceExtra,
};

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
                ComponentKind::of::<Sprite>("sprite").schema(sprite_schema),
                None,
            )
            .resource::<SheetMaterials>()
            .systems(Schedule::ViewSync, publish_sheets)
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

enum SheetMaterial {
    Made(MaterialHandle),
    Refused(Error),
}

#[derive(Resource, Default)]
struct SheetMaterials {
    made: BTreeMap<TextureHandle, SheetMaterial>,
}

impl SheetMaterials {
    fn material(&self, sheet: TextureHandle) -> Option<MaterialHandle> {
        match self.made.get(&sheet)? {
            SheetMaterial::Made(material) => Some(*material),
            SheetMaterial::Refused(_) => None,
        }
    }
}

fn sheet_material(sheet: TextureHandle) -> Result<Material, Error> {
    Ok(Material::new([1.0; 4])?
        .with_map(
            BASE_COLOR_MAP,
            TextureRef::new(sheet).with_sampler(Sampler::NEAREST),
        )?
        .with_slot(SurfaceSlot::BaseColor.name(), MAPPED)?
        .with_slot(SurfaceSlot::Opacity.name(), MAPPED)?
        .with_alpha_mode(AlphaMode::Mask {
            cutoff: SHEET_CUTOFF,
        })?
        .with_cull(Cull::None)
        .with_lighting(LightingModel::Unlit))
}

fn sprite_verdict(drawn: &Drawn, sprite: &Sprite, sheets: &SheetMaterials) -> Result<(), Error> {
    if drawn.kind != SPRITES {
        return Err(Error::new(ErrorCode::UsageConflict, "$.sprite")
            .got(drawn.kind.name())
            .nearest(SPRITES.name()));
    }
    if drawn.material.is_some() {
        return Ok(());
    }
    match sheets.made.get(&sprite.sheet) {
        Some(SheetMaterial::Refused(error)) => Err(error.clone()),
        Some(SheetMaterial::Made(_)) | None => Ok(()),
    }
}

fn forget_sprite(In(entity): In<Entity>, mut refusals: ResMut<Refusals>) {
    refusals.accept(entity, SPRITE_PART);
}

fn publish_sheets(
    sprites: Query<(Entity, &Sprite, &Drawn)>,
    (globals, residency): (Res<GlobalTransforms>, Res<AssetResidency>),
    mut assets: ResMut<AssetLibrary>,
    mut sheets: ResMut<SheetMaterials>,
    mut refusals: ResMut<Refusals>,
    own: Query<(Entity, &AssetUses), With<Sheets>>,
    mut spawner: Spawner<SpritesBrick>,
) {
    let mut wanted: Vec<TextureHandle> = sprites
        .iter()
        .filter(|(entity, _, drawn)| drawn.material.is_none() && globals.pose(*entity).is_some())
        .map(|(_, sprite, _)| sprite.sheet)
        .collect();
    wanted.sort_unstable();
    wanted.dedup();
    sheets.made.retain(|sheet, held| {
        let made = match &*held {
            SheetMaterial::Made(material) => Some(*material),
            SheetMaterial::Refused(_) => None,
        };
        let live = made.is_none_or(|material| assets.material(material).is_some());
        let kept = live && wanted.binary_search(sheet).is_ok();
        if let (false, Some(material)) = (kept, made) {
            assets.remove_material(material);
        }
        kept
    });
    for &sheet in &wanted {
        sheets.made.entry(sheet).or_insert_with(|| {
            match sheet_material(sheet).and_then(|material| assets.insert_material(material)) {
                Ok(material) => SheetMaterial::Made(material),
                Err(error) => SheetMaterial::Refused(error),
            }
        });
    }
    for (entity, sprite, drawn) in &sprites {
        refusals.judge(entity, SPRITE_PART, sprite_verdict(drawn, sprite, &sheets));
    }
    let mut held: Vec<Asset> = sheets
        .made
        .values()
        .filter_map(|made| match made {
            SheetMaterial::Made(material) => Some(Asset::Material(*material)),
            SheetMaterial::Refused(_) => None,
        })
        .collect();
    held.sort_unstable();
    match own.single() {
        Ok((entity, uses)) => {
            if uses.assets() != held.as_slice() {
                spawner.insert(entity, AssetUses::new(&residency, held));
            }
        }
        Err(_) => {
            spawner.spawn((Sheets, AssetUses::new(&residency, held)));
        }
    }
}

struct SpriteRun {
    material: MaterialHandle,
    blended: bool,
    instances: Range<u32>,
}

struct SpritesGpu {
    form: Form<Self>,
    instances: Buffer<Self>,
    staged: Vec<SpriteInstance>,
    runs: Vec<SpriteRun>,
    rows: u32,
}

impl GpuPart for SpritesGpu {
    type Reads = (
        (Query<'static, 'static, &'static Sprite>, Res<'static, DrawnTable>),
        (Res<'static, GlobalTransforms>, Res<'static, DrawnPoses>),
        (
            Res<'static, SheetMaterials>,
            Res<'static, AssetLibrary>,
            Res<'static, AssetResidency>,
        ),
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::sprites::draw;
        let instances = cx.buffer("instances", Lifetime::Kept);
        let form = cx.form(&FORM, &[(draw::INSTANCES, Bind::Buffer(instances))])?;
        Ok(Self {
            form,
            instances,
            staged: Vec::new(),
            runs: Vec::new(),
            rows: 0,
        })
    }

    fn prepare(
        &mut self,
        ((sprites, table), (globals, poses), (sheets, assets, residency)): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.rows = stage(
            (&sprites, &table),
            (&globals, &poses),
            (&sheets, &assets, &residency),
            (&mut self.staged, &mut self.runs),
        );
        let bytes = std::mem::size_of_val(self.staged.as_slice()) as u64;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.instances, bytes);
        cx.write(self.instances, 0, &self.staged);
    }
}

impl SpritesGpu {
    fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if self.runs.is_empty() {
            return if self.rows == 0 {
                0
            } else {
                draw.skip("sprite materials are not resident yet")
            };
        }
        let Some(pipeline) = draw.form(&self.form) else {
            return draw.skip("pass has no sprite form");
        };
        let blended = scene.role == PassRole::Blended;
        let mut drawn = 0;
        for run in self.runs.iter().filter(|run| run.blended == blended) {
            draw.draw(
                &pipeline,
                Shape::Pulled {
                    vertices: 0..6,
                    instances: run.instances.clone(),
                    material: Some(run.material),
                },
            );
            drawn += 1;
        }
        match (drawn, blended) {
            (0, true) => draw.skip("no sprite material blends"),
            (0, false) => draw.skip("every sprite material blends"),
            _ => drawn,
        }
    }
}

fn stage(
    (sprites, table): (&Query<&'static Sprite>, &DrawnTable),
    (globals, poses): (&GlobalTransforms, &DrawnPoses),
    (sheets, assets, residency): (&SheetMaterials, &AssetLibrary, &AssetResidency),
    (staged, runs): (&mut Vec<SpriteInstance>, &mut Vec<SpriteRun>),
) -> u32 {
    staged.clear();
    runs.clear();
    let mut rows = 0;
    let mut sorted: Vec<(bool, MaterialHandle, SpriteInstance)> = Vec::new();
    for (slot, entity, drawn) in table.rows(SPRITES) {
        rows += 1;
        let (Ok(sprite), Some(pose)) = (sprites.get(entity), poses.drawn_pose(globals, entity))
        else {
            continue;
        };
        let Some(material) = drawn.material.or_else(|| sheets.material(sprite.sheet)) else {
            continue;
        };
        if !residency.material_resident(material) {
            continue;
        }
        let blended = assets
            .material(material)
            .is_some_and(|held| held.alpha_mode().blended());
        let place = [
            pose.w_axis.x,
            pose.w_axis.y,
            pose.w_axis.z,
            pose.x_axis.truncate().length(),
        ];
        sorted.push((
            blended,
            material,
            SpriteInstance {
                place,
                rect: sprite.rect,
                tint: sprite.tint,
                drawn: [slot, 0, 0, 0],
            },
        ));
    }
    sorted.sort_by_key(|(blended, material, _)| (*blended, *material));
    for (blended, material, instance) in sorted {
        let at = staged.len() as u32;
        staged.push(instance);
        match runs.last_mut() {
            Some(run) if run.material == material => run.instances.end = at + 1,
            Some(_) | None => runs.push(SpriteRun {
                material,
                blended,
                instances: at..at + 1,
            }),
        }
    }
    rows
}
