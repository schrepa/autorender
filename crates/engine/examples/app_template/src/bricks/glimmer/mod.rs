engine::brick!(glimmer);

use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::{
    Attributes, Brick, Bricks, Camera, Clear, Cursor, DrawKind, Drawn, DrawnTable, EMISSIVE_MAP,
    Error, ErrorCode, Events, Expected, Feed, FrameTime, GpuStruct, Interest, KindDecl, Node, Pass,
    PassName, PassViews, Rect, Refusals, Requests, Schedule, Source, Spawner, SurfaceSlot, Ticket,
    Transform, View, gpu_struct, settings, stats,
};
use engine_render::{
    Allocation, Bind, Buffer, DepthUse, Draw, GpuPart, Install, Lifetime, Prepare, RenderBricks,
    Scene, wgpu,
};

use crate::bricks::scene_pipeline::{ColorPass, LIT};
use crate::bricks::surfaces::{
    FaceCull, Form, FormDraw, FormSide, FragmentUse, KindForm, LightingModule, ModelForm,
    PassFragment, PointInput, SlotModule, SurfaceBricks, SurfaceForms, Varyings,
};

pub const GLIMMER: DrawKind = DrawKind("glimmer");

const GENERATED: &[&str] = &[<GlimmerLook as GpuStruct>::WGSL];

pub const LOOK: &str = "glimmer";

const ADDED_LIGHT: PointInput = PointInput::new("added_light", "vec3<f32>", "vec3<f32>(0.0)");

pub const GLIMMER_PART: &str = "glimmer";

const FORM: KindForm = KindForm {
    kind: GLIMMER,
    placed: FormSide::pulled(shaders::glimmer::draw::VS_GLIMMER),
    varyings: Varyings::SurfaceExtra(FragmentUse::Masked),
};

const RING: PassFragment = PassFragment {
    pass: Scene::of::<GlimmerPass>(),
    side: Some(FormSide::pulled(shaders::glimmer::draw::VS_RING)),
    entry: shaders::glimmer::draw::FS_RING,
    cull: FaceCull::None,
    blend: None,
    depth: DepthUse::Compare {
        compare: wgpu::CompareFunction::GreaterEqual,
        write: false,
        bias: wgpu::DepthBiasState {
            constant: 0,
            slope_scale: 0.0,
            clamp: 0.0,
        },
    },
};

settings! {
    #[section("glimmer")]
    #[switch(on)]
    pub struct Glimmer {
        pub size: f32 = 0.25 => range(0.01, 100.0),
        pub ring: [f32; 4] = [1.0, 0.85, 0.3, 1.0] => range(0.0, 1.0),
        pub events: u32 = 256 => at_least(1),
    }
}

stats! {
    #[section("glimmer_stats")]
    pub struct GlimmerStats {
        pub ready: u32,
        pub lit: u32,
        pub missed: u32,
    }
}

gpu_struct! {
    pub struct GlimmerLook {
        pub ring: [f32; 4],
        pub size: f32,
        pub pad_0: u32,
        pub pad_1: u32,
        pub pad_2: u32,
    }
}

pub struct GlimmerPass;

impl PassName for GlimmerPass {
    const NAME: &'static str = "glimmer";
    const PASS: Pass = Pass {
        writes: LIT,
        views: PassViews::Main,
    };
}

#[derive(Component)]
pub struct GlimmerView;

#[derive(Resource)]
pub struct GlimmerBook {
    sparks: Requests<Entity, f64>,
    held: BTreeMap<Entity, Interest<Entity>>,
    lit: Feed<Entity>,
    seen: Cursor,
}

impl Default for GlimmerBook {
    fn default() -> Self {
        Self {
            sparks: Requests::default(),
            held: BTreeMap::new(),
            lit: Feed::new(1),
            seen: Cursor::default(),
        }
    }
}

pub struct GlimmerBrick;

impl Brick for GlimmerBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .switch::<Glimmer>()
            .stats::<GlimmerStats>()
            .resource::<GlimmerBook>()
            .draw_kind(KindDecl {
                kind: GLIMMER,
                needs: Attributes::NONE,
                streams: &[],
                passes: &[ColorPass::NAME, GlimmerPass::NAME],
            })
            .pass::<GlimmerPass>()
            .material_slot(SlotModule {
                slot: SurfaceSlot::Emissive.name(),
                name: LOOK,
                module: shaders::glimmer::glow::MODULE,
                maps: &[EMISSIVE_MAP],
                values: &[LOOK],
                reach: 0.0,
            })
            .material_slot(SlotModule {
                slot: SurfaceSlot::Opacity.name(),
                name: LOOK,
                module: shaders::glimmer::disc::MODULE,
                maps: &[],
                values: &[],
                reach: 0.0,
            })
            .lighting_model(LightingModule {
                name: LOOK,
                module: shaders::glimmer::bands::MODULE,
                form: ModelForm::Shades,
            })
            .point_input(ADDED_LIGHT)
            .generated(GENERATED)
            .draws::<GlimmerGpu>(GLIMMER, GlimmerGpu::draw)
            .systems(Schedule::FrameStart, (open_view, judge))
            .systems(Schedule::Presentation, (ask, answer));
    }
}

fn open_view(
    table: Res<DrawnTable>,
    opened: Query<(), With<GlimmerView>>,
    mut spawner: Spawner<GlimmerBrick>,
) {
    if !opened.is_empty() || table.batches(GLIMMER).next().is_none() {
        return;
    }
    let camera = spawner.spawn((
        Transform::looking_at(Vec3::new(0.0, 40.0, 30.0), Vec3::ZERO),
        Camera::default(),
        Node::default(),
    ));
    spawner.spawn((
        GlimmerView,
        View {
            viewport: Rect {
                x: 0.75,
                y: 0.0,
                width: 0.25,
                height: 0.25,
            },
            clear: Clear::DepthOnly,
            order: 1,
            ..View::new(camera)
        },
    ));
}

fn judge(drawn: Query<(Entity, &Drawn)>, mut refusals: ResMut<Refusals>) {
    refusals.keep_part(GLIMMER_PART, |entity| {
        drawn
            .get(entity)
            .is_ok_and(|(_, thing)| thing.kind == GLIMMER)
    });
    for (entity, thing) in &drawn {
        if thing.kind == GLIMMER {
            refusals.judge(entity, GLIMMER_PART, glimmer_verdict(thing));
        }
    }
}

fn glimmer_verdict(thing: &Drawn) -> Result<(), Error> {
    match thing.material {
        Some(_) => Ok(()),
        None => Err(Error::new(ErrorCode::MissingField, "$.drawn.material")
            .expected(Expected::Present)
            .got(GLIMMER.name())),
    }
}

fn ask(drawn: Query<(Entity, &Drawn)>, mut book: ResMut<GlimmerBook>) {
    let GlimmerBook { sparks, held, .. } = &mut *book;
    held.retain(|entity, _| {
        drawn
            .get(*entity)
            .is_ok_and(|(_, thing)| thing.kind == GLIMMER)
    });
    for (entity, thing) in &drawn {
        if thing.kind == GLIMMER && !held.contains_key(&entity) {
            held.insert(entity, sparks.ask(entity));
        }
    }
}

fn answer(
    time: Res<FrameTime>,
    settings: Res<Glimmer>,
    mut book: ResMut<GlimmerBook>,
    mut stats: ResMut<GlimmerStats>,
) {
    let GlimmerBook {
        sparks, lit, seen, ..
    } = &mut *book;
    sparks.settle(time.real_elapsed);
    let asked: Vec<(Entity, Ticket)> = sparks
        .asked()
        .map(|(entity, ticket)| (*entity, ticket))
        .collect();
    lit.resize(settings.events as usize);
    for (entity, ticket) in asked {
        if sparks.ready(&entity, ticket, time.real_elapsed) {
            lit.push(entity);
        }
    }
    match lit.events(seen) {
        Events::Listed(listed) => stats.lit += listed.len() as u32,
        Events::Missed(missed) => stats.missed += missed as u32,
    }
    stats.ready = sparks.census().ready;
}

struct GlimmerGpu {
    look: Buffer<Self>,
    sent: Option<GlimmerLook>,
    form: Form<Self>,
}

impl GpuPart for GlimmerGpu {
    type Reads = Res<'static, Glimmer>;

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let look = cx.buffer("glimmer_look", Lifetime::Kept);
        let bound = [(shaders::glimmer::draw::GLIMMER_LOOK, Bind::Buffer(look))];
        Ok(Self {
            look,
            sent: None,
            form: cx.form_owning(&FORM, &[RING], &bound)?,
        })
    }

    fn prepare(
        &mut self,
        settings: SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let look = GlimmerLook {
            ring: settings.ring,
            size: settings.size,
            ..GlimmerLook::default()
        };
        let bytes = size_of::<GlimmerLook>() as u64;
        if cx.allocate_buffer(self.look, bytes) == Allocation::Fresh || self.sent != Some(look) {
            cx.write(self.look, 0, &[look]);
            self.sent = Some(look);
        }
    }
}

impl GlimmerGpu {
    fn draw(&self, draw: &mut Draw<'_, Self, Scene>) {
        if let Some(pipeline) = draw.form(&self.form) {
            draw.batches(&pipeline, 0..6);
        }
    }
}
