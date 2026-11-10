engine::brick!(plates);

use bevy_ecs::prelude::*;
use engine_core::{
    AssetResidency, AttachPoint, Attributes, BlendedPass, Brick, Bricks, ColorPass, DrawKind,
    DrawnPoses, DrawnTable, Error, GlobalTransforms, GpuStruct, KindDecl, KindSkip, MaterialHandle,
    Pass, PassName, PassPurpose, PassViews, SlotSet, Source, SurfaceSlot, Writes, gpu_struct,
};
use engine_render::{
    Allocation, Bind, Buffer, Draw, Form, FormSide, GpuPart, Install, KindForm, Lifetime, Prepare,
    RenderBricks, Scene, SceneDraw, Shape, SlotModule, Varyings,
};

pub const PLATE: DrawKind = DrawKind("plate");

pub const PLATE_MODULE: &str = "plate";

const GENERATED: &[&str] = &[<PlateInstance as GpuStruct>::WGSL];

const SKIPS: &[KindSkip] = &[
    KindSkip {
        pass: ColorPass::NAME,
        reason: "plates draw in their own pass",
    },
    KindSkip {
        pass: BlendedPass::NAME,
        reason: "plates are opaque",
    },
];

const FORM: KindForm = KindForm {
    kind: PLATE,
    placed: FormSide {
        vertex: shaders::plates::draw::VS_PLATE,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    },
    owned: None,
    varyings: Varyings::Surface,
};

gpu_struct! {
    pub struct PlateInstance {
        pub center: [f32; 4],
        pub previous: [f32; 4],
        pub across: [f32; 4],
        pub along: [f32; 4],
        pub drawn: [u32; 4],
    }
}

pub struct PlatesPass;

impl PassName for PlatesPass {
    const NAME: &'static str = "plates";
    const PASS: Pass = Pass {
        writes: Writes::LIT,
        reads: SlotSet::ALL,
        views: PassViews::Main,
        at: AttachPoint::Opaque,
        purpose: PassPurpose::Scene,
        takes: plated,
    };
}

fn plated(kind: &KindDecl) -> bool {
    kind.kind == PLATE
}

pub struct PlatesBrick;

impl Brick for PlatesBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .draw_kind(KindDecl {
                kind: PLATE,
                needs: Attributes::NONE,
                streams: &[],
                skips: SKIPS,
            })
            .pass::<PlatesPass>()
            .material_slot(SlotModule {
                slot: SurfaceSlot::BaseColor.name(),
                name: PLATE_MODULE,
                module: shaders::plates::checker::MODULE,
                maps: &[],
                values: &[],
                reach: 0.0,
            })
            .generated(GENERATED)
            .draws::<PlatesGpu>(PLATE, PlatesGpu::draw);
    }
}

struct PlatesGpu {
    instances: Buffer<Self>,
    form: Form<Self>,
    material: Option<MaterialHandle>,
    staged: Vec<PlateInstance>,
}

impl GpuPart for PlatesGpu {
    type Reads = (
        Res<'static, DrawnTable>,
        (Res<'static, GlobalTransforms>, Res<'static, DrawnPoses>),
        Res<'static, AssetResidency>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::plates::draw;
        let instances = cx.buffer("plates", Lifetime::Kept);
        let form = cx.form(&FORM, &[(draw::PLATES, Bind::Buffer(instances))])?;
        Ok(Self {
            instances,
            form,
            material: None,
            staged: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (table, (globals, poses), residency): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.material = None;
        self.staged.clear();
        for (slot, entity, drawn) in table.rows(PLATE) {
            let Some(pose) = poses.drawn_pose(&globals, entity) else {
                continue;
            };
            let previous = poses.previous_drawn_pose(&globals, entity).unwrap_or(pose);
            if self.material.is_none() {
                self.material = drawn
                    .material
                    .filter(|material| residency.material_resident(*material));
            }
            self.staged.push(PlateInstance {
                center: pose.w_axis.to_array(),
                previous: previous.w_axis.to_array(),
                across: pose.x_axis.to_array(),
                along: pose.z_axis.to_array(),
                drawn: [slot, 0, 0, 0],
            });
        }
        let bytes = self.staged.len().max(1) * std::mem::size_of::<PlateInstance>();
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.instances, bytes as u64);
        cx.write(self.instances, 0, &self.staged);
    }
}

impl PlatesGpu {
    fn draw(&self, _: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        let count = self.staged.len() as u32;
        if count == 0 {
            return draw.skip("no plate has a pose yet");
        }
        let Some(material) = self.material else {
            return draw.skip("the plate material is not resident yet");
        };
        let Some(pipeline) = draw.form(&self.form) else {
            return draw.skip("pass has no plate form");
        };
        draw.draw(
            &pipeline,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..count,
                material: Some(material),
            },
        );
        count
    }
}
