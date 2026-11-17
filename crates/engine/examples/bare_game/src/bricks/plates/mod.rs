engine::brick!(plates);

use engine_core::{
    Attributes, Brick, Bricks, DrawKind, Error, KindDecl, Pass, PassName, PassViews, SlotSet,
    Source, SurfaceSlot, Writes,
};
use engine_render::{
    Draw, Form, FormSide, FragmentUse, GpuPart, Install, KindForm, Prepare, RenderBricks, Scene,
    SlotModule, Varyings,
};

pub const PLATE: DrawKind = DrawKind("plate");

pub const PLATE_MODULE: &str = "plate";

const FORM: KindForm = KindForm {
    kind: PLATE,
    placed: FormSide::pulled(shaders::plates::draw::VS_PLATE),
    varyings: Varyings::Surface(FragmentUse::Masked),
};

pub struct PlatesPass;

impl PassName for PlatesPass {
    const NAME: &'static str = "plates";
    const PASS: Pass = Pass {
        writes: Writes::LIT,
        reads: SlotSet::ALL,
        views: PassViews::Main,
    };
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
                passes: &[PlatesPass::NAME],
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
            .draws::<PlatesGpu>(PLATE, PlatesGpu::draw);
    }
}

struct PlatesGpu {
    form: Form<Self>,
}

impl GpuPart for PlatesGpu {
    type Reads = ();

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        Ok(Self {
            form: cx.form(&FORM, &[])?,
        })
    }

    fn prepare(&mut self, (): (), _: &mut Prepare<'_, Self>) {}
}

impl PlatesGpu {
    fn draw(&self, draw: &mut Draw<'_, Self, Scene>) {
        if let Some(pipeline) = draw.form(&self.form) {
            draw.batches(&pipeline, 0..6);
        }
    }
}
