use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParamItem;
use engine_core::{AttachPoint, ColorPass, CullPlan, DrawKind, Entry, Error, gpu_bits};
use engine_places_rendering::impostors::{ImpostorRecord, ImpostorRecords};
use engine_render::{
    Allocation, Attach, Bind, Buffer, ComputePipeline, Count, Draw, DrawInput, Encode, FaceCull,
    Form, FormSide, GpuPart, Install, KindForm, Lifetime, Own, OwnFragments, Pass, Pipeline,
    Prepare, RoleFragments, Scene, SceneDraw, Shape, Varyings, wgpu,
};

use crate::meshes::MeshCards;
use crate::shaders::impostors::{cards, counts};

pub(crate) const CARD: DrawKind = DrawKind("impostor");

const DRAW_ARGS: u64 = size_of::<wgpu::util::DrawIndirectArgs>() as u64;

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CardFrames {
    #[bits(8)]
    pub first: u32,
    #[bits(8)]
    pub second: u32,
    #[bits(8)]
    pub third: u32,
    #[bits(8)]
    pub origin: u32,
}

const CARD_FRAGMENTS: OwnFragments = OwnFragments {
    placed: RoleFragments {
        lit: Some(cards::FS_CARD),
        depth: Some(cards::FS_CARD_DEPTH),
        blended: None,
        shown: Some(cards::FS_DEBUG_CARD),
    },
    owned: RoleFragments::NONE,
    cull: FaceCull::None,
};

const fn card_form(vertex: Entry) -> KindForm {
    KindForm {
        kind: CARD,
        placed: FormSide {
            vertex,
            streams: &[],
            depth_bias: FormSide::UNBIASED,
        },
        owned: None,
        varyings: Varyings::Own(&CARD_FRAGMENTS),
    }
}

const CARD_FORM: KindForm = card_form(cards::VS_CARD);
const FADING_CARD_FORM: KindForm = card_form(cards::VS_FADING_CARD);
const LATE_CARD_FORM: KindForm = card_form(cards::VS_LATE_CARD);

pub(crate) struct CardGpu {
    counted: Pass<Self>,
    late_counted: Pass<Self>,
    late: Pass<Self>,
    records: Buffer<Self>,
    args: Buffer<Self>,
    late_args: Buffer<Self>,
    count: ComputePipeline<Self>,
    count_late: ComputePipeline<Self>,
    late_pipeline: Pipeline<Self, Own>,
    late_input: DrawInput<Self, Own>,
    placed: Form<Self>,
    fading: Form<Self>,
    inputs: Vec<(&'static str, DrawInput<Self, Scene>)>,
    drawn: bool,
}

impl GpuPart for CardGpu {
    type Reads = (
        Query<'static, 'static, &'static ImpostorRecords, With<MeshCards>>,
        Res<'static, CullPlan>,
    );
    const NAME: &'static str = "mesh_cards";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let records = cx.buffer("mesh_card_records", Lifetime::Kept);
        let args = cx.buffer("mesh_card_args", Lifetime::Kept);
        let late_args = cx.buffer("mesh_card_late_args", Lifetime::Kept);
        let counted = cx.pass("mesh_cards").id();
        let late_counted = cx
            .pass("mesh_cards_late")
            .at(AttachPoint::LateOpaque)
            .id();
        let late = cx
            .pass("mesh_cards_occluded")
            .at(AttachPoint::LateOpaque)
            .after(late_counted)
            .id();
        let count = cx.compute_pipeline(
            counted,
            counts::COUNT_CARDS,
            &[(counts::CARD_ARGS, Bind::Buffer(args))],
        )?;
        let count_late = cx.compute_pipeline(
            late_counted,
            counts::COUNT_LATE_CARDS,
            &[(counts::LATE_CARD_ARGS, Bind::Buffer(late_args))],
        )?;
        let bound = [(cards::CARD_RECORDS, Bind::Buffer(records))];
        let late_pipeline = cx.form_as::<ColorPass>(late, &LATE_CARD_FORM, &bound)?;
        let late_input = cx.draw_input(late, late_args);
        let placed = cx.form(&CARD_FORM, &bound)?;
        let fading = cx.form(&FADING_CARD_FORM, &bound)?;
        let inputs = placed
            .passes()
            .map(|pass| (pass.scene.name(), cx.draw_input(pass.scene, args)))
            .collect();
        Ok(Self {
            counted,
            late_counted,
            late,
            records,
            args,
            late_args,
            count,
            count_late,
            late_pipeline,
            late_input,
            placed,
            fading,
            inputs,
            drawn: false,
        })
    }

    fn prepare(
        &mut self,
        (records, plan): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let held = records
            .single()
            .map_or(&[][..], |records| records.records.as_slice());
        self.drawn = held.iter().any(ImpostorRecord::framed);
        let sizes = [
            (
                self.records,
                (held.len().max(1) * size_of::<ImpostorRecord>()) as u64,
            ),
            (self.args, 2 * DRAW_ARGS),
            (self.late_args, DRAW_ARGS),
        ];
        for (buffer, bytes) in sizes {
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(buffer, bytes);
        }
        if !self.drawn {
            return;
        }
        cx.write(self.records, 0, held);
        cx.run(self.counted);
        if plan.occlusion {
            cx.run(self.late_counted);
            cx.run(self.late);
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.counted {
            cx.dispatch(&self.count, Count::Groups([1, 1, 1]));
        } else if pass == self.late_counted {
            cx.dispatch(&self.count_late, Count::Groups([1, 1, 1]));
        } else if pass == self.late {
            let mut draw = cx.render(Attach::Declared);
            draw.draw(
                &self.late_pipeline,
                Shape::Indirect {
                    vertices: &[],
                    indices: None,
                    args: self.late_input,
                    offset: 0,
                    material: None,
                },
            );
            return 1;
        }
        0
    }
}

impl CardGpu {
    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if scene.view.is_some() {
            return draw.skip("impostor cards are listed for the main view only");
        }
        if !self.drawn {
            return draw.skip("no impostor is framed");
        }
        let (Some(placed), Some(fading)) = (draw.form(&self.placed), draw.form(&self.fading))
        else {
            return draw.skip("impostor cards draw opaque");
        };
        let Some(&(_, args)) = self
            .inputs
            .iter()
            .find(|(name, _)| *name == scene.pass.name())
        else {
            return draw.skip("pass has no impostor card list");
        };
        for (pipeline, offset) in [(placed, 0), (fading, DRAW_ARGS)] {
            draw.draw(
                &pipeline,
                Shape::Indirect {
                    vertices: &[],
                    indices: None,
                    args,
                    offset,
                    material: None,
                },
            );
        }
        2
    }
}
