use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParamItem;
use engine_core::{
    AttachPoint, CullPlan, DrawKind, Entry, Error, PassViews, ScenePass, TargetName, gpu_bits,
};
use engine_places_rendering::impostors::{ImpostorRecord, ImpostorRecords};
use engine_render::{
    Allocation, Attach, Bind, Buffer, ComputePipeline, Count, DEBUG_BLEND, DEBUG_TARGETS,
    DepthUse, Draw, DrawInput, Encode, GpuPart, Install, Lifetime, Own, Pass, Pipeline, Prepare,
    RenderPipeline, Scene, SceneDraw, Scope, Shape, wgpu,
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

type Placed = (
    &'static str,
    [Pipeline<CardGpu, Scene>; 2],
    DrawInput<CardGpu, Scene>,
);

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
    scenes: Vec<Placed>,
    drawn: bool,
}

fn card_pipeline<S: Scope<CardGpu>>(
    cx: &mut Install<'_, CardGpu>,
    scope: S,
    (vertex, fragment, shown): (Entry, Entry, bool),
    records: Buffer<CardGpu>,
) -> Result<Pipeline<CardGpu, S::Tag>, Error> {
    cx.render_pipeline(
        scope,
        &RenderPipeline {
            vertex,
            fragment: Some(fragment),
            vertices: &[],
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth: DepthUse::Write {
                bias: wgpu::DepthBiasState::default(),
            },
            blend: shown.then_some(DEBUG_BLEND),
            writes: wgpu::ColorWrites::ALL,
        },
        &[(cards::CARD_RECORDS, Bind::Buffer(records))],
    )
}

fn install_scenes(
    cx: &mut Install<'_, CardGpu>,
    records: Buffer<CardGpu>,
    args: Buffer<CardGpu>,
) -> Result<Vec<Placed>, Error> {
    let listed: Vec<ScenePass> = cx
        .passes()
        .listed()
        .iter()
        .filter(|listed| {
            listed.bit.is_some()
                && cx.passes().takes(listed, CARD)
                && listed.pass.views != PassViews::Owned
                && listed.pass.at != AttachPoint::Transparent
        })
        .copied()
        .collect();
    let mut scenes = Vec::with_capacity(listed.len());
    for listed in listed {
        let scene = Scene::named(listed.name);
        let writes = listed.pass.writes;
        let shown = writes.contains(DEBUG_TARGETS[0]);
        let fragment = if shown {
            cards::FS_DEBUG_CARD
        } else if writes.contains(TargetName::COLOR) {
            cards::FS_CARD
        } else {
            cards::FS_CARD_DEPTH
        };
        let pipelines = [
            card_pipeline(cx, scene, (cards::VS_CARD, fragment, shown), records)?,
            card_pipeline(cx, scene, (cards::VS_FADING_CARD, fragment, shown), records)?,
        ];
        scenes.push((listed.name, pipelines, cx.draw_input(scene, args)));
    }
    Ok(scenes)
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
        let late_pipeline =
            card_pipeline(cx, late, (cards::VS_LATE_CARD, cards::FS_CARD, false), records)?;
        let late_input = cx.draw_input(late, late_args);
        let scenes = install_scenes(cx, records, args)?;
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
            scenes,
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
                },
            );
            return 1;
        }
        0
    }
}

impl CardGpu {
    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if !self.drawn {
            return 0;
        }
        let Some((_, pipelines, args)) = self
            .scenes
            .iter()
            .find(|(name, _, _)| *name == scene.pass.name())
        else {
            return 0;
        };
        for (pipeline, offset) in pipelines.iter().zip([0, DRAW_ARGS]) {
            draw.draw(
                pipeline,
                Shape::Indirect {
                    vertices: &[],
                    indices: None,
                    args: *args,
                    offset,
                },
            );
        }
        2
    }
}
