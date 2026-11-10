use engine_core::{GeometryHandle, MaterialHandle};
use engine_render::{Draw, Pipeline, Scene, SceneDraw, Shape};

use crate::debug::DrawRecord;
use crate::gpu::{BlendedStreams, Drawing, Forms, MeshesGpu, Streams, ViewStreams};
use crate::layouts::LOD_LEVELS;
use crate::presentation::{PHASE_ONE, PHASE_TWO, ViewSlot};

type ScenePipelines = Forms<Option<Pipeline<MeshesGpu, Scene>>>;

impl MeshesGpu {
    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        self.draw_kind(scene, draw, false)
    }

    pub(crate) fn draw_skins(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        self.draw_kind(scene, draw, true)
    }

    fn draw_kind(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>, skinned: bool) -> u32 {
        if self.state.instance_count == 0 {
            return 0;
        }
        let Some(joined) = draw.per_pass(&self.scenes) else {
            return 0;
        };
        let pipelines = Forms {
            plain: draw.form(&self.forms.plain),
            extended: draw.form(&self.forms.extended),
            skin: draw.form(&self.forms.skin),
        };
        match joined {
            Drawing::Placed(streams) => {
                let opaque = self
                    .state
                    .draws
                    .items
                    .iter()
                    .any(|item| !item.blended && item.look.skinned == skinned);
                if !opaque {
                    return draw.skip("every material blends");
                }
                self.draw_meshes(draw, streams, &pipelines, (PHASE_ONE, skinned, false))
            }
            Drawing::Views(streams) => {
                let Some(slot) = self.state.view_slot(scene.view.and(scene.index)) else {
                    return draw.skip("view has no visible list");
                };
                self.draw_view(draw, (streams, &pipelines), slot, skinned)
            }
            Drawing::Blended(streams) => {
                let blended = self
                    .state
                    .draws
                    .items
                    .iter()
                    .any(|item| item.blended && item.look.skinned == skinned);
                if !blended {
                    return draw.skip("no material blends");
                }
                self.draw_blended(draw, (streams, &pipelines), skinned)
            }
        }
    }

    fn draw_blended(
        &self,
        draw: &mut Draw<'_, Self, Scene>,
        (streams, pipelines): (&BlendedStreams, &ScenePipelines),
        skinned: bool,
    ) -> u32 {
        let first = self.draw_meshes(draw, &streams.one, pipelines, (PHASE_ONE, skinned, true));
        if !self.state.occlusion {
            return first;
        }
        first + self.draw_meshes(draw, &streams.two, pipelines, (PHASE_TWO, skinned, true))
    }

    pub(crate) fn record(&self, level: usize) -> u64 {
        DrawRecord::of(level).offset()
    }

    pub(crate) fn draw_meshes<S: 'static>(
        &self,
        draw: &mut Draw<'_, Self, S>,
        streams: &Streams<S>,
        pipelines: &Forms<Option<Pipeline<Self, S>>>,
        (phase, skinned, blended): (usize, bool, bool),
    ) -> u32 {
        let mut drawn = 0;
        for item in &self.state.draws.items {
            if item.blended != blended || item.look.skinned != skinned {
                continue;
            }
            let Some(pipeline) = pipelines.of(item.look) else {
                continue;
            };
            let (mesh, material) = (item.mesh, item.material);
            for level in 0..LOD_LEVELS as usize {
                if item.levels & (1 << level) == 0 {
                    continue;
                }
                let (ids, args) =
                    self.state
                        .groups
                        .draw_source(item.start, item.batch as usize, level);
                draw.draw_timed(
                    pipeline,
                    Shape::MeshIndirect {
                        mesh,
                        material,
                        args: streams.args,
                        offset: args,
                        streams: &[(streams.ids, ids), (streams.records, self.record(level))],
                    },
                    || draw_label((mesh, material), (phase, level)),
                );
                drawn += 1;
            }
        }
        drawn
    }

    fn draw_view(
        &self,
        draw: &mut Draw<'_, Self, Scene>,
        (streams, pipelines): (&ViewStreams, &ScenePipelines),
        slot: &ViewSlot,
        skinned: bool,
    ) -> u32 {
        let mut drawn = 0;
        for item in &self.state.draws.items {
            if item.look.skinned != skinned {
                continue;
            }
            let Some(pipeline) = pipelines.of(item.look) else {
                continue;
            };
            for level in slot.levels(item.levels) {
                let (ids, args) = self.state.groups.view_source(slot, item, level);
                draw.draw(
                    pipeline,
                    Shape::MeshIndirect {
                        mesh: item.mesh,
                        material: item.material,
                        args: streams.args,
                        offset: args,
                        streams: &[(streams.ids, ids)],
                    },
                );
                drawn += 1;
            }
        }
        drawn
    }
}

fn draw_label(
    (mesh, material): (GeometryHandle, Option<MaterialHandle>),
    (phase, level): (usize, usize),
) -> String {
    let material =
        material.map_or_else(|| "none".to_owned(), |material| material.slot().to_string());
    let mesh = mesh.slot();
    format!("mesh {mesh} material {material} phase {phase} level {level}")
}
