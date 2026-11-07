use engine_core::{MaterialHandle, MeshHandle};
use engine_render::{Draw, MeshPass, Pipeline, Scene, SceneDraw, Shape};

use crate::debug::{DrawKind, DrawRecord};
use crate::gpu::{BlendedStreams, Drawing, Looked, MeshesGpu, Streams, ViewStreams};
use crate::layouts::{DRAW_ARGS_BYTES, LOD_LEVELS};
use crate::presentation::{PHASE_ONE, ViewSlot};

const FADE_CARDS_START: u64 = 16;

impl MeshesGpu {
    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if self.state.instance_count == 0 {
            return 0;
        }
        let Some(joined) = self
            .scenes
            .iter()
            .find(|joined| joined.name == scene.pass.name())
        else {
            return 0;
        };
        match &joined.drawing {
            Drawing::Placed {
                looks,
                card,
                streams,
            } => {
                self.draw_meshes(draw, streams, looks, PHASE_ONE)
                    + self.draw_cards(draw, streams, card, PHASE_ONE)
            }
            Drawing::Views { looks, streams } => {
                let Some(slot) = self.state.view_slot(scene.pass.name(), scene.view) else {
                    return 0;
                };
                self.draw_view(draw, streams, &looks[usize::from(slot.mirrored)], slot)
            }
            Drawing::Blended { looks, streams } => self.draw_blended(draw, streams, looks),
        }
    }

    fn draw_blended(
        &self,
        draw: &mut Draw<'_, Self, Scene>,
        streams: &BlendedStreams,
        looks: &Looked<Pipeline<Self, Scene>>,
    ) -> u32 {
        let records = (streams.records, self.record(DrawKind::Mesh, 0));
        let mut drawn = 0;
        for batch in &self.state.blended.batches {
            draw.draw(
                looks.of(batch.look),
                Shape::MeshIndirect {
                    mesh: batch.mesh,
                    material: batch.material,
                    args: streams.args,
                    offset: batch.args,
                    streams: &[(streams.ids, batch.ids), records],
                },
            );
            drawn += 1;
        }
        drawn
    }

    pub(crate) fn record(&self, kind: DrawKind, level: usize) -> u64 {
        DrawRecord::of(kind, level).offset()
    }

    pub(crate) fn draw_meshes<S: 'static>(
        &self,
        draw: &mut Draw<'_, Self, S>,
        streams: &Streams<S>,
        looks: &Looked<Pipeline<Self, S>>,
        phase: usize,
    ) -> u32 {
        let mut drawn = 0;
        for item in &self.state.draws.items {
            if item.blended {
                continue;
            }
            let pipeline = looks.of(item.look);
            let (mesh, material) = (item.mesh, item.material);
            for level in 0..LOD_LEVELS as usize {
                if item.levels & (1 << level) == 0 {
                    continue;
                }
                let (ids, args) =
                    self.state
                        .groups
                        .draw_source(item.start, item.batch as usize, level);
                draw.draw_timed_with(
                    pipeline,
                    (
                        Shape::MeshIndirect {
                            mesh,
                            material,
                            args: streams.args,
                            offset: args,
                            streams: &[
                                (streams.ids, ids),
                                (streams.records, self.record(DrawKind::Mesh, level)),
                            ],
                        },
                        MeshPass::default(),
                    ),
                    || draw_label((mesh, material), (phase, level)),
                );
                drawn += 1;
            }
        }
        drawn
    }

    pub(crate) fn draw_cards<S: 'static>(
        &self,
        draw: &mut Draw<'_, Self, S>,
        streams: &Streams<S>,
        card: &Pipeline<Self, S>,
        phase: usize,
    ) -> u32 {
        if !self.state.cards {
            return 0;
        }
        let records = (
            streams.records,
            self.record(DrawKind::Card, LOD_LEVELS as usize),
        );
        let args = u64::from(LOD_LEVELS) * u64::from(self.state.groups.batch_count);
        draw.draw(
            card,
            Shape::IndirectAt {
                vertices: &[(streams.cards, 0), records],
                indices: None,
                args: streams.args,
                offset: args * DRAW_ARGS_BYTES,
            },
        );
        if phase != PHASE_ONE {
            return 1;
        }
        draw.draw(
            card,
            Shape::IndirectAt {
                vertices: &[(streams.fading, FADE_CARDS_START), records],
                indices: None,
                args: streams.fading,
                offset: 0,
            },
        );
        2
    }

    fn draw_view(
        &self,
        draw: &mut Draw<'_, Self, Scene>,
        streams: &ViewStreams,
        looks: &Looked<Pipeline<Self, Scene>>,
        slot: &ViewSlot,
    ) -> u32 {
        let mut drawn = 0;
        for item in &self.state.draws.items {
            for level in slot.levels(item.levels) {
                let (ids, args) = self.state.groups.view_source(slot, item, level);
                draw.draw(
                    looks.of(item.look),
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
    (mesh, material): (MeshHandle, Option<MaterialHandle>),
    (phase, level): (usize, usize),
) -> String {
    let material =
        material.map_or_else(|| "none".to_owned(), |material| material.slot().to_string());
    let mesh = mesh.slot();
    format!("mesh {mesh} material {material} phase {phase} level {level}")
}
