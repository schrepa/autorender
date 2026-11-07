use engine_core::{MaterialHandle, MeshHandle};
use engine_render::{Draw, MeshPass, Pipeline, Scene, SceneDraw, Shape};

use crate::debug::{DrawKind, DrawRecord};
use crate::gpu::{Drawing, Looked, MeshesGpu, Streams, ViewStreams};
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
            Drawing::Lit {
                looks,
                card,
                streams,
            } => {
                self.draw_meshes(draw, streams, looks, (PHASE_ONE, false))
                    + self.draw_cards(draw, streams, card, PHASE_ONE)
            }
            Drawing::Depth { looks, streams } => {
                if !self.state.draws.masked {
                    return 0;
                }
                self.draw_meshes(draw, streams, looks, (PHASE_ONE, true))
            }
            Drawing::Casters { casting, streams } => {
                let Some(slot) = self.state.view_slot(scene.pass.name(), scene.view) else {
                    return 0;
                };
                self.draw_casters(draw, streams, casting, slot)
            }
            Drawing::Views { looks, streams } => {
                let Some(slot) = self.state.view_slot(scene.pass.name(), scene.view) else {
                    return 0;
                };
                self.draw_view(draw, streams, &looks[usize::from(slot.mirrored)], slot)
            }
        }
    }

    pub(crate) fn record(&self, kind: DrawKind, level: usize) -> u64 {
        DrawRecord::of(kind, level, self.state.casts).offset()
    }

    pub(crate) fn draw_meshes<S: 'static>(
        &self,
        draw: &mut Draw<'_, Self, S>,
        streams: &Streams<S>,
        looks: &Looked<Pipeline<Self, S>>,
        (phase, prepass): (usize, bool),
    ) -> u32 {
        let mut drawn = 0;
        for item in &self.state.draws.items {
            if prepass && !item.look.masked {
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
                    || draw_label((mesh, material), (phase, level), prepass),
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

    fn draw_casters(
        &self,
        draw: &mut Draw<'_, Self, Scene>,
        streams: &ViewStreams,
        casting: &[Pipeline<Self, Scene>; 2],
        slot: &ViewSlot,
    ) -> u32 {
        let mut drawn = 0;
        for item in &self.state.draws.items {
            let masked = item.look.masked;
            for level in slot.levels(item.levels) {
                let (ids, args) = self.state.groups.view_source(slot, item, level);
                draw.draw(
                    &casting[usize::from(masked)],
                    Shape::MeshIndirect {
                        mesh: item.mesh,
                        material: item.material.filter(|_| masked),
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
    prepass: bool,
) -> String {
    let material =
        material.map_or_else(|| "none".to_owned(), |material| material.slot().to_string());
    let mesh = mesh.slot();
    let stage = if prepass { " depth pre-pass" } else { "" };
    format!("mesh {mesh} material {material} phase {phase} level {level}{stage}")
}
