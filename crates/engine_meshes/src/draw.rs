use engine_places::probes::{ProbeView, ProbeViews};
use engine_places::shadows::{CasterView, ShadowCasters};
use engine_core::{MaterialHandle, MeshHandle};
use engine_render::{Draw, MainView, MeshPass, Shape};

use crate::casters::{map_factor, map_start};
use crate::debug::{DrawKind, DrawRecord};
use crate::gpu::MeshesGpu;
use crate::layouts::{DRAW_ARGS_BYTES, LOD_LEVELS, MOVING_MAP};
use crate::presentation::{PHASE_ONE, PHASE_TWO};

const FADE_CARDS_START: u64 = 16;

impl MeshesGpu {
    pub(crate) fn draw_prepass(&self, _: &(), draw: &mut Draw<'_, Self, MainView>) -> u32 {
        if self.state.instance_count == 0 || !self.state.draws.masked {
            return 0;
        }
        self.draw_meshes(draw, PHASE_ONE, true)
    }

    pub(crate) fn draw_opaque(&self, _: &(), draw: &mut Draw<'_, Self, MainView>) -> u32 {
        if self.state.instance_count == 0 {
            return 0;
        }
        self.draw_meshes(draw, PHASE_ONE, false) + self.draw_cards(draw, PHASE_ONE)
    }

    pub(crate) fn draw_second(&self, _: &(), draw: &mut Draw<'_, Self, MainView>) -> u32 {
        if !self.state.occlusion || self.state.instance_count == 0 {
            return 0;
        }
        let mut drawn = 0;
        if self.state.draws.masked {
            drawn += self.draw_meshes(draw, PHASE_TWO, true);
        }
        drawn + self.draw_meshes(draw, PHASE_TWO, false) + self.draw_cards(draw, PHASE_TWO)
    }

    fn record(&self, kind: DrawKind, level: usize) -> u64 {
        DrawRecord::of(kind, level, self.state.casts).offset()
    }

    fn draw_meshes(&self, draw: &mut Draw<'_, Self, MainView>, phase: usize, prepass: bool) -> u32 {
        let records = self.inputs.records;
        let mut drawn = 0;
        for item in &self.state.draws.items {
            if prepass && !item.look.masked {
                continue;
            }
            let pipeline = if prepass {
                self.pipelines.prepass.of(item.look)
            } else {
                self.pipelines.opaque.of(item.look)
            };
            let pass = MeshPass {
                opaque: !prepass,
                prepass,
                ..MeshPass::default()
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
                draw.draw_timed_with(
                    pipeline,
                    (
                        Shape::MeshIndirect {
                            mesh,
                            material,
                            args: self.inputs.args[phase],
                            offset: args,
                            streams: &[
                                (self.inputs.ids[phase], ids),
                                (records, self.record(DrawKind::Mesh, level)),
                            ],
                        },
                        pass,
                    ),
                    || draw_label((mesh, material), (phase, level), prepass),
                );
                drawn += 1;
            }
        }
        drawn
    }

    fn draw_cards(&self, draw: &mut Draw<'_, Self, MainView>, phase: usize) -> u32 {
        if !self.state.cards {
            return 0;
        }
        let records = (
            self.inputs.records,
            self.record(DrawKind::Card, LOD_LEVELS as usize),
        );
        let args = u64::from(LOD_LEVELS) * u64::from(self.state.groups.batch_count);
        draw.draw(
            &self.pipelines.card,
            Shape::IndirectAt {
                vertices: &[(self.inputs.card_ids[phase], 0), records],
                indices: None,
                args: self.inputs.args[phase],
                offset: args * DRAW_ARGS_BYTES,
            },
        );
        if phase != PHASE_ONE {
            return 1;
        }
        draw.draw(
            &self.pipelines.card,
            Shape::IndirectAt {
                vertices: &[(self.inputs.fade_cards, FADE_CARDS_START), records],
                indices: None,
                args: self.inputs.fade_cards,
                offset: 0,
            },
        );
        2
    }

    pub(crate) fn draw_probe_face(
        &self,
        at: &ProbeView,
        draw: &mut Draw<'_, Self, ProbeViews>,
    ) -> u32 {
        let face = at.face as usize;
        if !self.state.faces.contains(&at.face) {
            return 0;
        }
        let (Some(&args), Some(&ids)) =
            (self.inputs.face_args.get(face), self.inputs.face_ids.get(face))
        else {
            return 0;
        };
        let mut drawn = 0;
        for item in &self.state.draws.items {
            let (ids_offset, args_offset) =
                self.state
                    .groups
                    .draw_source(item.start, item.batch as usize, 0);
            draw.draw(
                self.pipelines.probe.of(item.look),
                Shape::MeshIndirect {
                    mesh: item.mesh,
                    material: item.material,
                    args,
                    offset: args_offset,
                    streams: &[(ids, ids_offset)],
                },
            );
            drawn += 1;
        }
        drawn
    }

    pub(crate) fn draw_casters(
        &self,
        at: &CasterView,
        draw: &mut Draw<'_, Self, ShadowCasters>,
    ) -> u32 {
        let casters = &self.state.casters;
        if !casters.runs {
            return 0;
        }
        let Some(Some(map)) = casters.maps.get(at.region as usize).copied() else {
            return 0;
        };
        let set = if map == MOVING_MAP as usize {
            &self.pipelines.moving
        } else {
            &self.pipelines.casters
        };
        let groups = casters.groups.len() as u64;
        let start = map_start(map, casters.id_capacity);
        let mut drawn = 0;
        for choice in [false, true] {
            for (index, &(mesh, material, group_start)) in casters.groups.iter().enumerate() {
                if material.is_some() != choice {
                    continue;
                }
                let ids = if casters.regions_in_args {
                    0
                } else {
                    (start + map_factor(map) as usize * group_start as usize) as u64 * 4
                };
                draw.draw(
                    &set[usize::from(choice)],
                    Shape::MeshIndirect {
                        mesh,
                        material,
                        args: self.inputs.caster_args,
                        offset: (map as u64 * groups + index as u64) * DRAW_ARGS_BYTES,
                        streams: &[(self.inputs.caster_ids, ids)],
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
