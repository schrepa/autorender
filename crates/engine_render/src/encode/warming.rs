use super::*;
use crate::install::Faces;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Asked {
    Drawn,
    Waited,
    Warmed,
}

pub(super) struct WarmShape {
    pub(super) formats: Vec<Option<wgpu::TextureFormat>>,
    pub(super) depth: Option<(wgpu::TextureFormat, bool)>,
    pub(super) shown: (bool, bool),
    pub(super) variant: Option<crate::surface::MaterialVariant>,
}

pub(super) struct Warm {
    pub(super) pipeline: u32,
    pub(super) shape: Option<WarmShape>,
    pub(super) switches: Vec<(u32, bool)>,
    pub(super) listed: bool,
}

impl Warm {
    pub(super) fn same(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline
            && self.switches == other.switches
            && match (&self.shape, &other.shape) {
                (Some(first), Some(second)) => {
                    first.formats == second.formats
                        && first.depth == second.depth
                        && first.shown == second.shown
                        && first.variant == second.variant
                }
                (None, None) => true,
                (Some(_), None) | (None, Some(_)) => false,
            }
    }
}

#[derive(Default)]
pub(crate) struct Warming {
    pub(super) queue: std::collections::VecDeque<Warm>,
    pub(super) fed: Option<(u64, u64)>,
    open: bool,
    pub(super) made: u32,
    passes: Vec<bool>,
    pub(super) running: Vec<bool>,
}

impl Warming {
    pub(crate) fn left(&self) -> u32 {
        u32::try_from(self.queue.len())
            .unwrap_or(u32::MAX)
            .saturating_add(u32::from(self.fed.is_some()))
    }

    pub(crate) fn made(&self) -> u32 {
        self.made
    }

    pub(crate) fn open(&mut self, open: bool) {
        self.open = open;
    }

    pub(super) fn promote(&mut self, made: (u64, u64)) {
        if self.fed == Some(made) {
            self.fed = None;
        }
    }
}

impl Run {
    pub(super) fn feed_warming(&mut self) {
        let warming = &mut self.pipelines.warming;
        if let Some(fed) = warming.fed
            && !self.pipelines.making.waits(fed)
        {
            warming.fed = None;
        }
        if !warming.open || warming.fed.is_some() || self.pipelines.making.pending() > 0 {
            return;
        }
        while let Some(warm) = self.pipelines.warming.queue.pop_front() {
            self.make_warm(warm, Asked::Warmed);
            if self.pipelines.warming.fed.is_some() {
                return;
            }
        }
    }

    pub(super) fn make_warm(&mut self, warm: Warm, asked: Asked) {
        let real = std::mem::replace(&mut self.pipelines.switches, warm.switches);
        self.pipelines.listing = warm.listed;
        match warm.shape {
            Some(shape) => {
                let _ = self.split().0.compile_with(
                    warm.pipeline,
                    (&shape.formats, shape.depth),
                    shape.shown,
                    shape.variant,
                    asked,
                );
            }
            None => {
                let _ = self.compute_with(warm.pipeline, asked);
            }
        }
        self.pipelines.listing = false;
        self.pipelines.switches = real;
    }

    fn part_reached(&self, part: u32) -> bool {
        self.reached
            .parts
            .get(part as usize)
            .copied()
            .unwrap_or(true)
    }

    pub(crate) fn warm_reached(&mut self, world: &World) {
        let states = world.get_resource::<engine_registry::BrickStates>();
        let in_play: Vec<engine_registry::GateId> = self
            .installs
            .gates
            .iter()
            .zip(&self.machine.declared.parts)
            .filter(|(_, part)| !crate::DEV && part.turns_on_in_play)
            .filter_map(|(gate, _)| *gate)
            .collect();
        let side = |forced: bool| -> Vec<(u32, bool)> {
            self.machine
                .switches
                .iter()
                .zip(&self.pipelines.switches)
                .map(|(&(id, gate), &(_, on))| {
                    (id, if in_play.contains(&gate) { forced } else { on })
                })
                .collect()
        };
        self.reached = Reached {
            generation: Some(states.map_or(0, engine_registry::BrickStates::generation)),
            parts: self
                .installs
                .gates
                .iter()
                .map(|gate| {
                    gate.is_none_or(|gate| {
                        in_play.contains(&gate) || states.is_none_or(|states| states.on(gate))
                    })
                })
                .collect(),
            sides: [side(true), side(false)],
        };
        let mut passes = vec![true; self.machine.declared.passes.len()];
        for &(pass, gate) in &self.installs.pass_gates {
            let on = in_play.contains(&gate) || states.is_none_or(|states| states.on(gate));
            if let Some(reached) = passes.get_mut(pass as usize) {
                *reached &= on;
            }
        }
        let warming = &mut self.pipelines.warming;
        warming.queue.clear();
        warming.passes = passes;
        warming.running.clear();
        let making = &self.pipelines.making;
        self.pipelines.standing.retain(|&made| making.waits(made));
        for built in &mut self.machine.pipelines {
            built.ready.clear();
        }
        self.compile_first();
        self.at_reached_switches(|run| {
            run.compile_listed();
            run.queue_reached();
        });
    }

    fn pass_reached(&self, pass: u32) -> bool {
        self.pipelines
            .warming
            .passes
            .get(pass as usize)
            .copied()
            .unwrap_or(true)
            && self
                .machine
                .declared
                .passes
                .get(pass as usize)
                .is_some_and(|declared| self.part_reached(declared.part))
    }

    fn pipeline_reached(&self, pipeline: usize) -> bool {
        let Some(built) = self.machine.pipelines.get(pipeline) else {
            return false;
        };
        if !self.part_reached(built.decl.part) {
            return false;
        }
        let passes = self.scope_passes(pipeline);
        passes.is_empty() || passes.iter().any(|&pass| self.pass_reached(pass))
    }

    fn pass_warms(&self, pipeline: usize, pass: u32) -> Vec<Warm> {
        let switches = &self.pipelines.switches;
        let warm = |shape| Warm {
            pipeline: pipeline as u32,
            shape,
            switches: switches.clone(),
            listed: false,
        };
        let (formats, depth, shown) = self.pass_formats(pass);
        if self.mesh_drawn(pipeline, shown.0) {
            return Vec::new();
        }
        let mut warms = Vec::new();
        for layered in self.layered_formats(pipeline, pass, formats.len()) {
            if self.writes_fit(pipeline, &layered) {
                warms.push(warm(Some(WarmShape {
                    formats: layered,
                    depth: None,
                    shown,
                    variant: None,
                })));
            }
        }
        if warms.is_empty() && self.writes_fit(pipeline, &formats) {
            warms.insert(
                0,
                warm(Some(WarmShape {
                    formats,
                    depth,
                    shown,
                    variant: None,
                })),
            );
        }
        warms
    }

    fn mesh_drawn(&self, pipeline: usize, plain: bool) -> bool {
        let Some(built) = self.machine.pipelines.get(pipeline) else {
            return false;
        };
        let Program::Render(desc) = &built.decl.program else {
            return false;
        };
        let used = crate::surface::MaterialVariant::PLAIN
            .constants()
            .iter()
            .any(|&offered| {
                built
                    .stages
                    .iter()
                    .any(|stage| !stage.composed.constants(&[offered]).is_empty())
            });
        let styled = !plain && !desc.writes.is_empty();
        let faced = desc.faces != Faces::Declared;
        !built.geometry.is_empty() && (used || styled || faced)
    }

    fn reached_warms(&self) -> Vec<Warm> {
        let mut warms = Vec::new();
        for pipeline in 0..self.machine.pipelines.len() {
            if !self.pipeline_reached(pipeline) {
                continue;
            }
            if matches!(
                self.machine.pipelines[pipeline].decl.program,
                Program::Compute(_)
            ) {
                warms.push(Warm {
                    pipeline: pipeline as u32,
                    shape: None,
                    switches: self.pipelines.switches.clone(),
                    listed: false,
                });
                continue;
            }
            for pass in self.scope_passes(pipeline) {
                if self.pass_reached(pass) {
                    warms.extend(self.pass_warms(pipeline, pass));
                }
            }
        }
        warms
    }

    fn queue_reached(&mut self) {
        let warms = self.reached_warms();
        self.pipelines.warming.queue.extend(warms);
    }

    pub(crate) fn warm_running(&mut self) {
        let declared = self.machine.declared.passes.len();
        let warmed = &self.pipelines.warming.running;
        let running: Vec<usize> = self
            .planning
            .runs
            .iter()
            .filter(|&pass| pass < declared && !warmed.get(pass).copied().unwrap_or(false))
            .collect();
        if running.is_empty() {
            return;
        }
        self.pipelines.warming.running.resize(declared, false);
        for pass in running {
            self.pipelines.warming.running[pass] = true;
            if !self.drawable(self.machine.declared.passes[pass].part) {
                continue;
            }
            for pipeline in 0..self.machine.pipelines.len() {
                let built = &self.machine.pipelines[pipeline];
                if built.skipped || built.decl.scope != Scoped::Pass(pass as u32) {
                    continue;
                }
                let warms = match &built.decl.program {
                    Program::Compute(_) => vec![Warm {
                        pipeline: pipeline as u32,
                        shape: None,
                        switches: self.pipelines.switches.clone(),
                        listed: false,
                    }],
                    Program::Render(desc) if desc.faces == Faces::Declared => {
                        self.pass_warms(pipeline, pass as u32)
                    }
                    Program::Render(_) => Vec::new(),
                };
                for warm in warms {
                    self.make_warm(warm, Asked::Drawn);
                }
            }
        }
    }

    pub(super) fn at_reached_switches(&mut self, work: impl Fn(&mut Self)) {
        let [on, off] = self.reached.sides.clone();
        let differ = on != off;
        let real = std::mem::replace(&mut self.pipelines.switches, on);
        work(self);
        if differ {
            self.pipelines.switches = off;
            work(self);
        }
        self.pipelines.switches = real;
    }

    pub(super) fn compile_first(&mut self) {
        if self.pipelines.settled {
            return;
        }
        let Some(keys) = self.pipelines.listed.take() else {
            return;
        };
        for key in &keys {
            let Needed::First(switches) = &key.needed else {
                continue;
            };
            let found = self
                .machine
                .pipelines
                .iter()
                .position(|built| built.program_key == key.program)
                .filter(|&at| !self.machine.pipelines[at].decl.speedup_only)
                .filter(|&at| self.pipeline_reached(at));
            if let Some(pipeline) = found {
                let warm = Warm {
                    pipeline: pipeline as u32,
                    shape: Some(WarmShape {
                        formats: key.formats.clone(),
                        depth: key.depth,
                        shown: (key.plain, key.main),
                        variant: specialized(key),
                    }),
                    switches: switches.clone(),
                    listed: true,
                };
                self.make_warm(warm, Asked::Drawn);
            }
        }
        self.pipelines.listed = Some(keys);
    }

    pub(super) fn compile_listed(&mut self) {
        let Some(keys) = self.pipelines.listed.take() else {
            return;
        };
        for key in &keys {
            let found = self
                .machine
                .pipelines
                .iter()
                .position(|built| built.program_key == key.program)
                .filter(|&at| self.pipeline_reached(at));
            if let Some(pipeline) = found {
                self.pipelines.warming.queue.push_back(Warm {
                    pipeline: pipeline as u32,
                    shape: Some(WarmShape {
                        formats: key.formats.clone(),
                        depth: key.depth,
                        shown: (key.plain, key.main),
                        variant: specialized(key),
                    }),
                    switches: self.pipelines.switches.clone(),
                    listed: true,
                });
            }
        }
        self.pipelines.listed = Some(keys);
    }

    pub(crate) fn warm(&mut self) {
        for warm in self.reached_warms() {
            self.make_warm(warm, Asked::Drawn);
        }
    }
}
