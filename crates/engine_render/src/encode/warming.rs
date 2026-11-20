use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Asked {
    Drawn,
    Warmed,
}

impl Asked {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Drawn => "needed",
            Self::Warmed => "later",
        }
    }
}

#[derive(Default)]
pub(crate) struct Warming {
    queue: std::collections::VecDeque<u32>,
    pub(super) fed: Option<(u64, u64)>,
    open: bool,
    pub(super) made: u32,
    pub(super) stood_in: u64,
    passes: Vec<bool>,
    running: Vec<bool>,
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

    #[cfg(feature = "dev")]
    pub(crate) fn stood_in(&self) -> u64 {
        self.stood_in
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
        while let Some(pipeline) = self.pipelines.warming.queue.pop_front() {
            let _ = self.compute_with(pipeline, Asked::Warmed);
            if self.pipelines.making.pending() > 0 {
                return;
            }
        }
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
        self.reached = Reached {
            generation: Some(states.map_or(0, engine_registry::BrickStates::generation)),
            parts: self
                .installs
                .gates
                .iter()
                .map(|gate| gate.is_none_or(|gate| states.is_none_or(|states| states.on(gate))))
                .collect(),
        };
        let mut passes = vec![true; self.machine.declared.passes.len()];
        for &(pass, gate) in &self.installs.pass_gates {
            let on = states.is_none_or(|states| states.on(gate));
            if let Some(reached) = passes.get_mut(pass as usize) {
                *reached &= on;
            }
        }
        let warming = &mut self.pipelines.warming;
        warming.queue.clear();
        warming.passes = passes;
        warming.running.clear();
        self.reach(false);
    }

    pub(super) fn pass_reached(&self, pass: u32) -> bool {
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

    fn mesh_drawn(&self, pipeline: usize) -> bool {
        let Some(built) = self.machine.pipelines.get(pipeline) else {
            return false;
        };
        let Program::Render(desc) = &built.decl.program else {
            return false;
        };
        !built.geometry.is_empty() && desc.per_batch != crate::PerBatch::NONE
    }

    fn reach(&mut self, now: bool) {
        for pipeline in 0..self.machine.pipelines.len() {
            if !self.pipeline_reached(pipeline) {
                continue;
            }
            if matches!(
                self.machine.pipelines[pipeline].decl.program,
                Program::Compute(_)
            ) {
                if now {
                    let _ = self.compute_with(pipeline as u32, Asked::Drawn);
                } else {
                    self.pipelines.warming.queue.push_back(pipeline as u32);
                }
            } else if !self.mesh_drawn(pipeline) {
                let mut plain = std::mem::take(&mut self.machine.pipelines[pipeline].plain);
                self.extend(&mut plain);
                self.machine.pipelines[pipeline].plain = plain;
            }
        }
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
                if built.decl.scope == Scoped::Pass(pass as u32)
                    && matches!(built.decl.program, Program::Compute(_))
                {
                    let _ = self.compute_with(pipeline as u32, Asked::Drawn);
                }
            }
        }
    }

    #[cfg(feature = "dev")]
    pub(crate) fn warm(&mut self) {
        self.reach(true);
    }
}
