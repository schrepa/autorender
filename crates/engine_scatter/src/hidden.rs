use engine_core::{Error, Var};
use engine_render::{Bind, Buffer, ComputePipeline, Count, Encode, Install, Lifetime, Pass};

use crate::gpu::ScatterGpu;
use crate::layouts::SCATTER_GROUP;
use crate::shaders::scatter::cull;

pub(crate) struct Hidden {
    pub(crate) picks: Buffer<ScatterGpu>,
    colour: ComputePipeline<ScatterGpu>,
    pub(crate) view: Option<u32>,
    pub(crate) culled: bool,
    pub(crate) coloured: Option<u32>,
}

impl Hidden {
    pub(crate) fn install(
        cx: &mut Install<'_, ScatterGpu>,
        lay: Pass<ScatterGpu>,
        mut bound: Vec<(Var, Bind<ScatterGpu>)>,
    ) -> Result<Self, Error> {
        let picks = cx.buffer("picks", Lifetime::Kept);
        bound.push((cull::PICKS, Bind::Buffer(picks)));
        let colour = cx.compute_pipeline(lay, cull::CULL_COLOUR, &bound)?;
        Ok(Self {
            picks,
            colour,
            view: None,
            culled: false,
            coloured: None,
        })
    }

    pub(crate) fn begin(&mut self) {
        self.view = None;
        self.culled = false;
        self.coloured = None;
    }

    pub(crate) fn encode(&mut self, cx: &mut Encode<'_, ScatterGpu>, objects: u32) {
        let index = cx.view_index();
        if self.view != Some(index) || !self.culled || objects == 0 {
            return;
        }
        let groups = [objects.div_ceil(SCATTER_GROUP), 1, 1];
        if cx.dispatch(&self.colour, Count::Groups(groups)) != 0 {
            self.coloured = Some(index);
        }
    }
}
