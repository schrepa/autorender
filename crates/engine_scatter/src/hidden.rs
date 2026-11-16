use engine_core::{Error, Var};
use engine_render::{
    Allocation, Bind, Buffer, ComputePipeline, Count, Encode, Install, Lifetime, PYRAMID_GROUP,
    Pass, Prepare, TargetName, Texture, mip_levels, pyramid_extents, wgpu,
};
use glam::UVec2;

use crate::gpu::ScatterGpu;
use crate::layouts::SCATTER_GROUP;
use crate::shaders::scatter::{cull, hidden as pyramid_shaders};

const PYRAMID_LEVELS: usize = 16;

pub(crate) struct Hidden {
    pub(crate) picks: Buffer<ScatterGpu>,
    pyramid: Texture<ScatterGpu>,
    extents: Vec<Buffer<ScatterGpu>>,
    first: ComputePipeline<ScatterGpu>,
    down: Vec<ComputePipeline<ScatterGpu>>,
    colour: ComputePipeline<ScatterGpu>,
    written: Vec<[u32; 4]>,
    view: Option<u32>,
    pub(crate) culled: bool,
    pub(crate) coloured: Option<u32>,
}

fn base(render: UVec2) -> (u32, u32) {
    (
        render.x.div_ceil(2).max(1).next_power_of_two(),
        render.y.div_ceil(2).max(1).next_power_of_two(),
    )
}

pub(crate) fn levels(render: UVec2) -> u32 {
    let (width, height) = base(render);
    mip_levels(width, height).min(PYRAMID_LEVELS as u32)
}

impl Hidden {
    pub(crate) fn install(
        cx: &mut Install<'_, ScatterGpu>,
        lay: Pass<ScatterGpu>,
        mut bound: Vec<(Var, Bind<ScatterGpu>)>,
    ) -> Result<Self, Error> {
        let picks = cx.buffer("picks", Lifetime::Kept);
        let pyramid = cx.texture(
            "hidden_pyramid",
            wgpu::TextureFormat::R32Float,
            Lifetime::Kept,
        );
        let extents: Vec<Buffer<ScatterGpu>> = (0..PYRAMID_LEVELS)
            .map(|_| cx.buffer("hidden_extent", Lifetime::Kept))
            .collect();
        let slice = |level: u32| Bind::Slice {
            texture: pyramid,
            mips: level..level + 1,
            layers: 0..1,
        };
        let first = cx.compute_pipeline(
            lay,
            pyramid_shaders::HIDDEN_FIRST,
            &[
                (
                    pyramid_shaders::SOURCE_DEPTH,
                    Bind::Target(TargetName::DEPTH),
                ),
                (pyramid_shaders::TARGET_LEVEL, slice(0)),
                (pyramid_shaders::EXTENT, Bind::Buffer(extents[0])),
            ],
        )?;
        let mut down = Vec::with_capacity(PYRAMID_LEVELS - 1);
        for level in 1..PYRAMID_LEVELS as u32 {
            down.push(cx.compute_pipeline(
                lay,
                pyramid_shaders::HIDDEN_DOWN,
                &[
                    (pyramid_shaders::SOURCE_LEVEL, slice(level - 1)),
                    (pyramid_shaders::TARGET_LEVEL, slice(level)),
                    (
                        pyramid_shaders::EXTENT,
                        Bind::Buffer(extents[level as usize]),
                    ),
                ],
            )?);
        }
        bound.push((cull::PICKS, Bind::Buffer(picks)));
        bound.push((cull::HIDDEN_PYRAMID, Bind::Texture(pyramid)));
        let colour = cx.compute_pipeline(lay, cull::CULL_COLOUR, &bound)?;
        Ok(Self {
            picks,
            pyramid,
            extents,
            first,
            down,
            colour,
            written: Vec::new(),
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

    pub(crate) fn prepare_view(
        &mut self,
        cx: &mut Prepare<'_, ScatterGpu>,
        (index, render): (u32, UVec2),
    ) {
        let (width, height) = base(render);
        let count = levels(render);
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate(
            self.pyramid,
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            count,
        );
        let wanted = pyramid_extents((render.x, render.y), count);
        for (level, extent) in wanted.iter().enumerate() {
            let buffer = self.extents[level];
            let fresh = cx.allocate_buffer(buffer, 16) == Allocation::Fresh;
            if fresh || self.written.get(level) != Some(extent) {
                cx.write(buffer, 0, extent);
            }
        }
        self.written = wanted;
        self.view = Some(index);
    }

    pub(crate) fn encode(&mut self, cx: &mut Encode<'_, ScatterGpu>, objects: u32) {
        let index = cx.view_index();
        if self.view != Some(index) || !self.culled || objects == 0 {
            return;
        }
        let mut built = true;
        for (level, &[width, height, _, _]) in self.written.iter().enumerate() {
            let pipeline = match level.checked_sub(1) {
                None => &self.first,
                Some(below) => match self.down.get(below) {
                    Some(pipeline) => pipeline,
                    None => return,
                },
            };
            let groups = [width.div_ceil(PYRAMID_GROUP), height.div_ceil(PYRAMID_GROUP), 1];
            built &= cx.dispatch(pipeline, Count::Groups(groups)) != 0;
        }
        if !built || self.written.is_empty() {
            return;
        }
        let groups = [objects.div_ceil(SCATTER_GROUP), 1, 1];
        if cx.dispatch(&self.colour, Count::Groups(groups)) != 0 {
            self.coloured = Some(index);
        }
    }
}
