use bevy_ecs::prelude::*;
use engine_core::Error;
use engine_places::ground::PALETTE_ENTRIES;
use engine_render::{
    AfterRead, Bind, Buffer, ComputePipeline, Count, DepthUse, Draw, DrawInput, Encode, GpuPart,
    Install, Lifetime, MainView, Pass, Pipeline, Prepare, RenderPipeline, Shape, wgpu,
};

use crate::covers::CoverTable;
use crate::growth::{BLADE_BYTES, BladeCounts, CoverPlan};
use crate::layouts::{CoverFrame, CoverMaterial, CoverTerrain, Grow};
use crate::shaders::cover::{debug, draw, faces::cover as face, grow};

const DRAW_ARGS: u64 = size_of::<wgpu::util::DrawIndirectArgs>() as u64;
const COUNT_BYTES: u64 = 32;
const ROW_BYTES: u64 = size_of::<Grow>() as u64;
const FRAME_BYTES: u64 = size_of::<CoverFrame>() as u64;
const TERRAIN_BYTES: u64 = size_of::<CoverTerrain>() as u64;
const MATERIAL_BYTES: u64 = PALETTE_ENTRIES as u64 * size_of::<CoverMaterial>() as u64;

pub(crate) struct CoverGpu {
    clear_counts: Pass<Self>,
    grow: Pass<Self>,
    rows: Buffer<Self>,
    blades: Buffer<Self>,
    counts: Buffer<Self>,
    args: Buffer<Self>,
    frame: Buffer<Self>,
    terrains: Buffer<Self>,
    materials: Buffer<Self>,
    growing: ComputePipeline<Self>,
    stepping: ComputePipeline<Self>,
    finishing: ComputePipeline<Self>,
    lit: Pipeline<Self, MainView>,
    indirect: DrawInput<Self, MainView>,
    cells: Vec<u32>,
    blade_bytes: u64,
    row_bytes: u64,
    drawn: bool,
}

impl GpuPart for CoverGpu {
    type Reads = (Res<'static, CoverPlan>, Res<'static, CoverTable>);

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let rows = cx.buffer("cover_grow", Lifetime::Kept);
        let blades = cx.buffer("cover_blades", Lifetime::Kept);
        let counts = cx.buffer("cover_counts", Lifetime::Kept);
        let args = cx.buffer("cover_draw", Lifetime::Kept);
        let frame = cx.buffer("cover_frame", Lifetime::Kept);
        let terrains = cx.buffer("cover_terrains", Lifetime::Kept);
        let materials = cx.buffer("cover_materials", Lifetime::Kept);
        cx.frame(face::COVER_TERRAINS, Bind::Buffer(terrains))?;
        cx.frame(face::COVER_MATERIALS, Bind::Buffer(materials))?;
        let clear_counts = cx.clear("clear_cover_counts", counts);
        let grow_pass = cx.pass("ground_cover").after(clear_counts).id();
        let growing = cx.compute_pipeline(
            grow_pass,
            grow::GROW_COVER,
            &[
                (grow::GROWS, Bind::Buffer(rows)),
                (grow::BLADES, Bind::Buffer(blades)),
                (grow::COUNTS, Bind::Buffer(counts)),
            ],
        )?;
        let stepping = cx.compute_pipeline(
            grow_pass,
            grow::NEXT_GROWTH,
            &[(grow::COUNTS, Bind::Buffer(counts))],
        )?;
        let finishing = cx.compute_pipeline(
            grow_pass,
            grow::FINISH_COVER,
            &[
                (grow::GROWS, Bind::Buffer(rows)),
                (grow::COUNTS, Bind::Buffer(counts)),
                (grow::ARGS, Bind::Buffer(args)),
            ],
        )?;
        let lit = cx.render_pipeline(
            MainView,
            &RenderPipeline {
                vertex: draw::VS_COVER,
                fragment: Some(draw::FS_COVER),
                debug: Some(debug::FS_DEBUG_COVER),
                vertices: &[],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    cull_mode: None,
                    ..Default::default()
                },
                depth: DepthUse::Write {
                    bias: wgpu::DepthBiasState::default(),
                },
                blend: None,
                writes: wgpu::ColorWrites::ALL,
            },
            &[
                (draw::COVER_BLADES, Bind::Buffer(blades)),
                (draw::COVER_FRAME, Bind::Buffer(frame)),
            ],
        )?;
        let indirect = cx.draw_input(MainView, args);
        Ok(Self {
            clear_counts,
            grow: grow_pass,
            rows,
            blades,
            counts,
            args,
            frame,
            terrains,
            materials,
            growing,
            stepping,
            finishing,
            lit,
            indirect,
            cells: Vec::new(),
            blade_bytes: BLADE_BYTES,
            row_bytes: ROW_BYTES,
            drawn: false,
        })
    }

    fn prepare(
        &mut self,
        (plan, table): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if table.is_changed() {
            let terrains = size_of_val(table.terrains.as_slice()) as u64;
            cx.allocate_buffer(self.terrains, terrains.max(TERRAIN_BYTES));
            cx.allocate_buffer(self.materials, MATERIAL_BYTES);
            if !table.terrains.is_empty() {
                cx.write(self.terrains, 0, &table.terrains);
            }
            if !table.materials.is_empty() {
                cx.write(self.materials, 0, &table.materials);
            }
        }
        if !plan.shown {
            self.blade_bytes = BLADE_BYTES;
            self.row_bytes = ROW_BYTES;
        }
        self.drawn = plan.drawn && !plan.rows.is_empty();
        if self.drawn {
            let wanted = u64::from(plan.capacity) * BLADE_BYTES;
            let size = wanted.next_power_of_two().min(plan.room);
            if wanted > self.blade_bytes && size > self.blade_bytes {
                self.blade_bytes = size;
            }
            let rows = size_of_val(plan.rows.as_slice()) as u64;
            if rows > self.row_bytes {
                self.row_bytes = rows.next_power_of_two();
            }
        }
        cx.allocate_buffer(self.counts, COUNT_BYTES);
        cx.allocate_buffer(self.args, 2 * DRAW_ARGS);
        cx.allocate_buffer(self.frame, FRAME_BYTES);
        cx.allocate_buffer(self.blades, self.blade_bytes);
        cx.allocate_buffer(self.rows, self.row_bytes);
        self.cells.clear();
        if !self.drawn {
            return;
        }
        cx.write(self.rows, 0, &plan.rows);
        cx.write(self.frame, 0, &[plan.frame]);
        self.cells.extend_from_slice(&plan.cells);
        cx.run(self.clear_counts);
        cx.run(self.grow);
        cx.read_back::<BladeCounts>(0, self.counts, 0..COUNT_BYTES, AfterRead::Keep);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass != self.grow {
            return 0;
        }
        let mut count = 0;
        for (index, &cells) in self.cells.iter().enumerate() {
            if index > 0 {
                count += cx.dispatch(&self.stepping, Count::Groups([1, 1, 1]));
            }
            count += cx.dispatch(&self.growing, Count::Groups([cells, cells, 1]));
        }
        count + cx.dispatch(&self.finishing, Count::Groups([1, 1, 1]))
    }
}

impl CoverGpu {
    pub(crate) fn draw(&self, _: &(), draw: &mut Draw<'_, Self, MainView>) -> u32 {
        if !self.drawn {
            return 0;
        }
        for offset in [0, DRAW_ARGS] {
            draw.draw(
                &self.lit,
                Shape::Indirect {
                    vertices: &[],
                    indices: None,
                    args: self.indirect,
                    offset,
                },
            );
        }
        2
    }
}
