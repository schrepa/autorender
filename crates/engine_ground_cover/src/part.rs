use bevy_ecs::prelude::*;
use engine_core::{DrawKind, Error, PassRole};
use engine_open_world_kit::ground::PALETTE_ENTRIES;
use engine_render::{
    AfterRead, Allocation, Bind, Buffer, ComputePipeline, Count, Draw, DrawInput, Encode, Form,
    FormSide, FormTopology, GpuPart, Install, KindForm, Lifetime, Pass, Prepare, Scene, SceneDraw,
    Shape, Varyings, wgpu,
};

pub(crate) const COVER: DrawKind = DrawKind("cover");

use crate::covers::CoverTable;
use crate::growth::{BLADE_BYTES, BladeCounts, CoverPlan};
use crate::layouts::{CoverFrame, CoverMaterial, CoverTerrain, Grow};
use crate::settings::GroundCover;
use crate::shaders::cover::{draw, faces::cover as face, grow};
use crate::surface::{CoverSurface, SurfaceState};

const DRAW_ARGS: u64 = size_of::<wgpu::util::DrawIndirectArgs>() as u64;
const COUNT_BYTES: u64 = 32;
const ROW_BYTES: u64 = size_of::<Grow>() as u64;
const FRAME_BYTES: u64 = size_of::<CoverFrame>() as u64;
const TERRAIN_BYTES: u64 = size_of::<CoverTerrain>() as u64;
const MATERIAL_BYTES: u64 = PALETTE_ENTRIES as u64 * size_of::<CoverMaterial>() as u64;

const COVER_FORM: KindForm = KindForm {
    kind: COVER,
    placed: FormSide {
        vertex: draw::VS_COVER,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    },
    owned: None,
    varyings: Varyings::ObjectExtra,
};

pub(crate) struct CoverGpu {
    grow: Pass<Self>,
    rows: Buffer<Self>,
    blades: Buffer<Self>,
    counts: Buffer<Self>,
    args: Buffer<Self>,
    frame: Buffer<Self>,
    table: Buffer<Self>,
    growing: ComputePipeline<Self>,
    stepping: ComputePipeline<Self>,
    finishing: ComputePipeline<Self>,
    form: Form<Self>,
    inputs: Vec<(&'static str, DrawInput<Self, Scene>)>,
    surface: SurfaceState,
    shadows: bool,
    cells: Vec<u32>,
    blade_bytes: u64,
    row_bytes: u64,
    drawn: bool,
}

impl GpuPart for CoverGpu {
    type Reads = (
        Res<'static, CoverPlan>,
        Res<'static, CoverTable>,
        Res<'static, CoverSurface>,
        Res<'static, GroundCover>,
    );
    const NAME: &'static str = "ground_cover";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let rows = cx.buffer("cover_grow", Lifetime::Kept);
        let blades = cx.buffer("cover_blades", Lifetime::Kept);
        let counts = cx.buffer("cover_counts", Lifetime::Kept);
        let args = cx.buffer("cover_draw", Lifetime::Kept);
        let frame = cx.buffer("cover_frame", Lifetime::Kept);
        let table = cx.buffer("cover_table", Lifetime::Kept);
        cx.frame(face::COVER_TABLE, Bind::Buffer(table))?;
        let grow_pass = cx
            .pass("ground_cover")
            .label("engine/cover-grow")
            .span("ground_cover")
            .id();
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
        let form = cx.form_shaped(
            &COVER_FORM,
            FormTopology::Strip,
            &[
                (draw::COVER_BLADES, Bind::Buffer(blades)),
                (draw::COVER_FRAME, Bind::Buffer(frame)),
            ],
        )?;
        let inputs = form
            .passes()
            .map(|pass| (pass.scene.name(), cx.draw_input(pass.scene, args)))
            .collect();
        Ok(Self {
            grow: grow_pass,
            rows,
            blades,
            counts,
            args,
            frame,
            table,
            growing,
            stepping,
            finishing,
            form,
            inputs,
            surface: SurfaceState::Unmade,
            shadows: false,
            cells: Vec::new(),
            blade_bytes: BLADE_BYTES,
            row_bytes: ROW_BYTES,
            drawn: false,
        })
    }

    fn prepare(
        &mut self,
        (plan, table, surface, settings): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.surface = surface.state;
        self.shadows = settings.shadows;
        if table.is_changed() {
            let terrains = size_of_val(table.terrains.as_slice()) as u64;
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate_buffer(self.table, MATERIAL_BYTES + terrains.max(TERRAIN_BYTES));
            if !table.terrains.is_empty() {
                cx.write(self.table, MATERIAL_BYTES, &table.terrains);
            }
            if !table.materials.is_empty() {
                cx.write(self.table, 0, &table.materials);
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
        let sizes = [
            (self.counts, COUNT_BYTES),
            (self.args, 2 * DRAW_ARGS),
            (self.frame, FRAME_BYTES),
            (self.blades, self.blade_bytes),
            (self.rows, self.row_bytes),
        ];
        for (buffer, bytes) in sizes {
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(buffer, bytes);
        }
        self.cells.clear();
        if !self.drawn {
            return;
        }
        cx.write(self.rows, 0, &plan.rows);
        cx.write(self.frame, 0, &[plan.frame]);
        self.cells.extend_from_slice(&plan.cells);
        cx.run(self.grow);
        cx.read_back::<BladeCounts>(0, self.counts, 0..COUNT_BYTES, AfterRead::Keep);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass != self.grow {
            return 0;
        }
        cx.clear(self.counts);
        for (index, &cells) in self.cells.iter().enumerate() {
            if index > 0 {
                cx.dispatch(&self.stepping, Count::Groups([1, 1, 1]));
            }
            cx.dispatch(&self.growing, Count::Groups([cells, cells, 1]));
        }
        cx.dispatch(&self.finishing, Count::Groups([1, 1, 1]));
        0
    }
}

impl CoverGpu {
    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if scene.role == PassRole::Blended {
            return draw.skip("ground cover draws opaque");
        }
        if scene.view.is_some() && !self.shadows {
            return draw.skip("ground_cover.shadows is off");
        }
        if !self.drawn {
            return draw.skip("no cover grows this frame");
        }
        let material = match self.surface {
            SurfaceState::Resident(material) => material,
            SurfaceState::Waiting(_) => return draw.skip("cover material is not resident yet"),
            SurfaceState::Unmade => {
                return draw.skip("cover material is not in the asset library");
            }
        };
        let Some(pipeline) = draw.form(&self.form) else {
            return draw.skip("pass has no cover pipeline");
        };
        let Some(&(_, args)) = self
            .inputs
            .iter()
            .find(|(name, _)| *name == scene.pass.name())
        else {
            return draw.skip("pass has no cover list");
        };
        for offset in [0, DRAW_ARGS] {
            draw.draw(
                &pipeline,
                Shape::Indirect {
                    vertices: &[],
                    indices: None,
                    args,
                    offset,
                    material: Some(material),
                },
            );
        }
        2
    }
}
