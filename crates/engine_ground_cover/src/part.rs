use bevy_ecs::prelude::*;
use engine_core::{
    AttachPoint, DrawKind, Entry, Error, MaterialHandle, PassRole, PassViews, ScenePass,
    ScenePasses, TargetName, Views,
};
use engine_open_world_kit::ground::PALETTE_ENTRIES;
use engine_render::{
    AfterRead, Allocation, Bind, Buffer, ComputePipeline, Count, Draw, DrawInput, Encode,
    FaceCull, Form, FormColor, FormDepth, FormFragment, FormSide, FormTopology, FragmentPass,
    FragmentUse, GpuPart, Install, KindForm, Lifetime, Own, Pass, PassFragment, PerPass, Prepare,
    Scene, SceneDraw, Shape, Varyings, wgpu,
};

pub(crate) const COVER: DrawKind = DrawKind("cover");

const COVER_MATERIAL: &str = "the cover material";
const CARD_MATERIAL: &str = "a cover card material";

use crate::covers::CoverTable;
use crate::growth::{BLADE_BYTES, BladeCounts, CARD_BYTES, CARDED_WORD, CoverPlan};
use crate::layouts::{
    COVER_HEADER_ROWS, CoverFrame, CoverHeader, CoverMaterial, CoverTerrain, Grow, GrowSpot,
    LightRows, SeenRows, ShapeRows,
};
use crate::settings::GroundCover;
use crate::shaders::cover::{casts, draw, faces::cover as face, grow, matched};
use crate::surface::CoverSurface;

const DRAW_ARGS: u64 = size_of::<wgpu::util::DrawIndirectArgs>() as u64;
const BLADE_DRAWS: u64 = 2;
const COUNT_WORDS: u64 = CARDED_WORD as u64;
const WORD: u64 = size_of::<u32>() as u64;
const MEGABYTE: u64 = 1 << 20;
const ROW_BYTES: u64 = size_of::<Grow>() as u64;
const SPOT_BYTES: u64 = size_of::<GrowSpot>() as u64;
const RUN_BYTES: u64 = size_of::<[u32; 3]>() as u64;
const FRAME_BYTES: u64 = size_of::<CoverFrame>() as u64;
const TERRAIN_BYTES: u64 = size_of::<CoverTerrain>() as u64;
const HEADER_BYTES: u64 = COVER_HEADER_ROWS as u64 * size_of::<[u32; 4]>() as u64;
const MATERIAL_BYTES: u64 = PALETTE_ENTRIES as u64 * size_of::<CoverMaterial>() as u64;
const SHAPE_AT: u64 = MATERIAL_BYTES + size_of::<CoverHeader>() as u64;
const THIN_AT: u64 = SHAPE_AT - size_of::<f32>() as u64;
const LIGHT_AT: u64 = SHAPE_AT + size_of::<ShapeRows>() as u64;
const SEEN_AT: u64 = LIGHT_AT + size_of::<LightRows>() as u64;
const CARDS_AT: u64 = SEEN_AT + size_of::<SeenRows>() as u64;
const VIEWS_AT: u64 = (std::mem::offset_of!(Grow, room) + 3 * size_of::<u32>()) as u64;

const BLADE_FORM: KindForm = KindForm {
    kind: COVER,
    placed: FormSide {
        vertex: draw::VS_COVER,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    },
    owned: Some(FormSide::pulled(draw::VS_COVER_CAST)),
    varyings: Varyings::ObjectExtra(FragmentUse::Masked),
};

const CARD_FORM: KindForm = KindForm {
    kind: COVER,
    placed: FormSide {
        vertex: draw::VS_CARD,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    },
    owned: Some(FormSide::pulled(draw::VS_CARD_CAST)),
    varyings: Varyings::ObjectExtra(FragmentUse::Masked),
};

type Held = Result<Option<MaterialHandle>, Error>;

struct SpotLists {
    spots: Buffer<CoverGpu>,
    runs: Buffer<CoverGpu>,
    running: DrawInput<CoverGpu, Own>,
    finding: ComputePipeline<CoverGpu>,
    gathering: ComputePipeline<CoverGpu>,
    found: [u32; 3],
    started: Vec<[u32; 3]>,
    spot_bytes: u64,
    run_bytes: u64,
}

pub(crate) struct CoverGpu {
    grow: Pass<Self>,
    rows: Buffer<Self>,
    lists: SpotLists,
    blades: Buffer<Self>,
    cards: Buffer<Self>,
    counts: Buffer<Self>,
    casts: Buffer<Self>,
    args: Buffer<Self>,
    frame: Buffer<Self>,
    table: Buffer<Self>,
    growing: ComputePipeline<Self>,
    finishing: ComputePipeline<Self>,
    blade_form: Form<Self>,
    card_form: Form<Self>,
    inputs: PerPass<DrawInput<Self, Scene>>,
    blade: Held,
    card: Vec<Held>,
    shadows: bool,
    bins: usize,
    views: u32,
    row_count: u64,
    blade_bytes: u64,
    card_bytes: u64,
    row_bytes: u64,
    cast_bytes: u64,
    drawn: bool,
}

impl GpuPart for CoverGpu {
    type Reads = (
        Res<'static, CoverPlan>,
        Res<'static, CoverTable>,
        Res<'static, CoverSurface>,
        Res<'static, GroundCover>,
        Res<'static, Views>,
    );
    const NAME: &'static str = "ground_cover";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let rows = cx.buffer("cover_grow", Lifetime::Kept);
        let spots = cx.buffer("cover_spots", Lifetime::Kept);
        let runs = cx.buffer("cover_runs", Lifetime::Kept);
        let blades = cx.buffer("cover_blades", Lifetime::Kept);
        let cards = cx.buffer("cover_cards", Lifetime::Kept);
        let counts = cx.buffer("cover_counts", Lifetime::Kept);
        let casts = cx.buffer("cover_casts", Lifetime::Kept);
        let args = cx.buffer("cover_draw", Lifetime::Kept);
        let frame = cx.buffer("cover_frame", Lifetime::Kept);
        let table = cx.buffer("cover_table", Lifetime::Kept);
        cx.frame(face::COVER_TABLE, Bind::Buffer(table))?;
        let grow_pass = cx.pass("ground_cover").label("engine/cover-grow").id();
        let finding = cx.compute_pipeline(
            grow_pass,
            grow::FIND_CELLS,
            &[
                (grow::GROWS, Bind::Buffer(rows)),
                (grow::COUNTS, Bind::Buffer(counts)),
                (grow::GROW_SPOTS, Bind::Buffer(spots)),
                (grow::GROW_RUNS, Bind::Buffer(runs)),
            ],
        )?;
        let gathering = cx.compute_pipeline(
            grow_pass,
            grow::GATHER_CELLS,
            &[(grow::GROWS, Bind::Buffer(rows)), (grow::GROW_RUNS, Bind::Buffer(runs))],
        )?;
        let growing = cx.compute_pipeline(
            grow_pass,
            grow::GROW_COVER,
            &[
                (grow::GROWS, Bind::Buffer(rows)),
                (grow::BLADES, Bind::Buffer(blades)),
                (grow::COUNTS, Bind::Buffer(counts)),
                (grow::CARDS, Bind::Buffer(cards)),
                (grow::GROW_SPOTS, Bind::Buffer(spots)),
                (grow::GROW_STARTS, Bind::Buffer(runs)),
                (casts::COVER_CASTS, Bind::Buffer(casts)),
            ],
        )?;
        let running = cx.draw_input(grow_pass, runs);
        let finishing = cx.compute_pipeline(
            grow_pass,
            grow::FINISH_COVER,
            &[
                (grow::GROWS, Bind::Buffer(rows)),
                (grow::COUNTS, Bind::Buffer(counts)),
                (grow::ARGS, Bind::Buffer(args)),
                (casts::COVER_CASTS, Bind::Buffer(casts)),
            ],
        )?;
        let blade_form = cx.form_shaped(
            &BLADE_FORM,
            FormTopology::Strip,
            &matched_passes(cx.passes(), draw::VS_COVER_CAST),
            &[
                (draw::COVER_BLADES, Bind::Buffer(blades)),
                (draw::COVER_FRAME, Bind::Buffer(frame)),
            ],
        )?;
        let card_form = cx.form_shaped(
            &CARD_FORM,
            FormTopology::Strip,
            &matched_passes(cx.passes(), draw::VS_CARD_CAST),
            &[
                (draw::COVER_CARDS, Bind::Buffer(cards)),
                (draw::COVER_FRAME, Bind::Buffer(frame)),
            ],
        )?;
        let inputs = PerPass::of(blade_form.passes(), |pass| {
            cx.args_input(pass.scene, args)
        });
        Ok(Self {
            grow: grow_pass,
            rows,
            lists: SpotLists {
                spots,
                runs,
                running,
                finding,
                gathering,
                found: [0; 3],
                started: Vec::new(),
                spot_bytes: SPOT_BYTES,
                run_bytes: RUN_BYTES,
            },
            blades,
            cards,
            counts,
            casts,
            args,
            frame,
            table,
            growing,
            finishing,
            blade_form,
            card_form,
            inputs,
            blade: Ok(None),
            card: Vec::new(),
            shadows: false,
            bins: 0,
            views: 0,
            row_count: 0,
            blade_bytes: BLADE_BYTES,
            card_bytes: CARD_BYTES,
            row_bytes: ROW_BYTES,
            cast_bytes: WORD,
            drawn: false,
        })
    }

    fn prepare(
        &mut self,
        (plan, table, surface, settings, views): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.blade.clone_from(&surface.blade);
        self.card.clone_from(&surface.card);
        self.shadows = settings.shadows;
        if table.is_changed() {
            let terrains = size_of_val(table.terrains.as_slice()) as u64;
            let kinds = size_of_val(table.kinds.as_slice()) as u64;
            let first = MATERIAL_BYTES + HEADER_BYTES;
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate_buffer(self.table, first + terrains.max(TERRAIN_BYTES) + kinds);
            cx.write(self.table, MATERIAL_BYTES, &[table.header]);
            cx.write(self.table, SHAPE_AT, &[table.shape]);
            cx.write(self.table, LIGHT_AT, &[table.light]);
            cx.write(self.table, SEEN_AT, &[table.seen]);
            cx.write(self.table, CARDS_AT, &[table.cards]);
            if !table.terrains.is_empty() {
                cx.write(self.table, first, &table.terrains);
            }
            if !table.materials.is_empty() {
                cx.write(self.table, 0, &table.materials);
            }
            if !table.kinds.is_empty() {
                cx.write(self.table, first + terrains, &table.kinds);
            }
        }
        if !plan.shown {
            self.blade_bytes = BLADE_BYTES;
            self.card_bytes = CARD_BYTES;
            self.row_bytes = ROW_BYTES;
            self.lists.spot_bytes = SPOT_BYTES;
            self.lists.run_bytes = RUN_BYTES;
        }
        self.drawn = plan.drawn && !plan.rows.is_empty();
        cx.write(self.table, THIN_AT, &[if self.drawn { plan.thin } else { 0.0 }]);
        self.bins = plan.bins as usize;
        self.views = views.drawn().len() as u32;
        let bins = u64::from(plan.bins.max(1));
        let kinds = BLADE_DRAWS + u64::from(plan.bins);
        let grown = plan.cells.len() as u64;
        if self.drawn {
            let blades = u64::from(plan.capacity) * BLADE_BYTES;
            let blades = blades.next_multiple_of(MEGABYTE).min(plan.room);
            self.blade_bytes = self.blade_bytes.max(blades);
            let cards = bins * u64::from(plan.cards) * CARD_BYTES;
            let cards = cards.next_multiple_of(MEGABYTE).min(plan.room);
            self.card_bytes = self.card_bytes.max(cards);
            let needed = [
                (&mut self.row_bytes, size_of_val(plan.rows.as_slice()) as u64),
                (&mut self.lists.spot_bytes, u64::from(plan.spots) * SPOT_BYTES),
                (&mut self.lists.run_bytes, (grown + 1) * RUN_BYTES),
            ];
            for (held, bytes) in needed {
                if bytes > *held {
                    *held = bytes.next_power_of_two();
                }
            }
        }
        let counted = (COUNT_WORDS + bins) * WORD;
        let casts = 2 * kinds * u64::from(self.views) * WORD;
        self.cast_bytes = self.cast_bytes.max(casts);
        let sizes = [
            (self.counts, counted),
            (self.args, (kinds * (1 + u64::from(self.views))).max(BLADE_DRAWS + bins) * DRAW_ARGS),
            (self.casts, self.cast_bytes),
            (self.frame, FRAME_BYTES),
            (self.blades, self.blade_bytes),
            (self.cards, self.card_bytes),
            (self.rows, self.row_bytes),
            (self.lists.spots, self.lists.spot_bytes),
            (self.lists.runs, self.lists.run_bytes),
        ];
        for (buffer, bytes) in sizes {
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(buffer, bytes);
        }
        let lists = &mut self.lists;
        if !self.drawn {
            return;
        }
        cx.write(self.rows, 0, &plan.rows);
        cx.write(self.rows, VIEWS_AT, &[self.views]);
        cx.write(self.frame, 0, &[plan.frame]);
        lists.started.clear();
        lists.started.resize(plan.cells.len(), [0; 3]);
        lists.started.push([plan.widest, 0, 0]);
        cx.write(lists.runs, 0, &lists.started);
        lists.found = plan.finding;
        self.row_count = grown;
        cx.run(self.grow);
        let thin = plan.thin.to_bits();
        cx.read_back::<BladeCounts>(thin, self.counts, 0..counted, AfterRead::Keep);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass != self.grow {
            return 0;
        }
        cx.clear(self.counts);
        cx.clear(self.casts);
        cx.dispatch(&self.lists.finding, Count::Groups(self.lists.found));
        cx.dispatch(&self.lists.gathering, Count::Groups([1, 1, 1]));
        let run = Count::Indirect(self.lists.running, self.row_count * RUN_BYTES);
        cx.dispatch(&self.growing, run);
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
        let material = match &self.blade {
            Ok(Some(material)) => *material,
            Ok(None) => return draw.wait(COVER_MATERIAL),
            Err(error) => return draw.refuse(COVER_MATERIAL, error),
        };
        let (Some(blades), Some(cards)) =
            (draw.form(&self.blade_form), draw.form(&self.card_form))
        else {
            return draw.skip("pass has no cover pipeline");
        };
        let Some(&args) = draw.per_pass(&self.inputs) else {
            return draw.skip("pass has no cover list");
        };
        let kinds = BLADE_DRAWS + self.bins as u64;
        let first = match (scene.view, scene.index) {
            (Some(_), Some(index)) if index < self.views => {
                (1 + u64::from(index)) * kinds
            }
            (Some(_) | None, _) => 0,
        };
        let mut drawn = 0;
        for (at, form) in (first..first + BLADE_DRAWS).zip(["strips", "flats"]) {
            draw.draw_timed(
                &blades,
                Shape::Indirect {
                    vertices: &[],
                    indices: None,
                    args,
                    offset: at * DRAW_ARGS,
                    material: Some(material),
                },
                || draw_label(scene, form),
            );
            drawn += 1;
        }
        let mut refused = None;
        for (bin, held) in self.card.iter().take(self.bins).enumerate() {
            match held {
                Ok(Some(material)) => {
                    draw.draw_timed(
                        &cards,
                        Shape::Indirect {
                            vertices: &[],
                            indices: None,
                            args,
                            offset: (first + BLADE_DRAWS + bin as u64) * DRAW_ARGS,
                            material: Some(*material),
                        },
                        || draw_label(scene, &format!("cards {bin}")),
                    );
                    drawn += 1;
                }
                Ok(None) => {}
                Err(error) => refused = refused.or(Some(error)),
            }
        }
        match refused {
            Some(error) => draw.refuse(CARD_MATERIAL, error),
            None => drawn,
        }
    }
}

fn matched_passes(passes: &ScenePasses, cast: Entry) -> Vec<PassFragment> {
    let placed = |listed: &&ScenePass| {
        listed.bit.is_some()
            && listed.pass.views != PassViews::Owned
            && listed.pass.writes.contains(TargetName::DEPTH)
            && passes.takes(listed, COVER)
    };
    let lays = |listed: &ScenePass| {
        listed.pass.role() == PassRole::Depth && listed.pass.at == AttachPoint::Depth
    };
    if !passes.listed().iter().filter(placed).any(lays) {
        return Vec::new();
    }
    passes
        .listed()
        .iter()
        .filter(placed)
        .filter_map(|listed| {
            let pass = FragmentPass::Scene(Scene::named(listed.name));
            if lays(listed) {
                return Some(PassFragment {
                    pass,
                    side: Some(FormSide::pulled(cast)),
                    fragment: FormFragment::Role(PassRole::Depth),
                });
            }
            (listed.pass.role() == PassRole::Lit && listed.pass.at > AttachPoint::Depth).then_some(
                PassFragment {
                    pass,
                    side: None,
                    fragment: FormFragment::Entry {
                        entry: matched::FS_COVER_MATCHED,
                        cull: FaceCull::None,
                        depth: FormDepth::Matched,
                        color: FormColor::Written,
                    },
                },
            )
        })
        .collect()
}

fn draw_label(scene: &SceneDraw, form: &str) -> String {
    let role = scene.role;
    match scene.view {
        Some(view) => format!("ground_cover {form} {role:?} view {view}"),
        None => format!("ground_cover {form} {role:?}"),
    }
}
