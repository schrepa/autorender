use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::math::{max, min};
use engine_core::presentation_bounds::ROUNDING_MARGIN;
use engine_core::{
    AssetLibrary, DrawKind, Entry, Error, KindDecl, PassName, PassPurpose, PassViews, SlotSet, Var,
    ViewFacts, Views, Writes,
};
use engine_heightfield::Fall;
use engine_open_world_kit::ground::{GroundTiles, Heightfield};
use engine_render::{
    Allocation, AttachPoint, Bind, Buffer, ComputePipeline, Count, Draw, DrawInput, Encode,
    FaceCull, Form, FormSide, GpuPart, Install, KindForm, Lifetime, OwnFragments, Pass, PerPass,
    Prepare, RoleFragments, Scene, SceneDraw, Shape, TargetName, Varyings, VertexLayout,
};
use glam::{DVec2, DVec3, DVec4, UVec2, Vec3};

use crate::Water;
use crate::gpu::{
    FallVertex, WATER_LEAF_DRAW_WORDS, WATER_LEAF_FIRST, WATER_LIST_GROUP, WATER_LIST_GROUPS,
    WATER_RAW_VERTEX, WaterUniform,
};
use crate::look::looked;
use crate::settings::WaterFallSheets;
use crate::state::WaterState;
use crate::water_blocks::{BlocksGpu, INFO_FORMAT, SURFACE_FORMAT, surface_reach};
use crate::water_detail::DetailGpu;
use crate::water_falls::{FALL_LAYOUT, sheets};
use crate::waves::{Waves, WavesGpu};

pub(crate) const WATER_KIND: DrawKind = DrawKind("water");

const NO_WATER: &str = "no water is drawn";
const NO_FALL: &str = "no fall is built";

pub(crate) struct WaterPass;

impl PassName for WaterPass {
    const NAME: &'static str = "water";
    const PASS: engine_core::Pass = engine_core::Pass {
        writes: Writes::LIT,
        reads: SlotSet::ALL,
        views: PassViews::Main,
        at: AttachPoint::LateOpaque,
        purpose: PassPurpose::Scene,
        takes: takes_water,
    };
}

fn takes_water(kind: &KindDecl) -> bool {
    kind.kind == WATER_KIND
}

type BuiltFalls = (Entity, Vec3, Arc<[Fall]>, Option<u32>, Range<u32>);

const SURFACE_FRAGMENTS: OwnFragments = OwnFragments {
    placed: RoleFragments {
        lit: Some(crate::shaders::water::surface::FS_WATER),
        shown: Some(crate::shaders::water::debug::FS_DEBUG_WATER),
        ..RoleFragments::NONE
    },
    owned: RoleFragments::NONE,
    cull: FaceCull::None,
};

const FALL_FRAGMENTS: OwnFragments = OwnFragments {
    placed: RoleFragments {
        blended: Some(crate::shaders::water::falls::FS_FALL),
        shown: Some(crate::shaders::water::debug::FS_DEBUG_FALL),
        ..RoleFragments::NONE
    },
    owned: RoleFragments::NONE,
    cull: FaceCull::None,
};

const FALL_STREAMS: &[VertexLayout] = &[VertexLayout::Buffer(FALL_LAYOUT)];

const SURFACE_FORM: KindForm = KindForm {
    kind: WATER_KIND,
    placed: FormSide {
        vertex: crate::shaders::water::surface::VS_WATER,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    },
    owned: None,
    varyings: Varyings::Own(&SURFACE_FRAGMENTS),
};

const FALL_FORM: KindForm = KindForm {
    kind: WATER_KIND,
    placed: FormSide {
        vertex: crate::shaders::water::falls::VS_FALL_TEMPORAL,
        streams: FALL_STREAMS,
        depth_bias: FormSide::UNBIASED,
    },
    owned: None,
    varyings: Varyings::Own(&FALL_FRAGMENTS),
};

pub(crate) fn reached(
    cx: &mut Install<'_, WaterGpu>,
    entries: &[Entry],
    all: Vec<(Var, Bind<WaterGpu>)>,
) -> Vec<(Var, Bind<WaterGpu>)> {
    all.into_iter()
        .filter(|(var, _)| entries.iter().any(|&entry| cx.reaches(entry, *var)))
        .collect()
}

struct Leaves {
    leaves: DrawInput<WaterGpu, Scene>,
    pattern: DrawInput<WaterGpu, Scene>,
}

struct Sheets {
    vertices: DrawInput<WaterGpu, Scene>,
    indices: DrawInput<WaterGpu, Scene>,
}

struct LeafList {
    count: Pass<WaterGpu>,
    pass: Pass<WaterGpu>,
    counting: ComputePipeline<WaterGpu>,
    marking: ComputePipeline<WaterGpu>,
    scanning: ComputePipeline<WaterGpu>,
    listing: ComputePipeline<WaterGpu>,
    list: Buffer<WaterGpu>,
    draw: Buffer<WaterGpu>,
    room: u32,
}

impl LeafList {
    fn install(
        cx: &mut Install<'_, WaterGpu>,
        uniform: Buffer<WaterGpu>,
        (leaf_args, wet): (Buffer<WaterGpu>, Buffer<WaterGpu>),
    ) -> Result<Self, Error> {
        use crate::shaders::water::{surface, uniforms};
        let count = cx.pass("water_leaf_args").id();
        let pass = cx
            .pass("water_leaf_list")
            .at(AttachPoint::Opaque)
            .views(PassViews::Main)
            .label("engine/water-leaf-list")
            .id();
        let list = cx.buffer("water_leaf_list", Lifetime::Kept);
        let draw = cx.buffer("water_leaf_draw", Lifetime::Frame);
        let mut pipeline = |pass: Pass<WaterGpu>, entry: Entry| {
            let bound = reached(
                cx,
                &[entry],
                vec![
                    (uniforms::WATER, Bind::Buffer(uniform)),
                    (surface::WATER_LEAF_ARGS, Bind::Buffer(leaf_args)),
                    (surface::WATER_LEAF_SPANS, Bind::Buffer(leaf_args)),
                    (surface::WATER_LEAF_LIST, Bind::Buffer(list)),
                    (surface::WATER_LEAF_DRAW, Bind::Buffer(draw)),
                    (surface::WATER_BLOCK_WET, Bind::Buffer(wet)),
                ],
            );
            cx.compute_pipeline(pass, entry, &bound)
        };
        Ok(Self {
            counting: pipeline(count, surface::CS_WATER_LEAF_ARGS)?,
            marking: pipeline(pass, surface::CS_WATER_LEAF_MARK)?,
            scanning: pipeline(pass, surface::CS_WATER_LEAF_SCAN)?,
            listing: pipeline(pass, surface::CS_WATER_LEAF_LIST)?,
            count,
            pass,
            list,
            draw,
            room: 0,
        })
    }

    fn room(&mut self, leaves: u32, cx: &mut Prepare<'_, WaterGpu>) -> u32 {
        let wanted = leaves.max(WATER_LIST_GROUP).next_power_of_two();
        self.room = self.room.max(wanted.min(WATER_LIST_GROUP * WATER_LIST_GROUPS));
        let chunks = u64::from(self.room / WATER_LIST_GROUP);
        let words = u64::from(self.room) + 3 * chunks;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.list, words * 4);
        let words = u64::from(WATER_LEAF_DRAW_WORDS);
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.draw, words * 4);
        self.room
    }

    fn run(&self, cx: &mut Prepare<'_, WaterGpu>) {
        cx.run(self.count);
        cx.run(self.pass);
    }

    fn encode(&self, pass: Pass<WaterGpu>, cx: &mut Encode<'_, WaterGpu>) {
        if pass == self.count {
            cx.dispatch(&self.counting, Count::Groups([1, 1, 1]));
        }
        if pass != self.pass {
            return;
        }
        let groups = [self.room / WATER_LIST_GROUP, 1, 1];
        cx.dispatch(&self.marking, Count::Groups(groups));
        cx.dispatch(&self.scanning, Count::Groups([1, 1, 1]));
        cx.dispatch(&self.listing, Count::Groups(groups));
    }
}

pub(crate) struct WaterGpu {
    uniform: Buffer<Self>,
    leaf_args: Buffer<Self>,
    leaf_pattern: Buffer<Self>,
    fall_vertices: Buffer<Self>,
    fall_indices: Buffer<Self>,
    listing: LeafList,
    form: Form<Self>,
    falls: Form<Self>,
    leaves: PerPass<Leaves>,
    sheets: PerPass<Sheets>,
    blocks: BlocksGpu,
    detail: DetailGpu,
    waves: WavesGpu,
    built: Vec<BuiltFalls>,
    sheeted: Option<WaterFallSheets>,
    vertices: Vec<FallVertex>,
    indices: Vec<u32>,
    pattern_written: Option<Arc<[u32]>>,
    drawn: bool,
    surface: bool,
}

impl GpuPart for WaterGpu {
    type Reads = (
        Res<'static, Water>,
        Res<'static, WaterState>,
        Res<'static, GroundTiles>,
        Res<'static, Heightfield>,
        Res<'static, AssetLibrary>,
        Res<'static, Waves>,
        Res<'static, Views>,
    );
    const NAME: &'static str = "water";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::water::{faces::water as face, refraction, surface, uniforms};
        let uniform = cx.buffer("water", Lifetime::Kept);
        let leaf_args = cx.buffer("water_leaf_args", Lifetime::Frame);
        let leaf_pattern = cx.buffer("water_leaf_pattern", Lifetime::Kept);
        let fall_vertices = cx.buffer("water_falls", Lifetime::Kept);
        let fall_indices = cx.buffer("water_fall_indices", Lifetime::Kept);
        let refraction_sampler = cx.sampler(engine_core::Sampler::LINEAR_CLAMP);
        cx.frame(
            refraction::REFRACTION_SAMPLER,
            Bind::Sampler(refraction_sampler),
        )?;
        let blocks = BlocksGpu::new(
            (
                cx.texture("water_surfaces", SURFACE_FORMAT, Lifetime::Kept),
                cx.texture("water_info", INFO_FORMAT, Lifetime::Kept),
            ),
            cx.buffer("water_directory", Lifetime::Kept),
            cx.buffer("water_block_wet", Lifetime::Kept),
        );
        cx.frame(face::WATER_SURFACE, Bind::Texture(blocks.surface))?;
        cx.frame(face::WATER_INFO, Bind::Texture(blocks.info))?;
        cx.frame(face::WATER_DIRECTORY, Bind::Buffer(blocks.directory))?;
        let detail = DetailGpu::install(cx);
        let waves = WavesGpu::install(cx);
        let listing = LeafList::install(cx, uniform, (leaf_args, blocks.wet))?;
        let behind = || {
            let mut all = vec![
                (uniforms::WATER, Bind::Buffer(uniform)),
                (surface::WATER_LEAF_SPANS, Bind::Buffer(leaf_args)),
                (surface::WATER_LEAF_ORDER, Bind::Buffer(listing.list)),
                (surface::WATER_DEPTH, Bind::Target(TargetName::DEPTH)),
                (surface::WATER_SCENE, Bind::Target(TargetName::COLOR)),
            ];
            all.extend(detail.bound());
            all.extend(waves.bound());
            all
        };
        let surface_form = cx.form(&SURFACE_FORM, &behind())?;
        let fall_form = cx.form(&FALL_FORM, &behind())?;
        let leaves = PerPass::of(surface_form.passes(), |pass| {
            let listed = pass.scene.name() == WaterPass::NAME;
            Leaves {
                leaves: cx.args_input(
                    pass.scene,
                    if listed { listing.draw } else { leaf_args },
                ),
                pattern: cx.draw_input(pass.scene, leaf_pattern),
            }
        });
        let sheets = PerPass::of(fall_form.passes(), |pass| Sheets {
            vertices: cx.draw_input(pass.scene, fall_vertices),
            indices: cx.draw_input(pass.scene, fall_indices),
        });
        Ok(Self {
            uniform,
            leaf_args,
            leaf_pattern,
            fall_vertices,
            fall_indices,
            listing,
            form: surface_form,
            falls: fall_form,
            leaves,
            sheets,
            blocks,
            detail,
            waves,
            built: Vec::new(),
            sheeted: None,
            vertices: Vec::new(),
            indices: Vec::new(),
            pattern_written: None,
            drawn: false,
            surface: false,
        })
    }

    fn prepare(
        &mut self,
        reads: SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let (water, state, ground, heightfield, assets, waves, views) = reads;
        let pattern = &heightfield.leaf_indices;
        let made = cx.allocate_buffer(
            self.leaf_pattern,
            2 * std::mem::size_of_val(&pattern[..]) as u64,
        );
        let fresh = made == Allocation::Fresh
            || self
                .pattern_written
                .as_ref()
                .is_none_or(|written| written[..] != pattern[..]);
        if fresh {
            if !pattern.is_empty() {
                let raw = pattern.iter().map(|index| index + WATER_RAW_VERTEX);
                let both: Vec<u32> = pattern.iter().copied().chain(raw).collect();
                cx.write(self.leaf_pattern, 0, &both);
            }
            self.pattern_written = Some(Arc::clone(pattern));
        }
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.uniform, size_of::<WaterUniform>() as u64);
        self.blocks
            .prepare(&state.blocks, (&ground, &state.terrains), cx);
        self.detail.prepare(&water, &assets, cx);
        let swell = waves;
        let waves = self.waves.prepare(&swell, cx);
        let watering = !state.watered.is_empty();
        self.prepare_falls(&state, &ground, &water.fall_sheets, cx);
        let terrains = ground.tables.len() as u32;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(
            self.leaf_args,
            u64::from(WATER_LEAF_FIRST + 2 * terrains.max(1)) * 4,
        );
        self.drawn = watering;
        if !self.drawn {
            self.surface = false;
            cx.skip_kind(WATER_KIND, NO_WATER);
            return;
        }
        let reach = surface_reach(&state, &ground, swell.swell().reach());
        self.surface = views
            .main()
            .is_none_or(|view| surface_seen(view, state.blocks.wet_areas(), reach));
        let [red, green, blue] = water.color;
        let sediment = water.sediment_color;
        let room = self.listing.room(heightfield.water_leaves, cx);
        let frame = WaterUniform {
            color: [red, green, blue, water.clarity],
            absorption: absorption(water.color, water.clarity),
            sediment: [sediment[0], sediment[1], sediment[2], water.sediment],
            terrains: [terrains, room, 0, 0],
            reach,
            waves,
            ..WaterUniform::default()
        };
        let look = looked(&water, self.detail.measured(), frame);
        cx.write(self.uniform, 0, &[look]);
        if self.surface {
            self.listing.run(cx);
        } else if !self.falling() {
            cx.skip_kind(WATER_KIND, NO_FALL);
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        self.listing.encode(pass, cx);
        0
    }
}

impl WaterGpu {
    pub(crate) fn draw(&self, _: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if !self.drawn {
            return draw.skip(NO_WATER);
        }
        let mut draws = 0;
        let leaves = draw.per_pass(&self.leaves).filter(|_| self.surface);
        if let (Some(pipeline), Some(leaves)) = (draw.form(&self.form), leaves) {
            draw.draw_timed(
                &pipeline,
                Shape::Indirect {
                    vertices: &[],
                    indices: Some(leaves.pattern),
                    args: leaves.leaves,
                    offset: 0,
                    material: None,
                },
                || "water surface".to_owned(),
            );
            draws += 1;
        }
        let sheets = draw.per_pass(&self.sheets);
        if let (Some(pipeline), Some(sheets)) = (draw.form(&self.falls), sheets) {
            for (_, _, _, drawn, run) in &self.built {
                if drawn.is_none() || run.is_empty() {
                    continue;
                }
                draw.draw_timed(
                    &pipeline,
                    Shape::Buffers {
                        vertices: &[sheets.vertices],
                        indices: Some(sheets.indices),
                        elements: run.clone(),
                        instances: 0..1,
                    },
                    || "water falls".to_owned(),
                );
                draws += 1;
            }
        }
        if draws == 0 {
            return draw.skip(NO_FALL);
        }
        draws
    }

    fn falling(&self) -> bool {
        self.built
            .iter()
            .any(|(_, _, _, drawn, run)| drawn.is_some() && !run.is_empty())
    }

    fn prepare_falls(
        &mut self,
        state: &WaterState,
        ground: &GroundTiles,
        look: &WaterFallSheets,
        cx: &mut Prepare<'_, Self>,
    ) {
        let index_of = |entity: Entity| {
            ground
                .tables
                .iter()
                .find(|(terrain, _)| *terrain == entity)
                .map(|&(_, index)| index)
        };
        let falling = || {
            state.terrains.iter().filter_map(|terrain| {
                let falls = state.emitters.get(&terrain.entity)?.falling()?;
                let drawn =
                    index_of(terrain.entity).filter(|_| state.watered.contains(&terrain.entity));
                (!falls.is_empty()).then_some((terrain, falls, drawn))
            })
        };
        let same = self.sheeted.as_ref() == Some(look)
            && self.built.len() == falling().count()
            && self
                .built
                .iter()
                .zip(falling())
                .all(|(a, (b, falls, drawn))| {
                    a.0 == b.entity && a.1 == b.origin && a.2[..] == falls[..] && a.3 == drawn
                });
        if same || (self.built.is_empty() && falling().next().is_none()) {
            return;
        }
        self.sheeted = Some(*look);
        let runs = sheets(
            falling().map(|(terrain, falls, drawn)| (terrain.origin, &**falls, drawn.unwrap_or(0))),
            look,
            &mut self.vertices,
            &mut self.indices,
        );
        self.built = falling()
            .zip(runs)
            .map(|((terrain, falls, drawn), run)| {
                (
                    terrain.entity,
                    terrain.origin,
                    Arc::clone(falls),
                    drawn,
                    run,
                )
            })
            .collect();
        if self.indices.is_empty() {
            return;
        }
        let bytes = std::mem::size_of_val(&self.vertices[..]) as u64;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.fall_vertices, bytes);
        cx.write(self.fall_vertices, 0, &self.vertices);
        let bytes = std::mem::size_of_val(&self.indices[..]) as u64;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.fall_indices, bytes);
        cx.write(self.fall_indices, 0, &self.indices);
    }
}

fn surface_seen(
    view: &ViewFacts,
    mut areas: impl Iterator<Item = [f32; 4]>,
    [low, high, known, _]: [f32; 4],
) -> bool {
    if known == 0.0 {
        return true;
    }
    let (low, high) = (
        f64::from(min(low, view.eye.y)),
        f64::from(max(high, view.eye.y)),
    );
    let size = view.render.min(view.size()).max(UVec2::ONE).as_dvec2();
    let jitter = view
        .jitter
        .map_or(DVec2::ZERO, |offset| 2.0 * offset.as_dvec2().abs() / size);
    let clip = view.view_projection.as_dmat4().transpose();
    let planes = [
        DVec4::new(1.0, 0.0, 0.0, 1.0 + jitter.x),
        DVec4::new(-1.0, 0.0, 0.0, 1.0 + jitter.x),
        DVec4::new(0.0, 1.0, 0.0, 1.0 + jitter.y),
        DVec4::new(0.0, -1.0, 0.0, 1.0 + jitter.y),
        DVec4::new(0.0, 0.0, -1.0, 1.0),
        DVec4::new(0.0, 0.0, 1.0, 0.0),
    ]
    .map(|plane| clip * plane);
    let margin = f64::from(ROUNDING_MARGIN);
    areas.any(|area| {
        if !area.iter().all(|edge| edge.is_finite()) {
            return true;
        }
        let [x0, z0, x1, z1] = area.map(f64::from);
        let slack = margin * (x0.abs().max(x1.abs()) + z0.abs().max(z1.abs()));
        let corner_low = DVec3::new(x0 - slack, low, z0 - slack);
        let corner_high = DVec3::new(x1 + slack, high, z1 + slack);
        let extent = corner_low.abs().max(corner_high.abs());
        planes.iter().all(|plane| {
            let facing = plane.truncate();
            let support = DVec3::select(facing.cmpge(DVec3::ZERO), corner_high, corner_low);
            facing.dot(support) + plane.w >= -margin * (facing.abs().dot(extent) + plane.w.abs())
        })
    })
}

fn absorption(color: [f32; 3], clarity: f32) -> [f32; 4] {
    const LUMINANCE: [f32; 3] = [0.2126, 0.7152, 0.0722];
    let luminance = max(
        color[0] * LUMINANCE[0] + color[1] * LUMINANCE[1] + color[2] * LUMINANCE[2],
        1.0e-4,
    );
    let weights = color.map(|channel| luminance / max(channel, luminance * 0.125));
    let mean = weights[0] * LUMINANCE[0] + weights[1] * LUMINANCE[1] + weights[2] * LUMINANCE[2];
    let rate = std::f32::consts::LN_2 / max(clarity, 0.01) / mean;
    [weights[0] * rate, weights[1] * rate, weights[2] * rate, 0.0]
}
