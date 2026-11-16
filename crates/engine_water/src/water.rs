use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::math::max;
use engine_core::{
    AssetLibrary, DrawKind, Entry, Error, KindDecl, PassName, PassPurpose, PassViews, SlotSet, Var,
    Writes,
};
use engine_heightfield::Fall;
use engine_open_world_kit::ground::{GroundTiles, Heightfield};
use engine_render::{
    Allocation, AttachPoint, Bind, Buffer, ComputePipeline, Count, Draw, DrawInput, Encode,
    FaceCull, Form, FormSide, GpuPart, Install, KindForm, Lifetime, OwnFragments, Pass, PerPass,
    Prepare, RoleFragments, Scene, SceneDraw, Shape, TargetName, Varyings, VertexLayout,
};
use glam::Vec3;

use crate::Water;
use crate::gpu::{FallVertex, WATER_LEAF_FIRST, WaterUniform};
use crate::look::looked;
use crate::settings::WaterFallSheets;
use crate::state::WaterState;
use crate::water_blocks::{BlocksGpu, INFO_FORMAT, SURFACE_FORMAT};
use crate::water_detail::DetailGpu;
use crate::water_falls::{FALL_LAYOUT, sheets};
use crate::waves::{Waves, WavesGpu};

pub(crate) const WATER_KIND: DrawKind = DrawKind("water");

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

pub(crate) struct WaterGpu {
    uniform: Buffer<Self>,
    leaf_args: Buffer<Self>,
    leaf_pattern: Buffer<Self>,
    fall_vertices: Buffer<Self>,
    fall_indices: Buffer<Self>,
    count_leaves: Pass<Self>,
    counting: ComputePipeline<Self>,
    surface: Form<Self>,
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
}

impl GpuPart for WaterGpu {
    type Reads = (
        Res<'static, Water>,
        Res<'static, WaterState>,
        Res<'static, GroundTiles>,
        Res<'static, Heightfield>,
        Res<'static, AssetLibrary>,
        Res<'static, Waves>,
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
            cx.texture("water_surfaces", SURFACE_FORMAT, Lifetime::Kept),
            cx.texture("water_info", INFO_FORMAT, Lifetime::Kept),
            cx.buffer("water_directory", Lifetime::Kept),
        );
        cx.frame(face::WATER_SURFACE, Bind::Texture(blocks.surface))?;
        cx.frame(face::WATER_INFO, Bind::Texture(blocks.info))?;
        cx.frame(face::WATER_DIRECTORY, Bind::Buffer(blocks.directory))?;
        let detail = DetailGpu::install(cx);
        let waves = WavesGpu::install(cx);
        let count_leaves = cx.pass("water_leaf_args").id();
        let counting_bound = reached(
            cx,
            &[surface::CS_WATER_LEAF_ARGS],
            vec![
                (uniforms::WATER, Bind::Buffer(uniform)),
                (surface::WATER_LEAF_ARGS, Bind::Buffer(leaf_args)),
            ],
        );
        let counting =
            cx.compute_pipeline(count_leaves, surface::CS_WATER_LEAF_ARGS, &counting_bound)?;
        let behind = || {
            let mut all = vec![
                (uniforms::WATER, Bind::Buffer(uniform)),
                (surface::WATER_LEAF_SPANS, Bind::Buffer(leaf_args)),
                (surface::WATER_DEPTH, Bind::Target(TargetName::DEPTH)),
                (surface::WATER_SCENE, Bind::Target(TargetName::COLOR)),
            ];
            all.extend(detail.bound());
            all.extend(waves.bound());
            all
        };
        let surface_form = cx.form(&SURFACE_FORM, &behind())?;
        let fall_form = cx.form(&FALL_FORM, &behind())?;
        let leaves = PerPass::of(surface_form.passes(), |pass| Leaves {
            leaves: cx.draw_input(pass.scene, leaf_args),
            pattern: cx.draw_input(pass.scene, leaf_pattern),
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
            count_leaves,
            counting,
            surface: surface_form,
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
        })
    }

    fn prepare(
        &mut self,
        reads: SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let (water, state, ground, heightfield, assets, waves) = reads;
        let pattern = &heightfield.leaf_indices;
        let made = cx.allocate_buffer(
            self.leaf_pattern,
            std::mem::size_of_val(&pattern[..]) as u64,
        );
        let fresh = made == Allocation::Fresh
            || self
                .pattern_written
                .as_ref()
                .is_none_or(|written| written[..] != pattern[..]);
        if fresh {
            if !pattern.is_empty() {
                cx.write(self.leaf_pattern, 0, &pattern[..]);
            }
            self.pattern_written = Some(Arc::clone(pattern));
        }
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.uniform, size_of::<WaterUniform>() as u64);
        self.blocks
            .prepare(&state.blocks, (&ground, &state.terrains), cx);
        self.detail.prepare(&water, &assets, cx);
        let waves = self.waves.prepare(&waves, cx);
        let watering = !state.watered.is_empty();
        self.prepare_falls(&state, &ground, &water.fall_sheets, cx);
        let terrains = ground.tables.len() as u32;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(
            self.leaf_args,
            u64::from(WATER_LEAF_FIRST + 2 * terrains.max(1)) * 4,
        );
        self.drawn = watering;
        if !self.drawn {
            return;
        }
        let [red, green, blue] = water.color;
        let sediment = water.sediment_color;
        let frame = WaterUniform {
            color: [red, green, blue, water.clarity],
            absorption: absorption(water.color, water.clarity),
            sediment: [sediment[0], sediment[1], sediment[2], water.sediment],
            terrains: [terrains, 0, 0, 0],
            waves,
            ..WaterUniform::default()
        };
        let look = looked(&water, self.detail.measured(), frame);
        cx.write(self.uniform, 0, &[look]);
        cx.run(self.count_leaves);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.count_leaves {
            cx.dispatch(&self.counting, Count::Groups([1, 1, 1]));
        }
        0
    }
}

impl WaterGpu {
    pub(crate) fn draw(&self, _: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if !self.drawn {
            return draw.skip("no water is drawn");
        }
        let mut draws = 0;
        let leaves = draw.per_pass(&self.leaves);
        if let (Some(pipeline), Some(leaves)) = (draw.form(&self.surface), leaves) {
            draw.draw(
                &pipeline,
                Shape::Indirect {
                    vertices: &[],
                    indices: Some(leaves.pattern),
                    args: leaves.leaves,
                    offset: 0,
                    material: None,
                },
            );
            draws += 1;
        }
        let sheets = draw.per_pass(&self.sheets);
        if let (Some(pipeline), Some(sheets)) = (draw.form(&self.falls), sheets) {
            for (_, _, _, drawn, run) in &self.built {
                if drawn.is_none() || run.is_empty() {
                    continue;
                }
                draw.draw(
                    &pipeline,
                    Shape::Buffers {
                        vertices: &[sheets.vertices],
                        indices: Some(sheets.indices),
                        elements: run.clone(),
                        instances: 0..1,
                    },
                );
                draws += 1;
            }
        }
        if draws == 0 {
            return draw.skip("no fall is built");
        }
        draws
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
