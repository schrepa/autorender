use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::math::max;
use engine_core::{
    DrawKind, Entry, Error, KindDecl, PassName, PassPurpose, PassViews, SlotSet, Var, Writes,
};
use engine_heightfield::Fall;
use engine_open_world_kit::ground::{GroundTiles, Heightfield};
use engine_render::{
    Access, Allocation, Attach, AttachPoint, Bind, Buffer, ComputePipeline, Count, DepthUse, Draw,
    DrawInput, Encode, FaceCull, Form, FormSide, GpuPart, Install, KindForm, Lifetime, Own,
    OwnFragments, Pass, Pipeline, Prepare, RenderPipeline, RoleFragments, Scene, SceneDraw, Shape,
    TargetName, Texture, Varyings, VertexLayout, wgpu,
};
use glam::Vec3;

use crate::Water;
use crate::gpu::{FallVertex, WATER_LEAF_FIRST, WaterUniform};
use crate::state::WaterState;
use crate::water_blocks::{BlocksGpu, INFO_FORMAT, SURFACE_FORMAT};
use crate::water_falls::{FALL_LAYOUT, sheets};
use crate::water_particles::ParticlesGpu;
use crate::water_settle::{SettleFeedback, SettleGpu};
use crate::water_sim::SimGpu;

const FOAM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

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

pub(crate) fn copied<P>(bound: &Bind<P>) -> Bind<P> {
    match bound {
        Bind::Texture(texture) => Bind::Texture(*texture),
        Bind::Slice {
            texture,
            mips,
            layers,
        } => Bind::Slice {
            texture: *texture,
            mips: mips.clone(),
            layers: layers.clone(),
        },
        Bind::Buffer(buffer) => Bind::Buffer(*buffer),
        Bind::Sampler(sampler) => Bind::Sampler(*sampler),
        Bind::Target(view) => Bind::Target(*view),
        Bind::Asset => Bind::Asset,
    }
}

pub(crate) fn reached(
    cx: &mut Install<'_, WaterGpu>,
    entries: &[Entry],
    all: Vec<(Var, Bind<WaterGpu>)>,
) -> Vec<(Var, Bind<WaterGpu>)> {
    all.into_iter()
        .filter(|(var, _)| entries.iter().any(|&entry| cx.reaches(entry, *var)))
        .collect()
}

fn foam_pipeline(
    cx: &mut Install<'_, WaterGpu>,
    foam_pass: Pass<WaterGpu>,
    foam_all: Vec<(Var, Bind<WaterGpu>)>,
) -> Result<Pipeline<WaterGpu, Own>, Error> {
    use crate::shaders::water::particles;
    let primitive = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        cull_mode: None,
        ..Default::default()
    };
    let added = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    let foam_bound = reached(cx, &[particles::VS_FOAM, particles::FS_FOAM], foam_all);
    cx.render_pipeline(
        foam_pass,
        &RenderPipeline {
            vertex: particles::VS_FOAM,
            fragment: Some(particles::FS_FOAM),
            vertices: &[],
            primitive,
            depth: DepthUse::None,
            blend: Some(wgpu::BlendState {
                color: added,
                alpha: added,
            }),
            writes: wgpu::ColorWrites::ALL,
        },
        &foam_bound,
    )
}

struct Leaves {
    name: &'static str,
    leaves: DrawInput<WaterGpu, Scene>,
    pattern: DrawInput<WaterGpu, Scene>,
}

struct Sheets {
    name: &'static str,
    vertices: DrawInput<WaterGpu, Scene>,
    indices: DrawInput<WaterGpu, Scene>,
}

pub(crate) struct WaterGpu {
    uniform: Buffer<Self>,
    foam: Texture<Self>,
    leaf_args: Buffer<Self>,
    leaf_pattern: Buffer<Self>,
    fall_vertices: Buffer<Self>,
    fall_indices: Buffer<Self>,
    count_leaves: Pass<Self>,
    foam_pass: Pass<Self>,
    counting: ComputePipeline<Self>,
    foaming: Pipeline<Self, Own>,
    surface: Form<Self>,
    falls: Form<Self>,
    leaves: Vec<Leaves>,
    sheets: Vec<Sheets>,
    blocks: BlocksGpu,
    settle: SettleGpu,
    sim: SimGpu,
    particles: ParticlesGpu,
    built: Vec<BuiltFalls>,
    vertices: Vec<FallVertex>,
    indices: Vec<u32>,
    pattern_written: Option<Arc<[u32]>>,
    drawn: bool,
}

impl GpuPart for WaterGpu {
    type Reads = (
        Res<'static, Water>,
        Res<'static, WaterState>,
        Res<'static, SettleFeedback>,
        Res<'static, engine_core::Views>,
        Res<'static, GroundTiles>,
        Res<'static, Heightfield>,
    );
    const NAME: &'static str = "water";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::water::{faces::water as face, foam, refraction, surface};
        let uniform = cx.buffer("water", Lifetime::Kept);
        let foam_texture = cx.texture("water_foam", FOAM_FORMAT, Lifetime::Frame);
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
        let settle = SettleGpu::install(cx)?;
        let sim = SimGpu::install(cx)?;
        let shared = vec![
            (surface::WATER, Bind::Buffer(uniform)),
            (surface::WATER_LIVE, Bind::Buffer(sim.looks())),
        ];
        let mut particles_gpu = ParticlesGpu::install(cx, sim.after(), &shared)?;
        let count_leaves = cx.pass("water_leaf_args").id();
        let foam_pass = cx
            .pass("water_foam")
            .at(AttachPoint::Opaque)
            .after(particles_gpu.after())
            .color(foam_texture, Access::Clear)
            .label("engine/water-foam")
            .id();
        let counting_bound = reached(
            cx,
            &[surface::CS_WATER_LEAF_ARGS],
            vec![
                (surface::WATER, Bind::Buffer(uniform)),
                (surface::WATER_LEAF_ARGS, Bind::Buffer(leaf_args)),
            ],
        );
        let counting =
            cx.compute_pipeline(count_leaves, surface::CS_WATER_LEAF_ARGS, &counting_bound)?;
        let behind = || {
            let mut all = vec![
                (surface::WATER, Bind::Buffer(uniform)),
                (surface::WATER_LIVE, Bind::Buffer(sim.looks())),
                (surface::WATER_LEAF_SPANS, Bind::Buffer(leaf_args)),
                (surface::WATER_DEPTH, Bind::Target(TargetName::DEPTH)),
                (surface::WATER_SCENE, Bind::Target(TargetName::COLOR)),
                (foam::WATER_FOAM_DRAWN, Bind::Texture(foam_texture)),
            ];
            all.extend(particles_gpu.bound());
            all
        };
        let mut foam_all = vec![
            (surface::WATER, Bind::Buffer(uniform)),
            (surface::WATER_LIVE, Bind::Buffer(sim.looks())),
            (surface::WATER_DEPTH, Bind::Target(TargetName::DEPTH)),
        ];
        foam_all.extend(particles_gpu.bound());
        let foaming = foam_pipeline(cx, foam_pass, foam_all)?;
        let surface_form = cx.form(&SURFACE_FORM, &behind())?;
        let fall_form = cx.form(&FALL_FORM, &behind())?;
        let leaves = surface_form
            .passes()
            .map(|pass| Leaves {
                name: pass.scene.name(),
                leaves: cx.draw_input(pass.scene, leaf_args),
                pattern: cx.draw_input(pass.scene, leaf_pattern),
            })
            .collect();
        let sheets = fall_form
            .passes()
            .map(|pass| Sheets {
                name: pass.scene.name(),
                vertices: cx.draw_input(pass.scene, fall_vertices),
                indices: cx.draw_input(pass.scene, fall_indices),
            })
            .collect();
        particles_gpu.foam_args = Some(cx.draw_input(foam_pass, particles_gpu.args()));
        Ok(Self {
            uniform,
            foam: foam_texture,
            leaf_args,
            leaf_pattern,
            fall_vertices,
            fall_indices,
            count_leaves,
            foam_pass,
            counting,
            foaming,
            surface: surface_form,
            falls: fall_form,
            leaves,
            sheets,
            blocks,
            settle,
            sim,
            particles: particles_gpu,
            built: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            pattern_written: None,
            drawn: false,
        })
    }

    fn prepare(
        &mut self,
        (water, state, feedback, views, ground, heightfield): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
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
        self.settle.prepare(&feedback.work, cx);
        self.blocks
            .prepare(&state.blocks, (&ground, &state.terrains), cx);
        self.sim.prepare(&state.live, cx);
        self.particles.prepare(&state.particles, cx);
        let watering = !state.watered.is_empty();
        self.prepare_falls(&state, &ground, cx);
        let terrains = ground.tables.len() as u32;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(
            self.leaf_args,
            u64::from(WATER_LEAF_FIRST + 2 * terrains.max(1)) * 4,
        );
        let size = views.frame().display_size;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate(
            self.foam,
            wgpu::Extent3d {
                width: size.x.max(1),
                height: size.y.max(1),
                depth_or_array_layers: 1,
            },
            1,
        );
        self.drawn = watering;
        if !self.drawn {
            return;
        }
        let running = state.live.work.drawn.as_ref().map(|drawn| {
            let [wrap_x, wrap_z] = drawn.wrap;
            (
                [
                    drawn.origin[0],
                    drawn.origin[1],
                    drawn.cell,
                    drawn.cells as f32,
                ],
                [drawn.datum, wrap_x as f32, wrap_z as f32, drawn.pool as f32],
                [drawn.pixels, 0.0, 0.0, 0.0],
            )
        });
        let [red, green, blue] = water.color;
        let sediment = water.sediment_color;
        let values = WaterUniform {
            color: [red, green, blue, water.clarity],
            absorption: absorption(water.color, water.clarity),
            foam: state.particles.shading(),
            sim: running.map_or([0.0; 4], |(square, _, _)| square),
            sim_datum: running.map_or([0.0; 4], |(_, datum, _)| datum),
            sim_pixels: running.map_or([0.0; 4], |(_, _, pixels)| pixels),
            sediment: [sediment[0], sediment[1], sediment[2], water.sediment],
            lift: [
                heightfield.subdivision_error,
                water.subdivision_error,
                0.0,
                0.0,
            ],
            terrains: [terrains, 0, 0, 0],
        };
        cx.write(self.uniform, 0, &[values]);
        cx.run(self.count_leaves);
        cx.run(self.foam_pass);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if self.settle.encode(pass, cx)
            || self.sim.encode(pass, cx)
            || self.particles.encode(pass, cx)
        {
            return 0;
        }
        if pass == self.count_leaves {
            cx.dispatch(&self.counting, Count::Groups([1, 1, 1]));
            return 0;
        }
        if pass == self.foam_pass {
            let mut draw = cx.render(Attach::Color(self.foam, None));
            self.particles.draw_foam(&mut draw, &self.foaming);
        }
        0
    }
}

impl WaterGpu {
    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if !self.drawn {
            return draw.skip("no water is drawn");
        }
        let name = scene.pass.name();
        let mut draws = 0;
        let leaves = self.leaves.iter().find(|leaves| leaves.name == name);
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
        let sheets = self.sheets.iter().find(|sheets| sheets.name == name);
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
            state
                .terrains
                .iter()
                .filter(|terrain| !terrain.falls.is_empty())
                .map(|terrain| {
                    let drawn = index_of(terrain.entity)
                        .filter(|_| state.watered.contains(&terrain.entity));
                    (terrain, drawn)
                })
        };
        let same = self.built.len() == falling().count()
            && self.built.iter().zip(falling()).all(|(a, (b, drawn))| {
                a.0 == b.entity && a.1 == b.origin && a.2[..] == b.falls[..] && a.3 == drawn
            });
        if same || (self.built.is_empty() && falling().next().is_none()) {
            return;
        }
        let runs = sheets(
            falling().map(|(terrain, drawn)| (terrain.origin, &*terrain.falls, drawn.unwrap_or(0))),
            &mut self.vertices,
            &mut self.indices,
        );
        self.built = falling()
            .zip(runs)
            .map(|((terrain, drawn), run)| {
                (
                    terrain.entity,
                    terrain.origin,
                    terrain.falls.clone(),
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
