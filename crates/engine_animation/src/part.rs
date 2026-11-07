use std::collections::BTreeMap;
use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::math::max;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, ColorPass, Cull, DrawKind, DrawnPoses, Error,
    GlobalTransforms, MaterialHandle, MeshHandle, MeshVertex, Mobility, PassName, PassViews,
    PrimaryView, ScenePass, SceneView, SceneViews, gpu_struct,
};
use engine_places_rendering::animation::{Animator, Rig, RigPart, VertexSkin};
use engine_render::{
    Bind, Buffer, ComputePipeline, Count, DepthUse, Draw, DrawInput, Encode, GpuPart, Install,
    Lifetime, Pass, Pipeline, Prepare, RenderPipeline, Scene, SceneDraw, Shape, VertexLayout,
    pack_color, pack_surface, planes, wgpu,
};
use glam::{Mat4, Vec3, Vec4};

use crate::shaders;

pub(crate) const SKIN: DrawKind = DrawKind("skin");

engine_core::gpu_consts! {
    pub(crate) const SKIN_GROUP: u32 = 64;
}

gpu_struct! {
    pub(crate) struct SkinFrame {
        pub(crate) vertices: u32,
        pub(crate) parts: u32,
        pub(crate) pad0: u32,
        pub(crate) pad1: u32,
    }
}

gpu_struct! {
    pub(crate) struct SkinPart {
        pub(crate) rows_at: u32,
        pub(crate) previous_at: u32,
        pub(crate) rest_at: u32,
        pub(crate) skinned_at: u32,
    }
}

gpu_struct! {
    pub(crate) struct RestVertex {
        pub(crate) position_u: [f32; 4],
        pub(crate) normal_v: [f32; 4],
        pub(crate) joints: [u32; 4],
        pub(crate) weights: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct JointRows {
        pub(crate) r0: [f32; 4],
        pub(crate) r1: [f32; 4],
        pub(crate) r2: [f32; 4],
    }
}

const SKINNED_BYTES: u64 = 40;
const ARGS_BYTES: u64 = 20;
const OWNER_BYTES: u64 = 4;
const COLOR_BYTES: u64 = 16;

const CASTER_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState {
    constant: 1,
    slope_scale: 0.0,
    clamp: 0.0,
};

const LIT_STREAMS: &[VertexLayout] = &[VertexLayout::Buffer(wgpu::VertexBufferLayout {
    array_stride: SKINNED_BYTES,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Snorm16x2,
            offset: 12,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 16,
            shader_location: 7,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 24,
            shader_location: 3,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 36,
            shader_location: 2,
        },
    ],
})];

const CASTER_STREAMS: &[VertexLayout] = &[VertexLayout::Buffer(wgpu::VertexBufferLayout {
    array_stride: SKINNED_BYTES,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 16,
            shader_location: 7,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 36,
            shader_location: 3,
        },
    ],
})];

struct Slot {
    entity: Entity,
    mesh: MeshHandle,
    material: MaterialHandle,
    rest_at: u32,
    skinned_at: u32,
    reach: usize,
    masked: bool,
    double_sided: bool,
    bounds: (Vec3, Vec3),
}

type LayoutKey = Vec<(Entity, usize, u64, Option<u64>)>;

#[derive(Default)]
struct Held {
    key: Option<LayoutKey>,
    slots: Vec<Slot>,
    reaches: Vec<Vec<f32>>,
    vertices: u32,
    lit: Vec<u32>,
    cast: Vec<u32>,
    views: Vec<(&'static str, SceneView)>,
}

struct Buffers {
    frame: Buffer<SkinningGpu>,
    parts: Buffer<SkinningGpu>,
    owners: Buffer<SkinningGpu>,
    rest: Buffer<SkinningGpu>,
    rows: Buffer<SkinningGpu>,
    skinned: Buffer<SkinningGpu>,
    colors: Buffer<SkinningGpu>,
    args: Buffer<SkinningGpu>,
}

struct Streams {
    skinned: DrawInput<SkinningGpu, Scene>,
    args: DrawInput<SkinningGpu, Scene>,
}

enum Drawing {
    Lit([Pipeline<SkinningGpu, Scene>; 2]),
    Casters([Pipeline<SkinningGpu, Scene>; 2]),
}

struct ScenePipelines {
    name: &'static str,
    drawing: Drawing,
    streams: Streams,
}

pub(crate) struct SkinningGpu {
    skin: Pass<Self>,
    skinning: ComputePipeline<Self>,
    scenes: Vec<ScenePipelines>,
    buffers: Buffers,
    held: Held,
}

struct Posed {
    rows: Vec<JointRows>,
    spheres: Vec<Vec4>,
    ranges: Vec<(Entity, Range<u32>, bool)>,
}

fn rows_of(matrix: Mat4) -> JointRows {
    let rows = matrix.transpose();
    JointRows {
        r0: rows.x_axis.to_array(),
        r1: rows.y_axis.to_array(),
        r2: rows.z_axis.to_array(),
    }
}

fn rest_vertex((skin, vertex): (&VertexSkin, &MeshVertex)) -> RestVertex {
    let [x, y, z] = vertex.position;
    let [nx, ny, nz] = vertex.normal;
    let [u, v] = vertex.uv;
    let total: f32 = skin.weights.iter().sum();
    RestVertex {
        position_u: [x, y, z, u],
        normal_v: [nx, ny, nz, v],
        joints: skin.joints,
        weights: skin.weights.map(|weight| weight / total),
    }
}

fn reach_of(rig: &Rig, part: &RigPart, vertices: &[MeshVertex]) -> Vec<f32> {
    let mut reach = vec![-1.0; rig.palette.len()];
    for (skin, vertex) in part.skin.iter().zip(vertices) {
        let position = Vec3::from_array(vertex.position);
        for (&joint, &weight) in skin.joints.iter().zip(&skin.weights) {
            if weight <= 0.0 {
                continue;
            }
            let joint = joint as usize;
            if let (Some(bound), Some(held)) = (rig.palette.get(joint), reach.get_mut(joint)) {
                let inverse = Mat4::from_cols_array_2d(&bound.inverse_bind);
                *held = max(*held, inverse.transform_point3(position).length());
            }
        }
    }
    reach
}

fn layout_key(
    listed: &[(Entity, &Animator)],
    residency: &AssetResidency,
    assets: &AssetLibrary,
) -> LayoutKey {
    let mut key = Vec::new();
    for (entity, animator) in listed {
        let rig = animator.rig();
        for (index, part) in rig.parts.iter().enumerate() {
            let shaped = assets
                .mesh(part.mesh)
                .is_some_and(|mesh| mesh.vertices().len() == part.skin.len());
            if shaped && residency.resident(part.mesh) && residency.material_resident(part.material)
            {
                key.push((
                    *entity,
                    index,
                    animator.rig_revision(),
                    assets.revision_of(part.mesh),
                ));
            }
        }
    }
    key
}

fn outside(planes: &[[f32; 4]; 6], center: Vec3, half: Vec3) -> bool {
    planes.iter().any(|&[x, y, z, d]| {
        let normal = Vec3::new(x, y, z);
        normal.dot(center) + normal.abs().dot(half) + d < 0.0
    })
}

fn pose_all(
    listed: &[(Entity, &Animator)],
    slots: &[Slot],
    globals: &GlobalTransforms,
    poses: &DrawnPoses,
) -> Posed {
    let mut posed = Posed {
        rows: Vec::new(),
        spheres: Vec::new(),
        ranges: Vec::new(),
    };
    let mut previous = Vec::new();
    let mut nodes = Vec::new();
    let mut drawn = Vec::new();
    let mut before = Vec::new();
    for &(entity, animator) in listed {
        if slots
            .binary_search_by_key(&entity, |slot| slot.entity)
            .is_err()
        {
            continue;
        }
        let start = posed.rows.len() as u32;
        let joints = animator.joint_nodes();
        let palette = &animator.rig().palette;
        nodes.clear();
        nodes.extend(
            palette
                .iter()
                .filter_map(|bound| joints.get(bound.joint as usize).copied()),
        );
        let mut shown = globals.drawn(entity) && !joints.is_empty() && nodes.len() == palette.len();
        poses.drawn_poses(globals, &nodes, &mut drawn);
        poses.previous_drawn_poses(globals, &nodes, &mut before);
        for (at, bound) in palette.iter().enumerate() {
            let now = drawn.get(at).copied().flatten();
            shown &= now.is_some();
            let joint = now.unwrap_or(Mat4::IDENTITY);
            let was = before.get(at).copied().flatten().unwrap_or(joint);
            let inverse = Mat4::from_cols_array_2d(&bound.inverse_bind);
            posed.rows.push(rows_of(joint * inverse));
            previous.push(rows_of(was * inverse));
            let scale = Vec3::new(
                joint.x_axis.truncate().length_squared(),
                joint.y_axis.truncate().length_squared(),
                joint.z_axis.truncate().length_squared(),
            )
            .max_element()
            .sqrt();
            posed.spheres.push(joint.w_axis.truncate().extend(scale));
        }
        let range = start..posed.rows.len() as u32;
        posed.ranges.push((entity, range, shown));
    }
    posed.rows.append(&mut previous);
    posed
}

fn part_box(reach: &[f32], spheres: &[Vec4]) -> (Vec3, Vec3) {
    let mut low = Vec3::splat(f32::INFINITY);
    let mut high = Vec3::splat(f32::NEG_INFINITY);
    for (&radius, sphere) in reach.iter().zip(spheres) {
        if radius < 0.0 {
            continue;
        }
        let extent = Vec3::splat(radius * sphere.w);
        low = low.min(sphere.truncate() - extent);
        high = high.max(sphere.truncate() + extent);
    }
    if low.x > high.x {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    ((low + high) * 0.5, (high - low) * 0.5)
}

impl SkinningGpu {
    fn lay_out(
        &mut self,
        listed: &[(Entity, &Animator)],
        key: &LayoutKey,
        assets: &AssetLibrary,
        cx: &mut Prepare<'_, Self>,
    ) {
        let held = &mut self.held;
        held.slots.clear();
        held.reaches.clear();
        let reaches = &mut held.reaches;
        let mut shared: BTreeMap<(MeshHandle, Option<u64>), Vec<(&Rig, &RigPart, u32, usize)>> =
            BTreeMap::new();
        let mut rest: Vec<RestVertex> = Vec::new();
        let mut owners: Vec<u32> = Vec::new();
        let mut palette = 0_u64;
        for &(entity, index, _, revision) in key {
            let Ok(at) = listed.binary_search_by_key(&entity, |(known, _)| *known) else {
                continue;
            };
            let rig: &Rig = listed[at].1.rig();
            let Some((part, mesh)) = rig
                .parts
                .get(index)
                .and_then(|part| Some((part, assets.mesh(part.mesh)?)))
            else {
                continue;
            };
            let vertices = mesh.vertices();
            let same = shared.entry((part.mesh, revision)).or_default();
            let found = same.iter().find(|(other, kept, _, _)| {
                kept.skin == part.skin && other.palette == rig.palette
            });
            let (rest_at, reach) = match found {
                Some(&(_, _, rest_at, reach)) => (rest_at, reach),
                None => {
                    let rest_at = rest.len() as u32;
                    rest.extend(part.skin.iter().zip(vertices).map(rest_vertex));
                    reaches.push(reach_of(rig, part, vertices));
                    same.push((rig, part, rest_at, reaches.len() - 1));
                    (rest_at, reaches.len() - 1)
                }
            };
            let skinned_at = owners.len() as u32;
            let slot = held.slots.len() as u32;
            owners.resize(owners.len() + vertices.len(), slot);
            if held.slots.last().is_none_or(|last| last.entity != entity) {
                palette += rig.palette.len() as u64;
            }
            held.slots.push(Slot {
                entity,
                mesh: part.mesh,
                material: part.material,
                rest_at,
                skinned_at,
                reach,
                masked: false,
                double_sided: false,
                bounds: (Vec3::ZERO, Vec3::ZERO),
            });
        }
        held.vertices = owners.len() as u32;
        let buffers = &self.buffers;
        let parts = held.slots.len().max(1) as u64;
        cx.allocate_buffer(buffers.frame, size_of::<SkinFrame>() as u64);
        cx.allocate_buffer(buffers.owners, owners.len().max(1) as u64 * OWNER_BYTES);
        cx.allocate_buffer(
            buffers.rest,
            rest.len().max(1) as u64 * size_of::<RestVertex>() as u64,
        );
        cx.allocate_buffer(buffers.skinned, owners.len().max(1) as u64 * SKINNED_BYTES);
        cx.allocate_buffer(
            buffers.rows,
            (palette * 2).max(1) * size_of::<JointRows>() as u64,
        );
        cx.allocate_buffer(buffers.parts, parts * size_of::<SkinPart>() as u64);
        cx.allocate_buffer(buffers.colors, parts * COLOR_BYTES);
        cx.allocate_buffer(buffers.args, parts * ARGS_BYTES);
        if !owners.is_empty() {
            cx.write(buffers.owners, 0, &owners);
            cx.write(buffers.rest, 0, &rest);
        }
    }

    fn stage(
        &mut self,
        posed: &Posed,
        (residency, assets, view): (&AssetResidency, &AssetLibrary, &PrimaryView),
        cx: &mut Prepare<'_, Self>,
    ) {
        let held = &mut self.held;
        let Posed {
            rows,
            spheres,
            ranges,
        } = posed;
        let total = spheres.len() as u32;
        debug_assert_eq!(rows.len(), spheres.len() * 2, "skin_rows_hold_previous");
        let frustum = planes(view.view_projection());
        let count = held.slots.len();
        let mut parts = Vec::with_capacity(count);
        let mut colors = Vec::with_capacity(count);
        let mut args = Vec::with_capacity(count);
        for (index, slot) in held.slots.iter_mut().enumerate() {
            let found = ranges
                .binary_search_by_key(&slot.entity, |(entity, _, _)| *entity)
                .ok()
                .map(|at| &ranges[at]);
            let (start, shown) =
                found.map_or((0, false), |(_, range, shown)| (range.start, *shown));
            let reach = held.reaches.get(slot.reach).map_or(&[][..], Vec::as_slice);
            let from_spheres = spheres.get(start as usize..).unwrap_or_default();
            let (center, half) = part_box(reach, from_spheres);
            slot.bounds = (center, half);
            parts.push(SkinPart {
                rows_at: start,
                previous_at: start + total,
                rest_at: slot.rest_at,
                skinned_at: slot.skinned_at,
            });
            let (color, surface) = match assets.material(slot.material) {
                Some(material) => {
                    slot.masked = matches!(material.alpha_mode(), AlphaMode::Mask { .. });
                    slot.double_sided = material.cull() == Cull::None;
                    let surface = pack_surface(material.surface());
                    (pack_color(material.color()), surface)
                }
                None => ([0, 0], 0),
            };
            colors.push([color[0], color[1], surface, 0]);
            let (indices, first) = residency
                .mesh(slot.mesh)
                .map_or((0, 0), |mesh| (mesh.index_count(), mesh.first_index()));
            args.push([indices, 1, first, 0, 0]);
            if shown {
                held.cast.push(index as u32);
                if !outside(&frustum, center, half) {
                    held.lit.push(index as u32);
                }
            }
        }
        let buffers = &self.buffers;
        cx.write(buffers.rows, 0, rows);
        cx.write(buffers.parts, 0, &parts);
        cx.write(buffers.colors, 0, &colors);
        cx.write(buffers.args, 0, &args);
        cx.write(
            buffers.frame,
            0,
            &[SkinFrame {
                vertices: held.vertices,
                parts: count as u32,
                pad0: 0,
                pad1: 0,
            }],
        );
    }

    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        let Some(joined) = self
            .scenes
            .iter()
            .find(|joined| joined.name == scene.pass.name())
        else {
            return 0;
        };
        match &joined.drawing {
            Drawing::Lit(lit) => self.draw_parts(draw, &joined.streams, &self.held.lit, |slot| {
                (&lit[usize::from(slot.double_sided)], Some(slot.material))
            }),
            Drawing::Casters(casting) => {
                let Some(view) = self.cast_view(scene) else {
                    return 0;
                };
                let frustum = view.planes.unwrap_or_else(|| {
                    planes(Mat4::from_cols_array_2d(&view.view_projection))
                });
                let cast: Vec<u32> = self
                    .held
                    .cast
                    .iter()
                    .copied()
                    .filter(|&index| {
                        let (center, half) = self.held.slots[index as usize].bounds;
                        !outside(&frustum, center, half)
                    })
                    .collect();
                self.draw_parts(draw, &joined.streams, &cast, |slot| {
                    (
                        &casting[usize::from(slot.masked)],
                        slot.masked.then_some(slot.material),
                    )
                })
            }
        }
    }

    fn cast_view(&self, scene: &SceneDraw) -> Option<&SceneView> {
        let index = scene.view? as usize;
        self.held
            .views
            .iter()
            .filter(|(pass, _)| *pass == scene.pass.name())
            .nth(index)
            .map(|(_, view)| view)
            .filter(|view| view.takes.is_none_or(|takes| takes == Mobility::Moving))
    }

    fn draw_parts<'p>(
        &'p self,
        draw: &mut Draw<'_, Self, Scene>,
        streams: &Streams,
        listed: &[u32],
        chosen: impl Fn(&Slot) -> (&'p Pipeline<Self, Scene>, Option<MaterialHandle>),
    ) -> u32 {
        for &index in listed {
            let slot = &self.held.slots[index as usize];
            let (pipeline, material) = chosen(slot);
            draw.draw(
                pipeline,
                Shape::MeshIndirect {
                    mesh: slot.mesh,
                    material,
                    args: streams.args,
                    offset: u64::from(index) * ARGS_BYTES,
                    streams: &[(streams.skinned, u64::from(slot.skinned_at) * SKINNED_BYTES)],
                },
            );
        }
        listed.len() as u32
    }
}

fn lit_pipeline(
    cx: &mut Install<'_, SkinningGpu>,
    scene: Scene,
    double_sided: bool,
    colors: Buffer<SkinningGpu>,
) -> Result<Pipeline<SkinningGpu, Scene>, Error> {
    use shaders::animation::draw;
    cx.render_pipeline(
        scene,
        &RenderPipeline {
            vertex: draw::VS_SKINNED,
            fragment: Some(draw::FS_SKINNED),
            debug: None,
            vertices: LIT_STREAMS,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: (!double_sided).then_some(wgpu::Face::Back),
                ..Default::default()
            },
            depth: DepthUse::Write {
                bias: wgpu::DepthBiasState::default(),
            },
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        },
        &[(draw::COLORS, Bind::Buffer(colors))],
    )
}

fn caster_pipelines(
    cx: &mut Install<'_, SkinningGpu>,
    scene: Scene,
    colors: Buffer<SkinningGpu>,
) -> Result<[Pipeline<SkinningGpu, Scene>; 2], Error> {
    use shaders::animation::casters;
    let desc = |vertex, fragment| RenderPipeline {
        vertex,
        fragment,
        debug: None,
        vertices: CASTER_STREAMS,
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::Write { bias: CASTER_BIAS },
        blend: None,
        writes: wgpu::ColorWrites::empty(),
    };
    Ok([
        cx.render_pipeline(scene, &desc(casters::VS_CASTER, None), &[])?,
        cx.render_pipeline(
            scene,
            &desc(casters::VS_CASTER_MASKED, Some(casters::FS_CASTER_MASKED)),
            &[(casters::COLORS, Bind::Buffer(colors))],
        )?,
    ])
}

fn scene_pipelines(
    cx: &mut Install<'_, SkinningGpu>,
    buffers: &Buffers,
) -> Result<Vec<ScenePipelines>, Error> {
    let listed: Vec<ScenePass> = cx
        .passes()
        .listed()
        .iter()
        .filter(|listed| listed.bit.is_some() && (listed.pass.takes)(SKIN))
        .copied()
        .collect();
    let mut scenes = Vec::with_capacity(listed.len());
    for listed in listed {
        let scene = Scene::named(listed.name);
        let drawing = if listed.name == ColorPass::NAME {
            Drawing::Lit([
                lit_pipeline(cx, scene, false, buffers.colors)?,
                lit_pipeline(cx, scene, true, buffers.colors)?,
            ])
        } else if listed.pass.views == PassViews::Owned && !listed.pass.reads.base_color {
            Drawing::Casters(caster_pipelines(cx, scene, buffers.colors)?)
        } else {
            continue;
        };
        let streams = Streams {
            skinned: cx.draw_input(scene, buffers.skinned),
            args: cx.draw_input(scene, buffers.args),
        };
        scenes.push(ScenePipelines {
            name: listed.name,
            drawing,
            streams,
        });
    }
    Ok(scenes)
}

impl GpuPart for SkinningGpu {
    type Reads = (
        Query<'static, 'static, (Entity, &'static Animator)>,
        Res<'static, GlobalTransforms>,
        Res<'static, DrawnPoses>,
        Res<'static, AssetResidency>,
        Res<'static, AssetLibrary>,
        Res<'static, PrimaryView>,
        Query<'static, 'static, &'static SceneViews>,
    );
    const NAME: &'static str = "animation";
    const LAZY: bool = true;
    const TURNS_ON_IN_PLAY: bool = true;
    const LABELS: &'static [&'static str] = &["engine/animation"];

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::animation::skin;
        let buffers = Buffers {
            frame: cx.buffer("skin_frame", Lifetime::Kept),
            parts: cx.buffer("skin_parts", Lifetime::Kept),
            owners: cx.buffer("skin_owners", Lifetime::Kept),
            rest: cx.buffer("skin_rest", Lifetime::Kept),
            rows: cx.buffer("skin_rows", Lifetime::Kept),
            skinned: cx.buffer("skin_vertices", Lifetime::Kept),
            colors: cx.buffer("skin_colors", Lifetime::Kept),
            args: cx.buffer("skin_args", Lifetime::Kept),
        };
        let pass = cx
            .pass("skin")
            .label("engine/animation")
            .span("skinning")
            .id();
        let skinning = cx.compute_pipeline(
            pass,
            skin::SKIN_VERTICES,
            &[
                (skin::FRAME, Bind::Buffer(buffers.frame)),
                (skin::PARTS, Bind::Buffer(buffers.parts)),
                (skin::OWNERS, Bind::Buffer(buffers.owners)),
                (skin::REST, Bind::Buffer(buffers.rest)),
                (skin::ROWS, Bind::Buffer(buffers.rows)),
                (skin::SKINNED, Bind::Buffer(buffers.skinned)),
            ],
        )?;
        let scenes = scene_pipelines(cx, &buffers)?;
        Ok(Self {
            skin: pass,
            skinning,
            scenes,
            buffers,
            held: Held::default(),
        })
    }

    fn prepare(
        &mut self,
        (animators, globals, poses, residency, assets, view, published): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        let mut listed: Vec<(Entity, &Animator)> = animators.iter().collect();
        listed.sort_unstable_by_key(|(entity, _)| *entity);
        let key = layout_key(&listed, &residency, &assets);
        if self.held.key.as_ref() != Some(&key) {
            self.lay_out(&listed, &key, &assets, cx);
            self.held.key = Some(key);
        }
        self.held.lit.clear();
        self.held.cast.clear();
        self.held.views.clear();
        self.held.views.extend(
            published
                .iter()
                .flat_map(|views| views.views.iter().map(|view| (views.pass, *view))),
        );
        if self.held.slots.is_empty() {
            return;
        }
        let posed = pose_all(&listed, &self.held.slots, &globals, &poses);
        self.stage(&posed, (&*residency, &*assets, &*view), cx);
        cx.run(self.skin);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass != self.skin {
            return 0;
        }
        let vertices = self.held.vertices.div_ceil(SKIN_GROUP);
        cx.dispatch(&self.skinning, Count::Groups([vertices, 1, 1]));
        0
    }
}
