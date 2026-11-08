use std::collections::BTreeMap;
use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::math::max;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, AttachPoint, Cull, DrawKind, DrawnPoses, DrawnTable,
    Error, GlobalTransforms, MaterialHandle, MeshHandle, MeshVertex, PassViews, ScenePass,
    TargetName, Views, gpu_struct,
};
use engine_places_rendering::animation::{Animator, Rig, RigPart, VertexSkin};
use engine_render::{
    Bind, Buffer, ComputePipeline, Count, DepthUse, Draw, DrawInput, Encode, GpuPart, Install,
    Lifetime, Pass, Pipeline, Prepare, RenderPipeline, Scene, SceneDraw, Shape, VertexLayout,
    planes, wgpu,
};
use glam::{Mat4, Vec3, Vec4};

use crate::shaders;

type SharedParts<'a> = BTreeMap<(MeshHandle, Option<u64>), Vec<(&'a Rig, &'a RigPart, u32, usize)>>;

pub(crate) const SKIN: DrawKind = DrawKind("skin");

engine_core::gpu_consts! {
    pub(crate) const SKIN_GROUP: u32 = 64;
    pub(crate) const ARGS_WORDS: u32 = 5;
}

gpu_struct! {
    pub(crate) struct SkinFrame {
        pub(crate) vertices: u32,
        pub(crate) parts: u32,
        pub(crate) views: u32,
        pub(crate) pad: u32,
    }
}

gpu_struct! {
    pub(crate) struct SkinPart {
        pub(crate) rows_at: u32,
        pub(crate) previous_at: u32,
        pub(crate) rest_at: u32,
        pub(crate) skinned_at: u32,
        pub(crate) drawn: u32,
        pub(crate) indices: u32,
        pub(crate) first: u32,
        pub(crate) pad: u32,
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

gpu_struct! {
    pub(crate) struct SkinBound {
        pub(crate) center: [f32; 4],
        pub(crate) half: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct SkinView {
        pub(crate) planes: [[f32; 4]; 6],
    }
}

const SKINNED_BYTES: u64 = 40;
const ARGS_BYTES: u64 = ARGS_WORDS as u64 * 4;
const OWNER_BYTES: u64 = 4;

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
    drawn: u32,
    animator: Entity,
    mesh: MeshHandle,
    material: Option<MaterialHandle>,
    rest_at: u32,
    skinned_at: u32,
    reach: usize,
    masked: bool,
    double_sided: bool,
}

type LayoutKey = Vec<(Entity, u32, usize, MeshHandle, u64, Option<u64>)>;

#[derive(Default)]
struct Held {
    key: Option<LayoutKey>,
    slots: Vec<Slot>,
    reaches: Vec<Vec<f32>>,
    vertices: u32,
    views: u32,
}

struct Buffers {
    frame: Buffer<SkinningGpu>,
    parts: Buffer<SkinningGpu>,
    owners: Buffer<SkinningGpu>,
    rest: Buffer<SkinningGpu>,
    rows: Buffer<SkinningGpu>,
    skinned: Buffer<SkinningGpu>,
    bounds: Buffer<SkinningGpu>,
    views: Buffer<SkinningGpu>,
    args: Buffer<SkinningGpu>,
}

struct Streams {
    skinned: DrawInput<SkinningGpu, Scene>,
    args: DrawInput<SkinningGpu, Scene>,
}

enum Drawing {
    Placed([Pipeline<SkinningGpu, Scene>; 4]),
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
    arguments: ComputePipeline<Self>,
    scenes: Vec<ScenePipelines>,
    buffers: Buffers,
    held: Held,
}

struct Posed {
    rows: Vec<JointRows>,
    spheres: Vec<Vec4>,
    ranges: Vec<(Entity, Range<u32>, bool)>,
}

type Animators<'w, 's> = Query<'w, 's, (Entity, &'static Animator)>;

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

fn layout_key(table: &DrawnTable, animators: &Animators<'_, '_>, assets: &AssetLibrary) -> LayoutKey {
    let mut key = Vec::new();
    for (slot, entity, drawn) in table.rows(SKIN) {
        let Ok((_, animator)) = animators.get(drawn.node) else {
            continue;
        };
        let Some(index) = animator.part_nodes().iter().position(|part| *part == entity) else {
            continue;
        };
        let (Some(part), Some(geometry)) = (animator.rig().parts.get(index), drawn.geometry) else {
            continue;
        };
        let shaped = assets
            .mesh(geometry)
            .is_some_and(|mesh| mesh.vertices().len() == part.skin.len());
        if shaped {
            key.push((
                drawn.node,
                slot,
                index,
                geometry,
                animator.rig_revision(),
                assets.revision_of(geometry),
            ));
        }
    }
    key.sort_unstable_by_key(|&(animator, slot, ..)| (animator, slot));
    key
}

fn view_planes(views: &Views) -> Vec<SkinView> {
    views
        .drawn()
        .iter()
        .map(|view| SkinView {
            planes: view
                .planes
                .unwrap_or_else(|| planes(view.view_projection)),
        })
        .collect()
}

fn pose_all(
    animators: &Animators<'_, '_>,
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
    let mut last = None;
    for slot in slots {
        if last == Some(slot.animator) {
            continue;
        }
        last = Some(slot.animator);
        let Ok((entity, animator)) = animators.get(slot.animator) else {
            continue;
        };
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
        key: &LayoutKey,
        animators: &Animators<'_, '_>,
        assets: &AssetLibrary,
        cx: &mut Prepare<'_, Self>,
    ) {
        let held = &mut self.held;
        held.slots.clear();
        held.reaches.clear();
        let reaches = &mut held.reaches;
        let mut shared: SharedParts<'_> = BTreeMap::new();
        let mut rest: Vec<RestVertex> = Vec::new();
        let mut owners: Vec<u32> = Vec::new();
        let mut palette = 0_u64;
        for &(node, drawn, index, geometry, _, revision) in key {
            let Ok((_, animator)) = animators.get(node) else {
                continue;
            };
            let rig: &Rig = animator.rig();
            let Some((part, mesh)) = rig
                .parts
                .get(index)
                .and_then(|part| Some((part, assets.mesh(geometry)?)))
            else {
                continue;
            };
            let vertices = mesh.vertices();
            let same = shared.entry((geometry, revision)).or_default();
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
            if held.slots.last().is_none_or(|last| last.animator != node) {
                palette += rig.palette.len() as u64;
            }
            held.slots.push(Slot {
                drawn,
                animator: node,
                mesh: geometry,
                material: None,
                rest_at,
                skinned_at,
                reach,
                masked: false,
                double_sided: false,
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
        cx.allocate_buffer(buffers.bounds, parts * size_of::<SkinBound>() as u64);
        if !owners.is_empty() {
            cx.write(buffers.owners, 0, &owners);
            cx.write(buffers.rest, 0, &rest);
        }
    }

    fn stage(
        &mut self,
        posed: &Posed,
        (table, residency, assets): (&DrawnTable, &AssetResidency, &AssetLibrary),
        views: &[SkinView],
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
        let count = held.slots.len();
        let mut parts = Vec::with_capacity(count);
        let mut bounds = Vec::with_capacity(count);
        for slot in &mut held.slots {
            let found = ranges
                .binary_search_by_key(&slot.animator, |(entity, _, _)| *entity)
                .ok()
                .map(|at| &ranges[at]);
            let (start, shown) =
                found.map_or((0, false), |(_, range, shown)| (range.start, *shown));
            let reach = held.reaches.get(slot.reach).map_or(&[][..], Vec::as_slice);
            let from_spheres = spheres.get(start as usize..).unwrap_or_default();
            let (center, half) = part_box(reach, from_spheres);
            slot.material = table.drawn(slot.drawn).and_then(|drawn| drawn.material);
            let material = slot.material.and_then(|material| assets.material(material));
            slot.masked =
                material.is_some_and(|material| matches!(material.alpha_mode(), AlphaMode::Mask { .. }));
            slot.double_sided = material.is_some_and(|material| material.cull() == Cull::None);
            let (indices, first) = residency
                .mesh(slot.mesh)
                .map_or((0, 0), |mesh| (mesh.index_count(), mesh.first_index()));
            parts.push(SkinPart {
                rows_at: start,
                previous_at: start + total,
                rest_at: slot.rest_at,
                skinned_at: slot.skinned_at,
                drawn: slot.drawn,
                indices,
                first,
                pad: 0,
            });
            bounds.push(SkinBound {
                center: center.extend(f32::from(u8::from(shown))).to_array(),
                half: half.extend(0.0).to_array(),
            });
        }
        let buffers = &self.buffers;
        cx.allocate_buffer(
            buffers.views,
            views.len().max(1) as u64 * size_of::<SkinView>() as u64,
        );
        cx.allocate_buffer(
            buffers.args,
            (count * views.len()).max(1) as u64 * ARGS_BYTES,
        );
        cx.write(buffers.rows, 0, rows);
        cx.write(buffers.parts, 0, &parts);
        cx.write(buffers.bounds, 0, &bounds);
        cx.write(buffers.views, 0, views);
        cx.write(
            buffers.frame,
            0,
            &[SkinFrame {
                vertices: held.vertices,
                parts: count as u32,
                views: views.len() as u32,
                pad: 0,
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
        let Some(view) = scene.index.filter(|&index| index < self.held.views) else {
            return 0;
        };
        let base = u64::from(view) * self.held.slots.len() as u64;
        let mut drawn = 0;
        for (index, slot) in self.held.slots.iter().enumerate() {
            let (pipeline, material) = match &joined.drawing {
                Drawing::Placed(placed) => {
                    if slot.material.is_none() {
                        continue;
                    }
                    let look = usize::from(slot.masked) << 1 | usize::from(slot.double_sided);
                    (&placed[look], slot.material)
                }
                Drawing::Casters(casting) => (
                    &casting[usize::from(slot.masked)],
                    slot.material.filter(|_| slot.masked),
                ),
            };
            draw.draw(
                pipeline,
                Shape::MeshIndirect {
                    mesh: slot.mesh,
                    material,
                    args: joined.streams.args,
                    offset: (base + index as u64) * ARGS_BYTES,
                    streams: &[(
                        joined.streams.skinned,
                        u64::from(slot.skinned_at) * SKINNED_BYTES,
                    )],
                },
            );
            drawn += 1;
        }
        drawn
    }
}

fn placed_pipelines(
    cx: &mut Install<'_, SkinningGpu>,
    scene: Scene,
    lit: bool,
) -> Result<[Pipeline<SkinningGpu, Scene>; 4], Error> {
    use shaders::animation::draw;
    let desc = |masked: bool, double_sided: bool| RenderPipeline {
        vertex: draw::VS_SKINNED,
        fragment: if lit {
            Some(draw::FS_SKINNED)
        } else {
            masked.then_some(draw::FS_SKINNED_DEPTH)
        },
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
    };
    Ok([
        cx.render_pipeline(scene, &desc(false, false), &[])?,
        cx.render_pipeline(scene, &desc(false, true), &[])?,
        cx.render_pipeline(scene, &desc(true, false), &[])?,
        cx.render_pipeline(scene, &desc(true, true), &[])?,
    ])
}

fn caster_pipelines(
    cx: &mut Install<'_, SkinningGpu>,
    scene: Scene,
) -> Result<[Pipeline<SkinningGpu, Scene>; 2], Error> {
    use shaders::animation::casters;
    let desc = |vertex, fragment| RenderPipeline {
        vertex,
        fragment,
        vertices: CASTER_STREAMS,
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::Write {
            bias: wgpu::DepthBiasState::default(),
        },
        blend: None,
        writes: wgpu::ColorWrites::empty(),
    };
    Ok([
        cx.render_pipeline(scene, &desc(casters::VS_CASTER, None), &[])?,
        cx.render_pipeline(
            scene,
            &desc(casters::VS_CASTER_MASKED, Some(casters::FS_CASTER_MASKED)),
            &[],
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
        let (owned, lit) = (
            listed.pass.views == PassViews::Owned,
            listed.pass.writes.contains(TargetName::COLOR),
        );
        let drawing = match (owned, lit) {
            (true, false) => Drawing::Casters(caster_pipelines(cx, scene)?),
            (false, true) if listed.pass.at != AttachPoint::Opaque => continue,
            (false, _) if listed.pass.at != AttachPoint::Transparent => {
                Drawing::Placed(placed_pipelines(cx, scene, lit)?)
            }
            (false | true, true | false) => continue,
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
        Res<'static, DrawnTable>,
        Res<'static, GlobalTransforms>,
        Res<'static, DrawnPoses>,
        Res<'static, AssetResidency>,
        Res<'static, AssetLibrary>,
        Res<'static, Views>,
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
            bounds: cx.buffer("skin_bounds", Lifetime::Kept),
            views: cx.buffer("skin_views", Lifetime::Kept),
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
        let arguments = cx.compute_pipeline(
            pass,
            skin::SKIN_ARGS,
            &[
                (skin::FRAME, Bind::Buffer(buffers.frame)),
                (skin::PARTS, Bind::Buffer(buffers.parts)),
                (skin::BOUNDS, Bind::Buffer(buffers.bounds)),
                (skin::VIEWS, Bind::Buffer(buffers.views)),
                (skin::ARGS, Bind::Buffer(buffers.args)),
            ],
        )?;
        let scenes = scene_pipelines(cx, &buffers)?;
        Ok(Self {
            skin: pass,
            skinning,
            arguments,
            scenes,
            buffers,
            held: Held::default(),
        })
    }

    fn prepare(
        &mut self,
        (animators, table, globals, poses, residency, assets, views): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        let key = layout_key(&table, &animators, &assets);
        if self.held.key.as_ref() != Some(&key) {
            self.lay_out(&key, &animators, &assets, cx);
            self.held.key = Some(key);
        }
        let planes = view_planes(&views);
        self.held.views = planes.len() as u32;
        if self.held.slots.is_empty() {
            return;
        }
        let posed = pose_all(&animators, &self.held.slots, &globals, &poses);
        self.stage(&posed, (&*table, &*residency, &*assets), &planes, cx);
        cx.run(self.skin);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass != self.skin {
            return 0;
        }
        let vertices = self.held.vertices.div_ceil(SKIN_GROUP);
        cx.dispatch(&self.skinning, Count::Groups([vertices, 1, 1]));
        let parts = (self.held.slots.len() as u32).div_ceil(SKIN_GROUP);
        cx.dispatch(&self.arguments, Count::Groups([parts, self.held.views, 1]));
        0
    }
}
