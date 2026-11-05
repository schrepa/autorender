use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::math::max;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, Error, GlobalTransforms, GpuPresentation,
    MaterialHandle, MeshHandle, MeshVertex, NO_PARENT, NODE_ACTIVE, NODE_BLEND, NODE_PARENT_ACTIVE,
    NODE_VALID, NodePose, PoseRows, PresentationState, PresentedTransforms, Transform, View,
    gpu_struct,
};
use engine_places::animation::{Animator, Rig, RigPart, VertexSkin};
use engine_places::shadows::{CasterMap, CasterView, ShadowCasters};
use engine_render::{
    Bind, Buffer, ComputePipeline, Count, DepthUse, Draw, DrawInput, Encode, GpuPart, Install,
    Lifetime, MainView, Pass, Pipeline, Prepare, RenderPipeline, Shape, VertexLayout, pack_color,
    pack_surface, planes, wgpu,
};
use glam::{Mat4, Quat, Vec3, Vec4};

use crate::animator::{Ticked, presented_pose};
use crate::shaders;

engine_core::gpu_consts! {
    pub(crate) const SKIN_GROUP: u32 = 64;
    pub(crate) const TILE_ROOM: u32 = 64;
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
        pub(crate) center: [f32; 4],
        pub(crate) half: [f32; 4],
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

const CASTER_STREAMS: &[VertexLayout] = &[
    VertexLayout::Buffer(wgpu::VertexBufferLayout {
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
    }),
    VertexLayout::Buffer(wgpu::VertexBufferLayout {
        array_stride: 4,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 0,
            shader_location: 2,
        }],
    }),
];

#[derive(Resource, Default)]
pub(crate) struct Books {
    pub(crate) epoch: u64,
    pub(crate) ticked: HashMap<Entity, Ticked>,
}

struct Slot {
    entity: Entity,
    mesh: MeshHandle,
    material: MaterialHandle,
    rest_at: u32,
    skinned_at: u32,
    reach: usize,
    masked: bool,
    double_sided: bool,
}

#[derive(Default)]
struct Held {
    key: Option<Vec<(Entity, usize, usize)>>,
    slots: Vec<Slot>,
    reaches: Vec<Vec<f32>>,
    vertices: u32,
    last: Vec<(Entity, Range<u32>)>,
    last_rows: Vec<JointRows>,
    lit: Vec<u32>,
    cast: Vec<u32>,
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
    caster_args: Buffer<SkinningGpu>,
    tiles: Buffer<SkinningGpu>,
}

struct Inputs {
    skinned: DrawInput<SkinningGpu, MainView>,
    args: DrawInput<SkinningGpu, MainView>,
    caster_skinned: DrawInput<SkinningGpu, ShadowCasters>,
    caster_args: DrawInput<SkinningGpu, ShadowCasters>,
    caster_tiles: DrawInput<SkinningGpu, ShadowCasters>,
}

pub(crate) struct SkinningGpu {
    skin: Pass<Self>,
    skinning: ComputePipeline<Self>,
    selecting: ComputePipeline<Self>,
    lit: [Pipeline<Self, MainView>; 2],
    casting: [Pipeline<Self, ShadowCasters>; 2],
    buffers: Buffers,
    inputs: Inputs,
    epoch: u64,
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

fn rows_matrix(rows: &PoseRows) -> Mat4 {
    Mat4::from_cols_array_2d(&[rows[0], rows[1], rows[2], [0.0, 0.0, 0.0, 1.0]]).transpose()
}

fn node_local(pose: &NodePose) -> Transform {
    let [tx, ty, tz, sx, rx, ry, rz, rw, sy, sz] = *pose;
    Transform {
        translation: Vec3::new(tx, ty, tz),
        rotation: Quat::from_xyzw(rx, ry, rz, rw),
        scale: Vec3::new(sx, sy, sz),
    }
}

fn presented_node(nodes: &GpuPresentation<'_>, node: usize) -> Option<Mat4> {
    let [parent, flags] = *nodes.info.get(node)?;
    if flags & NODE_VALID == 0 {
        return None;
    }
    if flags & NODE_ACTIVE == 0 {
        return nodes.authority.get(node).map(rows_matrix);
    }
    let above = if flags & NODE_PARENT_ACTIVE != 0 {
        presented_node(nodes, parent as usize)?
    } else if parent != NO_PARENT {
        rows_matrix(nodes.authority.get(parent as usize)?)
    } else {
        Mat4::IDENTITY
    };
    let current = node_local(nodes.current.get(node)?);
    if flags & NODE_BLEND == 0 || nodes.alpha >= 1.0 {
        return Some(above * current.matrix());
    }
    let previous = node_local(nodes.previous.get(node)?);
    let local = Transform {
        translation: previous.translation.lerp(current.translation, nodes.alpha),
        rotation: previous.rotation.lerp(current.rotation, nodes.alpha),
        scale: previous.scale.lerp(current.scale, nodes.alpha),
    };
    Some(above * local.matrix())
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
) -> Vec<(Entity, usize, usize)> {
    let mut key = Vec::new();
    for (entity, animator) in listed {
        let rig = animator.rig();
        for (index, part) in rig.parts.iter().enumerate() {
            let shaped = assets
                .mesh(part.mesh)
                .is_some_and(|mesh| mesh.vertices().len() == part.skin.len());
            if shaped && residency.resident(part.mesh) && residency.material_resident(part.material)
            {
                key.push((*entity, index, Arc::as_ptr(rig) as usize));
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
    ticked: &HashMap<Entity, Ticked>,
    slots: &[Slot],
    alpha: f32,
    place: impl Fn(Entity) -> Option<Mat4>,
) -> Posed {
    let mut posed = Posed {
        rows: Vec::new(),
        spheres: Vec::new(),
        ranges: Vec::new(),
    };
    let mut model = Vec::new();
    for &(entity, animator) in listed {
        if slots
            .binary_search_by_key(&entity, |slot| slot.entity)
            .is_err()
        {
            continue;
        }
        let start = posed.rows.len() as u32;
        let world = place(entity);
        presented_pose(animator, ticked.get(&entity), alpha, &mut model);
        let placed = world.unwrap_or(Mat4::IDENTITY);
        for bound in &animator.rig().palette {
            let held = model.get(bound.joint as usize).copied();
            let joint = placed * held.unwrap_or(Mat4::IDENTITY);
            let inverse = Mat4::from_cols_array_2d(&bound.inverse_bind);
            posed.rows.push(rows_of(joint * inverse));
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
        posed.ranges.push((entity, range, world.is_some()));
    }
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

fn previous_rows(held: &mut Held, posed: &mut Posed) {
    let current = posed.rows.len();
    for (entity, range, _) in &posed.ranges {
        let earlier = held
            .last
            .binary_search_by_key(entity, |(known, _)| *known)
            .ok()
            .map(|at| held.last[at].1.clone())
            .filter(|earlier| earlier.len() == range.len());
        match earlier {
            Some(earlier) => {
                let span = earlier.start as usize..earlier.end as usize;
                posed.rows.extend_from_slice(&held.last_rows[span]);
            }
            None => {
                let span = range.start as usize..range.end as usize;
                posed.rows.extend_from_within(span);
            }
        }
    }
    held.last.clear();
    held.last.extend(
        posed
            .ranges
            .iter()
            .map(|(entity, range, _)| (*entity, range.clone())),
    );
    held.last_rows.clear();
    held.last_rows.extend_from_slice(&posed.rows[..current]);
}

impl SkinningGpu {
    fn lay_out(
        &mut self,
        listed: &[(Entity, &Animator)],
        key: &[(Entity, usize, usize)],
        assets: &AssetLibrary,
        cx: &mut Prepare<'_, Self>,
    ) {
        let held = &mut self.held;
        held.slots.clear();
        held.reaches.clear();
        held.last.clear();
        held.last_rows.clear();
        let reaches = &mut held.reaches;
        let mut shared: BTreeMap<(usize, usize), (u32, usize)> = BTreeMap::new();
        let mut rest: Vec<RestVertex> = Vec::new();
        let mut owners: Vec<u32> = Vec::new();
        let mut palette = 0_u64;
        for &(entity, index, rig_id) in key {
            let Ok(at) = listed.binary_search_by_key(&entity, |(known, _)| *known) else {
                continue;
            };
            let rig = listed[at].1.rig();
            let Some((part, mesh)) = rig
                .parts
                .get(index)
                .and_then(|part| Some((part, assets.mesh(part.mesh)?)))
            else {
                continue;
            };
            let vertices = mesh.vertices();
            let (rest_at, reach) = *shared.entry((rig_id, index)).or_insert_with(|| {
                let rest_at = rest.len() as u32;
                rest.extend(part.skin.iter().zip(vertices).map(rest_vertex));
                reaches.push(reach_of(rig, part, vertices));
                (rest_at, reaches.len() - 1)
            });
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
        cx.allocate_buffer(buffers.caster_args, parts * ARGS_BYTES);
        cx.allocate_buffer(buffers.tiles, parts * u64::from(TILE_ROOM) * 4);
        if !owners.is_empty() {
            cx.write(buffers.owners, 0, &owners);
            cx.write(buffers.rest, 0, &rest);
        }
    }

    fn stage(
        &mut self,
        posed: &Posed,
        (residency, assets, view): (&AssetResidency, &AssetLibrary, &View),
        cx: &mut Prepare<'_, Self>,
    ) {
        let held = &mut self.held;
        let Posed {
            rows,
            spheres,
            ranges,
        } = posed;
        let total = spheres.len() as u32;
        let frustum = planes(view.view_projection());
        let count = held.slots.len();
        let mut parts = Vec::with_capacity(count);
        let mut colors = Vec::with_capacity(count);
        let mut args = Vec::with_capacity(count);
        let mut caster_args = Vec::with_capacity(count);
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
            parts.push(SkinPart {
                rows_at: start,
                previous_at: start + total,
                rest_at: slot.rest_at,
                skinned_at: slot.skinned_at,
                center: center.extend(if shown { 1.0 } else { 0.0 }).to_array(),
                half: half.extend(0.0).to_array(),
            });
            let (color, surface) = match assets.material(slot.material) {
                Some(material) => {
                    slot.masked = matches!(material.alpha_mode(), AlphaMode::Mask { .. });
                    slot.double_sided = material.is_double_sided();
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
            caster_args.push([indices, 0, first, 0, 0]);
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
        cx.write(buffers.caster_args, 0, &caster_args);
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

    pub(crate) fn draw_lit(&self, _: &(), draw: &mut Draw<'_, Self, MainView>) -> u32 {
        for &index in &self.held.lit {
            let slot = &self.held.slots[index as usize];
            draw.draw(
                &self.lit[usize::from(slot.double_sided)],
                Shape::MeshIndirect {
                    mesh: slot.mesh,
                    material: Some(slot.material),
                    args: self.inputs.args,
                    offset: u64::from(index) * ARGS_BYTES,
                    streams: &[(
                        self.inputs.skinned,
                        u64::from(slot.skinned_at) * SKINNED_BYTES,
                    )],
                },
            );
        }
        self.held.lit.len() as u32
    }

    pub(crate) fn draw_casters(
        &self,
        at: &CasterView,
        draw: &mut Draw<'_, Self, ShadowCasters>,
    ) -> u32 {
        if at.map != CasterMap::Moving {
            return 0;
        }
        for &index in &self.held.cast {
            let slot = &self.held.slots[index as usize];
            draw.draw(
                &self.casting[usize::from(slot.masked)],
                Shape::MeshIndirect {
                    mesh: slot.mesh,
                    material: slot.masked.then_some(slot.material),
                    args: self.inputs.caster_args,
                    offset: u64::from(index) * ARGS_BYTES,
                    streams: &[
                        (
                            self.inputs.caster_skinned,
                            u64::from(slot.skinned_at) * SKINNED_BYTES,
                        ),
                        (
                            self.inputs.caster_tiles,
                            u64::from(index) * u64::from(TILE_ROOM) * 4,
                        ),
                    ],
                },
            );
        }
        self.held.cast.len() as u32
    }
}

fn lit_pipeline(
    cx: &mut Install<'_, SkinningGpu>,
    double_sided: bool,
    colors: Buffer<SkinningGpu>,
) -> Result<Pipeline<SkinningGpu, MainView>, Error> {
    use shaders::animation::draw;
    cx.render_pipeline(
        MainView,
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
    colors: Buffer<SkinningGpu>,
) -> Result<[Pipeline<SkinningGpu, ShadowCasters>; 2], Error> {
    use shaders::animation::casters;
    let desc = |vertex, fragment| RenderPipeline {
        vertex,
        fragment: Some(fragment),
        debug: None,
        vertices: CASTER_STREAMS,
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::Write { bias: CASTER_BIAS },
        blend: None,
        writes: wgpu::ColorWrites::empty(),
    };
    Ok([
        cx.render_pipeline(
            ShadowCasters,
            &desc(casters::VS_CASTER, casters::FS_CASTER),
            &[],
        )?,
        cx.render_pipeline(
            ShadowCasters,
            &desc(casters::VS_CASTER_MASKED, casters::FS_CASTER_MASKED),
            &[(casters::COLORS, Bind::Buffer(colors))],
        )?,
    ])
}

impl GpuPart for SkinningGpu {
    type Reads = (
        Query<'static, 'static, (Entity, &'static Animator)>,
        Res<'static, GlobalTransforms>,
        Res<'static, PresentedTransforms>,
        Res<'static, PresentationState>,
        Res<'static, AssetResidency>,
        Res<'static, AssetLibrary>,
        Res<'static, View>,
        Res<'static, Books>,
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
            caster_args: cx.buffer("skin_caster_args", Lifetime::Kept),
            tiles: cx.buffer("skin_tiles", Lifetime::Kept),
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
        let selecting = cx.compute_pipeline(
            pass,
            skin::SELECT_TILES,
            &[
                (skin::FRAME, Bind::Buffer(buffers.frame)),
                (skin::PARTS, Bind::Buffer(buffers.parts)),
                (skin::CASTER_ARGS, Bind::Buffer(buffers.caster_args)),
                (skin::TILES, Bind::Buffer(buffers.tiles)),
            ],
        )?;
        let lit = [
            lit_pipeline(cx, false, buffers.colors)?,
            lit_pipeline(cx, true, buffers.colors)?,
        ];
        let casting = caster_pipelines(cx, buffers.colors)?;
        let inputs = Inputs {
            skinned: cx.draw_input(MainView, buffers.skinned),
            args: cx.draw_input(MainView, buffers.args),
            caster_skinned: cx.draw_input(ShadowCasters, buffers.skinned),
            caster_args: cx.draw_input(ShadowCasters, buffers.caster_args),
            caster_tiles: cx.draw_input(ShadowCasters, buffers.tiles),
        };
        Ok(Self {
            skin: pass,
            skinning,
            selecting,
            lit,
            casting,
            buffers,
            inputs,
            epoch: 0,
            held: Held::default(),
        })
    }

    fn prepare(
        &mut self,
        (animators, globals, presented, presentation, residency, assets, view, books): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        if cx.fresh() || books.epoch != self.epoch {
            self.held = Held::default();
            self.epoch = books.epoch;
        }
        let mut listed: Vec<(Entity, &Animator)> = animators.iter().collect();
        listed.sort_unstable_by_key(|(entity, _)| *entity);
        let key = layout_key(&listed, &residency, &assets);
        if self.held.key.as_ref() != Some(&key) {
            self.lay_out(&listed, &key, &assets, cx);
            self.held.key = Some(key);
        }
        self.held.lit.clear();
        self.held.cast.clear();
        if self.held.slots.is_empty() {
            return;
        }
        let nodes = presented.gpu_presentation(&globals);
        let place = |entity| presented_node(&nodes, globals.node_index(entity)?);
        let mut posed = pose_all(
            &listed,
            &books.ticked,
            &self.held.slots,
            presentation.alpha(),
            place,
        );
        previous_rows(&mut self.held, &mut posed);
        self.stage(&posed, (&*residency, &*assets, &*view), cx);
        cx.run(self.skin);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass != self.skin {
            return 0;
        }
        let vertices = self.held.vertices.div_ceil(SKIN_GROUP);
        let parts = (self.held.slots.len() as u32).div_ceil(SKIN_GROUP);
        cx.dispatch(&self.skinning, Count::Groups([vertices, 1, 1]));
        cx.dispatch(&self.selecting, Count::Groups([parts, 1, 1]));
        0
    }
}
