use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, DrawnBounds, DrawnPoses, DrawnTable, Error, GeometryHandle, GlobalTransforms,
    Mesh, Spawner, gpu_struct,
};
use engine_places::skin::{JOINTS, WEIGHTS};
use engine_places_rendering::animation::{Animator, Bound};
use engine_render::{Allocation, Bind, Buffer, GpuPart, Install, Lifetime, Prepare};
use glam::{Mat4, Vec3, Vec4};

use crate::AnimationBrick;
use crate::shaders::animation::faces::skin as face;

gpu_struct! {
    pub(crate) struct JointRows {
        pub(crate) r0: [f32; 4],
        pub(crate) r1: [f32; 4],
        pub(crate) r2: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct SkinSlot {
        pub(crate) now: u32,
        pub(crate) was: u32,
        pub(crate) joints: u32,
        pub(crate) pad: u32,
    }
}

type ReachKey = (GeometryHandle, Option<u64>, u64);

struct Reach {
    key: ReachKey,
    radii: Option<Vec<f32>>,
    pass: u64,
}

#[derive(Resource, Default)]
pub struct Skins {
    rows: Vec<JointRows>,
    parts: Vec<(Entity, SkinSlot)>,
    reaches: BTreeMap<Entity, Reach>,
    pass: u64,
}

impl Skins {
    pub(crate) fn posed(&self, part: Entity) -> bool {
        self.parts.iter().any(|&(held, _)| held == part)
    }
}

#[derive(Default)]
pub(crate) struct Scratch {
    nodes: Vec<Entity>,
    now: Vec<Option<Mat4>>,
    was: Vec<Option<Mat4>>,
    spheres: Vec<Vec4>,
}

fn rows_of(matrix: Mat4) -> JointRows {
    let rows = matrix.transpose();
    JointRows {
        r0: rows.x_axis.to_array(),
        r1: rows.y_axis.to_array(),
        r2: rows.z_axis.to_array(),
    }
}

fn reach_of(palette: &[Bound], mesh: &Mesh) -> Option<Vec<f32>> {
    let joints: Vec<u32> = mesh
        .stream(JOINTS)?
        .words()
        .map(u32::from_le_bytes)
        .collect();
    let weights: Vec<f32> = mesh
        .stream(WEIGHTS)?
        .words()
        .map(f32::from_le_bytes)
        .collect();
    let inverses: Vec<Mat4> = palette
        .iter()
        .map(|bound| Mat4::from_cols_array_2d(&bound.inverse_bind))
        .collect();
    let mut radii = vec![-1.0; palette.len()];
    let lanes = joints
        .as_chunks::<4>()
        .0
        .iter()
        .zip(weights.as_chunks::<4>().0);
    for ((joints, weights), vertex) in lanes.zip(mesh.vertices()) {
        if joints.iter().any(|&joint| joint as usize >= palette.len()) {
            return None;
        }
        let position = Vec3::from_array(vertex.position);
        for (&joint, &weight) in joints.iter().zip(weights) {
            if weight <= 0.0 {
                continue;
            }
            let joint = joint as usize;
            let distance = inverses[joint].transform_point3(position).length();
            radii[joint] = engine_core::math::max(radii[joint], distance);
        }
    }
    Some(radii)
}

fn part_bounds(radii: &[f32], spheres: &[Vec4]) -> Option<DrawnBounds> {
    let mut low = Vec3::splat(f32::INFINITY);
    let mut high = Vec3::splat(f32::NEG_INFINITY);
    for (&radius, sphere) in radii.iter().zip(spheres) {
        if radius < 0.0 {
            continue;
        }
        let extent = Vec3::splat(radius * sphere.w);
        low = low.min(sphere.truncate() - extent);
        high = high.max(sphere.truncate() + extent);
    }
    (low.x <= high.x).then(|| DrawnBounds {
        min: low.to_array(),
        max: high.to_array(),
    })
}

fn pose(
    (entity, animator): (Entity, &Animator),
    (globals, poses): (&GlobalTransforms, &DrawnPoses),
    scratch: &mut Scratch,
    rows: &mut Vec<JointRows>,
) -> Option<SkinSlot> {
    let joints = animator.joint_nodes();
    let palette = &animator.rig().palette;
    scratch.nodes.clear();
    scratch.nodes.extend(
        palette
            .iter()
            .filter_map(|bound| joints.get(bound.joint as usize).copied()),
    );
    if !globals.drawn(entity) || palette.is_empty() || scratch.nodes.len() != palette.len() {
        return None;
    }
    poses.drawn_and_previous_poses(
        globals,
        &scratch.nodes,
        (&mut scratch.now, &mut scratch.was),
    );
    let to_local = poses.drawn_pose(globals, entity)?.inverse();
    if !to_local.is_finite() || scratch.now.iter().any(Option::is_none) {
        return None;
    }
    let now = rows.len() as u32;
    scratch.spheres.clear();
    for (bound, joint) in palette.iter().zip(&scratch.now) {
        let joint = joint.unwrap_or(Mat4::IDENTITY);
        rows.push(rows_of(
            joint * Mat4::from_cols_array_2d(&bound.inverse_bind),
        ));
        let local = to_local * joint;
        let scale = Vec3::new(
            local.x_axis.truncate().length_squared(),
            local.y_axis.truncate().length_squared(),
            local.z_axis.truncate().length_squared(),
        )
        .max_element()
        .sqrt();
        scratch.spheres.push(local.w_axis.truncate().extend(scale));
    }
    let was = rows.len() as u32;
    for ((bound, joint), before) in palette.iter().zip(&scratch.now).zip(&scratch.was) {
        let joint = before.or(*joint).unwrap_or(Mat4::IDENTITY);
        rows.push(rows_of(
            joint * Mat4::from_cols_array_2d(&bound.inverse_bind),
        ));
    }
    Some(SkinSlot {
        now,
        was,
        joints: palette.len() as u32,
        pad: 0,
    })
}

pub(crate) fn pose_skins(
    (animators, held): (Query<(Entity, &Animator)>, Query<&DrawnBounds>),
    (globals, poses, assets): (Res<GlobalTransforms>, Res<DrawnPoses>, Res<AssetLibrary>),
    mut skins: ResMut<Skins>,
    mut scratch: Local<Scratch>,
    mut spawner: Spawner<AnimationBrick>,
) {
    let skins = &mut *skins;
    skins.rows.clear();
    skins.parts.clear();
    skins.pass = skins.pass.wrapping_add(1);
    let pass = skins.pass;
    for (entity, animator) in &animators {
        let posed = pose(
            (entity, animator),
            (&globals, &poses),
            &mut scratch,
            &mut skins.rows,
        );
        let rig = animator.rig();
        for (part, &node) in rig.parts.iter().zip(animator.part_nodes()) {
            let key = (
                part.mesh,
                assets.revision_of(part.mesh),
                animator.rig_revision(),
            );
            let reach = skins.reaches.entry(node).or_insert(Reach {
                key,
                radii: None,
                pass,
            });
            if reach.key != key || reach.pass == pass {
                reach.key = key;
                reach.radii = assets
                    .mesh(part.mesh)
                    .and_then(|mesh| reach_of(&rig.palette, mesh));
            }
            reach.pass = pass;
            if let (Some(slot), Some(radii)) = (posed, &reach.radii) {
                skins.parts.push((node, slot));
                let bounds = part_bounds(radii, &scratch.spheres);
                if let Some(bounds) = bounds
                    && held.get(node).ok() != Some(&bounds)
                {
                    spawner.insert(node, bounds);
                }
            }
        }
    }
    skins.reaches.retain(|_, reach| reach.pass == pass);
}

pub(crate) struct SkinGpu {
    rows: Buffer<Self>,
    slots: Buffer<Self>,
    table: Vec<SkinSlot>,
    uploaded: Vec<SkinSlot>,
}

impl GpuPart for SkinGpu {
    type Reads = (Res<'static, Skins>, Res<'static, DrawnTable>);
    const NAME: &'static str = "animation";
    const LAZY: bool = true;
    const TURNS_ON_IN_PLAY: bool = true;
    const LABELS: &'static [&'static str] = &["engine/animation"];

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let rows = cx.buffer("skin_rows", Lifetime::Kept);
        let slots = cx.buffer("skin_slots", Lifetime::Kept);
        cx.frame(face::SKIN_ROWS, Bind::Buffer(rows))?;
        cx.frame(face::SKIN_SLOTS, Bind::Buffer(slots))?;
        Ok(Self {
            rows,
            slots,
            table: Vec::new(),
            uploaded: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (skins, table): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let live = table.records().len().max(1);
        self.table.clear();
        self.table
            .resize(live.max(self.uploaded.len()), SkinSlot::default());
        for &(part, slot) in &skins.parts {
            if let Some(held) = table
                .slot(part)
                .filter(|&at| (at as usize) < live)
                .and_then(|at| self.table.get_mut(at as usize))
            {
                *held = slot;
            }
        }
        let rows = skins.rows.len().max(1) * size_of::<JointRows>();
        let slots = (self.table.len() * size_of::<SkinSlot>()) as u64;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.rows, rows as u64);
        let slots = cx.allocate_buffer(self.slots, slots);
        if !skins.rows.is_empty() {
            cx.write(self.rows, 0, &skins.rows);
        }
        if slots == Allocation::Fresh || self.table != self.uploaded {
            cx.write(self.slots, 0, &self.table);
            std::mem::swap(&mut self.table, &mut self.uploaded);
        }
    }
}
