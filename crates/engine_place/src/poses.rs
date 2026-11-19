use std::fmt::Write as _;

use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Res, SystemParam};
use engine_math::{max, sin};
use glam::{Mat4, Vec4};

use crate::{Changes, Error, ErrorCode, GpuStruct, Pace};

pub type NodePose = [f32; 10];
pub type PoseRows = [[f32; 4]; 3];

const NO_PARENT: u32 = u32::MAX;
const NO_SLOT: u32 = u32::MAX;
pub const COEFFICIENT_LIMIT: f32 = 1_048_576.0;
pub const MIN_LINEAR_SCALE: f32 = 1.0 / 1024.0;
pub const COMMON_DETERMINANT_MARGIN: f32 = 1.0 / 512.0;

const LINEAR_ROTATION_DOT_THRESHOLD: f32 = 0.9995;
const POSE_FLOAT_LIMIT: f32 = f32::MAX;
const MOVED_READER: usize = 0;
const PLACED_READER: usize = 1;
const IDENTITY_ROWS: PoseRows = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

pub trait PoseNodes {
    fn count(&self) -> usize;
    fn local(&self, node: usize) -> &NodePose;
    fn rows(&self, node: usize) -> Option<&PoseRows>;
    fn parent(&self, node: usize) -> Option<usize>;
    fn children(&self, node: usize) -> impl Iterator<Item = usize> + '_;
    fn index_of(&self, node: Entity) -> Option<usize>;
    fn entity_at(&self, node: usize) -> Option<Entity>;
    fn moved(&self) -> &Changes;
    fn placed(&self) -> &Changes;
}

crate::gpu_struct! {
    @crate "::bytemuck";
    pub struct PoseEntry {
        pub slot: u32,
        pub parent: u32,
        pub blend: u32,
        pub pad: u32,
        pub base: [[f32; 4]; 3],
        pub previous: [[f32; 4]; 3],
        pub current: [[f32; 4]; 3],
    }
}

crate::gpu_struct! {
    @crate "::bytemuck";
    pub struct PoseFrame {
        pub alpha: f32,
        pub entries: u32,
        pub pad0: u32,
        pub pad1: u32,
    }
}

fn vec4f(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4::new(x, y, z, w)
}

fn mat4x4f(x: Vec4, y: Vec4, z: Vec4, w: Vec4) -> Mat4 {
    Mat4::from_cols(x, y, z, w)
}

fn select<T>(off: T, on: T, pick: bool) -> T {
    if pick { on } else { off }
}

fn normalize(value: Vec4) -> Vec4 {
    value.normalize()
}

fn dot(left: Vec4, right: Vec4) -> f32 {
    left.dot(right)
}

fn abs(value: f32) -> f32 {
    value.abs()
}

fn acos(value: f32) -> f32 {
    value.acos()
}

macro_rules! formula {
    (@rust f32) => { f32 };
    (@rust bool) => { bool };
    (@rust vec4f) => { Vec4 };
    (@rust mat4x4f) => { Mat4 };
    ($(
        fn $name:ident($($arg:ident: $ty:ident),*) -> $out:ident {
            $(let $step:ident = $value:expr;)*
            return $result:expr;
        }
    )*) => {
        $(
            fn $name($($arg: formula!(@rust $ty)),*) -> formula!(@rust $out) {
                $(let $step = $value;)*
                $result
            }
        )*

        const FORMULA_WGSL: &str = concat!($(
            "fn ", stringify!($name), "(",
            $(stringify!($arg), ": ", stringify!($ty), ", ",)*
            ") -> ", stringify!($out), " {\n",
            $("    let ", stringify!($step), " = ", stringify!($value), ";\n",)*
            "    return ", stringify!($result), ";\n}\n",
        )*);
    };
}

formula! {
    fn pose_trs(offset_sx: vec4f, rotation: vec4f, scale_yz: vec4f) -> mat4x4f {
        let x = rotation.x;
        let y = rotation.y;
        let z = rotation.z;
        let w = rotation.w;
        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;
        return mat4x4f(
            vec4f(1.0 - (yy + zz), xy + wz, xz - wy, 0.0) * offset_sx.w,
            vec4f(xy - wz, 1.0 - (xx + zz), yz + wx, 0.0) * scale_yz.x,
            vec4f(xz + wy, yz - wx, 1.0 - (xx + yy), 0.0) * scale_yz.y,
            vec4f(offset_sx.x, offset_sx.y, offset_sx.z, 1.0)
        );
    }

    fn blend_rotation(previous: vec4f, current: vec4f, weight: f32) -> vec4f {
        let p = normalize(previous);
        let unflipped = normalize(current);
        let d = dot(p, unflipped);
        let c = select(unflipped, -unflipped, d < 0.0);
        let cosine = abs(d);
        let near = normalize(p * (1.0 - weight) + c * weight);
        let theta = acos(cosine);
        let far = normalize(
            (p * sin(theta * (1.0 - weight)) + c * sin(theta * weight)) * (1.0 / sin(theta))
        );
        return select(far, near, cosine >= LINEAR_ROTATION_DOT_THRESHOLD);
    }

    fn pose_mix(was: vec4f, now: vec4f, weight: f32) -> vec4f {
        let mixed = was * (1.0 - weight) + now * weight;
        return select(select(mixed, was, weight == 0.0), now, weight >= 1.0);
    }

    fn pose_weight(alpha: f32, blend: bool) -> f32 {
        return select(1.0, alpha, blend);
    }

    fn drawn_local(
        was_offset_sx: vec4f,
        was_rotation: vec4f,
        was_scale_yz: vec4f,
        offset_sx: vec4f,
        rotation: vec4f,
        scale_yz: vec4f,
        weight: f32
    ) -> mat4x4f {
        let blended = blend_rotation(was_rotation, rotation, weight);
        let turned = select(select(blended, was_rotation, weight == 0.0), rotation, weight >= 1.0);
        return pose_trs(
            pose_mix(was_offset_sx, offset_sx, weight),
            turned,
            pose_mix(was_scale_yz, scale_yz, weight)
        );
    }

    fn pose_finite(value: vec4f) -> bool {
        return abs(value.x) <= POSE_FLOAT_LIMIT
            && abs(value.y) <= POSE_FLOAT_LIMIT
            && abs(value.z) <= POSE_FLOAT_LIMIT
            && abs(value.w) <= POSE_FLOAT_LIMIT;
    }

    fn pose_largest(value: vec4f) -> f32 {
        return max(abs(value.x), max(abs(value.y), abs(value.z)));
    }

    fn pose_admitted(x: vec4f, y: vec4f, z: vec4f, w: vec4f) -> bool {
        let finite = pose_finite(x) && pose_finite(y) && pose_finite(z) && pose_finite(w);
        let scale = max(pose_largest(x), max(pose_largest(y), pose_largest(z)));
        let determinant = x.x * (y.y * z.z - y.z * z.y)
            + x.y * (y.z * z.x - y.x * z.z)
            + x.z * (y.x * z.y - y.y * z.x);
        return finite
            && scale >= MIN_LINEAR_SCALE
            && pose_largest(w) <= COEFFICIENT_LIMIT
            && scale <= COEFFICIENT_LIMIT
            && abs(determinant) >= COMMON_DETERMINANT_MARGIN * ((scale * scale) * scale);
    }
}

pub fn poses_wgsl() -> String {
    let mut source = String::from(PoseEntry::WGSL);
    source.push_str(PoseFrame::WGSL);
    for (name, value) in [("NO_PARENT", NO_PARENT), ("NO_SLOT", NO_SLOT)] {
        let _ = writeln!(source, "const {name}: u32 = {value}u;");
    }
    for (name, value) in [
        ("COEFFICIENT_LIMIT", COEFFICIENT_LIMIT),
        ("MIN_LINEAR_SCALE", MIN_LINEAR_SCALE),
        ("COMMON_DETERMINANT_MARGIN", COMMON_DETERMINANT_MARGIN),
        (
            "LINEAR_ROTATION_DOT_THRESHOLD",
            LINEAR_ROTATION_DOT_THRESHOLD,
        ),
        ("POSE_FLOAT_LIMIT", POSE_FLOAT_LIMIT),
    ] {
        let _ = writeln!(source, "const {name}: f32 = {:?};", f64::from(value));
    }
    source.push_str(FORMULA_WGSL);
    source.push_str(include_str!("poses.wgsl"));
    source
}

fn packed(pose: &NodePose) -> PoseRows {
    let [tx, ty, tz, sx, rx, ry, rz, rw, sy, sz] = *pose;
    [[tx, ty, tz, sx], [rx, ry, rz, rw], [sy, sz, 0.0, 0.0]]
}

#[inline(always)]
pub fn rows_matrix(rows: &PoseRows) -> Mat4 {
    Mat4::from_cols(
        Vec4::from_array(rows[0]),
        Vec4::from_array(rows[1]),
        Vec4::from_array(rows[2]),
        Vec4::W,
    )
    .transpose()
}

#[inline(always)]
pub fn same_pose_bits(left: &NodePose, right: &NodePose) -> bool {
    let lanes = |pose: &NodePose| (Vec4::from_slice(&pose[..4]), Vec4::from_slice(&pose[4..8]));
    let ((a, b), (c, d)) = (lanes(left), lanes(right));
    let tail = left[8].to_bits() == right[8].to_bits() && left[9].to_bits() == right[9].to_bits();
    let signs = (a.is_negative_bitmask() ^ c.is_negative_bitmask())
        | (b.is_negative_bitmask() ^ d.is_negative_bitmask());
    if (a.cmpeq(c) & b.cmpeq(d)).all() && signs == 0 {
        return tail;
    }
    if !(a.is_nan() || b.is_nan() || c.is_nan() || d.is_nan()) {
        return false;
    }
    std::hint::cold_path();
    let bits = |value: Vec4| value.to_array().map(f32::to_bits);
    tail && bits(a) == bits(c) && bits(b) == bits(d)
}

#[inline(always)]
pub fn same_rows_bits(left: &PoseRows, right: &PoseRows) -> bool {
    left.as_flattened()
        .iter()
        .zip(right.as_flattened())
        .fold(0, |difference, (left, right)| {
            difference | (left.to_bits() ^ right.to_bits())
        })
        == 0
}

fn scales(pose: &NodePose) -> [f32; 3] {
    [pose[3], pose[8], pose[9]]
}

fn scales_blend(was: &NodePose, now: &NodePose) -> bool {
    scales(was).into_iter().zip(scales(now)).all(|(was, now)| {
        was * now > 0.0 && was.abs() >= MIN_LINEAR_SCALE && now.abs() >= MIN_LINEAR_SCALE
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Span {
    #[default]
    Still,
    Moving,
    Settling,
}

#[derive(Resource, Default)]
pub struct DrawnPoses {
    entities: Vec<Entity>,
    spans: Vec<Span>,
    ticked: Vec<NodePose>,
    previous: Vec<NodePose>,
    listed: Vec<u32>,
    asked: Vec<Entity>,
    stack: Vec<(usize, u32)>,
    entries: Vec<PoseEntry>,
    refused: Vec<Entity>,
    verdicts: Vec<(Entity, Result<(), Error>)>,
    alpha: f32,
    blending: bool,
}

impl DrawnPoses {
    pub fn fresh(&mut self, node: Entity) {
        self.asked.push(node);
    }

    pub fn capture(&mut self, nodes: &impl PoseNodes, pace: impl Fn(Entity) -> Option<Pace>) {
        self.index(nodes);
        let mut kept = 0;
        for at in 0..self.listed.len() {
            let index = self.listed[at] as usize;
            self.spans[index] = match self.spans[index] {
                Span::Moving => Span::Settling,
                Span::Settling | Span::Still => Span::Still,
            };
            if self.spans[index] != Span::Still {
                self.listed[kept] = index as u32;
                kept += 1;
            }
        }
        self.listed.truncate(kept);
        let before = std::mem::take(&mut self.refused);
        let changed = nodes.moved().read();
        if changed.all {
            for index in 0..self.spans.len() {
                self.observe(nodes, &pace, index);
            }
        } else {
            for &index in changed.listed {
                self.observe(nodes, &pace, index as usize);
            }
        }
        nodes.moved().take(MOVED_READER, changed.upto);
        for node in before {
            if !self.refused.contains(&node) {
                self.verdicts.push((node, Ok(())));
            }
        }
        self.apply_fresh(nodes);
    }

    pub fn present(&mut self, nodes: &impl PoseNodes, alpha: f32) {
        self.alpha = alpha;
        self.index(nodes);
        self.apply_fresh(nodes);
    }

    pub fn gather(&mut self, nodes: &impl PoseNodes, slot_of: impl Fn(Entity) -> Option<u32>) {
        self.index(nodes);
        self.apply_fresh(nodes);
        if self.listed.is_empty() && self.entries.is_empty() {
            return;
        }
        self.entries.clear();
        self.blending = false;
        for at in 0..self.listed.len() {
            let top = self.listed[at] as usize;
            if self.span(top) == Span::Still || self.top(nodes, top) != Some(top) {
                continue;
            }
            let base = nodes
                .parent(top)
                .and_then(|parent| nodes.rows(parent))
                .copied()
                .unwrap_or(IDENTITY_ROWS);
            self.stack.push((top, NO_PARENT));
            while let Some((index, parent)) = self.stack.pop() {
                if nodes.rows(index).is_none() {
                    continue;
                }
                let blend = self.span(index) == Span::Moving;
                let current = packed(nodes.local(index));
                let previous = if blend {
                    packed(&self.previous[index])
                } else {
                    current
                };
                self.blending |= blend;
                let entry = self.entries.len() as u32;
                self.entries.push(PoseEntry {
                    slot: nodes.entity_at(index).and_then(&slot_of).unwrap_or(NO_SLOT),
                    parent,
                    blend: u32::from(blend),
                    pad: 0,
                    base,
                    previous,
                    current,
                });
                self.stack
                    .extend(nodes.children(index).map(|child| (child, entry)));
            }
        }
    }

    pub fn fresh_all(&mut self, nodes: &impl PoseNodes) {
        self.index(nodes);
        self.asked.clear();
        for &index in &self.listed {
            let span = &mut self.spans[index as usize];
            if *span == Span::Moving {
                *span = Span::Settling;
            }
        }
        for index in 0..self.spans.len() {
            self.ticked[index] = *nodes.local(index);
            self.previous[index] = *nodes.local(index);
        }
    }

    fn drawn_pose(&self, nodes: &impl PoseNodes, node: Entity) -> Option<Mat4> {
        let index = nodes.index_of(node)?;
        let authority = rows_matrix(nodes.rows(index)?);
        let Some(top) = self.top(nodes, index) else {
            return Some(authority);
        };
        let base = nodes
            .parent(top)
            .and_then(|parent| nodes.rows(parent))
            .map_or(Mat4::IDENTITY, rows_matrix);
        let m = self.chain(nodes, (index, top), base, self.alpha);
        Some(if pose_admitted(m.x_axis, m.y_axis, m.z_axis, m.w_axis) {
            m
        } else {
            self.chain(nodes, (index, top), base, 1.0)
        })
    }

    pub fn entries(&self) -> &[PoseEntry] {
        &self.entries
    }

    pub fn frame(&self) -> PoseFrame {
        PoseFrame {
            alpha: if self.blending { self.alpha } else { 1.0 },
            entries: self.entries.len() as u32,
            pad0: 0,
            pad1: 0,
        }
    }

    pub fn take_verdicts(&mut self) -> std::vec::Drain<'_, (Entity, Result<(), Error>)> {
        self.verdicts.drain(..)
    }

    pub fn capacity_bytes(&self) -> usize {
        (self.ticked.capacity() + self.previous.capacity()) * size_of::<NodePose>()
            + (self.entities.capacity() + self.asked.capacity() + self.refused.capacity())
                * size_of::<Entity>()
            + self.spans.capacity() * size_of::<Span>()
            + self.listed.capacity() * size_of::<u32>()
            + self.stack.capacity() * size_of::<(usize, u32)>()
            + self.entries.capacity() * size_of::<PoseEntry>()
    }

    fn span(&self, index: usize) -> Span {
        self.spans.get(index).copied().unwrap_or_default()
    }

    fn top(&self, nodes: &impl PoseNodes, index: usize) -> Option<usize> {
        let mut top = None;
        let mut at = Some(index);
        while let Some(node) = at {
            if self.span(node) != Span::Still {
                top = Some(node);
            }
            at = nodes.parent(node);
        }
        top
    }

    fn local(&self, nodes: &impl PoseNodes, index: usize, alpha: f32) -> Mat4 {
        let blend = self.span(index) == Span::Moving;
        let [offset_sx, rotation, scale_yz] = packed(nodes.local(index)).map(Vec4::from_array);
        let [was_offset_sx, was_rotation, was_scale_yz] = if blend {
            packed(&self.previous[index]).map(Vec4::from_array)
        } else {
            [offset_sx, rotation, scale_yz]
        };
        drawn_local(
            was_offset_sx,
            was_rotation,
            was_scale_yz,
            offset_sx,
            rotation,
            scale_yz,
            pose_weight(alpha, blend),
        )
    }

    fn chain(
        &self,
        nodes: &impl PoseNodes,
        (first, top): (usize, usize),
        base: Mat4,
        alpha: f32,
    ) -> Mat4 {
        let mut m = self.local(nodes, first, alpha);
        let mut at = first;
        while at != top
            && let Some(parent) = nodes.parent(at)
        {
            at = parent;
            m = self.local(nodes, at, alpha) * m;
        }
        base * m
    }

    fn index(&mut self, nodes: &impl PoseNodes) {
        let count = nodes.count();
        if count < self.spans.len() {
            self.entities.truncate(count);
            self.spans.truncate(count);
            self.ticked.truncate(count);
            self.previous.truncate(count);
            self.listed.retain(|&index| (index as usize) < count);
        }
        for index in self.spans.len()..count {
            self.entities.push(Entity::PLACEHOLDER);
            self.spans.push(Span::Still);
            self.ticked.push(*nodes.local(index));
            self.previous.push(*nodes.local(index));
        }
        let placed = nodes.placed().read();
        if placed.all {
            for index in 0..count {
                self.assign(nodes, index);
            }
        } else {
            for &index in placed.listed {
                if (index as usize) < count {
                    self.assign(nodes, index as usize);
                }
            }
        }
        nodes.placed().take(PLACED_READER, placed.upto);
    }

    fn assign(&mut self, nodes: &impl PoseNodes, index: usize) {
        let entity = nodes.entity_at(index).unwrap_or(Entity::PLACEHOLDER);
        if self.entities[index] == entity {
            return;
        }
        self.entities[index] = entity;
        self.spans[index] = Span::Still;
        self.ticked[index] = *nodes.local(index);
        self.previous[index] = *nodes.local(index);
    }

    fn apply_fresh(&mut self, nodes: &impl PoseNodes) {
        for node in self.asked.drain(..) {
            if let Some(index) = nodes.index_of(node) {
                self.stack.push((index, NO_PARENT));
            }
        }
        while let Some((index, _)) = self.stack.pop() {
            if index >= self.spans.len() {
                continue;
            }
            if self.spans[index] == Span::Moving {
                self.spans[index] = Span::Settling;
            }
            self.ticked[index] = *nodes.local(index);
            self.previous[index] = *nodes.local(index);
            self.stack
                .extend(nodes.children(index).map(|child| (child, NO_PARENT)));
        }
    }

    fn observe(
        &mut self,
        nodes: &impl PoseNodes,
        pace: &impl Fn(Entity) -> Option<Pace>,
        index: usize,
    ) {
        if index >= self.spans.len() {
            return;
        }
        let now = *nodes.local(index);
        let was = self.ticked[index];
        if same_pose_bits(&was, &now) {
            return;
        }
        self.ticked[index] = now;
        let node = self.entities[index];
        if pace(node) != Some(Pace::Tick) {
            return;
        }
        if !scales_blend(&was, &now) {
            self.refused.push(node);
            self.verdicts.push((
                node,
                Err(Error::new(ErrorCode::InvalidWorldTransform, "$.scale")
                    .got([scales(&was), scales(&now)])),
            ));
            return;
        }
        self.previous[index] = was;
        if self.spans[index] == Span::Still {
            self.listed.push(index as u32);
        }
        self.spans[index] = Span::Moving;
    }
}

#[derive(SystemParam)]
pub struct Poses<'w, N: PoseNodes + Resource> {
    poses: Res<'w, DrawnPoses>,
    nodes: Res<'w, N>,
}

impl<N: PoseNodes + Resource> Poses<'_, N> {
    pub fn drawn_pose(&self, node: Entity) -> Option<Mat4> {
        self.poses.drawn_pose(&*self.nodes, node)
    }
}
