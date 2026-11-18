use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use engine_place::Changes;
use glam::{Mat4, Vec3, Vec4};
use serde::Serialize;

use crate::hierarchy::same_pose_bits;
use crate::math::{max, sin};
use crate::presentation_bounds as bounds;
use crate::{FixedTime, FrameTime, GlobalTransforms, GpuBits, MAX_HIERARCHY_DEPTH, Revision};

pub type PoseRows = [[f32; 4]; 3];
pub type NodePose = [f32; 10];
pub type NodeInfo = [u32; 2];

pub const NO_PARENT: u32 = u32::MAX;
pub const PRESENTATION_WGSL: &str = include_str!("presentation.wgsl");

const SPARSE_SHARE: usize = 3;

#[inline]
pub fn pose_rows(model: Mat4) -> PoseRows {
    let rows = model.transpose();
    [
        rows.x_axis.to_array(),
        rows.y_axis.to_array(),
        rows.z_axis.to_array(),
    ]
}

#[inline]
fn rows_matrix(rows: &PoseRows) -> Mat4 {
    Mat4::from_cols(
        Vec4::from_array(rows[0]),
        Vec4::from_array(rows[1]),
        Vec4::from_array(rows[2]),
        Vec4::W,
    )
    .transpose()
}

#[crate::gpu_bits(crate = crate)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoseWord {
    pub valid: bool,
    pub blend: bool,
    pub composed: bool,
    pub parent_composed: bool,
}

#[derive(Clone, Copy)]
struct Pose {
    translation: Vec3,
    rotation: Vec4,
    scale: Vec3,
}

impl Pose {
    fn of(pose: &NodePose) -> Self {
        let [tx, ty, tz, sx, rx, ry, rz, rw, sy, sz] = *pose;
        Self {
            translation: Vec3::new(tx, ty, tz),
            rotation: Vec4::new(rx, ry, rz, rw),
            scale: Vec3::new(sx, sy, sz),
        }
    }

    fn trs(self) -> Mat4 {
        let (x, y, z, w) = (
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
            self.rotation.w,
        );
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, xy, xz) = (x * x2, x * y2, x * z2);
        let (yy, yz, zz) = (y * y2, y * z2, z * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        Mat4::from_cols(
            Vec4::new(1.0 - (yy + zz), xy + wz, xz - wy, 0.0) * self.scale.x,
            Vec4::new(xy - wz, 1.0 - (xx + zz), yz + wx, 0.0) * self.scale.y,
            Vec4::new(xz + wy, yz - wx, 1.0 - (xx + yy), 0.0) * self.scale.z,
            self.translation.extend(1.0),
        )
    }
}

fn blend_rotation(previous: Vec4, current: Vec4, alpha: f32) -> Vec4 {
    let p = previous.normalize();
    let mut c = current.normalize();
    let d = p.dot(c);
    if d.abs() >= bounds::LINEAR_ROTATION_DOT_THRESHOLD {
        if d < 0.0 {
            c = -c;
        }
        return (p * (1.0 - alpha) + c * alpha).normalize();
    }
    let mut cosine = d;
    if cosine < 0.0 {
        c = -c;
        cosine = -cosine;
    }
    let theta = cosine.acos();
    let start = sin(theta * (1.0 - alpha));
    let end = sin(theta * alpha);
    ((p * start + c * end) * (1.0 / sin(theta))).normalize()
}

pub fn drawn_local(previous: &NodePose, current: &NodePose, alpha: f32, blend: bool) -> Mat4 {
    let (previous, current) = (Pose::of(previous), Pose::of(current));
    if !blend || alpha >= 1.0 {
        return current.trs();
    }
    if alpha == 0.0 {
        return previous.trs();
    }
    let inverse = 1.0 - alpha;
    Pose {
        translation: previous.translation * inverse + current.translation * alpha,
        rotation: blend_rotation(previous.rotation, current.rotation, alpha),
        scale: previous.scale * inverse + current.scale * alpha,
    }
    .trs()
}

fn largest(value: Vec3) -> f32 {
    max(value.x.abs(), max(value.y.abs(), value.z.abs()))
}

pub fn pose_admitted(m: Mat4) -> bool {
    let finite = |column: Vec4| {
        column
            .abs()
            .cmple(Vec4::splat(bounds::POSE_FLOAT_LIMIT))
            .all()
    };
    if !(finite(m.x_axis) && finite(m.y_axis) && finite(m.z_axis) && finite(m.w_axis)) {
        return false;
    }
    let (x, y, z) = (
        m.x_axis.truncate(),
        m.y_axis.truncate(),
        m.z_axis.truncate(),
    );
    let scale = max(largest(x), max(largest(y), largest(z)));
    let determinant = x.dot(y.cross(z));
    (bounds::MIN_LINEAR_SCALE..=bounds::COEFFICIENT_LIMIT).contains(&scale)
        && largest(m.w_axis.truncate()) <= bounds::COEFFICIENT_LIMIT
        && determinant.abs() >= bounds::COMMON_DETERMINANT_MARGIN * ((scale * scale) * scale)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Span {
    #[default]
    Fresh,
    Moving {
        from: u64,
    },
    Still,
}

pub(crate) fn check_alpha(alpha: f32) -> Result<(), engine_settings::Error> {
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(engine_settings::Error::new(
            engine_settings::ErrorCode::InvalidPresentationAlpha,
            "$.alpha",
        )
        .range(0.0, 1.0)
        .got(alpha));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Presented {
    #[default]
    Tick,
    Between {
        alpha: f32,
    },
    Paced {
        alpha: f32,
    },
}

impl Presented {
    pub fn alpha(self) -> f32 {
        match self {
            Self::Tick => 1.0,
            Self::Between { alpha } | Self::Paced { alpha } => alpha,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PresentationState {
    presented: Presented,
    frame: Option<u64>,
    tick: Option<u64>,
    nodes: usize,
    moving: usize,
    fresh: usize,
    composed: usize,
    blended_nodes: usize,
    largest_blend_distance: f32,
    bytes: usize,
}

impl PresentationState {
    pub fn presented(&self) -> Presented {
        self.presented
    }
    pub fn alpha(&self) -> f32 {
        self.presented.alpha()
    }
    pub fn frame(&self) -> Option<u64> {
        self.frame
    }
    pub fn tick(&self) -> Option<u64> {
        self.tick
    }
    pub fn nodes(&self) -> usize {
        self.nodes
    }
    pub fn moving(&self) -> usize {
        self.moving
    }
    pub fn fresh(&self) -> usize {
        self.fresh
    }
    pub fn composed(&self) -> usize {
        self.composed
    }
    pub fn blended_nodes(&self) -> usize {
        self.blended_nodes
    }
    pub fn largest_blend_distance(&self) -> f32 {
        self.largest_blend_distance
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

#[derive(Clone, Copy)]
pub struct NodeStreams<'a> {
    pub alpha: f32,
    pub current: &'a [NodePose],
    pub previous: &'a [NodePose],
    pub info: &'a [NodeInfo],
    pub authority: &'a [PoseRows],
    pub parents: &'a [u32],
    pub levels: &'a [u32],
    pub blended: usize,
}

#[derive(Clone, Copy)]
struct Drawn {
    display: Mat4,
    current: Mat4,
    fallback: bool,
    composed: bool,
}

#[derive(Resource)]
pub struct DrawnPoses {
    topology: Option<Revision>,
    entities: Vec<Entity>,
    spans: Vec<Span>,
    fresh_at: Vec<u64>,
    ticked: Vec<NodePose>,
    previous: Vec<NodePose>,
    words: Vec<PoseWord>,
    info: Vec<NodeInfo>,
    composed: Vec<u32>,
    spare: Vec<u32>,
    parents: Vec<u32>,
    levels: Vec<u32>,
    sorted: Vec<(u8, u32)>,
    moving: Vec<u32>,
    moved: Vec<u32>,
    fresh: Vec<u32>,
    touched: Vec<u32>,
    asked: Vec<Entity>,
    stack: Vec<usize>,
    marks: Vec<u32>,
    mark: u32,
    buckets: Vec<Vec<u32>>,
    whole: bool,
    last_revision: u64,
    node_changes: Changes,
    alpha: f32,
    presented: Presented,
    blended: (usize, f32),
    frame: u64,
}

impl Default for DrawnPoses {
    fn default() -> Self {
        Self {
            topology: None,
            entities: Vec::new(),
            spans: Vec::new(),
            fresh_at: Vec::new(),
            ticked: Vec::new(),
            previous: Vec::new(),
            words: Vec::new(),
            info: Vec::new(),
            composed: Vec::new(),
            spare: Vec::new(),
            parents: Vec::new(),
            levels: Vec::new(),
            sorted: Vec::new(),
            moving: Vec::new(),
            moved: Vec::new(),
            fresh: Vec::new(),
            touched: Vec::new(),
            asked: Vec::new(),
            stack: Vec::new(),
            marks: Vec::new(),
            mark: 0,
            buckets: Vec::new(),
            whole: true,
            last_revision: 0,
            node_changes: Changes::new(1),
            alpha: 1.0,
            presented: Presented::Tick,
            blended: (0, 0.0),
            frame: 0,
        }
    }
}

fn tick_paced(world: &World, node: Entity) -> bool {
    world
        .get::<crate::Mover>(node)
        .is_some_and(|mover| matches!(mover.pace, crate::Pace::Tick))
}

fn enqueue(
    (marks, mark): (&mut [u32], u32),
    buckets: &mut [Vec<u32>],
    globals: &GlobalTransforms,
    index: usize,
) {
    if index < marks.len() && marks[index] != mark {
        marks[index] = mark;
        buckets[usize::from(globals.links()[index].depth())].push(index as u32);
    }
}

impl DrawnPoses {
    pub fn fresh(&mut self, node: Entity) {
        self.asked.push(node);
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn presented(&self) -> Presented {
        self.presented
    }

    pub fn span(&self, globals: &GlobalTransforms, node: Entity) -> Option<Span> {
        self.index_of(globals, node).map(|index| self.spans[index])
    }

    pub fn fresh_at(&self, globals: &GlobalTransforms, node: Entity) -> Option<u64> {
        self.index_of(globals, node)
            .and_then(|index| self.fresh_at.get(index).copied())
    }

    pub fn blends(&self, globals: &GlobalTransforms, node: Entity) -> bool {
        matches!(self.span(globals, node), Some(Span::Moving { .. }))
    }

    pub fn drawn_pose(&self, globals: &GlobalTransforms, node: Entity) -> Option<Mat4> {
        match self.index_of(globals, node) {
            Some(index) => self.drawn(globals, index, 0).map(|drawn| drawn.display),
            None => globals.get(node)?.matrix(),
        }
    }

    pub fn streams<'a>(&'a self, globals: &'a GlobalTransforms) -> NodeStreams<'a> {
        let aligned = self.topology == Some(globals.topology_revision())
            && self.info.len() == globals.links().len();
        let nodes = if aligned { self.info.len() } else { 0 };
        NodeStreams {
            alpha: self.alpha,
            current: &globals.locals()[..nodes],
            previous: &self.previous[..nodes],
            info: &self.info[..nodes],
            authority: &globals.global_rows()[..nodes],
            parents: if aligned { &self.parents } else { &[] },
            levels: if aligned { &self.levels } else { &[] },
            blended: if aligned { self.blended.0 } else { 0 },
        }
    }

    pub fn node_changes(&self) -> &Changes {
        &self.node_changes
    }

    pub fn capacity_bytes(&self) -> usize {
        (self.ticked.capacity() + self.previous.capacity()) * size_of::<NodePose>()
            + self.info.capacity() * size_of::<NodeInfo>()
            + self.spans.capacity() * size_of::<Span>()
            + self.entities.capacity() * size_of::<Entity>()
            + self.words.capacity() * size_of::<PoseWord>()
            + (self.composed.capacity()
                + self.spare.capacity()
                + self.parents.capacity()
                + self.levels.capacity()
                + self.moving.capacity()
                + self.moved.capacity()
                + self.fresh.capacity()
                + self.touched.capacity()
                + self.marks.capacity())
                * size_of::<u32>()
    }

    fn index_of(&self, globals: &GlobalTransforms, node: Entity) -> Option<usize> {
        globals
            .node_index(node)
            .filter(|&index| self.entities.get(index) == Some(&node))
    }

    fn drawn(&self, globals: &GlobalTransforms, index: usize, depth: usize) -> Option<Drawn> {
        let link = globals.links().get(index)?;
        if !link.topology_valid() || !link.has_global() || depth > MAX_HIERARCHY_DEPTH {
            return None;
        }
        let global = rows_matrix(&globals.global_rows()[index]);
        let above = match link.parent() {
            Some(parent) => Some(self.drawn(globals, parent, depth + 1)?),
            None => None,
        };
        let blend = matches!(self.spans[index], Span::Moving { .. });
        let parent_composed = above.is_some_and(|above| above.composed);
        if !blend && !parent_composed {
            return Some(Drawn {
                display: global,
                current: global,
                fallback: false,
                composed: false,
            });
        }
        let (base_display, base_current, parent_fallback) = match above {
            Some(above) if above.composed => (above.display, above.current, above.fallback),
            Some(above) => (above.display, above.display, false),
            None => (Mat4::IDENTITY, Mat4::IDENTITY, false),
        };
        let local = &globals.locals()[index];
        let current = base_current * Pose::of(local).trs();
        let display = if parent_fallback {
            None
        } else {
            Some(base_display * drawn_local(&self.previous[index], local, self.alpha, blend))
                .filter(|display| pose_admitted(*display))
        };
        Some(match display {
            Some(display) => Drawn {
                display,
                current,
                fallback: false,
                composed: true,
            },
            None => Drawn {
                display: current,
                current,
                fallback: true,
                composed: true,
            },
        })
    }

    fn next_mark(&mut self) {
        self.mark = self.mark.wrapping_add(1);
        if self.mark == 0 {
            self.marks.fill(0);
            self.mark = 1;
        }
    }

    fn index(&mut self, globals: &GlobalTransforms) {
        let nodes = globals.links().len();
        let placed = globals.placed().read();
        let known = self.spans.len();
        if nodes < known {
            self.shrink(nodes);
        }
        for local in &globals.locals()[known.min(nodes)..nodes] {
            self.entities.push(Entity::PLACEHOLDER);
            self.spans.push(Span::Fresh);
            self.fresh_at.push(self.frame);
            self.ticked.push(*local);
            self.previous.push(*local);
            self.words.push(PoseWord::default());
            self.info.push([NO_PARENT, 0]);
            self.marks.push(0);
        }
        if placed.all {
            self.whole = true;
            self.node_changes.mark_all();
            for index in 0..nodes {
                self.assign(globals, index);
            }
        } else {
            for &index in placed.listed {
                if (index as usize) < nodes {
                    self.assign(globals, index as usize);
                }
            }
        }
        globals.placed().take(1, placed.upto);
        self.topology = Some(globals.topology_revision());
    }

    fn assign(&mut self, globals: &GlobalTransforms, index: usize) {
        let entity = globals
            .entity_at(index as u32)
            .unwrap_or(Entity::PLACEHOLDER);
        if self.entities[index] == entity {
            return;
        }
        let local = globals.locals()[index];
        self.entities[index] = entity;
        self.spans[index] = Span::Fresh;
        self.fresh_at[index] = self.frame;
        self.ticked[index] = local;
        self.previous[index] = local;
        self.fresh.push(index as u32);
        self.touched.push(index as u32);
        self.node_changes.mark(index as u32);
    }

    fn shrink(&mut self, nodes: usize) {
        self.entities.truncate(nodes);
        self.spans.truncate(nodes);
        self.fresh_at.truncate(nodes);
        self.ticked.truncate(nodes);
        self.previous.truncate(nodes);
        self.words.truncate(nodes);
        self.info.truncate(nodes);
        self.marks.truncate(nodes);
        let kept = |index: &u32| (*index as usize) < nodes;
        self.moving.retain(kept);
        self.fresh.retain(kept);
        self.composed.retain(kept);
        self.touched.retain(kept);
    }

    fn apply_fresh(&mut self, globals: &GlobalTransforms) {
        if self.asked.is_empty() {
            return;
        }
        let mut stack = std::mem::take(&mut self.stack);
        stack.extend(
            self.asked
                .drain(..)
                .filter_map(|node| globals.node_index(node)),
        );
        while let Some(index) = stack.pop() {
            if index >= self.spans.len() {
                continue;
            }
            let now = globals.locals()[index];
            self.spans[index] = Span::Fresh;
            self.fresh_at[index] = self.frame;
            self.ticked[index] = now;
            self.previous[index] = now;
            self.fresh.push(index as u32);
            self.touched.push(index as u32);
            self.node_changes.mark(index as u32);
            stack.extend(globals.children(index));
        }
        self.stack = stack;
    }

    fn fresh_all(&mut self, globals: &GlobalTransforms) {
        self.index(globals);
        self.asked.clear();
        self.moving.clear();
        self.fresh.clear();
        for (index, local) in globals.locals().iter().enumerate().take(self.spans.len()) {
            self.spans[index] = Span::Fresh;
            self.fresh_at[index] = self.frame;
            self.ticked[index] = *local;
            self.previous[index] = *local;
            self.fresh.push(index as u32);
        }
        self.whole = true;
        self.node_changes.mark_all();
    }

    fn age_fresh(&mut self, world: &World) {
        let mut fresh = std::mem::take(&mut self.fresh);
        fresh.retain(|&index| {
            let at = index as usize;
            if self.spans.get(at) != Some(&Span::Fresh) {
                return false;
            }
            if self.fresh_at[at] < self.frame && !tick_paced(world, self.entities[at]) {
                self.spans[at] = Span::Still;
                self.touched.push(index);
                return false;
            }
            true
        });
        self.fresh = fresh;
    }

    fn observe(&mut self, world: &World, globals: &GlobalTransforms, index: usize, tick: u64) {
        if index >= self.spans.len() || self.marks[index] == self.mark {
            return;
        }
        self.marks[index] = self.mark;
        if !globals.links()[index].topology_valid() {
            return;
        }
        let now = globals.locals()[index];
        if same_pose_bits(&self.ticked[index], &now) {
            return;
        }
        if tick_paced(world, self.entities[index]) {
            self.previous[index] = self.ticked[index];
            self.spans[index] = Span::Moving {
                from: tick.saturating_sub(1),
            };
            self.moving.push(index as u32);
            self.touched.push(index as u32);
            self.node_changes.mark(index as u32);
        } else if self.spans[index] != Span::Still {
            self.spans[index] = Span::Still;
            self.touched.push(index as u32);
        }
        self.ticked[index] = now;
    }

    fn capture(&mut self, world: &World, globals: &GlobalTransforms, tick: u64) {
        let _profile = engine_profile::span("presentation", "capture_tick");
        self.index(globals);
        let mut fresh = std::mem::take(&mut self.fresh);
        for &index in &fresh {
            let index = index as usize;
            if self.spans.get(index) == Some(&Span::Fresh) {
                self.spans[index] = Span::Still;
                self.touched.push(index as u32);
            }
        }
        fresh.clear();
        self.fresh = fresh;
        self.next_mark();
        let mut moved = std::mem::take(&mut self.moved);
        std::mem::swap(&mut moved, &mut self.moving);
        let changed = globals.moved().read();
        if changed.all {
            for index in 0..self.spans.len() {
                self.observe(world, globals, index, tick);
            }
        } else {
            for &index in changed.listed {
                self.observe(world, globals, index as usize, tick);
            }
        }
        globals.moved().take(0, changed.upto);
        for &index in &moved {
            let index = index as usize;
            if let Some(Span::Moving { from }) = self.spans.get(index).copied()
                && from.saturating_add(1) != tick
            {
                self.spans[index] = Span::Still;
                self.touched.push(index as u32);
            }
        }
        moved.clear();
        self.moved = moved;
        self.apply_fresh(globals);
    }

    fn update(&mut self, globals: &GlobalTransforms, index: usize) -> bool {
        let link = &globals.links()[index];
        let valid = link.topology_valid() && link.has_global();
        let parent = link.parent().filter(|_| link.topology_valid());
        let parent_composed = valid && parent.is_some_and(|parent| self.words[parent].composed);
        let blend = valid && matches!(self.spans[index], Span::Moving { .. });
        let moved = valid
            && globals.local_revisions()[index] > self.last_revision
            && self.spans[index] != Span::Fresh;
        let word = PoseWord {
            valid,
            blend,
            composed: blend || parent_composed || moved,
            parent_composed,
        };
        let info = [
            parent.map_or(NO_PARENT, |parent| parent as u32),
            word.pack(),
        ];
        if self.info[index] != info {
            self.info[index] = info;
            self.node_changes.mark(index as u32);
        }
        let before = std::mem::replace(&mut self.words[index], word);
        before.composed != word.composed || before.valid != word.valid
    }

    fn walk_all(&mut self, globals: &GlobalTransforms) {
        let mut composed = std::mem::take(&mut self.spare);
        composed.clear();
        for &index in globals.order() {
            if index >= self.words.len() {
                continue;
            }
            self.update(globals, index);
            if self.words[index].composed {
                composed.push(index as u32);
            }
        }
        self.spare = std::mem::replace(&mut self.composed, composed);
    }

    fn walk_seeds(&mut self, globals: &GlobalTransforms, listed: &[u32]) {
        let mark = self.mark;
        let mut buckets = std::mem::take(&mut self.buckets);
        buckets.resize_with(MAX_HIERARCHY_DEPTH + 1, Vec::new);
        let seeds = listed.iter().chain(&self.composed).chain(&self.touched);
        for &index in seeds {
            enqueue(
                (&mut self.marks, mark),
                &mut buckets,
                globals,
                index as usize,
            );
        }
        let mut composed = std::mem::take(&mut self.spare);
        composed.clear();
        for depth in 0..buckets.len() {
            let mut bucket = std::mem::take(&mut buckets[depth]);
            for &index in &bucket {
                let index = index as usize;
                if self.update(globals, index) {
                    for child in globals.children(index) {
                        enqueue((&mut self.marks, mark), &mut buckets, globals, child);
                    }
                }
                if self.words[index].composed {
                    composed.push(index as u32);
                }
            }
            bucket.clear();
            buckets[depth] = bucket;
        }
        self.buckets = buckets;
        self.spare = std::mem::replace(&mut self.composed, composed);
    }

    fn gather(&mut self, globals: &GlobalTransforms) {
        self.node_changes.settle();
        self.index(globals);
        self.apply_fresh(globals);
        self.next_mark();
        let changed = globals.moved().read();
        let listed = if changed.all {
            self.whole = true;
            self.node_changes.mark_all();
            &[][..]
        } else {
            changed.listed
        };
        let seeds = listed.len() + self.composed.len() + self.touched.len();
        if self.whole || seeds > self.words.len() / SPARSE_SHARE {
            self.walk_all(globals);
        } else {
            self.walk_seeds(globals, listed);
        }
        for &index in listed {
            self.node_changes.mark(index);
        }
        globals.moved().take(1, changed.upto);
        self.whole = false;
        self.touched.clear();
        self.order_parents(globals);
        self.last_revision = globals.revision();
    }

    fn order_parents(&mut self, globals: &GlobalTransforms) {
        let links = globals.links();
        let locals = globals.locals();
        self.sorted.clear();
        self.blended = (0, 0.0);
        for &index in &self.composed {
            let node = index as usize;
            if links[node].has_children() {
                self.sorted.push((links[node].depth(), index));
            }
            if self.words[node].blend {
                let (current, previous) = (&locals[node], &self.previous[node]);
                let distance = Vec3::new(
                    current[0] - previous[0],
                    current[1] - previous[1],
                    current[2] - previous[2],
                )
                .length();
                self.blended.0 += 1;
                self.blended.1 = max(self.blended.1, distance);
            }
        }
        self.sorted.sort_unstable();
        self.parents.clear();
        self.levels.clear();
        let deepest = self
            .sorted
            .last()
            .map_or(0, |&(depth, _)| usize::from(depth) + 1);
        let mut at = 0;
        for depth in 0..deepest {
            while at < self.sorted.len() && usize::from(self.sorted[at].0) == depth {
                self.parents.push(self.sorted[at].1);
                at += 1;
            }
            self.levels.push(self.parents.len() as u32);
        }
    }

    fn counted(&self, state: &mut PresentationState) {
        state.nodes = self.spans.len();
        state.moving = self.moving.len();
        state.fresh = self.fresh.len();
        state.composed = self.composed.len();
        (state.blended_nodes, state.largest_blend_distance) = self.blended;
        state.bytes = self.capacity_bytes();
    }
}

#[derive(SystemParam)]
pub struct Poses<'w> {
    poses: Res<'w, DrawnPoses>,
    globals: Res<'w, GlobalTransforms>,
}

impl Poses<'_> {
    pub fn drawn_pose(&self, node: Entity) -> Option<Mat4> {
        self.poses.drawn_pose(&self.globals, node)
    }

    pub fn span(&self, node: Entity) -> Option<Span> {
        self.poses.span(&self.globals, node)
    }

    pub fn alpha(&self) -> f32 {
        self.poses.alpha()
    }
}

pub fn make_fresh(world: &mut World, nodes: &[Entity]) {
    if let Some(mut poses) = world.get_resource_mut::<DrawnPoses>() {
        for &node in nodes {
            poses.fresh(node);
        }
    }
}

pub(crate) fn capture(world: &mut World) {
    let tick = world
        .get_resource::<FixedTime>()
        .map_or(0, |time| time.tick);
    let frame = world
        .get_resource::<FrameTime>()
        .map_or(0, |time| time.frame);
    world.resource_scope(|world, mut poses: Mut<DrawnPoses>| {
        let world: &World = world;
        poses.frame = frame;
        if let Some(globals) = world.get_resource::<GlobalTransforms>() {
            poses.capture(world, globals, tick);
        }
    });
}

pub(crate) fn present(world: &mut World, presented: Presented) {
    let frame = world.get_resource::<FrameTime>().map(|time| time.frame);
    let tick = world.get_resource::<FixedTime>().map(|time| time.tick);
    world.resource_scope(|world, mut poses: Mut<DrawnPoses>| {
        poses.presented = presented;
        poses.alpha = presented.alpha();
        if let Some(frame) = frame {
            poses.frame = frame;
        }
        {
            let world: &World = &*world;
            if let Some(globals) = world.get_resource::<GlobalTransforms>() {
                poses.index(globals);
                poses.age_fresh(world);
                poses.apply_fresh(globals);
            }
        }
        let mut state = world.resource_mut::<PresentationState>();
        state.presented = presented;
        state.frame = frame;
        state.tick = tick;
    });
}

pub(crate) fn gather(world: &mut World) {
    let _profile = engine_profile::span("presentation", "gather");
    world.resource_scope(|world, mut poses: Mut<DrawnPoses>| {
        if let Some(globals) = world.get_resource::<GlobalTransforms>() {
            poses.gather(globals);
        }
        engine_profile::counter("presentation.composed", poses.composed.len() as f64);
        engine_profile::counter("memory.presentation_bytes", poses.capacity_bytes() as f64);
        poses.counted(&mut world.resource_mut::<PresentationState>());
    });
}

pub(crate) fn fresh_all(world: &mut World) {
    world.resource_scope(|world, mut poses: Mut<DrawnPoses>| {
        if let Some(globals) = world.get_resource::<GlobalTransforms>() {
            poses.fresh_all(globals);
        }
    });
}
