use engine_core::{Error, ErrorCode, Expected};

pub(crate) const MAX_COLORS: usize = 4;
pub(crate) const MAX_PASSES: usize = 64;
const SET_WORDS: usize = 2;
pub(crate) const MAX_RESOURCES: usize = 64 * SET_WORDS;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    Prepare,
    Opaque,
    Deferred,
    OpaqueLate,
    Occlusion,
    OpaqueSecond,
    AfterOpaque,
    Surfaces,
    Sky,
    Transparent,
    Exposure,
    PostScene,
    PostLight,
    Composite,
    AfterFrame,
}

impl Stage {
    pub const ALL: [Self; 15] = [
        Self::Prepare,
        Self::Opaque,
        Self::Deferred,
        Self::OpaqueLate,
        Self::Occlusion,
        Self::OpaqueSecond,
        Self::AfterOpaque,
        Self::Surfaces,
        Self::Sky,
        Self::Transparent,
        Self::Exposure,
        Self::PostScene,
        Self::PostLight,
        Self::Composite,
        Self::AfterFrame,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Prepare => "prepare",
            Self::Opaque => "opaque",
            Self::Deferred => "deferred",
            Self::OpaqueLate => "opaque_late",
            Self::Occlusion => "occlusion",
            Self::OpaqueSecond => "opaque_second",
            Self::AfterOpaque => "after_opaque",
            Self::Surfaces => "surfaces",
            Self::Sky => "sky",
            Self::Transparent => "transparent",
            Self::Exposure => "exposure",
            Self::PostScene => "post_scene",
            Self::PostLight => "post_light",
            Self::Composite => "composite",
            Self::AfterFrame => "after_frame",
        }
    }

    pub const fn batch(self) -> u8 {
        match self {
            Self::Prepare => 0,
            Self::Opaque
            | Self::Deferred
            | Self::OpaqueLate
            | Self::Occlusion
            | Self::OpaqueSecond => 1,
            Self::AfterOpaque
            | Self::Surfaces
            | Self::Sky
            | Self::Transparent
            | Self::Exposure
            | Self::PostScene
            | Self::PostLight
            | Self::Composite
            | Self::AfterFrame => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifetime {
    Frame,
    Kept,
    Output,
}

#[derive(Clone, Copy, Debug)]
pub struct ResourceDecl {
    pub name: &'static str,
    pub lifetime: Lifetime,
}

#[derive(Clone, Copy, Debug)]
pub struct PassDecl {
    pub name: &'static str,
    pub stage: Stage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PassId(u16);

impl PassId {
    pub fn index(self) -> usize {
        usize::from(self.0)
    }

    fn bit(self) -> u64 {
        1 << self.0
    }
}

impl ResourceId {
    pub fn index(self) -> usize {
        usize::from(self.0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Set([u64; SET_WORDS]);

impl Set {
    fn contains(self, resource: ResourceId) -> bool {
        let index = resource.index();
        self.0[index / 64] & (1 << (index % 64)) != 0
    }

    fn insert(&mut self, resource: ResourceId) {
        let index = resource.index();
        self.0[index / 64] |= 1 << (index % 64);
    }

    fn remove(&mut self, resource: ResourceId) {
        let index = resource.index();
        self.0[index / 64] &= !(1 << (index % 64));
    }

    fn union(self, other: Self) -> Self {
        let mut words = self.0;
        for (word, more) in words.iter_mut().zip(other.0) {
            *word |= more;
        }
        Self(words)
    }

    fn intersect(self, other: Self) -> Self {
        let mut words = self.0;
        for (word, more) in words.iter_mut().zip(other.0) {
            *word &= more;
        }
        Self(words)
    }

    fn without(self, other: Self) -> Self {
        let mut words = self.0;
        for (word, more) in words.iter_mut().zip(other.0) {
            *word &= !more;
        }
        Self(words)
    }

    fn is_empty(self) -> bool {
        self.0.iter().all(|&word| word == 0)
    }

    fn first(self) -> Option<ResourceId> {
        self.0.iter().enumerate().find_map(|(word, &bits)| {
            (bits != 0).then(|| ResourceId((word * 64) as u16 + bits.trailing_zeros() as u16))
        })
    }

    fn iter(self) -> impl Iterator<Item = ResourceId> {
        (0..MAX_RESOURCES as u16)
            .map(ResourceId)
            .filter(move |&resource| self.contains(resource))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Clear {
    Color([f64; 4]),
    Depth(f32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Access {
    Clear(Clear),
    Draw(Clear),
    Load,
    ReadOnly,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Load {
    Clear(Clear),
    Load,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ops {
    pub load: Load,
    pub store: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Targets {
    Canvas,
    Post,
    Temporal,
    Debug,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameShape {
    pub gpu: bool,
    pub targets: Targets,
    pub canvas_clear: [f64; 4],
    pub phases: u8,
    pub deferred: bool,
}

pub const SCENE_CLEAR: [f64; 4] = [0.009, 0.015, 0.025, 1.0];
pub const TRANSPARENT: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
pub const WHITE: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
pub const BLACK: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
pub const FAR: f32 = 0.0;

#[derive(Clone, Copy, Debug)]
struct ResourceEntry {
    name: &'static str,
    lifetime: Lifetime,
    owner: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct PassEntry {
    pub name: &'static str,
    pub span: &'static str,
    pub stage: Stage,
    pub part: usize,
    pub index: usize,
}

#[derive(Debug, Default)]
pub struct GraphTable {
    resources: Vec<ResourceEntry>,
    passes: Vec<PassEntry>,
    names: &'static [&'static str],
}

impl GraphTable {
    pub fn add_part(
        &mut self,
        part: usize,
        brick: &'static str,
        passes: &'static [PassDecl],
        resources: &'static [ResourceDecl],
    ) -> Result<(), Error> {
        for resource in resources {
            if self.resource(resource.name).is_some() {
                return Err(Error::new(
                    ErrorCode::DuplicateBrick,
                    format!("$.bricks.{brick}.resources.{}", resource.name),
                )
                .expected(Expected::Unique)
                .got(resource.name));
            }
            if self.resources.len() >= MAX_RESOURCES {
                return Err(Error::new(
                    ErrorCode::RenderFailed,
                    format!("$.bricks.{brick}.resources"),
                )
                .at_most(MAX_RESOURCES as u64)
                .got(self.resources.len() + 1));
            }
            self.resources.push(ResourceEntry {
                name: resource.name,
                lifetime: resource.lifetime,
                owner: part,
            });
        }
        for (index, pass) in passes.iter().enumerate() {
            if self.passes.iter().any(|known| known.name == pass.name) {
                return Err(Error::new(
                    ErrorCode::DuplicateBrick,
                    format!("$.bricks.{brick}.passes.{}", pass.name),
                )
                .expected(Expected::Unique)
                .got(pass.name));
            }
            if self.passes.len() >= MAX_PASSES {
                return Err(
                    Error::new(ErrorCode::RenderFailed, format!("$.bricks.{brick}.passes"))
                        .at_most(MAX_PASSES as u64)
                        .got(self.passes.len() + 1),
                );
            }
            self.passes.push(PassEntry {
                name: pass.name,
                span: leak(format!("stage_{}", pass.name)),
                stage: pass.stage,
                part,
                index,
            });
        }
        let names: Vec<&'static str> = self.passes.iter().map(|pass| pass.name).collect();
        self.names = Box::leak(names.into_boxed_slice());
        Ok(())
    }

    pub fn resource(&self, name: &str) -> Option<ResourceId> {
        self.resources
            .iter()
            .position(|resource| resource.name == name)
            .map(|index| ResourceId(index as u16))
    }

    pub fn resource_name(&self, resource: ResourceId) -> &'static str {
        self.resources[resource.index()].name
    }

    pub fn lifetime(&self, resource: ResourceId) -> Lifetime {
        self.resources[resource.index()].lifetime
    }

    pub fn owner(&self, resource: ResourceId) -> usize {
        self.resources[resource.index()].owner
    }

    pub fn pass(&self, part: usize, index: usize) -> Option<PassId> {
        self.passes
            .iter()
            .position(|pass| pass.part == part && pass.index == index)
            .map(|position| PassId(position as u16))
    }

    pub fn pass_named(&self, name: &str) -> Option<PassId> {
        self.passes
            .iter()
            .position(|pass| pass.name == name)
            .map(|position| PassId(position as u16))
    }

    pub fn entry(&self, pass: PassId) -> PassEntry {
        self.passes[pass.index()]
    }

    pub fn names(&self) -> &'static [&'static str] {
        self.names
    }

    pub fn passes(&self) -> impl Iterator<Item = (PassId, PassEntry)> + '_ {
        self.passes
            .iter()
            .enumerate()
            .map(|(index, &pass)| (PassId(index as u16), pass))
    }

    pub fn names_of(&self, kept: u64) -> engine_core::Names {
        engine_core::Names {
            bits: kept,
            names: self.names,
        }
    }

    fn resources(&self) -> impl Iterator<Item = (ResourceId, ResourceEntry)> + '_ {
        self.resources
            .iter()
            .enumerate()
            .map(|(index, &resource)| (ResourceId(index as u16), resource))
    }
}

fn leak(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

#[derive(Clone, Debug)]
pub struct Node {
    pass: PassId,
    stage: Stage,
    reads: Set,
    written_reads: Set,
    writes: Set,
    colors: [Option<(ResourceId, Access)>; MAX_COLORS],
    depth: Option<(ResourceId, Access)>,
    kept: bool,
    color_ops: [Option<Ops>; MAX_COLORS],
    depth_ops: Option<Ops>,
}

impl Node {
    pub fn pass(&self) -> PassId {
        self.pass
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }

    pub fn colors(&self) -> impl Iterator<Item = (ResourceId, Ops)> + '_ {
        self.colors
            .iter()
            .zip(&self.color_ops)
            .filter_map(|(color, ops)| Some((color.as_ref()?.0, (*ops)?)))
    }

    pub fn depth(&self) -> Option<(ResourceId, Option<Ops>)> {
        self.depth.map(|(resource, _)| (resource, self.depth_ops))
    }

    fn inputs(&self) -> Set {
        let mut inputs = self.reads;
        for (resource, access) in self.attachments() {
            if !matches!(access, Access::Clear(_)) {
                inputs.insert(resource);
            }
        }
        inputs
    }

    fn required(&self) -> Set {
        let mut required = self.reads;
        for (resource, access) in self.attachments() {
            if matches!(access, Access::Load | Access::ReadOnly) {
                required.insert(resource);
            }
        }
        required
    }

    fn outputs(&self) -> Set {
        let mut outputs = self.writes;
        for (resource, access) in self.attachments() {
            if access != Access::ReadOnly {
                outputs.insert(resource);
            }
        }
        outputs
    }

    fn reads_only(&self) -> Set {
        self.inputs().without(self.outputs())
    }

    fn attachments(&self) -> impl Iterator<Item = (ResourceId, Access)> + '_ {
        self.colors.iter().flatten().chain(&self.depth).copied()
    }
}

pub struct PassBuilder<'g> {
    node: &'g mut Node,
}

impl PassBuilder<'_> {
    #[must_use]
    pub fn read(self, resource: impl Into<Option<ResourceId>>) -> Self {
        if let Some(resource) = resource.into() {
            self.node.reads.insert(resource);
        }
        self
    }

    #[must_use]
    pub fn read_if(self, condition: bool, resource: impl Into<Option<ResourceId>>) -> Self {
        if condition { self.read(resource) } else { self }
    }

    #[must_use]
    pub fn read_written(self, resource: impl Into<Option<ResourceId>>) -> Self {
        if let Some(resource) = resource.into() {
            self.node.written_reads.insert(resource);
        }
        self
    }

    #[must_use]
    pub fn read_written_if(
        self,
        condition: bool,
        resource: impl Into<Option<ResourceId>>,
    ) -> Self {
        if condition {
            self.read_written(resource)
        } else {
            self
        }
    }

    #[must_use]
    pub fn write(self, resource: impl Into<Option<ResourceId>>) -> Self {
        if let Some(resource) = resource.into() {
            self.node.writes.insert(resource);
        }
        self
    }

    #[must_use]
    pub fn write_if(self, condition: bool, resource: impl Into<Option<ResourceId>>) -> Self {
        if condition { self.write(resource) } else { self }
    }

    #[must_use]
    pub fn color(self, resource: ResourceId, access: Access) -> Self {
        if let Some(slot) = self.node.colors.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some((resource, access));
        }
        self
    }

    #[must_use]
    pub fn depth(self, resource: ResourceId, access: Access) -> Self {
        self.node.depth = Some((resource, access));
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    Twice(PassId),
    Unwritten {
        pass: PassId,
        resource: ResourceId,
    },
    Ambiguous {
        first: PassId,
        second: PassId,
        resource: ResourceId,
    },
    Cycle(PassId),
}

impl GraphError {
    pub fn error(self, table: &GraphTable) -> Error {
        let pass = |pass: PassId| table.entry(pass).name;
        match self {
            Self::Twice(twice) => Error::new(
                ErrorCode::RenderFailed,
                format!("$.passes.{}", pass(twice)),
            )
            .expected(Expected::Unique),
            Self::Unwritten {
                pass: reader,
                resource,
            } => Error::new(
                ErrorCode::RenderFailed,
                format!("$.passes.{}", pass(reader)),
            )
            .expected(Expected::Present)
            .got(table.resource_name(resource)),
            Self::Ambiguous {
                first,
                second,
                resource,
            } => Error::new(
                ErrorCode::AmbiguousPasses,
                format!(
                    "$.passes.{}.writes.{}",
                    pass(second),
                    table.resource_name(resource)
                ),
            )
            .expected(Expected::Sorted)
            .got(pass(first)),
            Self::Cycle(cycled) => Error::new(
                ErrorCode::AmbiguousPasses,
                format!("$.passes.{}", pass(cycled)),
            )
            .expected(Expected::Acyclic),
        }
    }
}

#[derive(Debug, Default)]
pub struct FrameGraph {
    nodes: Vec<Node>,
}

impl FrameGraph {
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    pub fn add(&mut self, table: &GraphTable, pass: PassId) -> PassBuilder<'_> {
        self.nodes.push(Node {
            pass,
            stage: table.entry(pass).stage,
            reads: Set::default(),
            written_reads: Set::default(),
            writes: Set::default(),
            colors: [None; MAX_COLORS],
            depth: None,
            kept: false,
            color_ops: [None; MAX_COLORS],
            depth_ops: None,
        });
        let last = self.nodes.len() - 1;
        PassBuilder {
            node: &mut self.nodes[last],
        }
    }

    pub fn compile(&mut self, table: &GraphTable) -> Result<(), GraphError> {
        let mut written = Set::default();
        for node in &self.nodes {
            written = written.union(node.outputs());
        }
        for node in &mut self.nodes {
            node.reads = node.reads.union(node.written_reads.intersect(written));
            node.written_reads = Set::default();
        }
        self.nodes.sort_by_key(|node| (node.stage, node.pass));
        if let Some(twice) = self
            .nodes
            .windows(2)
            .find(|pair| pair[0].pass == pair[1].pass)
        {
            return Err(GraphError::Twice(twice[0].pass));
        }
        let mut ordered = Vec::with_capacity(self.nodes.len());
        let mut start = 0;
        while start < self.nodes.len() {
            let stage = self.nodes[start].stage;
            let end = start
                + self.nodes[start..]
                    .iter()
                    .take_while(|node| node.stage == stage)
                    .count();
            ordered.extend(order_stage(&self.nodes[start..end])?.into_iter().map(|at| start + at));
            start = end;
        }
        let nodes = std::mem::take(&mut self.nodes);
        let mut slots: Vec<Option<Node>> = nodes.into_iter().map(Some).collect();
        self.nodes = ordered
            .into_iter()
            .filter_map(|index| slots[index].take())
            .collect();
        self.plan_operations(table)
    }

    fn plan_operations(&mut self, table: &GraphTable) -> Result<(), GraphError> {
        let mut needed = Set::default();
        for (resource, entry) in table.resources() {
            if entry.lifetime != Lifetime::Frame {
                needed.insert(resource);
            }
        }
        for node in self.nodes.iter_mut().rev() {
            let outputs = node.outputs();
            node.kept = !outputs.intersect(needed).is_empty();
            if !node.kept {
                continue;
            }
            for (slot, ops) in node.colors.iter().zip(&mut node.color_ops) {
                *ops = slot
                    .and_then(|(resource, access)| store_ops(access, needed.contains(resource)));
            }
            node.depth_ops = node
                .depth
                .and_then(|(resource, access)| store_ops(access, needed.contains(resource)));
            for resource in outputs.iter() {
                if table.lifetime(resource) == Lifetime::Frame {
                    needed.remove(resource);
                }
            }
            needed = needed.union(node.inputs());
        }
        let mut written = Set::default();
        for (resource, entry) in table.resources() {
            if entry.lifetime == Lifetime::Kept {
                written.insert(resource);
            }
        }
        for node in self.nodes.iter_mut().filter(|node| node.kept) {
            if let Some(resource) = node.required().without(written).first() {
                return Err(GraphError::Unwritten {
                    pass: node.pass,
                    resource,
                });
            }
            let colors = node.colors;
            for (slot, ops) in colors.iter().zip(&mut node.color_ops) {
                if let (Some((resource, access)), Some(ops)) = (slot, ops.as_mut()) {
                    ops.load = load_op(*access, written.contains(*resource));
                }
            }
            if let (Some((resource, access)), Some(ops)) = (node.depth, node.depth_ops.as_mut()) {
                ops.load = load_op(access, written.contains(resource));
            }
            written = written.union(node.outputs());
        }
        Ok(())
    }

    pub fn kept_passes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(|node| node.kept)
    }

    pub fn kept(&self) -> u64 {
        self.kept_passes()
            .fold(0, |mask, node| mask | node.pass.bit())
    }
}

fn order_stage(nodes: &[Node]) -> Result<Vec<usize>, GraphError> {
    let count = nodes.len();
    let mut before = vec![0_u64; count];
    for (late, reader) in nodes.iter().enumerate() {
        let reads = reader.reads_only();
        for (early, writer) in nodes.iter().enumerate() {
            if early != late && !writer.outputs().intersect(reads).is_empty() {
                before[late] |= 1 << early;
            }
        }
    }
    let mut placed = 0_u64;
    let mut order = Vec::with_capacity(count);
    while order.len() < count {
        let next = (0..count)
            .filter(|&index| placed & (1 << index) == 0 && before[index] & !placed == 0)
            .min_by_key(|&index| nodes[index].pass);
        let Some(next) = next else {
            let stuck = (0..count)
                .filter(|&index| placed & (1 << index) == 0)
                .min_by_key(|&index| nodes[index].pass)
                .map_or(nodes[0].pass, |index| nodes[index].pass);
            return Err(GraphError::Cycle(stuck));
        };
        placed |= 1 << next;
        order.push(next);
    }
    let mut reach = before.clone();
    for &index in &order {
        let mut through = reach[index];
        for earlier in 0..count {
            if through & (1 << earlier) != 0 {
                through |= reach[earlier];
            }
        }
        reach[index] = through;
    }
    for first in 0..count {
        for second in first + 1..count {
            let shared = nodes[first].outputs().intersect(nodes[second].outputs());
            let Some(resource) = shared.first() else {
                continue;
            };
            let ordered =
                reach[second] & (1 << first) != 0 || reach[first] & (1 << second) != 0;
            if !ordered {
                let (early, late) = if nodes[first].pass < nodes[second].pass {
                    (first, second)
                } else {
                    (second, first)
                };
                return Err(GraphError::Ambiguous {
                    first: nodes[early].pass,
                    second: nodes[late].pass,
                    resource,
                });
            }
        }
    }
    Ok(order)
}

fn store_ops(access: Access, read: bool) -> Option<Ops> {
    (access != Access::ReadOnly).then_some(Ops {
        load: Load::Load,
        store: read,
    })
}

fn load_op(access: Access, written: bool) -> Load {
    match access {
        Access::Clear(clear) => Load::Clear(clear),
        Access::Draw(clear) if !written => Load::Clear(clear),
        Access::Draw(_) | Access::Load | Access::ReadOnly => Load::Load,
    }
}
