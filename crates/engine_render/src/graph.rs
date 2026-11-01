use engine_core::{Error, ErrorCode};

use crate::Layer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Life {
    Frame,
    Kept,
    Previous,
    Output,
}

impl From<crate::Lifetime> for Life {
    fn from(lifetime: crate::Lifetime) -> Self {
        match lifetime {
            crate::Lifetime::Frame => Self::Frame,
            crate::Lifetime::Kept => Self::Kept,
            crate::Lifetime::Previous => Self::Previous,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PassId(u32);

impl ResourceId {
    pub fn new(index: usize) -> Self {
        Self(index as u32)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl PassId {
    pub fn new(index: usize) -> Self {
        Self(index as u32)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bits(Vec<u64>);

impl Bits {
    pub fn of(width: usize) -> Self {
        Self(vec![0; width.div_ceil(64)])
    }

    pub fn contains(&self, index: usize) -> bool {
        self.0
            .get(index / 64)
            .is_some_and(|word| word & (1 << (index % 64)) != 0)
    }

    pub fn insert(&mut self, index: usize) {
        if let Some(word) = self.0.get_mut(index / 64) {
            *word |= 1 << (index % 64);
        }
    }

    fn remove(&mut self, index: usize) {
        if let Some(word) = self.0.get_mut(index / 64) {
            *word &= !(1 << (index % 64));
        }
    }

    fn union(&self, other: &Self) -> Self {
        Self(self.0.iter().zip(&other.0).map(|(a, b)| a | b).collect())
    }

    fn intersect(&self, other: &Self) -> Self {
        Self(self.0.iter().zip(&other.0).map(|(a, b)| a & b).collect())
    }

    fn without(&self, other: &Self) -> Self {
        Self(self.0.iter().zip(&other.0).map(|(a, b)| a & !b).collect())
    }

    fn is_empty(&self) -> bool {
        self.0.iter().all(|&word| word == 0)
    }

    fn first(&self) -> Option<usize> {
        self.0.iter().enumerate().find_map(|(word, &bits)| {
            (bits != 0).then(|| word * 64 + bits.trailing_zeros() as usize)
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(word, &bits)| {
            (0..64)
                .filter(move |bit| bits & (1 << bit) != 0)
                .map(move |bit| word * 64 + bit)
        })
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

pub const SCENE_CLEAR: [f64; 4] = [0.009, 0.015, 0.025, 1.0];
pub const TRANSPARENT: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
pub const WHITE: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
pub const BLACK: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
pub const FAR: f32 = 0.0;

#[derive(Clone, Copy, Debug)]
pub struct ResourceEntry {
    pub label: &'static str,
    pub part: u32,
    pub life: Life,
    pub view: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PassEntry {
    pub label: &'static str,
    pub part: u32,
    pub local: u32,
    pub package: &'static str,
    pub registered: u32,
    pub layer: Option<Layer>,
    pub shared: bool,
}

#[derive(Clone, Debug)]
pub struct Declared {
    pub entry: PassEntry,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
    pub colors: Vec<(ResourceId, Access)>,
    pub depth: Option<(ResourceId, Access)>,
    pub after: Vec<PassId>,
}

#[derive(Clone, Debug)]
struct Node {
    entry: PassEntry,
    reads: Bits,
    writes: Bits,
    colors: Vec<(ResourceId, Access)>,
    depth: Option<(ResourceId, Access)>,
}

impl Node {
    fn attachments(&self) -> impl Iterator<Item = (ResourceId, Access)> + '_ {
        self.colors.iter().chain(&self.depth).copied()
    }

    fn inputs(&self) -> Bits {
        let mut inputs = self.reads.clone();
        for (resource, access) in self.attachments() {
            if !matches!(access, Access::Clear(_)) {
                inputs.insert(resource.index());
            }
        }
        inputs
    }

    fn required(&self) -> Bits {
        let mut required = self.reads.clone();
        for (resource, access) in self.attachments() {
            if matches!(access, Access::Load | Access::ReadOnly) {
                required.insert(resource.index());
            }
        }
        required
    }

    fn outputs(&self) -> Bits {
        let mut outputs = self.writes.clone();
        for (resource, access) in self.attachments() {
            if access != Access::ReadOnly {
                outputs.insert(resource.index());
            }
        }
        outputs
    }

    fn reads_only(&self) -> Bits {
        self.inputs().without(&self.outputs())
    }

    fn key(&self, index: usize) -> (u8, u8, &'static str, u32, usize) {
        (
            self.entry.layer.map_or(0, |layer| layer as u8 + 1),
            u8::from(!self.entry.shared),
            self.entry.package,
            self.entry.registered,
            index,
        )
    }

    fn batch(&self, positioned_before: bool) -> u8 {
        match self.entry.layer {
            None if positioned_before => 2,
            None => 0,
            Some(Layer::Occluders | Layer::Opaque | Layer::Late) => 1,
            Some(Layer::Backdrop) if self.entry.shared => 1,
            Some(Layer::Backdrop | Layer::Media | Layer::Over | Layer::Blend) => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Planned {
    pub pass: PassId,
    pub batch: u8,
    pub colors: Vec<(ResourceId, Ops)>,
    pub depth: Option<(ResourceId, Option<Ops>)>,
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub passes: Vec<Planned>,
    pub written: Bits,
}

#[derive(Debug, Default)]
pub struct StaticGraph {
    resources: Vec<ResourceEntry>,
    nodes: Vec<Node>,
    order: Vec<usize>,
    batches: Vec<u8>,
}

fn refused(code: ErrorCode, path: String) -> Error {
    Error::new(code, path)
}

impl StaticGraph {
    pub fn build(resources: Vec<ResourceEntry>, passes: Vec<Declared>) -> Result<Self, Error> {
        let width = resources.len();
        let count = passes.len();
        let mut nodes = Vec::with_capacity(count);
        let mut after = Vec::with_capacity(count);
        for declared in passes {
            let mut reads = Bits::of(width);
            let mut writes = Bits::of(width);
            for resource in &declared.reads {
                reads.insert(resource.index());
            }
            for resource in &declared.writes {
                writes.insert(resource.index());
            }
            after.push(declared.after);
            nodes.push(Node {
                entry: declared.entry,
                reads,
                writes,
                colors: declared.colors,
                depth: declared.depth,
            });
        }
        let mut views = Bits::of(width);
        for (index, resource) in resources.iter().enumerate() {
            if resource.view {
                views.insert(index);
            }
        }
        for node in &nodes {
            if node.entry.layer.is_none() && !node.outputs().intersect(&views).is_empty() {
                return Err(refused(
                    ErrorCode::ViewWriterUnplaced,
                    format!("$.passes.{}", node.entry.label),
                ));
            }
        }
        let mut before = vec![Bits::of(count); count];
        for (late, reader) in nodes.iter().enumerate() {
            let reads = reader.reads_only();
            let inputs = reader.inputs();
            for (early, writer) in nodes.iter().enumerate() {
                if early == late {
                    continue;
                }
                let outputs = writer.outputs();
                for resource in outputs.intersect(&inputs).iter() {
                    let entry = &resources[resource];
                    if entry.view {
                        let placed_earlier = reader.entry.layer.is_none()
                            || writer.key(early) < reader.key(late);
                        if placed_earlier {
                            before[late].insert(early);
                        } else if reads.contains(resource) {
                            before[early].insert(late);
                        }
                    } else if entry.life == Life::Previous {
                        if reads.contains(resource) {
                            before[early].insert(late);
                        }
                    } else if reads.contains(resource) {
                        let kept_after = entry.life == Life::Kept
                            && after[early].contains(&PassId::new(late));
                        if !kept_after {
                            before[late].insert(early);
                        }
                    }
                }
                if reader.entry.layer.is_some() && writer.entry.layer.is_some() {
                    let shared_view = !reader.outputs().intersect(&outputs).intersect(&views).is_empty();
                    if shared_view && writer.key(early) < reader.key(late) {
                        before[late].insert(early);
                    }
                }
            }
            for pass in &after[late] {
                before[late].insert(pass.index());
            }
        }
        let mut placed = Bits::of(count);
        let mut order = Vec::with_capacity(count);
        while order.len() < count {
            let next = (0..count)
                .filter(|&index| {
                    !placed.contains(index) && before[index].without(&placed).is_empty()
                })
                .min_by_key(|&index| nodes[index].key(index));
            let Some(next) = next else {
                let stuck = (0..count)
                    .filter(|&index| !placed.contains(index))
                    .min_by_key(|&index| nodes[index].key(index))
                    .unwrap_or(0);
                return Err(refused(
                    ErrorCode::AmbiguousPasses,
                    format!("$.passes.{}", nodes[stuck].entry.label),
                ));
            };
            placed.insert(next);
            order.push(next);
        }
        let mut reach = before.clone();
        for &index in &order {
            let mut through = reach[index].clone();
            for earlier in reach[index].iter().collect::<Vec<_>>() {
                through = through.union(&reach[earlier]);
            }
            reach[index] = through;
        }
        for first in 0..count {
            for second in first + 1..count {
                let shared = nodes[first]
                    .outputs()
                    .intersect(&nodes[second].outputs())
                    .without(&views);
                let Some(resource) = shared.first() else {
                    continue;
                };
                let ordered = reach[second].contains(first) || reach[first].contains(second);
                if !ordered {
                    return Err(refused(
                        ErrorCode::AmbiguousPasses,
                        format!(
                            "$.passes.{}.writes.{}[{}]",
                            nodes[second].entry.label,
                            resources[resource].label,
                            nodes[first].entry.label
                        ),
                    ));
                }
            }
        }
        for (late, reader) in nodes.iter().enumerate() {
            for resource in reader.reads_only().iter() {
                if resources[resource].life != Life::Frame || resources[resource].view {
                    continue;
                }
                let written = (0..count).any(|early| {
                    reach[late].contains(early) && nodes[early].outputs().contains(resource)
                });
                if !written {
                    return Err(refused(
                        ErrorCode::ReadBeforeWrite,
                        format!("$.passes.{}.reads.{}", reader.entry.label, resources[resource].label),
                    ));
                }
            }
        }
        let mut batches = Vec::with_capacity(count);
        let mut positioned = false;
        let mut last = 0;
        for &index in &order {
            let node = &nodes[index];
            let batch = node.batch(positioned).max(last);
            positioned |= node.entry.layer.is_some();
            last = batch;
            batches.push(batch);
        }
        Ok(Self {
            resources,
            nodes,
            order,
            batches,
        })
    }

    pub fn passes(&self) -> usize {
        self.nodes.len()
    }

    pub fn resources(&self) -> usize {
        self.resources.len()
    }

    pub fn entry(&self, pass: PassId) -> PassEntry {
        self.nodes[pass.index()].entry
    }

    pub fn resource(&self, resource: ResourceId) -> ResourceEntry {
        self.resources[resource.index()]
    }

    pub fn plan(&self, runs: &Bits, outputs: &Bits) -> Result<Plan, Error> {
        let mut needed = outputs.clone();
        for (index, resource) in self.resources.iter().enumerate() {
            if resource.life != Life::Frame {
                needed.insert(index);
            }
        }
        let mut kept: Vec<Option<Planned>> = vec![None; self.order.len()];
        for (at, &index) in self.order.iter().enumerate().rev() {
            if !runs.contains(index) {
                continue;
            }
            let node = &self.nodes[index];
            let produced = node.outputs();
            if produced.intersect(&needed).is_empty() {
                continue;
            }
            let colors = node
                .colors
                .iter()
                .filter_map(|&(resource, access)| {
                    Some((resource, store_ops(access, needed.contains(resource.index()))?))
                })
                .collect();
            let depth = node.depth.map(|(resource, access)| {
                (resource, store_ops(access, needed.contains(resource.index())))
            });
            kept[at] = Some(Planned {
                pass: PassId::new(index),
                batch: self.batches[at],
                colors,
                depth,
            });
            for resource in produced.iter() {
                if self.resources[resource].life == Life::Frame {
                    needed.remove(resource);
                }
            }
            needed = needed.union(&node.inputs());
        }
        let mut written = Bits::of(self.resources.len());
        for (index, resource) in self.resources.iter().enumerate() {
            if matches!(resource.life, Life::Kept | Life::Previous) {
                written.insert(index);
            }
        }
        let mut passes = Vec::new();
        for mut planned in kept.into_iter().flatten() {
            let node = &self.nodes[planned.pass.index()];
            if let Some(resource) = node.required().without(&written).first() {
                return Err(refused(
                    ErrorCode::ReadBeforeWrite,
                    format!("$.passes.{}.reads.{}", node.entry.label, self.resources[resource].label),
                ));
            }
            for (resource, ops) in &mut planned.colors {
                if let Some(&(_, access)) = node.colors.iter().find(|(known, _)| *known == *resource) {
                    ops.load = load_op(access, written.contains(resource.index()));
                }
            }
            if let (Some((resource, Some(ops))), Some((_, access))) = (planned.depth.as_mut(), node.depth) {
                ops.load = load_op(access, written.contains(resource.index()));
            }
            written = written.union(&node.outputs());
            passes.push(planned);
        }
        Ok(Plan { passes, written })
    }
}

pub const PLANS: usize = 16;

#[derive(Debug, Default)]
pub struct PlanCache {
    plans: Vec<(Bits, Bits, Plan)>,
    misses: u64,
}

impl PlanCache {
    pub fn plan(
        &mut self,
        graph: &StaticGraph,
        runs: &Bits,
        outputs: &Bits,
        most: usize,
    ) -> Result<&Plan, Error> {
        let found = self
            .plans
            .iter()
            .position(|(known, read, _)| known == runs && read == outputs);
        let index = match found {
            Some(index) => index,
            None => {
                let plan = graph.plan(runs, outputs)?;
                self.misses += 1;
                if self.plans.len() >= most.max(1) {
                    self.plans.remove(0);
                }
                self.plans.push((runs.clone(), outputs.clone(), plan));
                self.plans.len() - 1
            }
        };
        Ok(&self.plans[index].2)
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn clear(&mut self) {
        self.plans.clear();
    }
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
