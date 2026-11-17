use engine_core::{AttachPoint, Error, ErrorCode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Life {
    Frame,
    Kept,
    Output,
}

impl From<crate::Lifetime> for Life {
    fn from(lifetime: crate::Lifetime) -> Self {
        match lifetime {
            crate::Lifetime::Frame => Self::Frame,
            crate::Lifetime::Kept => Self::Kept,
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

    pub fn remove(&mut self, index: usize) {
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

pub const TRANSPARENT: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
pub const BLACK: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
pub const FAR: f32 = 0.0;

#[derive(Clone, Copy, Debug)]
pub struct ResourceEntry {
    pub label: &'static str,
    pub life: Life,
    pub view: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PassEntry {
    pub label: &'static str,
    pub brick: &'static str,
    pub registered: u32,
    pub layer: Option<AttachPoint>,
    pub shared: bool,
    pub group: Option<&'static str>,
    pub blends: bool,
}

#[derive(Clone, Debug)]
pub struct Declared {
    pub entry: PassEntry,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
    pub colors: Vec<(ResourceId, Access)>,
    pub depth: Option<(ResourceId, Access)>,
    pub after: Vec<PassId>,
    pub current: Vec<ResourceId>,
}

#[derive(Clone, Debug)]
struct Node {
    entry: PassEntry,
    reads: Bits,
    writes: Bits,
    current: Bits,
    colors: Vec<(ResourceId, Access)>,
    depth: Option<(ResourceId, Access)>,
    composes: bool,
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
        let mut required = self.reads.without(&self.outputs());
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

    fn place(&self) -> (u8, u8) {
        (
            self.entry.layer.map_or(0, |layer| layer as u8 + 1),
            u8::from(!self.entry.shared),
        )
    }

    fn key(&self, index: usize) -> (u8, u8, bool, u32, usize) {
        let (layer, member) = self.place();
        (layer, member, self.composes, self.entry.registered, index)
    }

    fn peer(&self, other: &Self) -> bool {
        self.entry.layer.is_some() && !self.entry.shared && self.place() == other.place()
    }

    fn batch(&self, positioned_before: bool) -> u8 {
        match self.entry.layer {
            None if positioned_before => 2,
            None => 0,
            Some(AttachPoint::Inputs | AttachPoint::Opaque) => 1,
            Some(AttachPoint::Transparent | AttachPoint::Overlay) => 2,
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
}

#[derive(Debug, Default)]
pub struct StaticGraph {
    resources: Vec<ResourceEntry>,
    nodes: Vec<Node>,
    order: Vec<usize>,
    batches: Vec<u8>,
    starved: Bits,
}

#[derive(Clone, Copy, Debug)]
enum Tie {
    Read {
        writer: usize,
        reader: usize,
    },
    Write {
        first: usize,
        second: usize,
    },
}

fn refused(code: ErrorCode, path: String) -> Error {
    Error::new(code, path)
}

fn follows_of(before: &[Bits]) -> Vec<Bits> {
    let count = before.len();
    (0..count)
        .map(|start| {
            let mut follows = Bits::of(count);
            let mut open: Vec<usize> = before[start].iter().collect();
            while let Some(pass) = open.pop() {
                if follows.contains(pass) {
                    continue;
                }
                follows.insert(pass);
                open.extend(before[pass].iter());
            }
            follows
        })
        .collect()
}

fn nodes_of(passes: Vec<Declared>, width: usize, views: &Bits) -> (Vec<Node>, Vec<Vec<PassId>>) {
    let count = passes.len();
    let mut nodes = Vec::with_capacity(count);
    let mut after = Vec::with_capacity(count);
    for declared in passes {
        let mut reads = Bits::of(width);
        let mut writes = Bits::of(width);
        let mut current = Bits::of(width);
        for resource in &declared.reads {
            reads.insert(resource.index());
        }
        for resource in &declared.writes {
            writes.insert(resource.index());
        }
        for resource in &declared.current {
            current.insert(resource.index());
        }
        after.push(declared.after);
        let mut node = Node {
            entry: declared.entry,
            reads,
            writes,
            current,
            colors: declared.colors,
            depth: declared.depth,
            composes: false,
        };
        node.composes =
            !node.entry.shared && node.entry.blends && !node.outputs().intersect(views).is_empty();
        nodes.push(node);
    }
    (nodes, after)
}

fn view_bits(resources: &[ResourceEntry]) -> Bits {
    let mut views = Bits::of(resources.len());
    for (index, resource) in resources.iter().enumerate() {
        if resource.view {
            views.insert(index);
        }
    }
    views
}

fn passes_before(
    resources: &[ResourceEntry],
    nodes: &[Node],
    after: &[Vec<PassId>],
    views: &Bits,
) -> Vec<Bits> {
    let count = nodes.len();
    let mut before = vec![Bits::of(count); count];
    let mut kept = Vec::new();
    let mut ties = Vec::new();
    for (late, reader) in nodes.iter().enumerate() {
        let reads = reader.reads_only();
        let inputs = reader.inputs();
        let made = reader.outputs();
        for (early, writer) in nodes.iter().enumerate() {
            if early == late {
                continue;
            }
            let outputs = writer.outputs();
            for resource in outputs.intersect(&inputs).iter() {
                let entry = &resources[resource];
                if entry.view {
                    if reader.peer(writer) {
                        if reads.contains(resource) {
                            ties.push(Tie::Read {
                                writer: early,
                                reader: late,
                            });
                        }
                    } else if reader.entry.layer.is_none() || writer.key(early) < reader.key(late) {
                        before[late].insert(early);
                    } else if reads.contains(resource) {
                        before[early].insert(late);
                    }
                } else if reads.contains(resource) {
                    if entry.life == Life::Kept && !reader.current.contains(resource) {
                        kept.push((early, late));
                    } else {
                        before[late].insert(early);
                    }
                }
            }
            if reader.entry.layer.is_some()
                && writer.entry.layer.is_some()
                && made.intersect(&outputs).intersect(views).first().is_some()
            {
                if reader.peer(writer) {
                    if early < late {
                        ties.push(Tie::Write {
                            first: early,
                            second: late,
                        });
                    }
                } else if writer.key(early) < reader.key(late) {
                    before[late].insert(early);
                }
            }
            if writer.entry.shared
                && !reader.entry.shared
                && reader.entry.layer.is_some()
                && reader.entry.layer == writer.entry.layer
                && writer.attachments().any(|(resource, _)| {
                    views.contains(resource.index()) && inputs.contains(resource.index())
                })
            {
                before[late].insert(early);
            }
        }
        for pass in &after[late] {
            before[late].insert(pass.index());
        }
    }
    settle_ties(nodes, &mut before, &ties);
    let follows = follows_of(&before);
    for (early, late) in kept {
        if !follows[early].contains(late) {
            before[late].insert(early);
        }
    }
    before
}

fn settle_ties(nodes: &[Node], before: &mut [Bits], ties: &[Tie]) {
    for &tie in ties {
        if let Tie::Read { writer, reader } = tie
            && !earlier(before, reader, writer)
        {
            before[reader].insert(writer);
        }
    }
    for &tie in ties {
        let Tie::Write { first, second } = tie else {
            continue;
        };
        if earlier(before, first, second) || earlier(before, second, first) {
            continue;
        }
        let (one, other) = (&nodes[first], &nodes[second]);
        let (early, late) = if one.key(first) < other.key(second) {
            (first, second)
        } else {
            (second, first)
        };
        before[late].insert(early);
    }
}

fn earlier(before: &[Bits], first: usize, then: usize) -> bool {
    let mut seen = Bits::of(before.len());
    let mut open: Vec<usize> = before[then].iter().collect();
    while let Some(pass) = open.pop() {
        if pass == first {
            return true;
        }
        if seen.contains(pass) {
            continue;
        }
        seen.insert(pass);
        open.extend(before[pass].iter());
    }
    false
}

fn placed_order(nodes: &[Node], before: &[Bits]) -> Result<Vec<usize>, Error> {
    let count = nodes.len();
    let mut placed = Bits::of(count);
    let mut order: Vec<usize> = Vec::with_capacity(count);
    while order.len() < count {
        let last = order.last().map(|&last| &nodes[last].entry);
        let continues = |index: usize| {
            let entry = &nodes[index].entry;
            last.is_some_and(|last| {
                entry.group.is_some()
                    && entry.group == last.group
                    && entry.brick == last.brick
                    && entry.layer == last.layer
            })
        };
        let next = (0..count)
            .filter(|&index| !placed.contains(index) && before[index].without(&placed).is_empty())
            .min_by_key(|&index| (!continues(index), nodes[index].key(index)));
        let Some(next) = next else {
            let stuck = (0..count)
                .filter(|&index| !placed.contains(index))
                .min_by_key(|&index| nodes[index].key(index))
                .unwrap_or(0);
            let waiting: Vec<String> = (0..count)
                .filter(|&index| !placed.contains(index))
                .map(|index| {
                    let on: Vec<&str> = before[index]
                        .without(&placed)
                        .iter()
                        .map(|earlier| nodes[earlier].entry.label)
                        .collect();
                    format!("{} after {}", nodes[index].entry.label, on.join("+"))
                })
                .collect();
            return Err(refused(
                ErrorCode::AmbiguousPasses,
                format!("$.passes.{}", nodes[stuck].entry.label),
            )
            .got(waiting));
        };
        placed.insert(next);
        order.push(next);
    }
    Ok(order)
}

fn reach_of(before: &[Bits], order: &[usize]) -> Vec<Bits> {
    let mut reach = before.to_vec();
    for &index in order {
        let mut through = reach[index].clone();
        for earlier in reach[index].iter().collect::<Vec<_>>() {
            through = through.union(&reach[earlier]);
        }
        reach[index] = through;
    }
    reach
}

fn check_writers_ordered(
    resources: &[ResourceEntry],
    nodes: &[Node],
    reach: &[Bits],
    views: &Bits,
) -> Result<(), Error> {
    let count = nodes.len();
    for first in 0..count {
        for second in first + 1..count {
            let shared = nodes[first]
                .outputs()
                .intersect(&nodes[second].outputs())
                .without(views);
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
    Ok(())
}

fn unmet_of(resources: &[ResourceEntry], nodes: &[Node], reach: &[Bits]) -> Vec<Bits> {
    let mut unmet = vec![Bits::of(resources.len()); nodes.len()];
    for (late, reader) in nodes.iter().enumerate() {
        for resource in reader.required().iter() {
            let entry = &resources[resource];
            if !matches!(entry.life, Life::Frame | Life::Output) {
                continue;
            }
            let written = (0..nodes.len()).any(|early| {
                reach[late].contains(early) && nodes[early].outputs().contains(resource)
            });
            if written {
                continue;
            }
            unmet[late].insert(resource);
        }
    }
    unmet
}

fn starved_of(resources: &[ResourceEntry], nodes: &[Node], unmet: &[Bits]) -> Bits {
    let mut starved = Bits::of(nodes.len());
    for (pass, unmet) in unmet.iter().enumerate() {
        if unmet.is_empty() {
            continue;
        }
        starved.insert(pass);
        if nodes[pass].entry.brick.is_empty() {
            let made = nodes[pass].outputs();
            for (reader, node) in nodes.iter().enumerate() {
                if !node.reads_only().intersect(&made).is_empty() {
                    starved.insert(reader);
                }
            }
        }
    }
    loop {
        let mut grew = false;
        for (late, reader) in nodes.iter().enumerate() {
            if starved.contains(late) {
                continue;
            }
            let unfed = reader.required().iter().any(|resource| {
                let mut writers = nodes.iter().enumerate().filter(|(early, writer)| {
                    *early != late && writer.outputs().contains(resource)
                });
                matches!(
                    resources[resource].life,
                    Life::Frame | Life::Kept | Life::Output
                ) && writers
                    .clone()
                    .any(|(_, writer)| writer.entry.brick == reader.entry.brick)
                    && writers.all(|(early, _)| starved.contains(early))
            });
            if unfed {
                starved.insert(late);
                grew = true;
            }
        }
        if !grew {
            return starved;
        }
    }
}

fn batches_of(nodes: &[Node], order: &[usize]) -> Vec<u8> {
    let mut batches = Vec::with_capacity(order.len());
    let mut positioned = false;
    let mut last = 0;
    for &index in order {
        let node = &nodes[index];
        let batch = node.batch(positioned).max(last);
        positioned |= node.entry.layer.is_some();
        last = batch;
        batches.push(batch);
    }
    batches
}

impl StaticGraph {
    pub fn build(resources: Vec<ResourceEntry>, passes: Vec<Declared>) -> Result<Self, Error> {
        let views = view_bits(&resources);
        let (nodes, after) = nodes_of(passes, resources.len(), &views);
        for node in &nodes {
            if node.entry.layer.is_none() && !node.outputs().intersect(&views).is_empty() {
                return Err(refused(
                    ErrorCode::ViewWriterUnplaced,
                    format!("$.passes.{}", node.entry.label),
                ));
            }
        }
        let before = passes_before(&resources, &nodes, &after, &views);
        let order = placed_order(&nodes, &before)?;
        let reach = reach_of(&before, &order);
        check_writers_ordered(&resources, &nodes, &reach, &views)?;
        let unmet = unmet_of(&resources, &nodes, &reach);
        let starved = starved_of(&resources, &nodes, &unmet);
        let batches = batches_of(&nodes, &order);
        Ok(Self {
            resources,
            nodes,
            order,
            batches,
            starved,
        })
    }

    pub fn starved(&self, pass: usize) -> bool {
        self.starved.contains(pass)
    }

    pub fn passes(&self) -> usize {
        self.nodes.len()
    }

    pub fn resources(&self) -> usize {
        self.resources.len()
    }

    pub fn view_resources(&self) -> Bits {
        view_bits(&self.resources)
    }

    pub fn writes(&self, pass: usize) -> Bits {
        self.nodes
            .get(pass)
            .map_or_else(|| Bits::of(self.resources.len()), Node::outputs)
    }

    pub fn spread(&self, mut passes: Bits, mut varying: Bits) -> Bits {
        let mut frame = Bits::of(self.resources.len());
        for (index, resource) in self.resources.iter().enumerate() {
            if resource.life == Life::Frame {
                frame.insert(index);
            }
        }
        loop {
            let mut grew = false;
            for (index, node) in self.nodes.iter().enumerate() {
                let touches = !node
                    .inputs()
                    .union(&node.outputs())
                    .intersect(&varying)
                    .is_empty();
                if !passes.contains(index) && !touches {
                    continue;
                }
                let widened = varying.union(&node.outputs().intersect(&frame));
                grew |= !passes.contains(index) || widened != varying;
                passes.insert(index);
                varying = widened;
            }
            if !grew {
                return passes;
            }
        }
    }

    pub fn plan(&self, runs: &Bits, outputs: &Bits, ready: &Bits) -> Result<Plan, Error> {
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
                    Some((
                        resource,
                        store_ops(access, needed.contains(resource.index()))?,
                    ))
                })
                .collect();
            let depth = node.depth.map(|(resource, access)| {
                (
                    resource,
                    store_ops(access, needed.contains(resource.index())),
                )
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
        let mut written = ready.clone();
        for (index, resource) in self.resources.iter().enumerate() {
            if resource.life == Life::Kept {
                written.insert(index);
            }
        }
        let planned_writes = kept.iter().flatten().fold(ready.clone(), |all, planned| {
            all.union(&self.nodes[planned.pass.index()].outputs())
        });
        let mut passes = Vec::new();
        for mut planned in kept.into_iter().flatten() {
            let node = &self.nodes[planned.pass.index()];
            let refused_read = node.required().without(&written).iter().find(|&resource| {
                let unwritten = matches!(self.resources[resource].life, Life::Frame | Life::Output)
                    && !planned_writes.contains(resource);
                let elsewhere = self
                    .nodes
                    .iter()
                    .filter(|other| other.outputs().contains(resource))
                    .all(|other| other.entry.brick != node.entry.brick);
                !(unwritten && elsewhere)
            });

            if let Some(resource) = refused_read {
                return Err(refused(
                    ErrorCode::ReadBeforeWrite,
                    format!(
                        "$.passes.{}.reads.{}",
                        node.entry.label, self.resources[resource].label
                    ),
                ));
            }
            for (resource, ops) in &mut planned.colors {
                if let Some(&(_, access)) =
                    node.colors.iter().find(|(known, _)| *known == *resource)
                {
                    ops.load = load_op(access, written.contains(resource.index()));
                }
            }
            if let (Some((resource, Some(ops))), Some((_, access))) =
                (planned.depth.as_mut(), node.depth)
            {
                ops.load = load_op(access, written.contains(resource.index()));
            }
            written = written.union(&node.outputs());
            passes.push(planned);
        }
        Ok(Plan { passes })
    }
}

pub const PLANS: usize = 16;

#[derive(Debug, Default)]
pub struct PlanCache {
    plans: Vec<(Bits, Bits, Bits, std::sync::Arc<Plan>)>,
}

impl PlanCache {
    pub fn plan(
        &mut self,
        graph: &StaticGraph,
        (runs, outputs, ready): (&Bits, &Bits, &Bits),
        most: usize,
    ) -> Result<std::sync::Arc<Plan>, Error> {
        let found = self
            .plans
            .iter()
            .position(|(known, read, held, _)| known == runs && read == outputs && held == ready);
        let index = match found {
            Some(index) => index,
            None => {
                let plan = graph.plan(runs, outputs, ready)?;
                if self.plans.len() >= most.max(1) {
                    self.plans.remove(0);
                }
                self.plans.push((
                    runs.clone(),
                    outputs.clone(),
                    ready.clone(),
                    std::sync::Arc::new(plan),
                ));
                self.plans.len() - 1
            }
        };
        Ok(std::sync::Arc::clone(&self.plans[index].3))
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
