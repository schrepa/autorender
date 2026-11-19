use engine_core::{Error, ErrorCode};

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

    pub fn assign(&mut self, other: &Self) {
        self.0.clear();
        self.0.extend_from_slice(&other.0);
    }

    pub fn union_with(&mut self, other: &Self) {
        for (word, more) in self.0.iter_mut().zip(&other.0) {
            *word |= more;
        }
    }

    pub fn remove_all(&mut self, other: &Self) {
        for (word, gone) in self.0.iter_mut().zip(&other.0) {
            *word &= !gone;
        }
    }

    pub fn clear(&mut self) {
        self.0.fill(0);
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
    pub current: Vec<ResourceId>,
}

#[derive(Clone, Debug, Default)]
struct Derived {
    inputs: Bits,
    required: Bits,
    outputs: Bits,
    reads_only: Bits,
}

#[derive(Clone, Debug)]
struct Node {
    entry: PassEntry,
    reads: Bits,
    writes: Bits,
    current: Bits,
    colors: Vec<(ResourceId, Access)>,
    depth: Option<(ResourceId, Access)>,
    derived: Derived,
}

impl Node {
    fn attachments(&self) -> impl Iterator<Item = (ResourceId, Access)> + '_ {
        self.colors.iter().chain(&self.depth).copied()
    }

    fn derive(&mut self) {
        let mut inputs = self.reads.clone();
        let mut outputs = self.writes.clone();
        for (resource, access) in self.attachments() {
            if !matches!(access, Access::Clear(_)) {
                inputs.insert(resource.index());
            }
            if access != Access::ReadOnly {
                outputs.insert(resource.index());
            }
        }
        let mut required = self.reads.without(&outputs);
        for (resource, access) in self.attachments() {
            if matches!(access, Access::Load | Access::ReadOnly) {
                required.insert(resource.index());
            }
        }
        let reads_only = inputs.without(&outputs);
        self.derived = Derived {
            inputs,
            required,
            outputs,
            reads_only,
        };
    }

    fn inputs(&self) -> &Bits {
        &self.derived.inputs
    }

    fn required(&self) -> &Bits {
        &self.derived.required
    }

    fn outputs(&self) -> &Bits {
        &self.derived.outputs
    }

    fn reads_only(&self) -> &Bits {
        &self.derived.reads_only
    }

    fn key(&self, index: usize) -> (u32, usize) {
        (self.entry.registered, index)
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
    pub opaque: Option<(usize, usize)>,
}

#[derive(Debug, Default)]
pub struct StaticGraph {
    resources: Vec<ResourceEntry>,
    nodes: Vec<Node>,
    order: Vec<usize>,
    batches: Vec<u8>,
    starved: Bits,
    empty: Bits,
    #[cfg(any(feature = "dev", feature = "profiling"))]
    after: Vec<Vec<PassId>>,
    #[cfg(any(feature = "dev", feature = "profiling"))]
    unmet: Vec<Bits>,
    #[cfg(any(feature = "dev", feature = "profiling"))]
    unread: Bits,
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

fn nodes_of(passes: Vec<Declared>, width: usize) -> (Vec<Node>, Vec<Vec<PassId>>) {
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
            derived: Derived::default(),
        };
        node.derive();
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
        let made = reader.outputs();
        for (early, writer) in nodes.iter().enumerate() {
            if early == late {
                continue;
            }
            let outputs = writer.outputs();
            for resource in outputs.intersect(reads).iter() {
                if resources[resource].life == Life::Kept && !reader.current.contains(resource) {
                    kept.push((early, late));
                } else {
                    before[late].insert(early);
                }
            }
            if early < late && !made.intersect(outputs).intersect(views).is_empty() {
                ties.push((early, late));
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

fn settle_ties(nodes: &[Node], before: &mut [Bits], ties: &[(usize, usize)]) {
    for &(first, second) in ties {
        if earlier(before, first, second) || earlier(before, second, first) {
            continue;
        }
        let (early, late) = if nodes[first].key(first) < nodes[second].key(second) {
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
        let next = (0..count)
            .filter(|&index| !placed.contains(index) && before[index].without(&placed).is_empty())
            .min_by_key(|&index| nodes[index].key(index));
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
                .intersect(nodes[second].outputs())
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
                if !node.reads_only().intersect(made).is_empty() {
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

fn batches_of(nodes: &[Node], order: &[usize], views: &Bits) -> Vec<u8> {
    let mut batches = Vec::with_capacity(order.len());
    let mut batch = 0;
    for &index in order {
        let node = &nodes[index];
        let draws = !node.outputs().intersect(views).is_empty();
        let looks = !node.reads_only().intersect(views).is_empty();
        batch = match batch {
            0 if draws => 1,
            1 if looks => 2,
            held => held,
        };
        batches.push(batch);
    }
    batches
}

impl StaticGraph {
    pub fn build(resources: Vec<ResourceEntry>, passes: Vec<Declared>) -> Result<Self, Error> {
        let views = view_bits(&resources);
        let (nodes, after) = nodes_of(passes, resources.len());
        let before = passes_before(&resources, &nodes, &after, &views);
        let order = placed_order(&nodes, &before)?;
        let reach = reach_of(&before, &order);
        check_writers_ordered(&resources, &nodes, &reach, &views)?;
        let unmet = unmet_of(&resources, &nodes, &reach);
        let starved = starved_of(&resources, &nodes, &unmet);
        let batches = batches_of(&nodes, &order, &views);
        #[cfg(any(feature = "dev", feature = "profiling"))]
        let unread = facts::unread_of(&resources, &nodes, &order);
        Ok(Self {
            resources,
            nodes,
            order,
            batches,
            starved,
            empty: Bits::default(),
            #[cfg(any(feature = "dev", feature = "profiling"))]
            after,
            #[cfg(any(feature = "dev", feature = "profiling"))]
            unmet,
            #[cfg(any(feature = "dev", feature = "profiling"))]
            unread,
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

    pub fn writes(&self, pass: usize) -> &Bits {
        self.nodes.get(pass).map_or(&self.empty, Node::outputs)
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
                    .union(node.outputs())
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
            if resource.life == Life::Kept {
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
                if self.resources[resource].life != Life::Kept {
                    needed.remove(resource);
                }
            }
            needed.union_with(node.inputs());
        }
        let mut written = ready.clone();
        for (index, resource) in self.resources.iter().enumerate() {
            if resource.life == Life::Kept {
                written.insert(index);
            }
        }
        let planned_writes = kept.iter().flatten().fold(ready.clone(), |all, planned| {
            all.union(self.nodes[planned.pass.index()].outputs())
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
            written.union_with(node.outputs());
            passes.push(planned);
        }
        let opaque = |planned: &Planned| {
            planned.batch == 1 && self.nodes[planned.pass.index()].entry.shared
        };
        let opaque = passes
            .iter()
            .position(opaque)
            .zip(passes.iter().rposition(opaque));
        Ok(Plan { passes, opaque })
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

#[cfg(any(feature = "dev", feature = "profiling"))]
mod facts {
    use serde_json::{Map, Value, json};

    use super::{Access, Bits, Life, Node, ResourceEntry, StaticGraph};

    impl Life {
        pub fn name(self) -> &'static str {
            match self {
                Self::Frame => "frame",
                Self::Kept => "kept",
                Self::Output => "output",
            }
        }
    }

    pub fn keyed(map: &mut Map<String, Value>, name: &str, value: Value) {
        map.insert(free_key(map, name), value);
    }

    fn free_key(map: &Map<String, Value>, name: &str) -> String {
        let mut key = name.to_owned();
        let mut repeat = 1;
        while map.contains_key(&key) {
            repeat += 1;
            key = format!("{name}#{repeat}");
        }
        key
    }

    pub(super) fn unread_of(resources: &[ResourceEntry], nodes: &[Node], order: &[usize]) -> Bits {
        let mut needed = Bits::of(resources.len());
        for (index, resource) in resources.iter().enumerate() {
            if matches!(resource.life, Life::Kept | Life::Output) {
                needed.insert(index);
            }
        }
        let mut unread = Bits::of(nodes.len());
        for &index in order.iter().rev() {
            let node = &nodes[index];
            if node.outputs().intersect(&needed).is_empty() {
                unread.insert(index);
            } else {
                needed = needed.union(node.inputs());
            }
        }
        unread
    }

    impl StaticGraph {
        pub fn resource(&self, index: usize) -> Option<&ResourceEntry> {
            self.resources.get(index)
        }

        fn keys(&self) -> Vec<String> {
            let mut seen = Map::new();
            self.resources
                .iter()
                .map(|resource| {
                    if resource.view {
                        return resource.label.to_owned();
                    }
                    let key = free_key(&seen, resource.label);
                    seen.insert(key.clone(), Value::Null);
                    key
                })
                .collect()
        }

        fn labels<'a>(keys: &'a [String], bits: &Bits) -> Vec<&'a str> {
            bits.iter()
                .filter_map(|index| keys.get(index))
                .map(String::as_str)
                .collect()
        }

        fn dropped(&self, pass: usize, node: &Node, keys: &[String]) -> Value {
            let labels = |bits: &Bits| Self::labels(keys, bits);
            if let Some(unmet) = self.unmet.get(pass).filter(|unmet| !unmet.is_empty()) {
                return json!({"reason": "reads_unwritten", "resources": labels(unmet)});
            }
            if self.starved.contains(pass) {
                let mut fed = Bits::of(self.resources.len());
                for resource in node.required().iter() {
                    let mut writers = self
                        .nodes
                        .iter()
                        .enumerate()
                        .filter(|(early, writer)| {
                            *early != pass && writer.outputs().contains(resource)
                        })
                        .peekable();
                    if writers.peek().is_some()
                        && writers.all(|(early, _)| self.starved.contains(early))
                    {
                        fed.insert(resource);
                    }
                }
                return json!({"reason": "writer_dropped", "resources": labels(&fed)});
            }
            if self.unread.contains(pass) {
                return json!({"reason": "never_read", "resources": labels(node.outputs())});
            }
            Value::Null
        }

        pub fn pass_facts(&self, pass: usize) -> Map<String, Value> {
            let mut facts = Map::new();
            let Some(node) = self.nodes.get(pass) else {
                return facts;
            };
            let keys = self.keys();
            let mut targets = Map::new();
            for (resource, access) in node.attachments() {
                if let Some(key) = keys.get(resource.index()) {
                    let access = match access {
                        Access::Clear(_) => "clear",
                        Access::Draw(_) => "draw",
                        Access::Load => "load",
                        Access::ReadOnly => "read_only",
                    };
                    targets.insert(key.clone(), json!(access));
                }
            }
            let after: Vec<&'static str> = self
                .after
                .get(pass)
                .into_iter()
                .flatten()
                .filter_map(|after| self.nodes.get(after.index()))
                .map(|after| after.entry.label)
                .collect();
            let order = self.order.iter().position(|&placed| placed == pass);
            facts.insert("order".to_owned(), json!(order));
            facts.insert("targets".to_owned(), Value::Object(targets));
            facts.insert("reads".to_owned(), json!(Self::labels(&keys, &node.reads)));
            facts.insert("writes".to_owned(), json!(Self::labels(&keys, &node.writes)));
            facts.insert("after".to_owned(), json!(after));
            facts.insert("dropped".to_owned(), self.dropped(pass, node, &keys));
            facts
        }

        pub fn links(&self) -> Map<String, Value> {
            let mut resources = Map::new();
            let mut targets = Map::new();
            for (index, resource) in self.resources.iter().enumerate() {
                let passes = |touches: &dyn Fn(&Node) -> bool| -> Vec<&'static str> {
                    self.nodes
                        .iter()
                        .filter(|node| touches(node))
                        .map(|node| node.entry.label)
                        .collect()
                };
                let writers = passes(&|node| node.outputs().contains(index));
                let readers = passes(&|node| node.inputs().contains(index));
                keyed(
                    &mut resources,
                    resource.label,
                    json!({"writers": writers, "readers": readers}),
                );
                if resource.view {
                    let attaching = passes(&|node| {
                        node.attachments()
                            .any(|(attached, _)| attached.index() == index)
                    });
                    keyed(&mut targets, resource.label, json!(attaching));
                }
            }
            let mut links = Map::new();
            links.insert("resources".to_owned(), Value::Object(resources));
            links.insert("targets".to_owned(), Value::Object(targets));
            links
        }
    }
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub use facts::keyed;
