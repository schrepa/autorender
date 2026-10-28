#![expect(
    clippy::expect_used,
    reason = "a node holds at most four color attachments"
)]

pub(crate) use crate::render_features::FeaturePass;
pub(crate) use crate::screen_effects::Effects;

macro_rules! resources {
    ($($(#[$doc:meta])* $resource:ident: $lifetime:ident,)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub(crate) enum Resource {
            $($(#[$doc])* $resource,)*
        }

        const RESOURCES: &[Resource] = &[$(Resource::$resource,)*];

        impl Resource {
            pub(crate) fn lifetime(self) -> Lifetime {
                match self {
                    $(Self::$resource => Lifetime::$lifetime,)*
                }
            }
        }
    };
}

resources! {
    Canvas: Output,
    Depth: Frame,
    Scene: Frame,
    Glow: Frame,
    Surface: Frame,
    Motion: Frame,
    History: Kept,
    Accumulated: Frame,
    Revealed: Frame,
    Poses: Kept,
    Visible: Kept,
    Placed: Frame,
    Occupancy: Kept,
    Casters: Frame,
    ShadowMaps: Kept,
    ShadowMarks: Kept,
    Lights: Frame,
    Pyramid: Kept,
    Effects: Frame,
    Haze: Kept,
    Exposure: Kept,
    Probes: Kept,
    CloudView: Frame,
    CloudDome: Kept,
    Tracks: Kept,
    Bounce: Kept,
    Leaves: Kept,
    Cover: Kept,
    TerrainIds: Frame,
    DebugIds: Frame,
    DebugCount: Frame,
    DebugTerrain: Frame,
}

const _: () = assert!(RESOURCES.len() <= 64, "a resource set is a u64");

impl Resource {
    fn bit(self) -> u64 {
        1 << self as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lifetime {
    Frame,
    Kept,
    Output,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Set(u64);

impl Set {
    fn contains(self, resource: Resource) -> bool {
        self.0 & resource.bit() != 0
    }

    fn insert(&mut self, resource: Resource) {
        self.0 |= resource.bit();
    }

    fn remove(&mut self, resource: Resource) {
        self.0 &= !resource.bit();
    }

    fn iter(self) -> impl Iterator<Item = Resource> {
        RESOURCES
            .iter()
            .copied()
            .filter(move |resource| self.contains(*resource))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Clear {
    Color([f64; 4]),
    Depth(f32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Access {
    Clear(Clear),
    Draw(Clear),
    Load,
    ReadOnly,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Load {
    Clear(Clear),
    Load,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Ops {
    pub(crate) load: Load,
    pub(crate) store: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    One,
    Two,
}

impl Phase {
    fn first(count: u8) -> impl Iterator<Item = Self> {
        [Self::One, Self::Two].into_iter().take(usize::from(count))
    }

    pub(crate) fn index(self) -> usize {
        match self {
            Self::One => 0,
            Self::Two => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pass {
    Presentation,
    Subdivision,
    ShadowSelect,
    ShadowMaps,
    Deformation,
    Bounce,
    Opaque { phase: Phase },
    TerrainIds,
    TerrainResolve,
    OpaqueLate,
    OcclusionTest,
    ShadowMarks,
    Feature(FeaturePass),
    ProbeCaptures,
}

macro_rules! passes {
    ($($name:ident: $variant:ident $({ phase: $phase:path })? $(($feature:path))?,)*) => {
        #[expect(non_camel_case_types, reason = "rows go by their passes' report names")]
        #[derive(Clone, Copy)]
        enum Row {
            $($name,)*
        }

        const PASS_NAMES: &[&str] = &[$(stringify!($name),)*];

        const PASS_STAGES: &[&str] = &[$(concat!("stage_", stringify!($name)),)*];

        impl Pass {
            pub(crate) const fn index(self) -> usize {
                match self {
                    $(Pass::$variant $({ phase: $phase })? $(($feature))? => Row::$name as usize,)*
                }
            }
        }
    };
}

passes! {
    presentation: Presentation,
    subdivision: Subdivision,
    scatter: Feature(FeaturePass::Scatter),
    occupancy: Feature(FeaturePass::Occupancy),
    shadow_select: ShadowSelect,
    shadow_maps: ShadowMaps,
    lights: Feature(FeaturePass::Lights),
    cloud_dome: Feature(FeaturePass::CloudDome),
    deformation: Deformation,
    bounce: Bounce,
    ground_cover: Feature(FeaturePass::GroundCover),
    opaque: Opaque { phase: Phase::One },
    terrain_ids: TerrainIds,
    terrain_resolve: TerrainResolve,
    opaque_late: OpaqueLate,
    occlusion_test: OcclusionTest,
    opaque_phase_two: Opaque { phase: Phase::Two },
    shadow_marks: ShadowMarks,
    decals: Feature(FeaturePass::Decals),
    water: Feature(FeaturePass::Water),
    cloud_march: Feature(FeaturePass::CloudMarch),
    cloud_composite: Feature(FeaturePass::CloudComposite),
    transparent: Feature(FeaturePass::Transparent),
    transparent_composite: Feature(FeaturePass::TransparentComposite),
    exposure: Feature(FeaturePass::Exposure),
    effects: Feature(FeaturePass::Effects),
    composite: Feature(FeaturePass::Composite),
    temporal_resolve: Feature(FeaturePass::TemporalResolve),
    debug_view: Feature(FeaturePass::DebugView),
    probe_captures: ProbeCaptures,
}

const _: () = assert!(PASS_NAMES.len() <= 64, "a pass mask is a u64");

impl Pass {
    pub(crate) fn name(self) -> &'static str {
        PASS_NAMES[self.index()]
    }

    pub(crate) fn stage(self) -> &'static str {
        PASS_STAGES[self.index()]
    }

    fn bit(self) -> u64 {
        1 << self.index()
    }
}

pub(crate) fn render_passes(mask: u64) -> engine_core::Names {
    engine_core::Names {
        bits: mask,
        names: PASS_NAMES,
    }
}

const MAX_COLORS: usize = 4;

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pass: Pass,
    reads: Set,
    writes: Set,
    colors: [Option<(Resource, Access)>; MAX_COLORS],
    depth: Option<(Resource, Access)>,
    kept: bool,
    color_ops: [Option<Ops>; MAX_COLORS],
    depth_ops: Option<Ops>,
}

impl Node {
    pub(crate) fn pass(&self) -> Pass {
        self.pass
    }

    pub(crate) fn colors(&self) -> impl Iterator<Item = (Resource, Ops)> + '_ {
        self.colors
            .iter()
            .zip(&self.color_ops)
            .filter_map(|(color, ops)| Some((color.as_ref()?.0, (*ops)?)))
    }

    pub(crate) fn depth(&self) -> Option<(Resource, Option<Ops>)> {
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

    fn attachments(&self) -> impl Iterator<Item = (Resource, Access)> + '_ {
        self.colors.iter().flatten().chain(&self.depth).copied()
    }
}

pub(crate) struct NodeBuilder<'g> {
    node: &'g mut Node,
}

impl NodeBuilder<'_> {
    pub(crate) fn read(self, resource: Resource) -> Self {
        self.node.reads.insert(resource);
        self
    }

    pub(crate) fn read_if(self, condition: bool, resource: Resource) -> Self {
        if condition { self.read(resource) } else { self }
    }

    pub(crate) fn write(self, resource: Resource) -> Self {
        self.node.writes.insert(resource);
        self
    }

    pub(crate) fn write_if(self, condition: bool, resource: Resource) -> Self {
        if condition {
            self.write(resource)
        } else {
            self
        }
    }

    pub(crate) fn color(self, resource: Resource, access: Access) -> Self {
        let slot = self
            .node
            .colors
            .iter_mut()
            .find(|slot| slot.is_none())
            .expect("at most four color attachments");
        *slot = Some((resource, access));
        self
    }

    pub(crate) fn depth(self, resource: Resource, access: Access) -> Self {
        self.node.depth = Some((resource, access));
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Invalid {
    Twice(Pass),
    Unwritten { pass: Pass, resource: Resource },
}

impl std::fmt::Display for Invalid {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Twice(pass) => write!(
                formatter,
                "The render graph's {} pass is declared twice.",
                pass.name()
            ),
            Self::Unwritten { pass, resource } => write!(
                formatter,
                "The render graph's {} pass reads {resource:?} before any pass writes it.",
                pass.name()
            ),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct RenderGraph {
    nodes: Vec<Node>,
}

impl RenderGraph {
    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
    }

    pub(crate) fn add(&mut self, pass: Pass) -> NodeBuilder<'_> {
        self.nodes.push(Node {
            pass,
            reads: Set::default(),
            writes: Set::default(),
            colors: [None; MAX_COLORS],
            depth: None,
            kept: false,
            color_ops: [None; MAX_COLORS],
            depth_ops: None,
        });
        NodeBuilder {
            node: self.nodes.last_mut().expect("pushed above"),
        }
    }

    pub(crate) fn compile(&mut self) -> Result<(), Invalid> {
        self.nodes.sort_unstable_by_key(|node| node.pass.index());
        if let Some(twice) = self
            .nodes
            .windows(2)
            .find(|pair| pair[0].pass == pair[1].pass)
        {
            return Err(Invalid::Twice(twice[0].pass));
        }
        let mut needed = Set::default();
        for &resource in RESOURCES {
            if resource.lifetime() != Lifetime::Frame {
                needed.insert(resource);
            }
        }
        for node in self.nodes.iter_mut().rev() {
            let outputs = node.outputs();
            node.kept = RESOURCES
                .iter()
                .copied()
                .any(|resource| outputs.contains(resource) && needed.contains(resource));
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
                if resource.lifetime() == Lifetime::Frame {
                    needed.remove(resource);
                }
            }
            needed.0 |= node.inputs().0;
        }
        let mut written = Set::default();
        for &resource in RESOURCES {
            if resource.lifetime() == Lifetime::Kept {
                written.insert(resource);
            }
        }
        for node in self.nodes.iter_mut().filter(|node| node.kept) {
            if let Some(resource) = node
                .required()
                .iter()
                .find(|resource| !written.contains(*resource))
            {
                return Err(Invalid::Unwritten {
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
            written.0 |= node.outputs().0;
        }
        Ok(())
    }

    pub(crate) fn kept_passes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(|node| node.kept)
    }

    pub(crate) fn kept(&self) -> u64 {
        self.kept_passes()
            .fold(0, |mask, node| mask | node.pass.bit())
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Targets {
    Canvas,
    Post,
    Temporal,
    Debug,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FrameShape {
    pub(crate) gpu: bool,
    pub(crate) targets: Targets,
    pub(crate) canvas_clear: [f64; 4],
    pub(crate) phases: u8,
    pub(crate) shadows: bool,
    pub(crate) deformation: bool,
    pub(crate) bounce: bool,
    pub(crate) subdivision: bool,
    pub(crate) deferred: bool,
    pub(crate) scatter: bool,
    pub(crate) lights: bool,
    pub(crate) refraction: bool,
    pub(crate) effects: Effects,
    pub(crate) probes: bool,
}

impl FrameShape {
    pub(crate) fn scene(&self) -> Resource {
        match self.targets {
            Targets::Canvas => Resource::Canvas,
            Targets::Post | Targets::Temporal => Resource::Scene,
            Targets::Debug => Resource::DebugIds,
        }
    }

    pub(crate) fn main_targets(&self) -> &'static [Resource] {
        match self.targets {
            Targets::Canvas => &[Resource::Canvas],
            Targets::Post => &[Resource::Scene, Resource::Glow, Resource::Surface],
            Targets::Temporal => &[
                Resource::Scene,
                Resource::Motion,
                Resource::Glow,
                Resource::Surface,
            ],
            Targets::Debug => &[
                Resource::DebugIds,
                Resource::DebugCount,
                Resource::DebugTerrain,
            ],
        }
    }
}

pub(crate) const SCENE_CLEAR: [f64; 4] = [0.009, 0.015, 0.025, 1.0];
pub(crate) const TRANSPARENT: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
pub(crate) const WHITE: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
pub(crate) const BLACK: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
const FAR: f32 = 0.0;

fn plan_before_opaque(graph: &mut RenderGraph, shape: &FrameShape) {
    use Resource::*;
    let gpu = shape.gpu;
    if gpu {
        graph.add(Pass::Presentation).write(Poses).write(Visible);
    }
    if shape.subdivision {
        graph.add(Pass::Subdivision).write(Leaves);
    }
    if shape.shadows {
        graph
            .add(Pass::ShadowSelect)
            .read(Poses)
            .read_if(gpu, Visible)
            .write(Casters);
    }
    if shape.shadows {
        graph
            .add(Pass::ShadowMaps)
            .read(Poses)
            .read(Casters)
            .read_if(shape.scatter, Placed)
            .read_if(shape.subdivision, Leaves)
            .write(ShadowMaps);
    }
    if shape.deformation {
        graph.add(Pass::Deformation).write(Tracks);
    }
    if shape.bounce {
        graph.add(Pass::Bounce).read(CloudDome).write(Bounce);
    }
}

pub(crate) fn plan(graph: &mut RenderGraph, shape: &FrameShape) {
    use Resource::*;
    graph.clear();
    plan_before_opaque(graph, shape);
    let gpu = shape.gpu;
    let targets: &[(Resource, Clear)] = match shape.targets {
        Targets::Canvas => &[(Canvas, Clear::Color(shape.canvas_clear))],
        Targets::Post => &[
            (Scene, Clear::Color(SCENE_CLEAR)),
            (Glow, Clear::Color(TRANSPARENT)),
            (Surface, Clear::Color(TRANSPARENT)),
        ],
        Targets::Temporal => &[
            (Scene, Clear::Color(SCENE_CLEAR)),
            (Motion, Clear::Color(TRANSPARENT)),
            (Glow, Clear::Color(TRANSPARENT)),
            (Surface, Clear::Color(TRANSPARENT)),
        ],
        Targets::Debug => &[
            (DebugIds, Clear::Color(TRANSPARENT)),
            (DebugCount, Clear::Color(TRANSPARENT)),
            (DebugTerrain, Clear::Color(TRANSPARENT)),
        ],
    };
    for phase in Phase::first(shape.phases) {
        if phase == Phase::Two {
            graph
                .add(Pass::OcclusionTest)
                .read(Depth)
                .read(Poses)
                .write(Pyramid)
                .write(Visible);
        }
        let first = phase == Phase::One;
        let forward = first && !shape.deferred;
        let mut opaque = graph
            .add(Pass::Opaque { phase })
            .read(Poses)
            .read_if(gpu, Visible)
            .read_if(shape.scatter && first, Placed)
            .read(Occupancy)
            .read_if(shape.shadows, ShadowMaps)
            .read_if(shape.lights, Lights)
            .read(Probes)
            .read(Bounce)
            .read(CloudDome)
            .read_if(forward, Tracks)
            .read_if(forward, Cover)
            .read_if(forward && shape.subdivision, Leaves);
        for &(target, clear) in targets {
            opaque = opaque.color(target, Access::Draw(clear));
        }
        opaque.depth(Depth, Access::Draw(Clear::Depth(FAR)));
        if first && shape.deferred {
            deferred_terrain(graph, shape, targets);
        }
    }
    if shape.shadows {
        graph.add(Pass::ShadowMarks).read(Depth).write(ShadowMarks);
    }
    if shape.probes {
        graph
            .add(Pass::ProbeCaptures)
            .read(Poses)
            .read(Visible)
            .read_if(shape.subdivision, Leaves)
            .write(Probes);
    }
}

fn deferred_terrain(graph: &mut RenderGraph, shape: &FrameShape, targets: &[(Resource, Clear)]) {
    use Resource::*;
    graph
        .add(Pass::TerrainIds)
        .read(Tracks)
        .read(Leaves)
        .color(TerrainIds, Access::Clear(Clear::Color(TRANSPARENT)))
        .depth(Depth, Access::Load);
    let mut resolve = graph
        .add(Pass::TerrainResolve)
        .read(TerrainIds)
        .read(Occupancy)
        .read_if(shape.shadows, ShadowMaps)
        .read_if(shape.lights, Lights)
        .read(Probes)
        .read(Bounce)
        .read(CloudDome)
        .read(Tracks);
    for &(target, _) in targets {
        resolve = resolve.color(target, Access::Load);
    }
    resolve.depth(Depth, Access::Load);
    let mut late = graph
        .add(Pass::OpaqueLate)
        .read_if(shape.scatter, Placed)
        .read(Occupancy)
        .read_if(shape.shadows, ShadowMaps)
        .read_if(shape.lights, Lights)
        .read(Probes)
        .read(Bounce)
        .read(CloudDome)
        .read(Tracks)
        .read(Cover);
    for &(target, _) in targets {
        late = late.color(target, Access::Load);
    }
    late.depth(Depth, Access::Load);
}
