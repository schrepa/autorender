//! The frame as a render graph: the passes a frame runs, in
//! order, each declaring the resources it reads and writes and how it draws
//! into its attachments. Compiling the graph culls passes whose results no
//! later pass uses, derives every attachment's load and store operations,
//! and checks that each pass reads only what an earlier pass wrote or the
//! frame started with.
//!
//! WebGPU orders and synchronizes passes itself and has no memory aliasing,
//! so the graph holds no GPU objects: compiling puts the declared passes in
//! the frame order of the passes' table, whatever order they were declared
//! in, and the renderer encodes each kept pass by its [`Pass`] with the
//! derived operations. [`plan`] declares the renderer's frame from its
//! [`FrameShape`].

pub(crate) use crate::screen_effects::Effects;

/// Builds [`Resource`], the list of every resource and each one's
/// [`Lifetime`] from one table: `Name: Lifetime`.
macro_rules! resources {
    ($($(#[$doc:meta])* $resource:ident: $lifetime:ident,)*) => {
        /// A resource the frame's passes share.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub(crate) enum Resource {
            $($(#[$doc])* $resource,)*
        }

        /// Every resource, by its bit in a [`Set`].
        const RESOURCES: &[Resource] = &[$(Resource::$resource,)*];

        impl Resource {
            /// How long its contents live.
            pub(crate) fn lifetime(self) -> Lifetime {
                match self {
                    $(Self::$resource => Lifetime::$lifetime,)*
                }
            }
        }
    };
}

resources! {
    /// The canvas texture the frame presents.
    Canvas: Output,
    /// The depth buffer (reversed).
    Depth: Frame,
    /// The screen effects' scene color, glow and surface targets.
    Scene: Frame,
    Glow: Frame,
    Surface: Frame,
    /// Temporal antialiasing's motion target and history.
    Motion: Frame,
    History: Kept,
    /// The visibility buffer's triangle IDs and per-bin tile lists.
    Ids: Frame,
    Tiles: Frame,
    /// Transparent surfaces' accumulated color and revealage.
    Accumulated: Frame,
    Revealed: Frame,
    /// Retained slots' poses and colors.
    Poses: Kept,
    /// GPU presentation's visible IDs and indirect draws, which the next
    /// frame's first phase starts from.
    Visible: Kept,
    /// Placed objects' lists and draws for the camera and the shadow maps.
    Placed: Frame,
    /// What placed objects cover around the camera, kept between its draws.
    Occupancy: Kept,
    /// Shadow casters each map draws, and the maps (the kept map is kept
    /// between frames).
    Casters: Frame,
    ShadowMaps: Kept,
    /// Virtual geometry's selected clusters and the pages they request.
    Clusters: Kept,
    /// Local lights binned into view clusters.
    Lights: Frame,
    /// The depth pyramid occlusion culling tests against, which the next
    /// frame's cluster selection retests with.
    Pyramid: Kept,
    /// Ambient occlusion, bloom, reflections and sun shafts.
    Effects: Frame,
    /// Environment maps and reflection probes' cubes.
    Probes: Kept,
    /// The view's marched clouds.
    CloudView: Frame,
    /// Terrain's horizon maps, which its long sun shadows read.
    Horizons: Kept,
    /// The clouds' dome, which the sky and its environment map show, and
    /// the grid of their shadows.
    CloudDome: Kept,
    /// Runtime deformation's map of tracks.
    Tracks: Kept,
    /// The runtime virtual texture's baked pages.
    Pages: Kept,
    /// Ground cover's blades and their counts, read back later.
    Cover: Kept,
    /// Terrain's visibility target: each pixel's terrain and depth.
    TerrainIds: Frame,
    /// The pages terrain's resolves wanted of the virtual texture, read back
    /// later.
    Feedback: Kept,
}

// A set holds every resource as a bit of a u64.
const _: () = assert!(RESOURCES.len() <= 64, "a resource set is a u64");

impl Resource {
    fn bit(self) -> u64 {
        1 << self as u32
    }
}

/// How long a resource's contents live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lifetime {
    /// Within a frame: undefined until a pass writes it, and not stored
    /// after its last reader.
    Frame,
    /// Between frames (histories, caches, state the next frame starts
    /// from): defined when the frame starts, and every write is stored.
    Kept,
    /// The frame's output: undefined when the frame starts, and every write
    /// is stored.
    Output,
}

/// A set of resources.
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

/// A clear value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Clear {
    Color([f64; 4]),
    Depth(f32),
}

/// How a pass draws into an attachment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Access {
    /// Overwrites it: clears it, whatever earlier passes drew.
    Clear(Clear),
    /// Draws over what earlier passes drew, or over the clear value when
    /// no earlier pass drew it.
    Draw(Clear),
    /// Draws over what earlier passes drew, which must exist.
    Load,
    /// Tests depth against it without writing.
    ReadOnly,
}

/// How a pass begins with an attachment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Load {
    Clear(Clear),
    Load,
}

/// An attachment's derived operations: how the pass begins with it and
/// whether it stores what it drew. Read-only attachments have none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Ops {
    pub(crate) load: Load,
    pub(crate) store: bool,
}

/// An opaque phase: frames with occlusion culling draw a second, of what
/// the test after the first finds visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    One,
    Two,
}

impl Phase {
    /// The first `count` phases (one or two), in order.
    fn first(count: u8) -> impl Iterator<Item = Self> {
        [Self::One, Self::Two].into_iter().take(usize::from(count))
    }

    /// Its place among the phases: 0, then 1.
    pub(crate) fn index(self) -> usize {
        match self {
            Self::One => 0,
            Self::Two => 1,
        }
    }
}

/// The renderer's passes, each encoded by the renderer's method of the same
/// purpose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pass {
    /// GPU presentation's blending, composition, culling and compaction.
    Presentation,
    /// Placed objects' culling for the camera and the shadow maps.
    Scatter,
    /// What placed objects cover drawn around the camera.
    Occupancy,
    /// Shadow caster selection.
    ShadowSelect,
    /// Virtual geometry's cluster selection.
    VirtualSelect,
    /// Terrain's horizon maps where the sun or terrain changed.
    TerrainHorizons,
    /// The shadow maps.
    ShadowMaps,
    /// Light binning.
    Lights,
    /// This frame's slice of the clouds' dome, and their shadows' grid.
    CloudDome,
    /// Tracks pressed into the deformation map.
    Deformation,
    /// The runtime virtual texture's pages baked.
    VtBakes,
    /// Ground cover grown on the terrain.
    GroundCover,
    /// The opaque pass of `phase`.
    Opaque {
        phase: Phase,
    },
    /// Terrain shaded deferred in the first phase, after the meshes: its
    /// visibility, writing each pixel's terrain and depth; each terrain's
    /// resolve, shading its pixels once; then its cover, placed objects and
    /// the sky.
    TerrainIds,
    TerrainResolve,
    OpaqueLate,
    /// The pages the first phase's terrain resolves wanted of the runtime
    /// virtual texture.
    VtFeedback,
    /// The visibility buffer's triangle IDs, tile lists and resolve of
    /// `phase`.
    VisibilityIds {
        phase: Phase,
    },
    Classify {
        phase: Phase,
    },
    Resolve {
        phase: Phase,
    },
    /// The depth pyramid and occlusion test between the opaque phases.
    OcclusionTest,
    /// Decals over the opaque surfaces.
    Decals,
    /// Transparent surfaces and particles accumulated, then composited.
    Transparent,
    TransparentComposite,
    /// The screen effects.
    Effects,
    /// The screen effects' composite into the canvas.
    Composite,
    /// Temporal antialiasing's resolve into the canvas.
    TemporalResolve,
    /// Reflection probe faces captured after the frame.
    ProbeCaptures,
    /// Water over the opaque scene.
    Water,
    /// The view's clouds marched after the opaque scene and water, then
    /// laid over it.
    CloudMarch,
    CloudComposite,
}

/// Builds the frame order from one table: every pass, in the order frames
/// run them, under the name reports give it. Each row gives its pass's
/// [`Pass::index`], its place in the order and its bit in the masks frames
/// report, and the span profiling captures time it whole as,
/// `stage_<name>`. A pass without a row, two rows of one pass or two of one
/// name do not compile.
macro_rules! passes {
    ($($name:ident: $variant:ident $({ phase: $phase:path })?,)*) => {
        /// Every pass's row in the frame order, named as reports name it.
        #[expect(non_camel_case_types, reason = "rows go by their passes' report names")]
        #[derive(Clone, Copy)]
        enum Row {
            $($name,)*
        }

        /// Every pass's name in reports, by its row.
        const PASS_NAMES: &[&str] = &[$(stringify!($name),)*];

        /// The span profiling captures time each pass whole as, by its row.
        const PASS_STAGES: &[&str] = &[$(concat!("stage_", stringify!($name)),)*];

        impl Pass {
            /// The pass's row: its place in the frame order, and its bit in
            /// the masks frames report.
            pub(crate) const fn index(self) -> usize {
                match self {
                    $(Pass::$variant $({ phase: $phase })? => Row::$name as usize,)*
                }
            }
        }
    };
}

passes! {
    presentation: Presentation,
    scatter: Scatter,
    occupancy: Occupancy,
    shadow_select: ShadowSelect,
    virtual_select: VirtualSelect,
    terrain_horizons: TerrainHorizons,
    shadow_maps: ShadowMaps,
    lights: Lights,
    cloud_dome: CloudDome,
    deformation: Deformation,
    vt_bakes: VtBakes,
    ground_cover: GroundCover,
    opaque: Opaque { phase: Phase::One },
    terrain_ids: TerrainIds,
    terrain_resolve: TerrainResolve,
    opaque_late: OpaqueLate,
    vt_feedback: VtFeedback,
    visibility: VisibilityIds { phase: Phase::One },
    visibility_tiles: Classify { phase: Phase::One },
    visibility_resolve: Resolve { phase: Phase::One },
    occlusion_test: OcclusionTest,
    opaque_phase_two: Opaque { phase: Phase::Two },
    visibility_phase_two: VisibilityIds { phase: Phase::Two },
    visibility_tiles_phase_two: Classify { phase: Phase::Two },
    visibility_resolve_phase_two: Resolve { phase: Phase::Two },
    decals: Decals,
    water: Water,
    cloud_march: CloudMarch,
    cloud_composite: CloudComposite,
    transparent: Transparent,
    transparent_composite: TransparentComposite,
    effects: Effects,
    composite: Composite,
    temporal_resolve: TemporalResolve,
    probe_captures: ProbeCaptures,
}

// A mask holds every pass as a bit of a u64.
const _: () = assert!(PASS_NAMES.len() <= 64, "a pass mask is a u64");

impl Pass {
    /// The pass's name in reports.
    pub(crate) fn name(self) -> &'static str {
        PASS_NAMES[self.index()]
    }

    /// The span profiling captures time the whole pass as.
    pub(crate) fn stage(self) -> &'static str {
        PASS_STAGES[self.index()]
    }

    fn bit(self) -> u64 {
        1 << self.index()
    }
}

/// The passes in `mask`, such as a frame's [`RenderGraph::kept`], as the
/// stats report them: by name, in frame order.
pub(crate) fn render_passes(mask: u64) -> engine_core::RenderPasses {
    engine_core::RenderPasses {
        bits: mask,
        names: PASS_NAMES,
    }
}

/// Color attachments a pass may draw into.
const MAX_COLORS: usize = 4;

/// A declared pass and, once compiled, whether it runs and its attachments'
/// operations.
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

    /// The color attachments in target order, with their operations.
    pub(crate) fn colors(&self) -> impl Iterator<Item = (Resource, Ops)> + '_ {
        self.colors
            .iter()
            .zip(&self.color_ops)
            .filter_map(|(color, ops)| Some((color.as_ref()?.0, (*ops)?)))
    }

    /// The depth attachment with its operations, `None` when read-only.
    pub(crate) fn depth(&self) -> Option<(Resource, Option<Ops>)> {
        self.depth.map(|(resource, _)| (resource, self.depth_ops))
    }

    /// Resources whose earlier contents the pass reads where they exist,
    /// attachments it draws over included.
    fn inputs(&self) -> Set {
        let mut inputs = self.reads;
        for (resource, access) in self.attachments() {
            if !matches!(access, Access::Clear(_)) {
                inputs.insert(resource);
            }
        }
        inputs
    }

    /// Resources whose earlier contents must exist: its reads, and the
    /// attachments it loads or tests.
    fn required(&self) -> Set {
        let mut required = self.reads;
        for (resource, access) in self.attachments() {
            if matches!(access, Access::Load | Access::ReadOnly) {
                required.insert(resource);
            }
        }
        required
    }

    /// Resources the pass writes, its attachments included.
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

/// Declares a node's resources.
pub(crate) struct NodeBuilder<'g> {
    node: &'g mut Node,
}

impl NodeBuilder<'_> {
    /// Reads `resource` as a texture or buffer.
    pub(crate) fn read(self, resource: Resource) -> Self {
        self.node.reads.insert(resource);
        self
    }

    /// Reads `resource` when `condition` holds.
    pub(crate) fn read_if(self, condition: bool, resource: Resource) -> Self {
        if condition { self.read(resource) } else { self }
    }

    /// Writes `resource` as a storage buffer or texture.
    pub(crate) fn write(self, resource: Resource) -> Self {
        self.node.writes.insert(resource);
        self
    }

    /// Writes `resource` when `condition` holds.
    pub(crate) fn write_if(self, condition: bool, resource: Resource) -> Self {
        if condition {
            self.write(resource)
        } else {
            self
        }
    }

    /// Draws into the next color attachment.
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

    /// Tests against (and unless `ReadOnly`, writes) the depth attachment.
    pub(crate) fn depth(self, resource: Resource, access: Access) -> Self {
        self.node.depth = Some((resource, access));
        self
    }
}

/// What compiling found wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Invalid {
    /// A pass declared twice.
    Twice(Pass),
    /// A pass reading what no earlier pass wrote.
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

/// A frame's passes. The renderer keeps one and declares each frame into it.
#[derive(Debug, Default)]
pub(crate) struct RenderGraph {
    nodes: Vec<Node>,
}

impl RenderGraph {
    /// Forgets the last frame's passes.
    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Declares `pass`, which runs in its place in the frame order whatever
    /// was declared before it.
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

    /// Puts the declared passes in frame order, culls those whose results
    /// no later pass uses (passes writing kept or output resources always
    /// run), derives the kept passes' attachment operations and checks
    /// their reads. Attachments clear where nothing was drawn before and
    /// store what a later pass reads. A pass declared twice fails.
    pub(crate) fn compile(&mut self) -> Result<(), Invalid> {
        // In place: rows are unique, so the order is the same every frame
        // however the passes were declared.
        self.nodes.sort_unstable_by_key(|node| node.pass.index());
        if let Some(twice) = self
            .nodes
            .windows(2)
            .find(|pair| pair[0].pass == pair[1].pass)
        {
            return Err(Invalid::Twice(twice[0].pass));
        }
        // Backwards: what later kept passes read of each resource.
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
            // Earlier contents matter only to what this pass reads.
            for resource in outputs.iter() {
                if resource.lifetime() == Lifetime::Frame {
                    needed.remove(resource);
                }
            }
            needed.0 |= node.inputs().0;
        }
        // Forwards: what earlier kept passes wrote.
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

    /// The passes that run, in order.
    pub(crate) fn kept_passes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(|node| node.kept)
    }

    /// The passes that run, as a mask [`render_passes`] reads.
    pub(crate) fn kept(&self) -> u64 {
        self.kept_passes()
            .fold(0, |mask, node| mask | node.pass.bit())
    }
}

/// An attachment's operations before its load is known: stored when `read`
/// later, none when read-only.
fn store_ops(access: Access, read: bool) -> Option<Ops> {
    (access != Access::ReadOnly).then_some(Ops {
        load: Load::Load,
        store: read,
    })
}

/// How a pass of `access` begins with an attachment earlier passes
/// `written` (or not).
fn load_op(access: Access, written: bool) -> Load {
    match access {
        Access::Clear(clear) => Load::Clear(clear),
        Access::Draw(clear) if !written => Load::Clear(clear),
        Access::Draw(_) | Access::Load | Access::ReadOnly => Load::Load,
    }
}

/// Where a frame's main passes draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Targets {
    /// Straight into the canvas.
    Canvas,
    /// Into the screen effects' targets, composited into the canvas.
    Post,
    /// Into the screen effects' and motion targets, resolved with the
    /// history into the canvas.
    Temporal,
}

/// What a frame runs, from which [`plan`] declares its passes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FrameShape {
    /// Presented on the GPU.
    pub(crate) gpu: bool,
    pub(crate) targets: Targets,
    /// The canvas's clear color, which depends on whether it is sRGB.
    pub(crate) canvas_clear: [f64; 4],
    /// Opaque phases: two around the occlusion test.
    pub(crate) phases: u8,
    pub(crate) shadows: bool,
    /// Work before the passes reading it: terrain's horizon maps build
    /// (shadowed frames only), the clouds' dome marches, tracks press into
    /// the deformation map, the virtual texture bakes pages and ground cover
    /// grows.
    pub(crate) horizons: bool,
    pub(crate) cloud_dome: bool,
    pub(crate) deformation: bool,
    pub(crate) vt_bakes: bool,
    pub(crate) cover: bool,
    /// Terrain draws deferred in the first opaque phase: its visibility
    /// target holds the pages the virtual texture's feedback looks up.
    pub(crate) deferred: bool,
    /// Placed objects draw, and the occupancy map is drawn
    /// again.
    pub(crate) scatter: bool,
    pub(crate) occupancy: bool,
    /// Virtual geometry selects clusters, the shadow maps
    /// draw some, and the visibility buffer resolves them in each phase.
    pub(crate) clusters: bool,
    pub(crate) shadow_clusters: bool,
    pub(crate) resolves: [bool; 2],
    pub(crate) lights: bool,
    pub(crate) decals: bool,
    /// Water draws over the opaque scene.
    pub(crate) water: bool,
    /// The view's clouds march and cover the scene.
    pub(crate) clouds: bool,
    /// Transparent draws or particles, and glass reading
    /// the opaque scene.
    pub(crate) transparency: bool,
    pub(crate) refraction: bool,
    pub(crate) effects: Effects,
    /// Reflection probes capture faces after the frame.
    pub(crate) probes: bool,
}

/// The opaque scene's clear color in the effects' linear targets: the
/// canvas's in linear light.
pub(crate) const SCENE_CLEAR: [f64; 4] = [0.009, 0.015, 0.025, 1.0];
const TRANSPARENT: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
const WHITE: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
const BLACK: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
/// The far plane's depth (reversed): `gpu_data::FAR_DEPTH`.
const FAR: f32 = 0.0;

/// Declares the passes of a frame of `shape` into `graph`.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub(crate) fn plan(graph: &mut RenderGraph, shape: &FrameShape) {
    use Resource::*;
    graph.clear();
    let gpu = shape.gpu;
    if gpu {
        graph.add(Pass::Presentation).write(Poses).write(Visible);
    }
    if shape.scatter {
        graph.add(Pass::Scatter).write(Placed);
    }
    if shape.occupancy {
        graph
            .add(Pass::Occupancy)
            .read_if(shape.scatter, Placed)
            .write(Occupancy);
    }
    // Shadow casters first: virtual geometry selects the clusters of the
    // virtual ones with the camera's, then the maps draw.
    if shape.shadows {
        graph
            .add(Pass::ShadowSelect)
            .read(Poses)
            .read_if(gpu, Visible)
            .write(Casters);
    }
    if shape.clusters {
        graph
            .add(Pass::VirtualSelect)
            .read(Poses)
            .read_if(gpu, Visible)
            .read_if(shape.shadow_clusters, Casters)
            .read(Pyramid)
            .write(Clusters);
    }
    // Terrain's horizon maps, which shadowed frames build as the sun or
    // terrain changes, before the passes shading with them.
    if shape.shadows && shape.horizons {
        graph.add(Pass::TerrainHorizons).write(Horizons);
    }
    if shape.shadows {
        graph
            .add(Pass::ShadowMaps)
            .read(Poses)
            .read(Casters)
            .read_if(shape.scatter, Placed)
            .read_if(shape.shadow_clusters, Clusters)
            .write(ShadowMaps);
    }
    if shape.lights {
        graph.add(Pass::Lights).read(Poses).write(Lights);
    }
    // What the first opaque phase draws and shades with: the clouds' dome
    // (the sky) and their shadows, tracks, the virtual texture's pages and
    // ground cover, grown over the tracks.
    if shape.cloud_dome {
        graph.add(Pass::CloudDome).write(CloudDome);
    }
    if shape.deformation {
        graph.add(Pass::Deformation).write(Tracks);
    }
    if shape.vt_bakes {
        graph.add(Pass::VtBakes).write(Pages);
    }
    if shape.cover {
        graph.add(Pass::GroundCover).read(Tracks).write(Cover);
    }
    // The main passes' color targets, in pipeline target order.
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
    };
    let scene = if shape.targets == Targets::Canvas {
        Canvas
    } else {
        Scene
    };
    for phase in Phase::first(shape.phases) {
        if phase == Phase::Two {
            graph
                .add(Pass::OcclusionTest)
                .read(Depth)
                .read(Poses)
                .read_if(shape.clusters, Clusters)
                .write(Pyramid)
                .write(Visible)
                .write_if(shape.clusters, Clusters);
        }
        // Terrain, its cover and placed objects draw in the first phase,
        // after the meshes where terrain shades deferred.
        let first = phase == Phase::One;
        let forward = first && !shape.deferred;
        let mut opaque = graph
            .add(Pass::Opaque { phase })
            .read(Poses)
            .read_if(gpu, Visible)
            .read_if(shape.scatter && forward, Placed)
            .read(Occupancy)
            .read_if(shape.shadows, ShadowMaps)
            .read_if(shape.lights, Lights)
            .read_if(shape.clusters, Clusters)
            .read(Probes)
            .read(Horizons)
            .read(CloudDome)
            .read_if(forward, Tracks)
            .read_if(forward, Pages)
            .read_if(forward, Cover);
        for &(target, clear) in targets {
            opaque = opaque.color(target, Access::Draw(clear));
        }
        opaque.depth(Depth, Access::Draw(Clear::Depth(FAR)));
        if first && shape.deferred {
            deferred_terrain(graph, shape, targets);
        }
        if shape.resolves[phase.index()] {
            graph
                .add(Pass::VisibilityIds { phase })
                .read(Poses)
                .read(Clusters)
                .color(Ids, Access::Clear(Clear::Color(TRANSPARENT)))
                .depth(Depth, Access::Load);
            graph.add(Pass::Classify { phase }).read(Ids).write(Tiles);
            let mut resolve = graph
                .add(Pass::Resolve { phase })
                .read(Poses)
                .read(Clusters)
                .read(Ids)
                .read(Tiles)
                .read_if(shape.shadows, ShadowMaps)
                .read_if(shape.lights, Lights)
                .read(Probes);
            for &(target, _) in targets {
                resolve = resolve.color(target, Access::Load);
            }
            resolve.depth(Depth, Access::ReadOnly);
        }
    }
    if shape.decals {
        graph
            .add(Pass::Decals)
            .read(Depth)
            .read_if(shape.lights, Lights)
            .color(scene, Access::Load);
    }
    // Water blends over the opaque scene and writes its
    // depth, which transparent surfaces and screen effects then see.
    if shape.water {
        let mut water = graph
            .add(Pass::Water)
            .read_if(shape.shadows, ShadowMaps)
            .read(Probes);
        for &(target, _) in targets {
            water = water.color(target, Access::Load);
        }
        water.depth(Depth, Access::Load);
    }
    // The view's clouds march up to what the opaque scene
    // and water show, then cover it before transparent surfaces draw.
    if shape.clouds {
        graph.add(Pass::CloudMarch).read(Depth).write(CloudView);
        graph
            .add(Pass::CloudComposite)
            .read(CloudView)
            .read(Depth)
            .color(scene, Access::Load);
    }
    if shape.transparency {
        graph
            .add(Pass::Transparent)
            .read(Poses)
            .read_if(gpu, Visible)
            .read_if(shape.lights, Lights)
            .read_if(shape.shadows, ShadowMaps)
            .read_if(shape.refraction, Scene)
            .read(Probes)
            .color(Accumulated, Access::Clear(Clear::Color(TRANSPARENT)))
            .color(Revealed, Access::Clear(Clear::Color(WHITE)))
            .depth(Depth, Access::ReadOnly);
        let composite = graph
            .add(Pass::TransparentComposite)
            .read(Accumulated)
            .read(Revealed)
            .color(scene, Access::Load);
        let composite = if scene == Scene {
            composite.color(Glow, Access::Load)
        } else {
            composite
        };
        composite.depth(Depth, Access::ReadOnly);
    }
    let effects = shape.effects;
    if shape.targets != Targets::Canvas && effects.any() {
        effects.reads(graph.add(Pass::Effects)).write(Effects);
    }
    match shape.targets {
        Targets::Canvas => {}
        Targets::Post => {
            graph
                .add(Pass::Composite)
                .read(Scene)
                .read_if(effects.any(), Effects)
                .color(Canvas, Access::Clear(Clear::Color(BLACK)));
        }
        Targets::Temporal => {
            graph
                .add(Pass::TemporalResolve)
                .read(Scene)
                .read(Motion)
                .read(History)
                .read_if(effects.any(), Effects)
                .color(Canvas, Access::Clear(Clear::Color(BLACK)))
                .color(History, Access::Clear(Clear::Color(BLACK)));
        }
    }
    if shape.probes {
        graph
            .add(Pass::ProbeCaptures)
            .read(Poses)
            .read(Visible)
            .write(Probes);
    }
}

/// Declares terrain shaded deferred after the first phase's meshes, drawing
/// into the frame's `targets`: its visibility over their depth, each
/// terrain's resolve of the pixels it holds, its cover, placed objects and
/// the sky over them, and the pages its resolves wanted of the virtual
/// texture.
fn deferred_terrain(graph: &mut RenderGraph, shape: &FrameShape, targets: &[(Resource, Clear)]) {
    use Resource::*;
    graph
        .add(Pass::TerrainIds)
        .read(Tracks)
        .color(TerrainIds, Access::Clear(Clear::Color(TRANSPARENT)))
        .depth(Depth, Access::Load);
    let mut resolve = graph
        .add(Pass::TerrainResolve)
        .read(TerrainIds)
        .read(Occupancy)
        .read_if(shape.shadows, ShadowMaps)
        .read_if(shape.lights, Lights)
        .read(Probes)
        .read(Horizons)
        .read(CloudDome)
        .read(Tracks)
        .read(Pages);
    for &(target, _) in targets {
        resolve = resolve.color(target, Access::Load);
    }
    let mut late = graph
        .add(Pass::OpaqueLate)
        .read_if(shape.scatter, Placed)
        .read(Occupancy)
        .read_if(shape.shadows, ShadowMaps)
        .read_if(shape.lights, Lights)
        .read(Probes)
        .read(Horizons)
        .read(CloudDome)
        .read(Tracks)
        .read(Cover);
    for &(target, _) in targets {
        late = late.color(target, Access::Load);
    }
    late.depth(Depth, Access::Load);
    graph.add(Pass::VtFeedback).read(TerrainIds).write(Feedback);
}
