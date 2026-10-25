//! How what a flow watches ([`Watch`]) runs through a shader module: which
//! values depend on where the camera stands or how it looks, or on a size
//! on screen, and which decisions they reach. Each function is followed
//! once, callees before callers, into a summary of what its result, the
//! variables its pointers reach and the module's private variables depend
//! on in terms of its arguments' parts, and of the decisions its arguments
//! feed; a call applies its callee's summary to what it passes. A local or
//! variable holds every value stored into it, in any order, so a function
//! is followed until nothing it holds grows. Stages are followed apart:
//! what a vertex stage writes reaches no fragment stage, but positions
//! named `relative`, which are relative to the eye.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use naga::valid::{FunctionInfo, ModuleInfo};
use naga::{
    AddressSpace, Block, Expression, Function, GlobalVariable, Handle, Module, Span, Statement,
    Type, TypeInner,
};

mod evaluate;
mod walk;

/// The member name of eye-relative positions, which every shader here gives
/// them, between stages and within them.
const RELATIVE: &str = "relative";
/// Struct members nested deeper than this are followed as one.
const DEPTH: usize = 8;
/// Rounds a function is followed before it counts as unsettled, and passes
/// over a module while its private variables grow.
const ROUNDS: usize = 256;
const PASSES: usize = 16;

/// The detail eye's measures (`scene.wgsl`'s): what each gives depends on
/// where the camera stands no more, and is a distance or a size on screen
/// seen from the detail eye.
const SIZES: [&str; 3] = ["detail_distance", "detail_across", "detail_span"];

/// What a flow follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::gpu) enum Watch {
    /// Where the camera stands and how it looks: its eye, the detail eye
    /// and a pixel's angle read outside the measures, a camera matrix's eye
    /// or axes, and positions relative to the eye.
    Camera,
    /// Sizes on screen: what the detail eye's measures ([`SIZES`]) give, and
    /// a pixel's derivatives. A direction (`normalize`'s) holds none, and
    /// what a texture gives is its texels as filtering averages them,
    /// wherever a size moves the read; `saturate` decides, as the ramp a
    /// fade takes.
    Sizes,
}

/// What a value read from the camera, or a size on screen, is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Read {
    /// The camera's eye, `camera.position.xyz`.
    Eye,
    /// The detail eye read outside the measures, `camera.detail.xyz`.
    Detail,
    /// A pixel's angle read outside the measures, `camera.options.y`.
    PixelAngle,
    /// A part of a camera matrix: the view's eye or an axis.
    ViewAxis,
    /// A position relative to the eye: a value named `relative`.
    Relative,
    /// What one of the detail eye's measures ([`SIZES`]) gives.
    Measure,
    /// A pixel's derivatives: `dpdx`, `dpdy` or `fwidth`.
    Derivative,
}

/// A decision a value reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Decision {
    Smoothstep,
    Step,
    MixWeight,
    Select,
    If,
    Loop,
    BreakIf,
    Switch,
    Number,
    /// `saturate`, a decision where sizes on screen are watched.
    Saturate,
}

/// Where a function reads what the flow watches: what it reads, and the
/// earliest place it does.
pub(super) struct Source {
    pub read: Read,
    pub function: String,
    pub span: Span,
}

/// A decision a function makes by what the flow watches.
pub(super) struct Sink {
    pub decision: Decision,
    pub function: String,
    pub span: Span,
}

/// A call a dependence on what the flow watches passed through: the
/// caller, and where it calls.
#[derive(Clone, Debug)]
pub(super) struct Call {
    pub function: String,
    pub span: Span,
}

/// Where a dependence comes from: a read of the camera (an index into
/// [`Flow::sources`]) or a part of an argument of the function followed (an
/// index into its parts).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Origin {
    Source(usize),
    Part(usize),
}

type Taint = BTreeSet<Origin>;

/// Adds `more` to `taint`; whether it grew.
fn extend(taint: &mut Taint, more: &Taint) -> bool {
    let before = taint.len();
    taint.extend(more.iter().copied());
    taint.len() != before
}

/// What a value depends on, part by part where its type has parts.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Value {
    /// One dependence for all of it: scalars, arrays, and values whose
    /// parts are not told apart.
    Whole(Taint),
    /// A vector's lanes.
    Lanes(Vec<Taint>),
    /// A struct's members.
    Members(Vec<Value>),
    /// A camera matrix: nothing while it transforms whole, its dependence
    /// where a part of it is read.
    Matrix(Taint),
}

impl Value {
    fn clean() -> Self {
        Value::Whole(Taint::new())
    }

    /// Everything it depends on, all its parts together; a camera matrix
    /// used whole as a transform depends on nothing.
    fn collapse(&self) -> Taint {
        let mut all = Taint::new();
        self.gather(&mut all);
        all
    }

    fn gather(&self, into: &mut Taint) {
        match self {
            Value::Whole(taint) => into.extend(taint.iter().copied()),
            Value::Lanes(lanes) => {
                for lane in lanes {
                    into.extend(lane.iter().copied());
                }
            }
            Value::Members(members) => {
                for member in members {
                    member.gather(into);
                }
            }
            Value::Matrix(_) => {}
        }
    }

    /// What lane or element `index` depends on: a vector's lane, a camera
    /// matrix's part, or the whole of anything else.
    fn lane(&self, index: usize) -> Taint {
        match self {
            Value::Lanes(lanes) => lanes.get(index).cloned().unwrap_or_else(|| self.collapse()),
            Value::Whole(taint) | Value::Matrix(taint) => taint.clone(),
            Value::Members(_) => self.collapse(),
        }
    }

    /// Its part at `path` of struct members and vector lanes.
    fn at(&self, path: &[u32]) -> Value {
        let Some((&step, rest)) = path.split_first() else {
            return self.clone();
        };
        match self {
            Value::Members(members) => members
                .get(step as usize)
                .map_or_else(|| Value::Whole(self.collapse()), |member| member.at(rest)),
            _ => Value::Whole(self.lane(step as usize)),
        }
    }

    /// Its part the steps of a place reach.
    fn at_steps(&self, steps: &[Step]) -> Value {
        let Some((&step, rest)) = steps.split_first() else {
            return self.clone();
        };
        match (self, step) {
            (Value::Members(members), Step::Index(index)) => {
                members.get(index as usize).map_or_else(
                    || Value::Whole(self.collapse()),
                    |member| member.at_steps(rest),
                )
            }
            (Value::Lanes(_), Step::Index(index)) => Value::Whole(self.lane(index as usize)),
            (Value::Lanes(_) | Value::Members(_), Step::Any) => Value::Whole(self.collapse()),
            (Value::Whole(taint) | Value::Matrix(taint), _) => Value::Whole(taint.clone()),
        }
    }

    /// Joins `other` into it; whether it changed.
    fn join(&mut self, other: &Value) -> bool {
        match (&mut *self, other) {
            (Value::Whole(mine), Value::Whole(theirs))
            | (Value::Matrix(mine), Value::Matrix(theirs)) => extend(mine, theirs),
            (Value::Lanes(mine), Value::Lanes(theirs)) if mine.len() == theirs.len() => {
                let mut grew = false;
                for (lane, more) in mine.iter_mut().zip(theirs) {
                    grew |= extend(lane, more);
                }
                grew
            }
            (Value::Members(mine), Value::Members(theirs)) if mine.len() == theirs.len() => {
                let mut grew = false;
                for (member, more) in mine.iter_mut().zip(theirs) {
                    grew |= member.join(more);
                }
                grew
            }
            (Value::Lanes(mine), _) => {
                let all = other.collapse();
                let mut grew = false;
                for lane in mine.iter_mut() {
                    grew |= extend(lane, &all);
                }
                grew
            }
            (Value::Members(mine), _) => {
                let all = Value::Whole(other.collapse());
                let mut grew = false;
                for member in mine.iter_mut() {
                    grew |= member.join(&all);
                }
                grew
            }
            (Value::Whole(mine), Value::Lanes(_) | Value::Members(_)) => {
                // It takes the other's shape, what it held in every part.
                let mut shaped = other.clone();
                shaped.join(&Value::Whole(mine.clone()));
                let changed = shaped != *self;
                *self = shaped;
                changed
            }
            (Value::Whole(mine), Value::Matrix(theirs)) => {
                if mine.is_empty() {
                    *self = Value::Matrix(theirs.clone());
                    true
                } else {
                    extend(mine, theirs)
                }
            }
            (Value::Matrix(mine), _) => {
                let all = other.collapse();
                if all.is_empty() {
                    return false;
                }
                let mut whole = mine.clone();
                whole.extend(all);
                *self = Value::Whole(whole);
                true
            }
        }
    }

    /// Joins `other` into its part at `steps`; whether it changed.
    fn join_at(&mut self, steps: &[Step], other: &Value) -> bool {
        let Some((&step, rest)) = steps.split_first() else {
            return self.join(other);
        };
        match (&mut *self, step) {
            (Value::Members(members), Step::Index(index)) if (index as usize) < members.len() => {
                members[index as usize].join_at(rest, other)
            }
            (Value::Lanes(lanes), Step::Index(index)) if (index as usize) < lanes.len() => {
                extend(&mut lanes[index as usize], &other.collapse())
            }
            (Value::Whole(taint), _) => extend(taint, &other.collapse()),
            (Value::Matrix(_), _) => self.join(other),
            _ => self.join(&Value::Whole(other.collapse())),
        }
    }

    /// It with each part's dependence mapped by `map`.
    fn map(&self, map: &mut impl FnMut(&Taint) -> Taint) -> Value {
        match self {
            Value::Whole(taint) => Value::Whole(map(taint)),
            Value::Lanes(lanes) => Value::Lanes(lanes.iter().map(&mut *map).collect()),
            Value::Members(members) => {
                Value::Members(members.iter().map(|member| member.map(&mut *map)).collect())
            }
            Value::Matrix(taint) => Value::Matrix(map(taint)),
        }
    }

    /// It with `more` added to every part.
    fn with(&self, more: &Taint) -> Value {
        self.map(&mut |taint| taint.union(more).copied().collect())
    }

    /// It depending on nothing, its shape kept.
    fn cleared(&self) -> Value {
        self.map(&mut |_| Taint::new())
    }
}

/// A value of type `ty` whose parts (struct members and vector lanes, at
/// `path`) take what `leaf` says they depend on; a pointer's, what it points
/// at.
fn shaped(
    module: &Module,
    ty: Handle<Type>,
    path: &mut Vec<u32>,
    leaf: &mut impl FnMut(&[u32]) -> Taint,
) -> Value {
    match &module.types[ty].inner {
        TypeInner::Vector { size, .. } => Value::Lanes(
            (0..u32::from(u8::from(*size)))
                .map(|lane| {
                    path.push(lane);
                    let taint = leaf(path.as_slice());
                    path.pop();
                    taint
                })
                .collect(),
        ),
        TypeInner::Struct { members, .. } if path.len() < DEPTH => Value::Members(
            members
                .iter()
                .zip(0u32..)
                .map(|(member, index)| {
                    path.push(index);
                    let value = shaped(module, member.ty, path, leaf);
                    path.pop();
                    value
                })
                .collect(),
        ),
        TypeInner::Pointer { base, .. } => shaped(module, *base, path, leaf),
        _ => Value::Whole(leaf(path.as_slice())),
    }
}

/// A value of type `ty` depending on nothing.
fn clean(module: &Module, ty: Handle<Type>) -> Value {
    shaped(module, ty, &mut Vec::new(), &mut |_| Taint::new())
}

/// The lanes of a vector type; none for anything else.
fn lanes_of(inner: &TypeInner) -> Option<usize> {
    match inner {
        TypeInner::Vector { size, .. } => Some(usize::from(u8::from(*size))),
        _ => None,
    }
}

/// The type of expression `handle` of the function `info` describes.
fn type_of<'t>(
    info: &'t FunctionInfo,
    module: &'t Module,
    handle: Handle<Expression>,
) -> &'t TypeInner {
    info[handle].ty.inner_with(&module.types)
}

/// A step from a place into a part of it: a member, lane, column or
/// element, or one chosen as the shader runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Index(u32),
    Any,
}

/// What a pointer points at: a local, a module variable or what a pointer
/// argument points at, and the steps into it; `relative` where a step names
/// a member `relative`, and what the indices chosen as the shader runs
/// depend on.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Place {
    root: Root,
    steps: Vec<Step>,
    relative: bool,
    index: Taint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Root {
    Local(usize),
    Global(Handle<GlobalVariable>),
    Argument(usize),
}

impl Place {
    fn of(root: Root) -> Place {
        Place {
            root,
            steps: Vec::new(),
            relative: false,
            index: Taint::new(),
        }
    }

    fn step(&self, step: Step, relative: bool, index: &Taint) -> Place {
        let mut steps = self.steps.clone();
        steps.push(step);
        Place {
            root: self.root,
            steps,
            relative: self.relative || relative,
            index: self.index.union(index).copied().collect(),
        }
    }
}

/// What an expression stands for: a value, or a place for pointers.
#[derive(Clone, Debug)]
enum Slot {
    Value(Value),
    Place(Place),
}

/// What the camera's members are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Member {
    Eye,
    Detail,
    Options,
    Matrix,
    Other,
}

/// A function or an entry point of the module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Of {
    Function(usize),
    Entry(usize),
}

/// What calling a function does, in terms of its arguments' parts
/// (`parts`, which `Origin::Part` names: an argument and a path into it,
/// into what it points at for pointers): what its result depends on, what
/// it leaves where its pointer arguments point, what it stores in the
/// module's private variables, and the decisions its arguments reach, with
/// the calls they came in through.
struct Summary {
    parts: Vec<(usize, Vec<u32>)>,
    result: Value,
    pointees: Vec<Option<Value>>,
    globals: BTreeMap<Handle<GlobalVariable>, Value>,
    pending: BTreeMap<usize, (Taint, Vec<Call>)>,
}

impl Summary {
    /// A function trusted to depend on what the flow watches only as it
    /// should (the camera's measures and optics, the filters of sizes on
    /// screen): what it gives depends on nothing.
    fn trusted(arguments: usize) -> Summary {
        Summary {
            parts: Vec::new(),
            result: Value::clean(),
            pointees: vec![None; arguments],
            globals: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }
}

/// A function being followed: what its arguments, expressions, locals and
/// pointer arguments' targets hold so far, which expressions are constant
/// and which lie on marked lines, what its calls returned, and what it
/// returns, stores in private variables and feeds into decisions.
struct Frame<'f> {
    of: Of,
    name: String,
    function: &'f Function,
    info: &'f FunctionInfo,
    parts: Vec<(usize, Vec<u32>)>,
    arguments: Vec<Value>,
    pointees: Vec<Option<Value>>,
    locals: Vec<Value>,
    slots: Vec<Slot>,
    constant: Vec<bool>,
    marked: Vec<bool>,
    calls: BTreeMap<Handle<Expression>, Value>,
    result: Value,
    globals: BTreeMap<Handle<GlobalVariable>, Value>,
    pending: BTreeMap<usize, (Taint, Vec<Call>)>,
}

impl Frame<'_> {
    fn slot(&self, handle: Handle<Expression>) -> &Slot {
        &self.slots[handle.index()]
    }

    fn value(&self, handle: Handle<Expression>) -> Value {
        match self.slot(handle) {
            Slot::Value(value) => value.clone(),
            Slot::Place(_) => Value::clean(),
        }
    }

    fn taint(&self, handle: Handle<Expression>) -> Taint {
        match self.slot(handle) {
            Slot::Value(value) => value.collapse(),
            Slot::Place(_) => Taint::new(),
        }
    }
}

/// A shader module followed: where its functions read what the flow
/// watches, the decisions those reads reach (by sink and source, with the
/// calls the dependence came in through, innermost first), how many
/// decisions its functions make, and those whose flow never settled.
pub(super) struct Flow<'a> {
    module: &'a Module,
    watch: Watch,
    trusted: &'a dyn Fn(&str) -> bool,
    marked: &'a dyn Fn(Span) -> bool,
    camera: Option<(Handle<GlobalVariable>, Vec<Member>)>,
    summaries: Vec<Option<Rc<Summary>>>,
    globals: BTreeMap<Handle<GlobalVariable>, Value>,
    source_ids: BTreeMap<(Read, Of), usize>,
    sink_ids: BTreeMap<(Of, usize, Decision), usize>,
    decided: BTreeSet<(Of, usize, Decision)>,
    pub sources: Vec<Source>,
    pub sinks: Vec<Sink>,
    pub findings: BTreeMap<(usize, usize), Vec<Call>>,
    pub unsettled: Vec<String>,
}

/// Follows what `watch` names through every function and entry point of
/// `module`, which `info` describes. Functions `trusted` names are not
/// followed and give values depending on nothing, nor are the detail eye's
/// measures ([`SIZES`]), whose calls give sizes on screen where sizes are
/// watched; expressions on spans `marked` (the lines the check's mark
/// explains) depend on nothing too, and decisions there are not counted.
pub(super) fn follow<'a>(
    module: &'a Module,
    info: &'a ModuleInfo,
    watch: Watch,
    trusted: &'a dyn Fn(&str) -> bool,
    marked: &'a dyn Fn(Span) -> bool,
) -> Flow<'a> {
    let mut flow = Flow {
        module,
        watch,
        trusted,
        marked,
        camera: match watch {
            Watch::Camera => camera(module),
            Watch::Sizes => None,
        },
        summaries: vec![None; module.functions.len()],
        globals: BTreeMap::new(),
        source_ids: BTreeMap::new(),
        sink_ids: BTreeMap::new(),
        decided: BTreeSet::new(),
        sources: Vec::new(),
        sinks: Vec::new(),
        findings: BTreeMap::new(),
        unsettled: Vec::new(),
    };
    // Functions call only those before them; what a later function stores in
    // a private variable reaches earlier ones on the next pass.
    for _ in 0..PASSES {
        let before = flow.globals.clone();
        for (handle, function) in module.functions.iter() {
            let name = function.name.clone().unwrap_or_default();
            let summary = if SIZES.contains(&name.as_str()) || (flow.trusted)(&name) {
                Summary::trusted(function.arguments.len())
            } else {
                let of = Of::Function(handle.index());
                flow.function(of, name, function, &info[handle], false)
            };
            flow.summaries[handle.index()] = Some(Rc::new(summary));
        }
        for (index, entry) in module.entry_points.iter().enumerate() {
            if (flow.trusted)(&entry.name) {
                continue;
            }
            let info = info.get_entry_point(index);
            let name = entry.name.clone();
            flow.function(Of::Entry(index), name, &entry.function, info, true);
        }
        if flow.globals == before {
            break;
        }
    }
    flow
}

/// The camera's uniform (`scene.wgsl`'s `camera`), and what its members are.
fn camera(module: &Module) -> Option<(Handle<GlobalVariable>, Vec<Member>)> {
    let (handle, global) = module.global_variables.iter().find(|(_, global)| {
        global.name.as_deref() == Some("camera") && global.space == AddressSpace::Uniform
    })?;
    let TypeInner::Struct { members, .. } = &module.types[global.ty].inner else {
        return None;
    };
    let meanings = members
        .iter()
        .map(|member| match member.name.as_deref() {
            Some("position") => Member::Eye,
            Some("detail") => Member::Detail,
            Some("options") => Member::Options,
            Some("view_projection" | "previous_view_projection" | "inverse_view_projection") => {
                Member::Matrix
            }
            _ => Member::Other,
        })
        .collect();
    Some((handle, meanings))
}

impl Flow<'_> {
    /// How many decisions the module's followed functions make.
    pub(super) fn decisions(&self) -> usize {
        self.decided.len()
    }

    /// Follows `function` until what it holds settles, then gathers the
    /// decisions it makes, into its summary. Entry points read their inputs
    /// named `relative` as eye-relative where the camera is watched, and the
    /// rest as depending on nothing; other functions read their arguments'
    /// parts.
    fn function(
        &mut self,
        of: Of,
        name: String,
        function: &Function,
        info: &FunctionInfo,
        entry: bool,
    ) -> Summary {
        let mut frame = self.frame(of, name, function, info, entry);
        let mut settled = false;
        for _ in 0..ROUNDS {
            for (handle, expression) in function.expressions.iter() {
                let slot = self.evaluate(&frame, handle, expression);
                frame.slots[handle.index()] = slot;
            }
            if !self.walk(&mut frame, &function.body, false) {
                settled = true;
                break;
            }
        }
        if !settled {
            self.unsettled.push(frame.name.clone());
        }
        self.walk(&mut frame, &function.body, true);
        self.expression_decisions(&mut frame);
        Summary {
            parts: frame.parts,
            result: frame.result,
            pointees: frame.pointees,
            globals: frame.globals,
            pending: frame.pending,
        }
    }

    fn frame<'f>(
        &mut self,
        of: Of,
        name: String,
        function: &'f Function,
        info: &'f FunctionInfo,
        entry: bool,
    ) -> Frame<'f> {
        let module = self.module;
        let mut parts = Vec::new();
        let mut arguments = Vec::new();
        let mut pointees = Vec::new();
        for (index, argument) in function.arguments.iter().enumerate() {
            let value = if entry {
                let taint =
                    if self.watch == Watch::Camera && argument.name.as_deref() == Some(RELATIVE) {
                        self.source(Read::Relative, of, &name, argument_span(function, index))
                    } else {
                        Taint::new()
                    };
                shaped(module, argument.ty, &mut Vec::new(), &mut |_| taint.clone())
            } else {
                shaped(module, argument.ty, &mut Vec::new(), &mut |path| {
                    parts.push((index, path.to_vec()));
                    Taint::from([Origin::Part(parts.len() - 1)])
                })
            };
            if matches!(module.types[argument.ty].inner, TypeInner::Pointer { .. }) {
                arguments.push(Value::clean());
                pointees.push(Some(value));
            } else {
                arguments.push(value);
                pointees.push(None);
            }
        }
        let locals = function
            .local_variables
            .iter()
            .map(|(_, local)| clean(module, local.ty))
            .collect();
        let mut constant = vec![false; function.expressions.len()];
        for (handle, expression) in function.expressions.iter() {
            constant[handle.index()] = match expression {
                Expression::Literal(_)
                | Expression::Constant(_)
                | Expression::Override(_)
                | Expression::ZeroValue(_) => true,
                Expression::Compose { components, .. } => components
                    .iter()
                    .all(|component| constant[component.index()]),
                Expression::Splat { value, .. } => constant[value.index()],
                _ => false,
            };
        }
        let marked = function
            .expressions
            .iter()
            .map(|(handle, _)| (self.marked)(function.expressions.get_span(handle)))
            .collect();
        Frame {
            of,
            name,
            function,
            info,
            parts,
            arguments,
            pointees,
            locals,
            slots: vec![Slot::Value(Value::clean()); function.expressions.len()],
            constant,
            marked,
            calls: BTreeMap::new(),
            result: Value::clean(),
            globals: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }

    /// A read of the camera of kind `read` in the function `of`, named
    /// `name`, at `span`: what it gives depends on that read.
    fn source(&mut self, read: Read, of: Of, name: &str, span: Span) -> Taint {
        let start = |span: Span| span.to_range().map_or(usize::MAX, |range| range.start);
        let id = if let Some(&id) = self.source_ids.get(&(read, of)) {
            // Reports name the earliest place the function reads it.
            if start(span) < start(self.sources[id].span) {
                self.sources[id].span = span;
            }
            id
        } else {
            self.sources.push(Source {
                read,
                function: name.to_owned(),
                span,
            });
            self.source_ids.insert((read, of), self.sources.len() - 1);
            self.sources.len() - 1
        };
        Taint::from([Origin::Source(id)])
    }
}

/// Where argument `index` of `function` is first read.
fn argument_span(function: &Function, index: usize) -> Span {
    function
        .expressions
        .iter()
        .find(|(_, expression)| {
            matches!(
                expression,
                Expression::FunctionArgument(argument) if *argument as usize == index
            )
        })
        .map_or_else(Span::default, |(handle, _)| {
            function.expressions.get_span(handle)
        })
}

/// Whether an `if` with branches `accept` and `reject` is a loop's
/// condition: one branch empty, the other leaving the loop.
fn bounds_loop(accept: &Block, reject: &Block) -> bool {
    let accept: &[Statement] = accept;
    let reject: &[Statement] = reject;
    matches!(
        (accept, reject),
        ([], [Statement::Break]) | ([Statement::Break], [])
    )
}
