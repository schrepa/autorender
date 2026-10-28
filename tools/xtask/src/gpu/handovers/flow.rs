use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use naga::valid::{FunctionInfo, ModuleInfo};
use naga::{
    AddressSpace, Block, Expression, Function, GlobalVariable, Handle, Module, Span, Statement,
    Type, TypeInner,
};

mod evaluate;
mod walk;

const RELATIVE: &str = "relative";
const DEPTH: usize = 8;
const ROUNDS: usize = 256;
const PASSES: usize = 16;

const SIZES: [&str; 3] = ["detail_distance", "detail_across", "detail_span"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::gpu) enum Watch {
    Camera,
    Sizes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Read {
    Eye,
    Detail,
    PixelAngle,
    ViewAxis,
    Relative,
    Measure,
    Derivative,
}

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
    Saturate,
}

pub(super) struct Source {
    pub read: Read,
    pub function: String,
    pub span: Span,
}

pub(super) struct Sink {
    pub decision: Decision,
    pub function: String,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub(super) struct Call {
    pub function: String,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Origin {
    Source(usize),
    Part(usize),
}

type Taint = BTreeSet<Origin>;

fn extend(taint: &mut Taint, more: &Taint) -> bool {
    let before = taint.len();
    taint.extend(more.iter().copied());
    taint.len() != before
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Value {
    Whole(Taint),
    Lanes(Vec<Taint>),
    Members(Vec<Value>),
    Matrix(Taint),
}

impl Value {
    fn clean() -> Self {
        Value::Whole(Taint::new())
    }

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

    fn lane(&self, index: usize) -> Taint {
        match self {
            Value::Lanes(lanes) => lanes.get(index).cloned().unwrap_or_else(|| self.collapse()),
            Value::Whole(taint) | Value::Matrix(taint) => taint.clone(),
            Value::Members(_) => self.collapse(),
        }
    }

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

    fn with(&self, more: &Taint) -> Value {
        self.map(&mut |taint| taint.union(more).copied().collect())
    }

    fn cleared(&self) -> Value {
        self.map(&mut |_| Taint::new())
    }
}

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

fn clean(module: &Module, ty: Handle<Type>) -> Value {
    shaped(module, ty, &mut Vec::new(), &mut |_| Taint::new())
}

fn lanes_of(inner: &TypeInner) -> Option<usize> {
    match inner {
        TypeInner::Vector { size, .. } => Some(usize::from(u8::from(*size))),
        _ => None,
    }
}

fn type_of<'t>(
    info: &'t FunctionInfo,
    module: &'t Module,
    handle: Handle<Expression>,
) -> &'t TypeInner {
    info[handle].ty.inner_with(&module.types)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Index(u32),
    Any,
}

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

#[derive(Clone, Debug)]
enum Slot {
    Value(Value),
    Place(Place),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Member {
    Eye,
    Detail,
    Options,
    Matrix,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Of {
    Function(usize),
    Entry(usize),
}

struct Summary {
    parts: Vec<(usize, Vec<u32>)>,
    result: Value,
    pointees: Vec<Option<Value>>,
    globals: BTreeMap<Handle<GlobalVariable>, Value>,
    pending: BTreeMap<usize, (Taint, Vec<Call>)>,
}

impl Summary {
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
    pub(super) fn decisions(&self) -> usize {
        self.decided.len()
    }

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

    fn source(&mut self, read: Read, of: Of, name: &str, span: Span) -> Taint {
        let start = |span: Span| span.to_range().map_or(usize::MAX, |range| range.start);
        let id = if let Some(&id) = self.source_ids.get(&(read, of)) {
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

fn bounds_loop(accept: &Block, reject: &Block) -> bool {
    let accept: &[Statement] = accept;
    let reject: &[Statement] = reject;
    matches!(
        (accept, reject),
        ([], [Statement::Break]) | ([Statement::Break], [])
    )
}
