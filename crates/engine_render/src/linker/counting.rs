use serde_json::{Map, Value, json};
use wesl::syntax::{
    AssignmentOperator, Attribute, CompoundStatement, DeclarationKind, ElseClause, Expression,
    Function, FunctionCall, GlobalDeclaration, IfStatement, LiteralExpression, PathOrigin,
    Spanned, Statement, StatementNode, TranslationUnit, UnaryOperator,
};
use wesl::Mangler as _;

use super::Ty;
use crate::machine::Stage;

pub(crate) const INSTRUMENT: &str = "gpu.pipelines.work";
pub(crate) const GROUP: u32 = 3;
pub(crate) const CHANGES: &str = "fragment and compute entries run a counted copy that adds one to an atomic counter in group 3 at the start of every path; a counted fragment entry loses early depth tests, so it also runs for fragments the depth test then rejects; vertex entries run uncounted";
const COUNTED: &str = "counted_work";
const RUNS: &str = "gpu.pipelines.work.runs";
const OPERATIONS: &str = "gpu.pipelines.work.operations";
const SAMPLES: &str = "gpu.pipelines.work.samples";
const LIMITS: [&str; 4] = [
    "a counted fragment entry loses the early depth test: in depth-tested passes its runs include fragments the depth test then rejects; pixels are unchanged",
    "the right side of && and || is counted as if it always ran",
    "vertex entries are not counted: a vertex stage cannot write storage",
    "a for loop left by break counts its update once more than it ran",
];

pub(crate) fn names(world: &mut bevy_ecs::world::World) {
    use engine_registry::{Kind, NameEntry, register, unavailable};
    let names = [
        (INSTRUMENT, Kind::State, None),
        (RUNS, Kind::Count, Some("runs")),
        (OPERATIONS, Kind::Count, Some("operations")),
        (SAMPLES, Kind::Count, Some("samples")),
    ];
    for (name, kind, unit) in names {
        let entry = NameEntry::read(name, kind, move |reach, _| {
            Ok(reach.part::<crate::Renderer>().map_or_else(
                || unavailable("no_renderer"),
                |renderer| {
                    renderer
                        .counter()
                        .map_or_else(|| unavailable("never_switched_on"), |counter| counter.read(name))
                },
            ))
        });
        let entry = match unit {
            Some(unit) => entry.unit(unit),
            None => entry.changes_frame(CHANGES),
        };
        register(world, entry);
    }
}

pub(crate) struct Counter {
    counting: Counting,
    words: Vec<u32>,
    through_frame: Option<u64>,
}

impl Counter {
    pub(crate) fn new(counting: Counting) -> Self {
        Self {
            counting,
            words: Vec::new(),
            through_frame: None,
        }
    }

    pub(crate) fn counting(&self) -> &Counting {
        &self.counting
    }

    pub(crate) fn landed(&mut self, frame: u64, bytes: &[u8]) {
        if (bytes.len() as u64) < self.counting.bytes() {
            return;
        }
        self.words = (bytes.as_chunks::<4>().0.iter())
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        self.through_frame = Some(frame);
    }

    fn read(&self, name: &str) -> Value {
        let Some(frame) = self.through_frame else {
            return json!({"unavailable": "not_yet_read"});
        };
        let measure = match name {
            RUNS => Measure::Runs,
            OPERATIONS => Measure::Operations,
            SAMPLES => Measure::Samples,
            _ => {
                let mut state = self.counting.entries();
                state["through_frame"] = json!(frame);
                state["limits"] = json!(LIMITS);
                return state;
            }
        };
        self.counting.read(&self.words, measure)
    }
}

#[derive(Clone, Copy)]
enum Measure {
    Runs,
    Operations,
    Samples,
}

#[derive(Clone, Copy, Default)]
struct Cost {
    operations: u32,
    samples: u32,
}

impl Cost {
    const OPERATION: Self = Self {
        operations: 1,
        samples: 0,
    };
}

impl std::ops::Add for Cost {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            operations: self.operations + other.operations,
            samples: self.samples + other.samples,
        }
    }
}

struct Path {
    function: String,
    at: u32,
    cost: Cost,
}

struct Copied {
    text: String,
    base: u32,
    paths: Vec<Path>,
}

struct Copy {
    key: u64,
    label: String,
    entry: &'static str,
    stage: &'static str,
    pipelines: Vec<String>,
    taken: bool,
    counted: Result<Copied, &'static str>,
}

pub(crate) struct Counting {
    copies: Vec<Copy>,
    words: u32,
}

impl Counting {
    pub(crate) fn new<'a>(pipelines: impl IntoIterator<Item = (String, &'a [Stage])>) -> Self {
        let mut counting = Self {
            copies: Vec::new(),
            words: 0,
        };
        for (pipeline, stages) in pipelines {
            for stage in stages {
                let key = stage.text_key();
                if let Some(known) = counting.copies.iter_mut().find(|copy| copy.key == key) {
                    known.pipelines.push(pipeline.clone());
                    continue;
                }
                let entry = stage.entry.name();
                let main = super::path_of((stage.entry.module(), entry)).0;
                let (kind, taken, counted) =
                    copy(&stage.composed.text, (entry, main), counting.words);
                if let Ok(made) = &counted {
                    counting.words += 2 * made.paths.len() as u32;
                }
                let same = (counting.copies.iter())
                    .filter(|copy| copy.entry == entry)
                    .count();
                let label = match same {
                    0 => entry.to_owned(),
                    _ => format!("{entry}#{same}"),
                };
                counting.copies.push(Copy {
                    key,
                    label,
                    entry,
                    stage: kind,
                    pipelines: vec![pipeline.clone()],
                    taken,
                    counted,
                });
            }
        }
        counting
    }

    pub(crate) fn texts<'a>(&'a self, stages: &'a [Stage]) -> Option<Vec<&'a str>> {
        let mut counted = false;
        let mut texts = Vec::with_capacity(stages.len());
        for stage in stages {
            let key = stage.text_key();
            let copy = self.copies.iter().find(|copy| copy.key == key)?;
            if copy.taken {
                return None;
            }
            texts.push(match &copy.counted {
                Ok(made) => {
                    counted = true;
                    made.text.as_str()
                }
                Err(_) => stage.composed.text.as_str(),
            });
        }
        counted.then_some(texts)
    }

    pub(crate) fn bytes(&self) -> u64 {
        u64::from(self.words) * 4
    }

    pub(crate) fn layout_entry() -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: Ty::Storage { write: true }.binding(false, false),
            count: None,
        }
    }

    fn entries(&self) -> Value {
        let mut entries = Map::new();
        for copy in &self.copies {
            let mut about = json!({
                "entry": copy.entry,
                "stage": copy.stage,
                "pipelines": copy.pipelines,
            });
            match &copy.counted {
                Ok(made) => {
                    about["paths"] = (made.paths.iter())
                        .map(|path| json!(format!("{}#{}", path.function, path.at)))
                        .collect();
                }
                Err(reason) => about["unavailable"] = json!(reason),
            }
            entries.insert(copy.label.clone(), about);
        }
        json!({"words": self.words, "entries": entries})
    }

    fn read(&self, words: &[u32], measure: Measure) -> Value {
        if words.len() < self.words as usize {
            return json!({"unavailable": "not_yet_read"});
        }
        let runs = |at: u32| {
            u64::from(words[at as usize]) | (u64::from(words[at as usize + 1]) << 32)
        };
        let mut entries = Map::new();
        for copy in &self.copies {
            let Ok(made) = &copy.counted else {
                continue;
            };
            let (mut total, mut paths) = (0, Map::new());
            for (index, path) in made.paths.iter().enumerate() {
                let ran = runs(made.base + 2 * index as u32);
                let value = match measure {
                    Measure::Runs => ran,
                    Measure::Operations => ran * u64::from(path.cost.operations),
                    Measure::Samples => ran * u64::from(path.cost.samples),
                };
                let entered = path.at == 0 && path.function == copy.entry;
                match measure {
                    Measure::Runs if entered => total = ran,
                    Measure::Runs => {}
                    Measure::Operations | Measure::Samples => total += value,
                }
                paths.insert(format!("{}#{}", path.function, path.at), json!(value));
            }
            entries.insert(copy.label.clone(), json!({"total": total, "paths": paths}));
        }
        Value::Object(entries)
    }
}

fn copy(
    text: &str,
    (entry, main): (&str, &str),
    base: u32,
) -> (&'static str, bool, Result<Copied, &'static str>) {
    let Ok(mut unit) = text.parse::<TranslationUnit>() else {
        return ("unknown", false, Err("unparsed"));
    };
    let taken = unit.global_declarations.iter().any(|declaration| {
        let GlobalDeclaration::Declaration(item) = declaration.node() else {
            return false;
        };
        item.attributes.iter().any(|attribute| {
            matches!(attribute.node(), Attribute::Group(group) if literal(group) == Some(i64::from(GROUP)))
        })
    });
    let Some(kind) = stage_of(&unit, entry) else {
        return ("unknown", taken, Err("entry_missing"));
    };
    let counted = if kind == "vertex" {
        Err("vertex_stage_cannot_write_storage")
    } else if taken {
        Err("group_3_taken")
    } else if text.contains(COUNTED) {
        Err("name_taken")
    } else {
        counted(&mut unit, (entry, main), base).map(|paths| Copied {
            text: unit.to_string(),
            base,
            paths,
        })
    };
    (kind, taken, counted)
}

fn stage_of(unit: &TranslationUnit, entry: &str) -> Option<&'static str> {
    functions(unit)
        .find(|function| *function.ident.name() == entry)?
        .attributes
        .iter()
        .find_map(|attribute| {
            let stages = [
                (Attribute::Vertex, "vertex"),
                (Attribute::Fragment, "fragment"),
                (Attribute::Compute, "compute"),
            ];
            (stages.into_iter())
                .find(|(stage, _)| stage == attribute.node())
                .map(|(_, name)| name)
        })
}

fn source_name(name: &str, (entry, main): (&str, &str)) -> String {
    if name == entry {
        return name.to_owned();
    }
    let unmangled = wesl::mangler::EscapeMangler
        .unmangle(name)
        .filter(|(path, item)| {
            path.origin == PathOrigin::Absolute
                && wesl::mangler::EscapeMangler.mangle(path, item) == name
        });
    match unmangled {
        Some((path, item)) => format!("{}::{item}", path.components.join("::")),
        None => format!("{main}::{name}"),
    }
}

fn literal(expression: &Expression) -> Option<i64> {
    let Expression::Literal(literal) = expression else {
        return None;
    };
    match literal {
        LiteralExpression::AbstractInt(value) => Some(*value),
        LiteralExpression::I32(value) => Some(i64::from(*value)),
        LiteralExpression::U32(value) => Some(i64::from(*value)),
        LiteralExpression::Bool(_)
        | LiteralExpression::AbstractFloat(_)
        | LiteralExpression::F32(_)
        | LiteralExpression::F16(_) => None,
    }
}

fn functions(unit: &TranslationUnit) -> impl Iterator<Item = &Function> {
    (unit.global_declarations.iter()).filter_map(|declaration| {
        if let GlobalDeclaration::Function(function) = declaration.node() {
            Some(function)
        } else {
            None
        }
    })
}

fn counted(
    unit: &mut TranslationUnit,
    (entry, main): (&str, &str),
    base: u32,
) -> Result<Vec<Path>, &'static str> {
    let discarding = discarding(unit);
    let folded = folded(unit, entry, &discarding);
    let mut walker = Walker {
        base,
        paths: Vec::new(),
        discarding: &discarding,
        folded: &folded,
        function: String::new(),
        at: 0,
    };
    for declaration in &mut unit.global_declarations {
        if let GlobalDeclaration::Function(function) = declaration.node_mut() {
            let name = function.ident.name().clone();
            if folded.iter().any(|(known, _)| *known == name) {
                continue;
            }
            walker.function = source_name(&name, (entry, main));
            walker.at = 0;
            walker.compound(&mut function.body, Cost::default())?;
        }
    }
    let declared: GlobalDeclaration = format!(
        "@group({GROUP}) @binding(0) var<storage, read_write> {COUNTED}: array<atomic<u32>>;"
    )
    .parse()
    .map_err(|_| "counter_unparsed")?;
    unit.global_declarations.push(Spanned::from(declared));
    Ok(walker.paths)
}

type Folded = [(String, Cost)];

fn discarding(unit: &TranslationUnit) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    loop {
        let before = found.len();
        for function in functions(unit) {
            let name = function.ident.name().clone();
            let test = |called: &str| found.iter().any(|known| known == called);
            if !found.contains(&name) && block_ends(&function.body, KILLS, &test) {
                found.push(name);
            }
        }
        if found.len() == before {
            return found;
        }
    }
}

fn folded(unit: &TranslationUnit, entry: &str, discarding: &[String]) -> Vec<(String, Cost)> {
    let leaves = |called: &str| discarding.iter().any(|known| known == called);
    let straight: Vec<&Function> = functions(unit)
        .filter(|function| *function.ident.name() != entry)
        .filter(|function| {
            let statements = &function.body.statements;
            statements.iter().enumerate().all(|(index, statement)| {
                own_cost(statement.node(), &[]).is_some()
                    && (index + 1 == statements.len() || !ends(statement.node(), LEAVES, &leaves))
            })
        })
        .collect();
    let mut folded: Vec<(String, Cost)> = Vec::new();
    loop {
        let before = folded.len();
        for function in &straight {
            let name = function.ident.name().clone();
            let done = |called: &str| folded.iter().any(|(known, _)| known == called);
            let waits = |called: &str| {
                !done(called) && straight.iter().any(|other| *other.ident.name() == called)
            };
            if done(&name) || block_ends(&function.body, WAITS, &waits) {
                continue;
            }
            let cost = (function.body.statements.iter())
                .filter_map(|statement| own_cost(statement.node(), &folded))
                .fold(Cost::default(), |total, cost| total + cost);
            folded.push((name, cost));
        }
        if folded.len() == before {
            return folded;
        }
    }
}

#[derive(Clone, Copy)]
struct Inside {
    loops: bool,
    breaks: bool,
    returns: bool,
    discards: bool,
}

const KILLS: Inside = Inside {
    loops: true,
    breaks: true,
    returns: false,
    discards: true,
};

const LEAVES: Inside = Inside {
    loops: false,
    breaks: false,
    returns: true,
    discards: true,
};

const WAITS: Inside = Inside {
    loops: true,
    breaks: true,
    returns: false,
    discards: false,
};

type Test<'a> = &'a dyn Fn(&str) -> bool;

fn block_ends(block: &CompoundStatement, inside: Inside, test: Test<'_>) -> bool {
    (block.statements.iter()).any(|statement| ends(statement.node(), inside, test))
}

fn ends(statement: &Statement, inside: Inside, test: Test<'_>) -> bool {
    let calls = |expression: &Expression| calls(expression, test);
    let looped = Inside {
        loops: true,
        breaks: true,
        ..inside
    };
    let block = |block: &CompoundStatement, inside| block_ends(block, inside, test);
    match statement {
        Statement::Void | Statement::ConstAssert(_) => false,
        Statement::Break(_) => !inside.breaks,
        Statement::Continue(_) => !inside.loops,
        Statement::Discard(_) => inside.discards,
        Statement::Return(item) => {
            inside.returns || item.expression.as_ref().is_some_and(|e| calls(e))
        }
        Statement::Compound(item) => block(item, inside),
        Statement::Assignment(item) => calls(&item.lhs) || calls(&item.rhs),
        Statement::Increment(item) => calls(&item.expression),
        Statement::Decrement(item) => calls(&item.expression),
        Statement::FunctionCall(item) => call_calls(&item.call, test),
        Statement::Declaration(item) => item.initializer.as_ref().is_some_and(|e| calls(e)),
        Statement::If(item) => {
            calls(&item.if_clause.expression)
                || block(&item.if_clause.body, inside)
                || (item.else_if_clauses.iter())
                    .any(|clause| calls(&clause.expression) || block(&clause.body, inside))
                || (item.else_clause.as_ref()).is_some_and(|clause| block(&clause.body, inside))
        }
        Statement::Switch(item) => {
            let switched = Inside {
                breaks: true,
                ..inside
            };
            calls(&item.expression)
                || (item.clauses.iter()).any(|clause| block(&clause.body, switched))
        }
        Statement::Loop(item) => {
            block(&item.body, looped)
                || item.continuing.as_ref().is_some_and(|continuing| {
                    block(&continuing.body, looped)
                        || (continuing.break_if.as_ref())
                            .is_some_and(|test| calls(&test.expression))
                })
        }
        Statement::For(item) => {
            (item.initializer.as_ref()).is_some_and(|first| ends(first.node(), inside, test))
                || item.condition.as_ref().is_some_and(|e| calls(e))
                || (item.update.as_ref()).is_some_and(|next| ends(next.node(), looped, test))
                || block(&item.body, looped)
        }
        Statement::While(item) => calls(&item.condition) || block(&item.body, looped),
    }
}

fn calls(expression: &Expression, test: Test<'_>) -> bool {
    match expression {
        Expression::Literal(_) | Expression::TypeOrIdentifier(_) => false,
        Expression::Parenthesized(item) => calls(&item.expression, test),
        Expression::NamedComponent(item) => calls(&item.base, test),
        Expression::Indexing(item) => calls(&item.base, test) || calls(&item.index, test),
        Expression::Unary(item) => calls(&item.operand, test),
        Expression::Binary(item) => calls(&item.left, test) || calls(&item.right, test),
        Expression::FunctionCall(call) => call_calls(call, test),
    }
}

fn call_calls(call: &FunctionCall, test: Test<'_>) -> bool {
    test(&call.ty.ident.name()) || (call.arguments.iter()).any(|argument| calls(argument, test))
}

fn own_cost(statement: &Statement, folded: &Folded) -> Option<Cost> {
    let none = Cost::default();
    let cost = |expression: &Expression| cost(expression, folded);
    Some(match statement {
        Statement::Void
        | Statement::Break(_)
        | Statement::Continue(_)
        | Statement::Discard(_)
        | Statement::ConstAssert(_) => none,
        Statement::Assignment(item) => {
            let combined = !matches!(item.operator, AssignmentOperator::Equal);
            let operator = if combined { Cost::OPERATION } else { none };
            operator + cost(&item.lhs) + cost(&item.rhs)
        }
        Statement::Increment(item) => Cost::OPERATION + cost(&item.expression),
        Statement::Decrement(item) => Cost::OPERATION + cost(&item.expression),
        Statement::Return(item) => item.expression.as_ref().map_or(none, |e| cost(e)),
        Statement::FunctionCall(item) => call_cost(&item.call, folded),
        Statement::Declaration(item) => match (item.kind, &item.initializer) {
            (DeclarationKind::Const, _) | (_, None) => none,
            (_, Some(value)) => cost(value),
        },
        Statement::Compound(_)
        | Statement::If(_)
        | Statement::Switch(_)
        | Statement::Loop(_)
        | Statement::For(_)
        | Statement::While(_) => return None,
    })
}

fn cost(expression: &Expression, folded: &Folded) -> Cost {
    let inner = |expression: &Expression| cost(expression, folded);
    match expression {
        Expression::Literal(_) | Expression::TypeOrIdentifier(_) => Cost::default(),
        Expression::Parenthesized(item) => inner(&item.expression),
        Expression::NamedComponent(item) => inner(&item.base),
        Expression::Indexing(item) => Cost::OPERATION + inner(&item.base) + inner(&item.index),
        Expression::Unary(item) => match item.operator {
            UnaryOperator::AddressOf | UnaryOperator::Indirection => inner(&item.operand),
            UnaryOperator::LogicalNegation
            | UnaryOperator::Negation
            | UnaryOperator::BitwiseComplement => Cost::OPERATION + inner(&item.operand),
        },
        Expression::Binary(item) => Cost::OPERATION + inner(&item.left) + inner(&item.right),
        Expression::FunctionCall(call) => call_cost(call, folded),
    }
}

fn call_cost(call: &FunctionCall, folded: &Folded) -> Cost {
    let name = call.ty.ident.name();
    let sampled = name.starts_with("textureSample")
        || name.starts_with("textureGather")
        || *name == "textureLoad";
    let own = Cost {
        operations: 1,
        samples: u32::from(sampled),
    };
    let inlined = (folded.iter())
        .find(|(known, _)| *known == *name)
        .map_or(Cost::default(), |(_, cost)| *cost);
    (call.arguments.iter()).fold(own + inlined, |total, argument| {
        total + cost(argument, folded)
    })
}

struct Walker<'a> {
    base: u32,
    paths: Vec<Path>,
    discarding: &'a [String],
    folded: &'a Folded,
    function: String,
    at: u32,
}

impl Walker<'_> {
    fn open(
        &mut self,
        statements: &mut Vec<StatementNode>,
        head: Cost,
    ) -> Result<usize, &'static str> {
        let at = self.base + 2 * self.paths.len() as u32;
        let counter: Statement = format!(
            "if atomicAdd(&{COUNTED}[{at}u], 1u) == {}u {{ atomicAdd(&{COUNTED}[{}u], 1u); }}",
            u32::MAX,
            at + 1
        )
        .parse()
        .map_err(|_| "counter_unparsed")?;
        statements.push(Spanned::from(counter));
        self.paths.push(Path {
            function: self.function.clone(),
            at: self.at,
            cost: head,
        });
        self.at += 1;
        Ok(self.paths.len() - 1)
    }

    fn cost(&self, expression: &Expression) -> Cost {
        cost(expression, self.folded)
    }

    fn compound(
        &mut self,
        block: &mut CompoundStatement,
        head: Cost,
    ) -> Result<usize, &'static str> {
        let statements = std::mem::take(&mut block.statements);
        let last = statements.len();
        let mut path = self.open(&mut block.statements, head)?;
        for (index, mut statement) in statements.into_iter().enumerate() {
            let own = self.statement(statement.node_mut())?;
            self.paths[path].cost = self.paths[path].cost + own;
            let discarding = self.discarding;
            let leaves = |called: &str| discarding.iter().any(|known| known == called);
            let split = index + 1 < last && ends(statement.node(), LEAVES, &leaves);
            block.statements.push(statement);
            if split {
                path = self.open(&mut block.statements, Cost::default())?;
            }
        }
        Ok(path)
    }

    fn statement(&mut self, statement: &mut Statement) -> Result<Cost, &'static str> {
        if let Some(cost) = own_cost(statement, self.folded) {
            return Ok(cost);
        }
        let none = Cost::default();
        Ok(match statement {
            Statement::Compound(item) => {
                self.compound(item, none)?;
                none
            }
            Statement::If(item) => self.branches(item)?,
            Statement::Switch(item) => {
                for clause in &mut item.clauses {
                    self.compound(&mut clause.body, none)?;
                }
                self.cost(&item.expression)
            }
            Statement::Loop(item) => {
                self.compound(&mut item.body, none)?;
                if let Some(continuing) = &mut item.continuing {
                    let last = self.compound(&mut continuing.body, none)?;
                    if let Some(test) = &continuing.break_if {
                        let tested = self.cost(&test.expression);
                        self.paths[last].cost = self.paths[last].cost + tested;
                    }
                }
                none
            }
            Statement::For(item) => {
                let condition = item.condition.as_ref().map_or(none, |e| self.cost(e));
                let first = match &mut item.initializer {
                    Some(first) => self.statement(first.node_mut())?,
                    None => none,
                };
                let next = match &mut item.update {
                    Some(next) => self.statement(next.node_mut())?,
                    None => none,
                };
                self.compound(&mut item.body, condition + next)?;
                first + condition
            }
            Statement::While(item) => {
                let condition = self.cost(&item.condition);
                self.compound(&mut item.body, condition)?;
                condition
            }
            Statement::Void
            | Statement::Break(_)
            | Statement::Continue(_)
            | Statement::Discard(_)
            | Statement::ConstAssert(_)
            | Statement::Assignment(_)
            | Statement::Increment(_)
            | Statement::Decrement(_)
            | Statement::Return(_)
            | Statement::FunctionCall(_)
            | Statement::Declaration(_) => none,
        })
    }

    fn branches(&mut self, item: &mut IfStatement) -> Result<Cost, &'static str> {
        self.compound(&mut item.if_clause.body, Cost::default())?;
        let mut head = Cost::default();
        for clause in &mut item.else_if_clauses {
            head = head + self.cost(&clause.expression);
            self.compound(&mut clause.body, head)?;
        }
        if !item.else_if_clauses.is_empty() && item.else_clause.is_none() {
            item.else_clause = Some(ElseClause {
                attributes: Vec::new(),
                body: CompoundStatement::default(),
            });
        }
        if let Some(clause) = &mut item.else_clause {
            self.compound(&mut clause.body, head)?;
        }
        Ok(self.cost(&item.if_clause.expression))
    }
}
