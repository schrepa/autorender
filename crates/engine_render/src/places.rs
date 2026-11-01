use std::fmt::Write as _;

use engine_core::{Absent, BrickRegistry, Error, ErrorCode, Member, Place, PlaceKind};
use wesl::syntax::{
    AssignmentOperator, Attribute, DeclarationKind, Expression, Function, GlobalDeclaration,
    ImportContent, ImportStatement, PathOrigin, Statement, TranslationUnit, TypeExpression,
    UnaryOperator,
};

pub const PLACE_ROOT: &str = "place";

pub const PLACE_OWNER: &str = "place";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceShape {
    Reexport,
    Forward,
}

pub const FACE_SHAPE: FaceShape = FaceShape::Reexport;

const CONSTRUCTED: [&str; 17] = [
    "bool", "i32", "u32", "f32", "f16", "vec2", "vec3", "vec4", "mat2x2", "mat2x3", "mat2x4",
    "mat3x2", "mat3x3", "mat3x4", "mat4x2", "mat4x3", "mat4x4",
];

pub fn place_module(place: &str) -> String {
    format!("{PLACE_ROOT}::{place}")
}

pub fn place_of(module: &str) -> Option<&str> {
    let place = module.strip_prefix(PLACE_ROOT)?.strip_prefix("::")?;
    (!place.is_empty() && !place.contains("::")).then_some(place)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Answer {
    pub name: String,
    pub constant: bool,
    pub parameters: Vec<(String, String)>,
    pub returns: Option<String>,
}

impl Answer {
    pub fn signature(&self) -> String {
        let returns = self.returns.as_deref().unwrap_or_default();
        if self.constant {
            return format!("const {}: {returns}", self.name);
        }
        let parameters = self.spelled_parameters();
        match &self.returns {
            Some(returns) => format!("fn {}({parameters}) -> {returns}", self.name),
            None => format!("fn {}({parameters})", self.name),
        }
    }

    fn spelled_parameters(&self) -> String {
        self.parameters
            .iter()
            .map(|(name, ty)| format!("{name}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn arguments(&self) -> String {
        self.parameters
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Contract {
    pub imports: Vec<String>,
    pub types: Vec<String>,
    pub shared: Vec<Answer>,
    pub answers: Vec<Answer>,
}

impl Contract {
    pub fn names(&self) -> Vec<&str> {
        self.types
            .iter()
            .map(String::as_str)
            .chain(self.shared.iter().map(|answer| answer.name.as_str()))
            .chain(self.answers.iter().map(|answer| answer.name.as_str()))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answered {
    Filler {
        brick: &'static str,
        module: &'static str,
    },
    Nobody,
    Members(Vec<(&'static str, Member)>),
}

#[derive(Clone, Debug)]
pub struct LinkedPlace {
    pub place: Place,
    pub contract: Contract,
    pub answered: Option<Answered>,
    pub source: Option<String>,
}

fn parsed(source: &str) -> Result<TranslationUnit, String> {
    source
        .parse::<TranslationUnit>()
        .map_err(|error| error.to_string())
}

fn words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|letter: char| !(letter.is_ascii_alphanumeric() || letter == '_'))
        .filter(|word| !word.is_empty())
}

pub fn contract(source: &str) -> Result<Contract, String> {
    let unit = parsed(source)?;
    let mut contract = Contract {
        imports: unit.imports.iter().map(ToString::to_string).collect(),
        ..Contract::default()
    };
    let mut fields: Vec<String> = Vec::new();
    let mut constants: Vec<Answer> = Vec::new();
    for declaration in &unit.global_declarations {
        match declaration.node() {
            GlobalDeclaration::Struct(item) => {
                contract.types.push(item.ident.to_string());
                fields.extend(item.members.iter().map(|member| member.ty.to_string()));
            }
            GlobalDeclaration::TypeAlias(item) => {
                contract.types.push(item.ident.to_string());
                fields.push(item.ty.to_string());
            }
            GlobalDeclaration::Declaration(item) => constants.push(Answer {
                name: item.ident.to_string(),
                constant: true,
                parameters: Vec::new(),
                returns: item.ty.as_ref().map(ToString::to_string),
            }),
            GlobalDeclaration::Function(item) => contract.answers.push(Answer {
                name: item.ident.to_string(),
                constant: false,
                parameters: item
                    .parameters
                    .iter()
                    .map(|parameter| (parameter.ident.to_string(), parameter.ty.to_string()))
                    .collect(),
                returns: item.return_type.as_ref().map(ToString::to_string),
            }),
            GlobalDeclaration::Void
            | GlobalDeclaration::ConstAssert(_)
            | GlobalDeclaration::Compound(_) => {}
        }
    }
    for constant in constants {
        let typed = fields
            .iter()
            .any(|field| words(field).any(|word| word == constant.name));
        if typed {
            contract.shared.push(constant);
        } else {
            contract.answers.push(constant);
        }
    }
    Ok(contract)
}

pub fn contract_problems(
    source: &str,
    kind: PlaceKind,
    core: &dyn Fn(&str) -> bool,
) -> Vec<String> {
    let unit = match parsed(source) {
        Ok(unit) => unit,
        Err(error) => return vec![error],
    };
    let mut problems = Vec::new();
    for import in &unit.imports {
        for module in imported_modules(import) {
            if place_of(&module).is_none() && !core(&module) {
                problems.push(format!(
                    "imports {module}: a contract imports only core modules and contracts"
                ));
            }
        }
    }
    for declaration in &unit.global_declarations {
        match declaration.node() {
            GlobalDeclaration::Void
            | GlobalDeclaration::Struct(_)
            | GlobalDeclaration::TypeAlias(_) => {}
            GlobalDeclaration::Declaration(item) => {
                let constant = matches!(item.kind, DeclarationKind::Const)
                    && item.ty.is_some()
                    && item
                        .initializer
                        .as_ref()
                        .is_some_and(|value| constant_value(value));
                if !constant {
                    problems.push(format!(
                        "{}: a contract declares only typed constants of literal values",
                        item.ident
                    ));
                }
            }
            GlobalDeclaration::Function(item) => problems.extend(function_problems(item, kind)),
            GlobalDeclaration::ConstAssert(_) | GlobalDeclaration::Compound(_) => problems
                .push("a contract holds only structs, aliases, constants and functions".to_owned()),
        }
    }
    problems
}

fn imported_modules(import: &ImportStatement) -> Vec<String> {
    let mut modules = Vec::new();
    let Some(path) = &import.path else {
        return vec![import.to_string()];
    };
    if !matches!(path.origin, PathOrigin::Absolute) {
        return vec![import.to_string()];
    }
    gather_modules(&path.components, &import.content, &mut modules);
    modules
}

fn gather_modules(base: &[String], content: &ImportContent, modules: &mut Vec<String>) {
    match content {
        ImportContent::Item(_) => {
            let module = base.join("::");
            if !modules.contains(&module) {
                modules.push(module);
            }
        }
        ImportContent::Collection(imports) => {
            for import in imports {
                let mut path = base.to_vec();
                path.extend(import.path.iter().cloned());
                gather_modules(&path, &import.content, modules);
            }
        }
    }
}

fn function_problems(item: &Function, kind: PlaceKind) -> Vec<String> {
    let name = item.ident.to_string();
    let mut problems = Vec::new();
    let entry = item.attributes.iter().any(|attribute| {
        matches!(
            attribute.node(),
            Attribute::Vertex | Attribute::Fragment | Attribute::Compute
        )
    });
    if entry {
        problems.push(format!("{name}: a contract has no entry point"));
    }
    let statements = &item.body.statements;
    match kind {
        PlaceKind::List => {
            let identity = item.parameters.first().is_some_and(|first| {
                let folded = first.ident.to_string();
                item.return_type
                    .as_ref()
                    .is_some_and(|returns| returns.to_string() == first.ty.to_string())
                    && statements.len() == 1
                    && returned(statements[0].node()).is_some_and(|value| {
                        named(value).is_some_and(|returned| returned == folded)
                    })
            });
            if !identity {
                problems.push(format!(
                    "{name}: a list's function takes the folded value first, returns its type and \
                     returns it unchanged"
                ));
            }
        }
        PlaceKind::Slot => {
            let mut locals: Vec<String> = Vec::new();
            for (index, statement) in statements.iter().enumerate() {
                let last = index + 1 == statements.len();
                if !plain_statement(statement.node(), last, &mut locals) {
                    problems.push(format!(
                        "{name}: a contract body declares locals, assigns their fields and returns \
                         literals, parameters, constants or constructors (statement {index})"
                    ));
                }
            }
            let ends = statements
                .last()
                .is_some_and(|statement| returned(statement.node()).is_some());
            if item.return_type.is_some() && !ends {
                problems.push(format!("{name}: a contract body ends with its return"));
            }
        }
    }
    problems
}

fn returned(statement: &Statement) -> Option<&Expression> {
    if let Statement::Return(returned) = statement {
        return returned.expression.as_ref().map(|value| value.node());
    }
    None
}

fn plain_statement(statement: &Statement, last: bool, locals: &mut Vec<String>) -> bool {
    if let Statement::Return(returned) = statement {
        return last
            && returned
                .expression
                .as_ref()
                .is_none_or(|value| constant_value(value));
    }
    if let Statement::Declaration(local) = statement {
        locals.push(local.ident.to_string());
        return !last
            && matches!(local.kind, DeclarationKind::Var(_))
            && local
                .initializer
                .as_ref()
                .is_none_or(|value| constant_value(value));
    }
    if let Statement::Assignment(assigned) = statement {
        return !last
            && matches!(assigned.operator, AssignmentOperator::Equal)
            && local_field(&assigned.lhs, locals)
            && constant_value(&assigned.rhs);
    }
    false
}

fn local_field(expression: &Expression, locals: &[String]) -> bool {
    if let Expression::NamedComponent(component) = expression {
        let base: &Expression = &component.base;
        return named(base).is_some_and(|name| locals.contains(&name)) || local_field(base, locals);
    }
    false
}

fn named(expression: &Expression) -> Option<String> {
    if let Expression::TypeOrIdentifier(ty) = expression
        && ty.path.is_none()
        && ty.template_args.is_none()
    {
        return Some(ty.ident.to_string());
    }
    None
}

fn constructed(ty: &TypeExpression) -> bool {
    let name = ty.ident.to_string();
    CONSTRUCTED.contains(&name.as_str())
        || name == "array"
        || name.chars().next().is_some_and(char::is_uppercase)
}

fn constant_value(expression: &Expression) -> bool {
    if let Expression::Literal(_) | Expression::TypeOrIdentifier(_) = expression {
        return true;
    }
    if let Expression::Parenthesized(inner) = expression {
        return constant_value(&inner.expression);
    }
    if let Expression::Unary(unary) = expression {
        return matches!(unary.operator, UnaryOperator::Negation)
            && matches!(unary.operand.node(), Expression::Literal(_));
    }
    if let Expression::FunctionCall(call) = expression {
        return constructed(&call.ty)
            && call
                .arguments
                .iter()
                .all(|argument| constant_value(argument));
    }
    false
}

fn shared_imports(imports: &mut String, body: &mut String, module: &str, contract: &Contract) {
    let names: Vec<&str> = contract
        .types
        .iter()
        .map(String::as_str)
        .chain(contract.shared.iter().map(|answer| answer.name.as_str()))
        .collect();
    match FACE_SHAPE {
        FaceShape::Reexport => {
            if !names.is_empty() {
                let _ = writeln!(
                    imports,
                    "import package::{module}::{{{}}};",
                    names.join(", ")
                );
            }
        }
        FaceShape::Forward => {
            for name in &contract.types {
                let _ = writeln!(imports, "import package::{module}::{name} as place_{name};");
                let _ = writeln!(body, "alias {name} = place_{name};");
            }
            forwarded(imports, body, module, &contract.shared);
        }
    }
}

fn forwarded(imports: &mut String, body: &mut String, module: &str, answers: &[Answer]) {
    for answer in answers {
        let name = &answer.name;
        let _ = writeln!(
            imports,
            "import package::{module}::{name} as answer_{name};"
        );
        if answer.constant {
            let ty = answer.returns.as_deref().unwrap_or_default();
            let _ = writeln!(body, "const {name}: {ty} = answer_{name};");
            continue;
        }
        let arguments = answer.arguments();
        let _ = match &answer.returns {
            Some(_) => writeln!(
                body,
                "\n{} {{\n    return answer_{name}({arguments});\n}}",
                answer.signature()
            ),
            None => writeln!(
                body,
                "\n{} {{\n    answer_{name}({arguments});\n}}",
                answer.signature()
            ),
        };
    }
}

fn answers_from(imports: &mut String, body: &mut String, module: &str, answers: &[Answer]) {
    match FACE_SHAPE {
        FaceShape::Reexport => {
            let names: Vec<&str> = answers.iter().map(|answer| answer.name.as_str()).collect();
            if !names.is_empty() {
                let _ = writeln!(
                    imports,
                    "import package::{module}::{{{}}};",
                    names.join(", ")
                );
            }
        }
        FaceShape::Forward => forwarded(imports, body, module, answers),
    }
}

fn folds(
    imports: &mut String,
    body: &mut String,
    contract: &Contract,
    members: &[(&'static str, Member)],
) {
    for answer in contract.answers.iter().filter(|answer| !answer.constant) {
        let Some((folded, _)) = answer.parameters.first() else {
            continue;
        };
        let name = &answer.name;
        let fed: Vec<&'static str> = members
            .iter()
            .filter(|(_, member)| member.implements(name))
            .map(|(_, member)| member.module)
            .collect();
        for (index, module) in fed.iter().enumerate() {
            let _ = writeln!(
                imports,
                "import package::{module}::{name} as {name}_{index};"
            );
        }
        let rest: String = answer
            .parameters
            .iter()
            .skip(1)
            .map(|(parameter, _)| format!(", {parameter}"))
            .collect();
        let _ = writeln!(body, "\n{} {{", answer.signature());
        let _ = writeln!(body, "    var folded = {folded};");
        for index in 0..fed.len() {
            let _ = writeln!(body, "    folded = {name}_{index}(folded{rest});");
        }
        let _ = writeln!(body, "    return folded;\n}}");
    }
}

pub fn face_source(place: &str, contract: &Contract, answered: &Answered) -> String {
    let module = place_module(place);
    let mut imports = String::new();
    let mut body = String::new();
    match answered {
        Answered::Filler { module: filler, .. } => {
            shared_imports(&mut imports, &mut body, &module, contract);
            answers_from(&mut imports, &mut body, filler, &contract.answers);
        }
        Answered::Nobody => {
            shared_imports(&mut imports, &mut body, &module, contract);
            answers_from(&mut imports, &mut body, &module, &contract.answers);
        }
        Answered::Members(members) => {
            for import in &contract.imports {
                let _ = writeln!(imports, "{import}");
            }
            shared_imports(&mut imports, &mut body, &module, contract);
            folds(&mut imports, &mut body, contract, members);
        }
    }
    format!("{imports}{body}")
}

pub fn places_of(registry: &BrickRegistry) -> Result<Vec<LinkedPlace>, Error> {
    let places = registry.places();
    let mut linked = Vec::new();
    for place in places {
        let Some(source) = place.face.wgsl else {
            continue;
        };
        let contract = contract(source).map_err(|error| {
            Error::new(ErrorCode::ShaderLink, format!("$.places.{}", place.name)).got(error)
        })?;
        let answered = match place.kind {
            PlaceKind::Slot => {
                let filled = registry
                    .filled(place.name)
                    .and_then(|(brick, filled)| Some((brick, filled.wgsl?)));
                match filled {
                    Some((brick, module)) => Some(Answered::Filler { brick, module }),
                    None => (place.face.absent == Absent::Nobody).then_some(Answered::Nobody),
                }
            }
            PlaceKind::List => Some(Answered::Members(
                registry
                    .joined(place.name)
                    .into_iter()
                    .filter_map(|(brick, joined)| Some((brick, joined.wgsl?)))
                    .collect(),
            )),
        };
        let source = answered
            .as_ref()
            .map(|answered| face_source(place.name, &contract, answered));
        linked.push(LinkedPlace {
            place,
            contract,
            answered,
            source,
        });
    }
    Ok(linked)
}
