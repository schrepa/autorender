use std::fmt::Write as _;

use engine_core::{Error, ErrorCode};
use engine_place::PlaceKind;
use wesl::syntax::{
    AssignmentOperator, Attribute, DeclarationKind, Expression, Function, GlobalDeclaration,
    ImportContent, ImportStatement, PathOrigin, Statement, TranslationUnit, TypeExpression,
    UnaryOperator,
};

pub const PLACE_ROOT: &str = "place";
pub const FACE_ROOT: &str = "face";

const CONSTRUCTED: [&str; 17] = [
    "bool", "i32", "u32", "f32", "f16", "vec2", "vec3", "vec4", "mat2x2", "mat2x3", "mat2x4",
    "mat3x2", "mat3x3", "mat3x4", "mat4x2", "mat4x3", "mat4x4",
];

pub fn place_module(place: &str) -> String {
    format!("{PLACE_ROOT}::{place}")
}

pub fn face_module(place: &str) -> String {
    format!("{FACE_ROOT}::{place}")
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
        let parameters = self
            .parameters
            .iter()
            .map(|(name, ty)| format!("{name}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        match &self.returns {
            Some(returns) => format!("fn {}({parameters}) -> {returns}", self.name),
            None => format!("fn {}({parameters})", self.name),
        }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fed {
    pub module: String,
    pub switch: Option<u32>,
    pub implements: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answered {
    Filler(Fed),
    Nobody,
    Members(Vec<Fed>),
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
        PlaceKind::Slot | PlaceKind::Component => {
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

fn shared_imports(imports: &mut String, module: &str, contract: &Contract) {
    let names: Vec<&str> = contract
        .types
        .iter()
        .map(String::as_str)
        .chain(contract.shared.iter().map(|answer| answer.name.as_str()))
        .collect();
    if !names.is_empty() {
        let _ = writeln!(imports, "import package::{module}::{{{}}};", names.join(", "));
    }
}

fn switch(body: &mut String, id: u32) -> String {
    let name = format!("switch_{id}");
    let _ = writeln!(body, "@id({id}) override {name}: bool = true;");
    name
}

fn switched(imports: &mut String, body: &mut String, place: &str, filler: &Fed, contract: &Contract) {
    let own = |answer: &&Answer| !answer.constant || filler.implements.contains(&answer.name);
    let Some(id) = filler.switch else {
        let names: Vec<&str> = contract.answers.iter().filter(own).map(|answer| answer.name.as_str()).collect();
        if !names.is_empty() {
            let _ = writeln!(imports, "import package::{}::{{{}}};", filler.module, names.join(", "));
        }
        let taken: Vec<&str> = contract
            .answers
            .iter()
            .filter(|answer| !own(answer))
            .map(|answer| answer.name.as_str())
            .collect();
        if !taken.is_empty() {
            let _ = writeln!(imports, "import package::{}::{{{}}};", place_module(place), taken.join(", "));
        }
        return;
    };
    let on = switch(body, id);
    for answer in &contract.answers {
        let name = &answer.name;
        if !own(&answer) {
            let _ = writeln!(imports, "import package::{}::{name};", place_module(place));
            continue;
        }
        let _ = writeln!(imports, "import package::{}::{name} as answer_{name};", filler.module);
        if answer.constant {
            let ty = answer.returns.as_deref().unwrap_or_default();
            let _ = writeln!(body, "const {name}: {ty} = answer_{name};");
            continue;
        }
        let _ = writeln!(imports, "import package::{}::{name} as nobody_{name};", place_module(place));
        let arguments = answer.arguments();
        let _ = match &answer.returns {
            Some(_) => writeln!(
                body,
                "\n{} {{\n    if {on} {{\n        return answer_{name}({arguments});\n    }}\n    return nobody_{name}({arguments});\n}}",
                answer.signature()
            ),
            None => writeln!(
                body,
                "\n{} {{\n    if {on} {{\n        answer_{name}({arguments});\n    }} else {{\n        nobody_{name}({arguments});\n    }}\n}}",
                answer.signature()
            ),
        };
    }
}

fn folds(imports: &mut String, body: &mut String, contract: &Contract, members: &[Fed]) {
    let mut ons: Vec<(usize, String)> = Vec::new();
    for (index, member) in members.iter().enumerate() {
        if let Some(id) = member.switch {
            ons.push((index, switch(body, id)));
        }
    }
    for answer in contract.answers.iter().filter(|answer| !answer.constant) {
        let Some((folded, _)) = answer.parameters.first() else {
            continue;
        };
        let name = &answer.name;
        let fed: Vec<(usize, &Fed)> = members
            .iter()
            .enumerate()
            .filter(|(_, member)| member.implements.contains(name))
            .collect();
        for (index, member) in &fed {
            let _ = writeln!(imports, "import package::{}::{name} as {name}_{index};", member.module);
        }
        let rest: String = answer
            .parameters
            .iter()
            .skip(1)
            .map(|(parameter, _)| format!(", {parameter}"))
            .collect();
        let _ = writeln!(body, "\n{} {{", answer.signature());
        let _ = writeln!(body, "    var folded = {folded};");
        for (index, _) in &fed {
            match ons.iter().find(|(member, _)| member == index) {
                Some((_, on)) => {
                    let _ = writeln!(body, "    if {on} {{\n        folded = {name}_{index}(folded{rest});\n    }}");
                }
                None => {
                    let _ = writeln!(body, "    folded = {name}_{index}(folded{rest});");
                }
            }
        }
        let _ = writeln!(body, "    return folded;\n}}");
    }
}

pub fn face_source(place: &str, contract: &Contract, answered: &Answered) -> String {
    let module = place_module(place);
    let mut imports = String::new();
    let mut body = String::new();
    match answered {
        Answered::Filler(filler) => {
            for import in &contract.imports {
                let _ = writeln!(imports, "{import}");
            }
            shared_imports(&mut imports, &module, contract);
            switched(&mut imports, &mut body, place, filler, contract);
        }
        Answered::Nobody => {
            for import in &contract.imports {
                let _ = writeln!(imports, "{import}");
            }
            shared_imports(&mut imports, &module, contract);
            let names: Vec<&str> = contract.answers.iter().map(|answer| answer.name.as_str()).collect();
            if !names.is_empty() {
                let _ = writeln!(imports, "import package::{module}::{{{}}};", names.join(", "));
            }
        }
        Answered::Members(members) => {
            for import in &contract.imports {
                let _ = writeln!(imports, "{import}");
            }
            shared_imports(&mut imports, &module, contract);
            folds(&mut imports, &mut body, contract, members);
        }
    }
    format!("{imports}{body}")
}

pub fn checked(name: &str, source: &str, kind: PlaceKind, core: &dyn Fn(&str) -> bool) -> Result<Contract, Error> {
    let path = format!("$.places.{name}");
    let problems = contract_problems(source, kind, core);
    if !problems.is_empty() {
        return Err(Error::new(ErrorCode::ShaderLink, path).got(problems));
    }
    contract(source).map_err(|error| Error::new(ErrorCode::ShaderLink, path).got(error))
}
