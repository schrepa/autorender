use std::fs;
use std::path::Path;

use crate::techniques::{Located, Token, depth_change, files, mark, tokens, word};

const DECLARATIONS: &[&str] = &["KindDecl", "KindForm"];

const REGISTRATIONS: &[&str] = &["draw_kind", "draws"];

const UNBUILT_AFTER: &[&str] = &[
    "struct", "enum", "union", "trait", "impl", "for", "type", "let",
];

struct Declared {
    line: usize,
    kind: String,
}

pub fn check(
    label: &str,
    source: &Path,
    shown: impl Fn(&Path) -> String,
    problems: &mut Vec<String>,
) -> Result<(), String> {
    if !source.is_dir() {
        return Ok(());
    }
    let mut found = Vec::new();
    files(source, &mut found)?;
    for path in found
        .iter()
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
    {
        let text =
            fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
        for declared in declarations(&tokens(&text)) {
            problems.push(format!(
                "{label}: {}:{} declares the draw kind {}; a core crate declares no draw kind",
                shown(path),
                declared.line,
                declared.kind
            ));
        }
    }
    Ok(())
}

fn declarations(tokens: &[Located]) -> Vec<Declared> {
    (0..tokens.len())
        .filter_map(|at| built(tokens, at).or_else(|| registered(tokens, at)))
        .collect()
}

fn built(tokens: &[Located], at: usize) -> Option<Declared> {
    let name = word(tokens, at)?;
    if !DECLARATIONS.contains(&name) || !mark(tokens, at + 1, '{') {
        return None;
    }
    let close = closing(tokens, at + 1)?;
    if typed(tokens, path_start(tokens, at)) || pattern(tokens, close) {
        return None;
    }
    Some(Declared {
        line: tokens[at].line,
        kind: field(tokens, at + 1, close, "kind").unwrap_or_else(|| format!("of this {name}")),
    })
}

fn registered(tokens: &[Located], at: usize) -> Option<Declared> {
    let name = word(tokens, at)?;
    let called = mark(tokens, at.wrapping_sub(1), '.')
        || (mark(tokens, at.wrapping_sub(1), ':') && mark(tokens, at.wrapping_sub(2), ':'));
    if !REGISTRATIONS.contains(&name) || !called {
        return None;
    }
    let open = call_open(tokens, at + 1)?;
    if mark(tokens, open + 1, ')') || builds_declaration(tokens, open + 1) {
        return None;
    }
    Some(Declared {
        line: tokens[at].line,
        kind: argument(tokens, open + 1, tokens.len()),
    })
}

fn path_start(tokens: &[Located], mut at: usize) -> usize {
    while at >= 3
        && mark(tokens, at - 1, ':')
        && mark(tokens, at - 2, ':')
        && word(tokens, at - 3).is_some()
    {
        at -= 3;
    }
    while at >= 1 && (mark(tokens, at - 1, '&') || word(tokens, at - 1) == Some("mut")) {
        at -= 1;
    }
    at
}

fn typed(tokens: &[Located], start: usize) -> bool {
    let Some(before) = start.checked_sub(1) else {
        return false;
    };
    word(tokens, before).is_some_and(|word| UNBUILT_AFTER.contains(&word))
        || (mark(tokens, before, '>') && !mark(tokens, before.wrapping_sub(1), '='))
}

fn pattern(tokens: &[Located], close: usize) -> bool {
    if mark(tokens, close - 1, '.') && mark(tokens, close - 2, '.') {
        return true;
    }
    let mut after = close + 1;
    while mark(tokens, after, ')') || mark(tokens, after, ']') {
        after += 1;
    }
    let assigned = mark(tokens, after, '=') && !mark(tokens, after + 1, '=');
    assigned
        || mark(tokens, after, '|')
        || mark(tokens, after, ':')
        || matches!(word(tokens, after), Some("if" | "in"))
}

fn closing(tokens: &[Located], open: usize) -> Option<usize> {
    let mut depth = 0_isize;
    for (index, located) in tokens.iter().enumerate().skip(open) {
        depth += depth_change(&located.token);
        if depth == 0 {
            return Some(index);
        }
    }
    None
}

fn field(tokens: &[Located], open: usize, close: usize, name: &str) -> Option<String> {
    let mut depth = 0_isize;
    for at in open..close {
        depth += depth_change(&tokens[at].token);
        let starts = depth == 1
            && word(tokens, at) == Some(name)
            && (mark(tokens, at - 1, '{') || mark(tokens, at - 1, ','));
        if !starts {
            continue;
        }
        if mark(tokens, at + 1, ':') && !mark(tokens, at + 2, ':') {
            return Some(argument(tokens, at + 2, close));
        }
        return Some(name.to_owned());
    }
    None
}

fn call_open(tokens: &[Located], at: usize) -> Option<usize> {
    if mark(tokens, at, '(') {
        return Some(at);
    }
    if !(mark(tokens, at, ':') && mark(tokens, at + 1, ':') && mark(tokens, at + 2, '<')) {
        return None;
    }
    let mut angle = 0_isize;
    for index in at + 2..tokens.len() {
        match tokens[index].token {
            Token::Mark('<') => angle += 1,
            Token::Mark('>') if !mark(tokens, index - 1, '-') => angle -= 1,
            Token::Mark(_) | Token::Word(_) | Token::Text(_) => {}
        }
        if angle == 0 {
            return mark(tokens, index + 1, '(').then_some(index + 1);
        }
    }
    None
}

fn builds_declaration(tokens: &[Located], mut at: usize) -> bool {
    while word(tokens, at).is_some() && mark(tokens, at + 1, ':') && mark(tokens, at + 2, ':') {
        at += 3;
    }
    word(tokens, at).is_some_and(|name| DECLARATIONS.contains(&name)) && mark(tokens, at + 1, '{')
}

fn argument(tokens: &[Located], from: usize, end: usize) -> String {
    let mut depth = 0_isize;
    let mut text = String::new();
    let mut after_word = false;
    for located in tokens.get(from..end).unwrap_or_default() {
        depth += depth_change(&located.token);
        if depth < 0 || (depth == 0 && located.token == Token::Mark(',')) {
            break;
        }
        let is_word = matches!(located.token, Token::Word(_));
        if is_word && after_word {
            text.push(' ');
        }
        match &located.token {
            Token::Word(name) => text.push_str(name),
            Token::Text(quoted) => {
                text.push('"');
                text.push_str(quoted);
                text.push('"');
            }
            Token::Mark(character) => text.push(*character),
        }
        after_word = is_word;
    }
    text
}
