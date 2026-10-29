use std::collections::BTreeSet;
use std::path::Path;

use engine::Engine;
use engine::bricks::BrickRegistry;
use serde_json::{Value, json};

pub fn coverage(
    engine: &mut Engine,
    development: &engine_dev::schemas::Generated,
    smokes: &[(String, String)],
) -> Result<String, String> {
    let (protocol, stats) = (&development.summary, &development.frame_stats);
    let sections: Vec<(&'static str, &'static str)> = engine
        .app
        .world()
        .get_resource::<BrickRegistry>()
        .into_iter()
        .flat_map(BrickRegistry::sections)
        .map(|section| (section.key, section.name))
        .collect();
    let smokes: Vec<(String, Claims)> = smokes
        .iter()
        .map(|(name, source)| (name.clone(), Claims::of(source)))
        .collect();
    let naming = |kind: fn(&Claims) -> &BTreeSet<String>, word: &str| -> Vec<String> {
        smokes
            .iter()
            .filter(|(_, claims)| kind(claims).contains(word))
            .map(|(name, _)| name.clone())
            .collect()
    };
    let reported = |key: &str| stats["properties"].get(key).is_some();
    let tools = engine_dev::DevTools::with_schemas(&development.inspectors);
    let described: Value = serde_json::from_str(&tools.execute(
        engine,
        engine_dev::Context::default(),
        &engine_dev::request(engine_dev::Verb::Describe, json!({})),
    ))
    .expect("replies are JSON");
    let section_types: Vec<&str> = sections.iter().map(|(_, name)| *name).collect();
    let inspectors: Vec<Value> = described["result"]["types"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|ty| ["component", "resource"].contains(&ty["kind"].as_str().unwrap_or_default()))
        .filter(|ty| !section_types.contains(&ty["name"].as_str().unwrap_or_default()))
        .map(|ty| {
            let name = ty["name"].as_str().unwrap_or_default();
            json!({"inspector": name, "kind": ty["kind"], "smokes": naming(|claims| &claims.inspectors, name)})
        })
        .collect();
    let features: Vec<Value> = sections
        .iter()
        .map(|&(key, name)| {
            json!({
                "section": key,
                "inspector": name,
                "stats": reported(key),
                "smokes": naming(|claims| &claims.properties, key),
            })
        })
        .collect();
    let named = [
        engine_dev::Verb::Query,
        engine_dev::Verb::Control,
        engine_dev::Verb::Check,
    ]
    .map(engine_dev::Verb::name);
    let verbs = protocol["verbs"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|verb| !named.contains(&verb.as_str().unwrap_or_default()));
    let ops: Vec<Value> = verbs
        .chain(
            ["queries", "controls"]
                .iter()
                .flat_map(|kind| protocol[kind].as_array().cloned().unwrap_or_default()),
        )
        .map(|op| {
            let op = op.as_str().unwrap_or_default().to_owned();
            json!({"op": op, "smokes": naming(|claims| &claims.ops, &op)})
        })
        .collect();
    Ok(serde_json::to_string_pretty(&json!({
        "sections": features,
        "inspectors": inspectors,
        "ops": ops,
    }))
    .expect("coverage serializes")
        + "\n")
}

pub fn check(coverage: &str) -> Result<(), String> {
    let coverage: Value = serde_json::from_str(coverage).expect("coverage is JSON");
    let mut unchecked = Vec::new();
    for (list, key) in [
        ("inspectors", "inspector"),
        ("sections", "section"),
        ("ops", "op"),
    ] {
        for entry in coverage[list].as_array().into_iter().flatten() {
            if entry["smokes"].as_array().is_none_or(Vec::is_empty) {
                unchecked.push(format!("{key} {}", entry[key].as_str().unwrap_or_default()));
            }
        }
    }
    if unchecked.is_empty() {
        return Ok(());
    }
    Err(format!("unchecked: {}", unchecked.join(", ")))
}

#[derive(Debug, Default)]
pub struct Claims {
    pub ops: BTreeSet<String>,
    pub inspectors: BTreeSet<String>,
    pub properties: BTreeSet<String>,
}

impl Claims {
    pub fn of(source: &str) -> Self {
        let tokens = tokens(source);
        let mut claims = Self::default();
        for (at, token) in tokens.iter().enumerate() {
            let before = |back: usize| at.checked_sub(back).and_then(|at| tokens.get(at));
            let after = |ahead: usize| tokens.get(at + ahead);
            match token {
                Token::Text(value) => {
                    let keyed = after(1) == Some(&Token::Mark(':'))
                        && after(2) != Some(&Token::Mark(':'));
                    let indexed = before(1) == Some(&Token::Mark('['))
                        && after(1) == Some(&Token::Mark(']'));
                    if keyed || indexed {
                        claims.properties.insert(value.clone());
                    }
                    let field = match (before(2), before(1)) {
                        (Some(Token::Text(field)), Some(Token::Mark(':'))) => Some(field.as_str()),
                        _ => None,
                    };
                    if matches!(field, Some("name" | "op" | "verb")) {
                        claims.ops.insert(value.clone());
                    }
                    if field == Some("component") {
                        claims.inspectors.insert(value.clone());
                    }
                    match enclosing_call(&tokens, at) {
                        Some(call) if call == "request" && before(1) == Some(&Token::Mark('(')) => {
                            claims.ops.insert(value.clone());
                        }
                        Some(call) if call.starts_with("inspect") => {
                            claims.inspectors.insert(value.clone());
                        }
                        Some(call) if call.starts_with("query") || call.starts_with("control") => {
                            claims.ops.insert(value.clone());
                        }
                        Some(_) | None => {}
                    }
                }
                Token::Name(name) => {
                    let member = before(1) == Some(&Token::Mark('.'));
                    let field = after(1) == Some(&Token::Mark(':'))
                        && after(2) != Some(&Token::Mark(':'))
                        && before(1) != Some(&Token::Mark(':'));
                    if (member || field) && !name.starts_with(|c: char| c.is_ascii_digit()) {
                        claims.properties.insert(name.clone());
                    }
                }
                Token::Mark(_) => {}
            }
        }
        claims
    }
}

pub fn smokes(root: &Path) -> Result<Vec<(String, String)>, String> {
    let files: Vec<Source> = crate::files::listed(root, &["crates", "examples"])?
        .into_iter()
        .filter(|path| path.ends_with(".rs") && crate::files::ours(path))
        .filter_map(|path| {
            let (krate, _) = path.split_once("/src/")?;
            let krate = krate.to_owned();
            let text = std::fs::read_to_string(root.join(&path)).ok()?;
            let tokens = tokens(&text);
            Some(Source {
                path,
                krate,
                text,
                tokens,
            })
        })
        .collect();
    let mut smokes: Vec<(String, String)> = Vec::new();
    for file in &files {
        for (name, run) in registrations(&file.tokens) {
            if smokes.iter().any(|(known, _)| *known == name) {
                continue;
            }
            let Some((function, module)) = run.split_last() else {
                continue;
            };
            let defining: Vec<&Source> = files
                .iter()
                .filter(|other| other.krate == file.krate && defines(&other.tokens, function))
                .collect();
            let in_module: Vec<&Source> = defining
                .iter()
                .copied()
                .filter(|other| module.last().is_some_and(|module| other.is_module(module)))
                .collect();
            let chosen = if in_module.is_empty() {
                defining
            } else {
                in_module
            };
            if chosen.is_empty() {
                continue;
            }
            let source = chosen
                .iter()
                .map(|other| other.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            smokes.push((name, source));
        }
    }
    smokes.sort();
    Ok(smokes)
}

struct Source {
    path: String,
    krate: String,
    text: String,
    tokens: Vec<Token>,
}

impl Source {
    fn is_module(&self, module: &str) -> bool {
        let stem = self.path.strip_suffix(".rs").unwrap_or(&self.path);
        stem.ends_with(&format!("/{module}")) || stem.ends_with(&format!("/{module}/mod"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Text(String),
    Name(String),
    Mark(char),
}

fn registrations(tokens: &[Token]) -> Vec<(String, Vec<String>)> {
    let mut found = Vec::new();
    for (at, token) in tokens.iter().enumerate() {
        match token {
            Token::Mark('(') => {
                let (Some(Token::Text(name)), Some(Token::Mark(',')), Some(Token::Name(gpu))) =
                    (tokens.get(at + 1), tokens.get(at + 2), tokens.get(at + 3))
                else {
                    continue;
                };
                if !["true", "false"].contains(&gpu.as_str())
                    || tokens.get(at + 4) != Some(&Token::Mark(','))
                {
                    continue;
                }
                let run = path_at(tokens, at + 5);
                if !run.is_empty() {
                    found.push((name.clone(), run));
                }
            }
            Token::Name(name) if name == "SmokeEntry" => {
                if tokens.get(at + 1) != Some(&Token::Mark('{')) {
                    continue;
                }
                let end = closing(tokens, at + 1);
                let fields = &tokens[at + 2..end];
                let named = fields.windows(3).find_map(|window| match window {
                    [Token::Name(field), Token::Mark(':'), Token::Text(value)] if field == "name" => {
                        Some(value.clone())
                    }
                    _ => None,
                });
                let run = fields.windows(3).enumerate().find_map(|(index, window)| {
                    match window {
                        [Token::Name(field), Token::Mark(':'), Token::Name(_)] if field == "run" => {
                            Some(path_at(fields, index + 2))
                        }
                        _ => None,
                    }
                });
                if let (Some(name), Some(run)) = (named, run) {
                    found.push((name, run));
                }
            }
            Token::Text(_) | Token::Name(_) | Token::Mark(_) => {}
        }
    }
    found
}

fn path_at(tokens: &[Token], mut at: usize) -> Vec<String> {
    let mut path = Vec::new();
    while let Some(Token::Name(segment)) = tokens.get(at) {
        path.push(segment.clone());
        if tokens.get(at + 1) == Some(&Token::Mark(':'))
            && tokens.get(at + 2) == Some(&Token::Mark(':'))
        {
            at += 3;
        } else {
            break;
        }
    }
    path
}

fn closing(tokens: &[Token], open: usize) -> usize {
    let mut depth = 0usize;
    for (at, token) in tokens.iter().enumerate().skip(open) {
        match token {
            Token::Mark('(' | '[' | '{') => depth += 1,
            Token::Mark(')' | ']' | '}') => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return at;
                }
            }
            Token::Text(_) | Token::Name(_) | Token::Mark(_) => {}
        }
    }
    tokens.len()
}

fn defines(tokens: &[Token], function: &str) -> bool {
    tokens.windows(3).enumerate().any(|(at, window)| match window {
        [Token::Name(keyword), Token::Name(name), Token::Mark('(' | '<')]
            if keyword == "fn" && name == function =>
        {
            let end = closing(tokens, at + 2);
            tokens[at + 2..end]
                .iter()
                .any(|token| matches!(token, Token::Name(name) if name == "SmokeSession"))
        }
        _ => false,
    })
}

fn enclosing_call(tokens: &[Token], at: usize) -> Option<&str> {
    const KEYWORDS: [&str; 6] = ["if", "while", "for", "match", "return", "in"];
    let mut depth = 0usize;
    for before in (0..at).rev() {
        match &tokens[before] {
            Token::Mark(')' | ']' | '}') => depth += 1,
            Token::Mark('[' | '{') => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
            }
            Token::Mark('(') => {
                if depth == 0 {
                    return match before.checked_sub(1).and_then(|at| tokens.get(at)) {
                        Some(Token::Name(name)) if !KEYWORDS.contains(&name.as_str()) => {
                            Some(name.as_str())
                        }
                        Some(_) | None => None,
                    };
                }
                depth -= 1;
            }
            Token::Text(_) | Token::Name(_) | Token::Mark(_) => {}
        }
    }
    None
}

fn tokens(source: &str) -> Vec<Token> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut at = 0;
    while let Some(&c) = chars.get(at) {
        let next = chars.get(at + 1).copied();
        if c.is_whitespace() {
            at += 1;
        } else if c == '/' && next == Some('/') {
            while chars.get(at).is_some_and(|&c| c != '\n') {
                at += 1;
            }
        } else if c == '/' && next == Some('*') {
            at = comment_end(&chars, at);
        } else if let Some((text, end)) = literal(&chars, at) {
            tokens.push(Token::Text(text));
            at = end;
        } else if c == '\'' {
            at = quoted_end(&chars, at);
        } else if c.is_alphanumeric() || c == '_' {
            let start = at;
            while chars.get(at).is_some_and(|&c| c.is_alphanumeric() || c == '_') {
                at += 1;
            }
            tokens.push(Token::Name(chars[start..at].iter().collect()));
        } else {
            tokens.push(Token::Mark(c));
            at += 1;
        }
    }
    tokens
}

fn comment_end(chars: &[char], start: usize) -> usize {
    let mut depth = 0usize;
    let mut at = start;
    while at < chars.len() {
        match (chars[at], chars.get(at + 1).copied()) {
            ('/', Some('*')) => {
                depth += 1;
                at += 2;
            }
            ('*', Some('/')) => {
                depth = depth.saturating_sub(1);
                at += 2;
                if depth == 0 {
                    return at;
                }
            }
            _ => at += 1,
        }
    }
    at
}

fn literal(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut at = start;
    if matches!(chars.get(at), Some('b' | 'c')) {
        at += 1;
    }
    let raw = chars.get(at) == Some(&'r');
    if raw {
        at += 1;
    }
    let hashes = chars[at.min(chars.len())..]
        .iter()
        .take_while(|&&c| c == '#')
        .count();
    if hashes > 0 && !raw {
        return None;
    }
    at += hashes;
    if chars.get(at) != Some(&'"') {
        return None;
    }
    at += 1;
    let open = at;
    while at < chars.len() {
        match chars[at] {
            '\\' if !raw => at += 2,
            '"' if chars[at + 1..].iter().take(hashes).filter(|&&c| c == '#').count() == hashes => {
                let text = chars[open..at].iter().collect();
                return Some((text, at + 1 + hashes));
            }
            _ => at += 1,
        }
    }
    Some((chars[open.min(chars.len())..].iter().collect(), chars.len()))
}

fn quoted_end(chars: &[char], start: usize) -> usize {
    match (chars.get(start + 1), chars.get(start + 2)) {
        (Some('\\'), _) => chars
            .get(start + 3..)
            .and_then(|rest| rest.iter().position(|&c| c == '\''))
            .map_or(chars.len(), |offset| start + 4 + offset),
        (Some(_), Some('\'')) => start + 3,
        _ => start + 1,
    }
}
