use std::collections::BTreeSet;
use std::path::Path;

use serde_json::{Value, json};

macro_rules! section_names {
    ($($key:ident: $settings:ident { opt_in: $opt_in:literal, feature: $feature:ident }),* $(,)?) => {
        &[$((stringify!($key), stringify!($settings))),*]
    };
}

pub fn environment_sections() -> &'static [(&'static str, &'static str)] {
    engine::environment_sections!(section_names)
}

pub fn coverage(
    root: &Path,
    development: &engine_dev::schemas::Generated,
) -> Result<String, String> {
    let (protocol, stats) = (&development.summary, &development.frame_stats);
    let sections = environment_sections();
    let smokes: Vec<(String, Claims)> = crate::smoke::scripts(root)?
        .into_iter()
        .map(|(name, source)| (name, Claims::of(&source)))
        .collect();
    let naming = |kind: fn(&Claims) -> &BTreeSet<String>, word: &str| -> Vec<String> {
        smokes
            .iter()
            .filter(|(_, claims)| kind(claims).contains(word))
            .map(|(name, _)| name.clone())
            .collect()
    };
    let reported = |key: &str| stats["properties"].get(key).is_some();
    let mut engine = engine::Engine::default();
    let tools = engine_dev::DevTools::with_schemas(&development.inspectors);
    let described: Value = serde_json::from_str(&tools.execute(
        &mut engine,
        engine_dev::Context::default(),
        &json!({"version": engine_dev::PROTOCOL_VERSION, "op": "describe"}).to_string(),
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
    let ops: Vec<Value> = ["queries", "controls"]
        .iter()
        .flat_map(|kind| protocol[kind].as_array().cloned().unwrap_or_default())
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

pub fn check(root: &Path) -> Result<(), String> {
    let development = engine_dev::schemas::generated()?;
    let coverage: Value =
        serde_json::from_str(&coverage(root, &development)?).expect("coverage is JSON");
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
    Err(format!(
        "no smoke (web/scripts/*-smoke.mjs) checks {}: add one that does",
        unchecked.join(", ")
    ))
}
#[derive(Debug, Default)]
pub struct Claims {
    pub ops: BTreeSet<String>,
    pub inspectors: BTreeSet<String>,
    pub properties: BTreeSet<String>,
}

impl Claims {
    pub fn of(source: &str) -> Self {
        let code: Vec<char> = stripped(source, true).chars().collect();
        let bare: Vec<char> = stripped(source, false).chars().collect();
        assert_eq!(code.len(), bare.len(), "strippings keep positions");
        let mut claims = Self::default();
        let mut open: Option<(usize, char)> = None;
        for (at, &c) in bare.iter().enumerate() {
            if c != '"' && c != '\'' {
                continue;
            }
            match open {
                Some((start, quote)) if quote == c => {
                    open = None;
                    let value: String = code[start + 1..at].iter().collect();
                    let field = field_before(&bare, start);
                    if field == Some("op") || first_argument(&bare, start) {
                        claims.ops.insert(value.clone());
                    }
                    let inspecting = enclosing_call(&bare, start)
                        .is_some_and(|name| name.starts_with("inspect"));
                    if field == Some("component") || inspecting {
                        claims.inspectors.insert(value);
                    }
                }
                Some(_) => {}
                None => open = Some((at, c)),
            }
        }
        let mut at = 0;
        while at < bare.len() {
            if !is_name(bare[at]) {
                at += 1;
                continue;
            }
            let end = (at..bare.len())
                .find(|&end| !is_name(bare[end]))
                .unwrap_or(bare.len());
            let before = previous(&bare, at).map(|(_, c)| c);
            let after = (end..bare.len()).find(|&next| !bare[next].is_whitespace());
            let after_char = after.map(|next| bare[next]);
            let key =
                after_char == Some(':') && after.and_then(|next| bare.get(next + 1)) != Some(&':');
            let member = before == Some('.');
            let shorthand =
                matches!(before, Some('{' | ',')) && matches!(after_char, Some('}' | ','));
            if (key || member || shorthand) && !bare[at].is_ascii_digit() {
                claims.properties.insert(bare[at..end].iter().collect());
            }
            at = end;
        }
        claims
    }
}

fn is_name(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '$'
}

fn previous(code: &[char], at: usize) -> Option<(usize, char)> {
    (0..at)
        .rev()
        .find(|&before| !code[before].is_whitespace())
        .map(|before| (before, code[before]))
}

fn name_before(code: &[char], end: usize) -> Option<(usize, String)> {
    let (last, c) = previous(code, end)?;
    if !is_name(c) {
        return None;
    }
    let start = (0..=last)
        .rev()
        .take_while(|&at| is_name(code[at]))
        .last()?;
    Some((start, code[start..=last].iter().collect()))
}

fn field_before(code: &[char], at: usize) -> Option<&'static str> {
    let (colon, c) = previous(code, at)?;
    if c != ':' {
        return None;
    }
    let (_, name) = name_before(code, colon)?;
    ["op", "component"].into_iter().find(|field| *field == name)
}

fn first_argument(code: &[char], at: usize) -> bool {
    match previous(code, at) {
        Some((paren, '(')) => plain_call(code, paren).is_some(),
        _ => false,
    }
}

fn plain_call(code: &[char], paren: usize) -> Option<String> {
    const KEYWORDS: [&str; 9] = [
        "if", "while", "for", "switch", "return", "catch", "function", "typeof", "await",
    ];
    let (start, name) = name_before(code, paren)?;
    let method = previous(code, start).is_some_and(|(_, c)| c == '.');
    (!method && !KEYWORDS.contains(&name.as_str())).then_some(name)
}

fn enclosing_call(code: &[char], at: usize) -> Option<String> {
    let mut depth = 0usize;
    for before in (0..at).rev() {
        match code[before] {
            ')' | ']' | '}' => depth += 1,
            '[' | '{' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
            }
            '(' => {
                if depth == 0 {
                    return plain_call(code, before);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

fn stripped(source: &str, keep: bool) -> String {
    let chars: Vec<char> = source.chars().collect();
    let mut out = String::with_capacity(source.len());
    let text = |c: char| if keep { c } else { ' ' };
    let mut holes: Vec<usize> = Vec::new();
    let mut depth = 0usize;
    let mut at = 0;
    let mut template = false;
    while at < chars.len() {
        let c = chars[at];
        let next = chars.get(at + 1).copied();
        if template {
            at += 1;
            match c {
                '\\' => {
                    out.push(text(c));
                    if let Some(escaped) = next {
                        out.push(text(escaped));
                        at += 1;
                    }
                }
                '`' => {
                    out.push(c);
                    template = false;
                }
                '$' if next == Some('{') => {
                    out.push_str("${");
                    at += 1;
                    holes.push(depth);
                    depth += 1;
                    template = false;
                }
                _ => out.push(text(c)),
            }
            continue;
        }
        match c {
            '/' if next == Some('/') => {
                while at < chars.len() && chars[at] != '\n' {
                    at += 1;
                }
                out.push(' ');
            }
            '/' if next == Some('*') => {
                at += 2;
                while at < chars.len() && !(chars[at] == '*' && chars.get(at + 1) == Some(&'/')) {
                    at += 1;
                }
                at += 2;
                out.push(' ');
            }
            '"' | '\'' => {
                out.push(c);
                at += 1;
                while at < chars.len() && chars[at] != c && chars[at] != '\n' {
                    if chars[at] == '\\' && at + 1 < chars.len() {
                        out.push(text(chars[at]));
                        at += 1;
                    }
                    out.push(text(chars[at]));
                    at += 1;
                }
                if at < chars.len() && chars[at] == c {
                    out.push(c);
                    at += 1;
                }
            }
            '`' => {
                out.push(c);
                at += 1;
                template = true;
            }
            '/' if regex_starts(&out) => {
                out.push(c);
                at += 1;
                let mut class = false;
                while at < chars.len() && chars[at] != '\n' {
                    let inside = chars[at];
                    at += 1;
                    match inside {
                        '\\' if at < chars.len() => {
                            out.push(text(inside));
                            out.push(text(chars[at]));
                            at += 1;
                        }
                        '/' if !class => {
                            out.push(inside);
                            break;
                        }
                        _ => {
                            class = match inside {
                                '[' => true,
                                ']' => false,
                                _ => class,
                            };
                            out.push(text(inside));
                        }
                    }
                }
            }
            '{' => {
                depth += 1;
                out.push(c);
                at += 1;
            }
            '}' => {
                depth = depth.saturating_sub(1);
                out.push(c);
                at += 1;
                if holes.last() == Some(&depth) {
                    holes.pop();
                    template = true;
                }
            }
            _ => {
                out.push(c);
                at += 1;
            }
        }
    }
    out
}

fn regex_starts(before: &str) -> bool {
    let trimmed = before.trim_end();
    let Some(last) = trimmed.chars().next_back() else {
        return true;
    };
    if last.is_alphanumeric() || last == '_' || last == '$' {
        let word: String = trimmed
            .chars()
            .rev()
            .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        return matches!(
            word.as_str(),
            "return"
                | "typeof"
                | "case"
                | "of"
                | "in"
                | "void"
                | "yield"
                | "await"
                | "delete"
                | "throw"
                | "new"
                | "else"
                | "do"
                | "instanceof"
        );
    }
    !matches!(last, ')' | ']' | '}' | '"' | '\'' | '`')
}
