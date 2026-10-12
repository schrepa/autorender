//! What the smokes check of each engine feature: every environment
//! section, inspector and development op, whether frame stats report the
//! section, and which smokes name each in their code (a comment naming one
//! checks nothing). `cargo xtask schema` writes it as
//! web/src/generated/coverage.json; lint keeps it current, so a gap shows
//! where agents and reviewers look.

use std::path::Path;

use serde_json::{Value, json};

/// Each environment section's key and settings type, in authored order.
macro_rules! section_names {
    ($($key:ident: $settings:ident { since: $since:literal, opt_in: $opt_in:literal, feature: $feature:ident }),* $(,)?) => {
        &[$((stringify!($key), stringify!($settings))),*]
    };
}

pub fn coverage(root: &Path) -> Result<String, String> {
    let sections: &[(&str, &str)] = engine::environment_sections!(section_names);
    // What a smoke's comments mention it does not check.
    let smokes: Vec<(String, String)> = crate::smoke::scripts(root)?
        .into_iter()
        .map(|(name, _, source)| (name, uncommented(&source)))
        .collect();
    // Smokes naming `word` whole, or `"word"` quoted.
    let naming = |word: &str, quoted: bool| -> Vec<String> {
        smokes
            .iter()
            .filter(|(_, text)| {
                if quoted {
                    return text.contains(&format!("\"{word}\""));
                }
                text.match_indices(word).any(|(at, _)| {
                    let before = text[..at].chars().next_back();
                    let after = text[at + word.len()..].chars().next();
                    let part = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
                    !part(before) && !part(after)
                })
            })
            .map(|(name, _)| name.clone())
            .collect()
    };
    let stats: Value =
        serde_json::from_str(engine_dev::FRAME_STATS_SCHEMA).expect("generated schema is JSON");
    let reported = |key: &str| stats["properties"].get(key).is_some();
    // The protocol's own description of its types.
    let mut engine = engine::Engine::default();
    let tools = engine_dev::DevTools::default();
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
            json!({"inspector": name, "kind": ty["kind"], "smokes": naming(name, true)})
        })
        .collect();
    let features: Vec<Value> = sections
        .iter()
        .map(|&(key, name)| {
            json!({
                "section": key,
                "inspector": name,
                "stats": reported(key),
                "smokes": naming(key, false),
            })
        })
        .collect();
    let protocol: Value =
        serde_json::from_str(engine_dev::PROTOCOL_SUMMARY).expect("the summary is JSON");
    let ops: Vec<Value> = ["queries", "controls"]
        .iter()
        .flat_map(|kind| protocol[kind].as_array().cloned().unwrap_or_default())
        .map(|op| {
            let op = op.as_str().unwrap_or_default().to_owned();
            json!({"op": op, "smokes": naming(&op, true)})
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

/// Fails unless some smoke checks every inspector, environment section and
/// protocol op: what the engine offers agents is exercised.
pub fn check(root: &Path) -> Result<(), String> {
    let coverage: Value = serde_json::from_str(&coverage(root)?).expect("coverage is JSON");
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
/// JavaScript `source` without its comments: strings, template literals
/// (their `${}` holes included) and regular expressions stay as written,
/// and each comment leaves a space (a line comment its line's end).
fn uncommented(source: &str) -> String {
    let chars: Vec<char> = source.chars().collect();
    let mut out = String::with_capacity(source.len());
    // For each open template literal hole, the brace depth it opened at.
    let mut holes: Vec<usize> = Vec::new();
    let mut depth = 0usize;
    let mut at = 0;
    // Inside a template literal's text (not in one of its holes).
    let mut template = false;
    while at < chars.len() {
        let c = chars[at];
        let next = chars.get(at + 1).copied();
        if template {
            out.push(c);
            at += 1;
            match c {
                '\\' => {
                    if let Some(escaped) = next {
                        out.push(escaped);
                        at += 1;
                    }
                }
                '`' => template = false,
                '$' if next == Some('{') => {
                    out.push('{');
                    at += 1;
                    holes.push(depth);
                    depth += 1;
                    template = false;
                }
                _ => {}
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
                        out.push(chars[at]);
                        at += 1;
                    }
                    out.push(chars[at]);
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
                // A regular expression: to its closing slash outside a
                // character class.
                out.push(c);
                at += 1;
                let mut class = false;
                while at < chars.len() && chars[at] != '\n' {
                    let inside = chars[at];
                    out.push(inside);
                    at += 1;
                    match inside {
                        '\\' if at < chars.len() => {
                            out.push(chars[at]);
                            at += 1;
                        }
                        '[' => class = true,
                        ']' => class = false,
                        '/' if !class => break,
                        _ => {}
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

/// Whether a `/` after `before` (code so far) starts a regular expression
/// rather than dividing: after an operator, an opening bracket, or a
/// keyword such as `return`; not after a value.
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
