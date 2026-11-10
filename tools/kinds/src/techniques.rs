use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

pub const FREE_KINDS: &[&str] = &["core", "core-internal"];

const WORDS: &[&str] = &[
    "shadow",
    "probe",
    "roughness",
    "metalness",
    "skin",
    "terrain",
    "water",
];

const ENDINGS: &[&str] = &["", "s", "es", "ed", "ing", "ned", "ning"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Token {
    Word(String),
    Text(String),
    Mark(char),
}

pub(crate) struct Located {
    pub(crate) token: Token,
    pub(crate) line: usize,
}

struct Named<'a> {
    name: &'a str,
    line: usize,
    what: &'static str,
}

pub fn technique(name: &str) -> Option<&'static str> {
    words_of(name).iter().find_map(|word| {
        WORDS.iter().copied().find(|technique| {
            word.strip_prefix(technique)
                .is_some_and(|ending| ENDINGS.contains(&ending))
        })
    })
}

fn words_of(name: &str) -> Vec<String> {
    let mut words = Vec::new();
    for part in name.split(|character: char| !character.is_ascii_alphanumeric()) {
        let mut word = String::new();
        let mut previous: Option<char> = None;
        for character in part.chars() {
            let turns = previous.is_some_and(|previous| {
                (previous.is_ascii_lowercase() && character.is_ascii_uppercase())
                    || (previous.is_ascii_digit() != character.is_ascii_digit())
            });
            if turns && !word.is_empty() {
                words.push(std::mem::take(&mut word));
            }
            word.push(character.to_ascii_lowercase());
            previous = Some(character);
        }
        if !word.is_empty() {
            words.push(word);
        }
    }
    words
}

fn identifier(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_'
}

fn skip_comment(chars: &[char], at: usize, line: &mut usize) -> usize {
    if chars.get(at + 1) == Some(&'/') {
        return chars[at..]
            .iter()
            .position(|&character| character == '\n')
            .map_or(chars.len(), |found| at + found);
    }
    let mut depth = 0_usize;
    let mut index = at;
    while index < chars.len() {
        match (chars[index], chars.get(index + 1).copied()) {
            ('/', Some('*')) => {
                depth += 1;
                index += 2;
            }
            ('*', Some('/')) => {
                depth = depth.saturating_sub(1);
                index += 2;
                if depth == 0 {
                    return index;
                }
            }
            ('\n', _) => {
                *line += 1;
                index += 1;
            }
            _ => index += 1,
        }
    }
    index
}

fn hashes_after(chars: &[char], at: usize) -> usize {
    chars[at + 1..]
        .iter()
        .take_while(|&&character| character == '#')
        .count()
}

fn raw_text(chars: &[char], at: usize) -> bool {
    chars[at] == 'r' && chars.get(at + 1 + hashes_after(chars, at)) == Some(&'"')
}

fn quoted(chars: &[char], at: usize, line: &mut usize) -> (usize, String) {
    let raw = chars[at] == 'r';
    let hashes = if raw { hashes_after(chars, at) } else { 0 };
    let open = if raw { at + 1 + hashes } else { at };
    let mut text = String::new();
    let mut index = open + 1;
    while index < chars.len() {
        let character = chars[index];
        if character == '\n' {
            *line += 1;
        }
        if !raw && character == '\\' {
            index += 2;
            continue;
        }
        let closes = character == '"'
            && chars[index + 1..]
                .iter()
                .take(hashes)
                .filter(|&&hash| hash == '#')
                .count()
                == hashes;
        if closes {
            return (index + 1 + hashes, text);
        }
        text.push(character);
        index += 1;
    }
    (index, text)
}

fn tick(chars: &[char], at: usize) -> usize {
    match (chars.get(at + 1), chars.get(at + 2)) {
        (Some('\\'), _) => chars
            .get(at + 3..)
            .and_then(|rest| rest.iter().position(|&character| character == '\''))
            .map_or(chars.len(), |found| at + 4 + found),
        (Some(_), Some('\'')) => at + 3,
        _ => {
            let length = chars[at + 1..]
                .iter()
                .take_while(|&&character| identifier(character))
                .count();
            at + 1 + length
        }
    }
}

pub(crate) fn tokens(source: &str) -> Vec<Located> {
    let chars: Vec<char> = source.chars().collect();
    let mut found = Vec::new();
    let mut line = 1;
    let mut at = 0;
    while at < chars.len() {
        let character = chars[at];
        let next = chars.get(at + 1).copied();
        let raw = raw_text(&chars, at);
        let bytes = character == 'b' && next == Some('"');
        if character == '\n' {
            line += 1;
            at += 1;
        } else if character == '/' && matches!(next, Some('/' | '*')) {
            at = skip_comment(&chars, at, &mut line);
        } else if character == '"' || raw || bytes {
            let start_line = line;
            let (end, text) = quoted(&chars, if bytes { at + 1 } else { at }, &mut line);
            found.push(Located {
                token: Token::Text(text),
                line: start_line,
            });
            at = end;
        } else if character == '\'' {
            at = tick(&chars, at);
        } else if character.is_ascii_alphabetic() || character == '_' {
            let length = chars[at..]
                .iter()
                .take_while(|&&character| identifier(character))
                .count();
            found.push(Located {
                token: Token::Word(chars[at..at + length].iter().collect()),
                line,
            });
            at += length;
        } else if character.is_ascii_digit() {
            at += chars[at..]
                .iter()
                .take_while(|&&character| identifier(character) || character == '.')
                .count();
        } else {
            if !character.is_whitespace() {
                found.push(Located {
                    token: Token::Mark(character),
                    line,
                });
            }
            at += 1;
        }
    }
    found
}

pub(crate) fn word(tokens: &[Located], at: usize) -> Option<&str> {
    match &tokens.get(at)?.token {
        Token::Word(word) => Some(word),
        Token::Text(_) | Token::Mark(_) => None,
    }
}

pub(crate) fn mark(tokens: &[Located], at: usize, wanted: char) -> bool {
    tokens
        .get(at)
        .is_some_and(|located| located.token == Token::Mark(wanted))
}

pub(crate) fn depth_change(token: &Token) -> isize {
    match token {
        Token::Mark('{' | '(' | '[') => 1,
        Token::Mark('}' | ')' | ']') => -1,
        Token::Mark(_) | Token::Word(_) | Token::Text(_) => 0,
    }
}

fn body_start(tokens: &[Located], from: usize) -> Option<usize> {
    let mut angle = 0_isize;
    for (index, located) in tokens.iter().enumerate().skip(from) {
        match located.token {
            Token::Mark('<') => angle += 1,
            Token::Mark('>') if !mark(tokens, index.wrapping_sub(1), '-') => angle -= 1,
            Token::Mark('{') if angle <= 0 => return Some(index),
            Token::Mark('(' | ';') if angle <= 0 => return None,
            Token::Mark(_) | Token::Word(_) | Token::Text(_) => {}
        }
    }
    None
}

fn body_names<'a>(tokens: &'a [Located], open: usize, variants: bool, names: &mut Vec<Named<'a>>) {
    let mut depth = 0_isize;
    for index in open..tokens.len() {
        depth += depth_change(&tokens[index].token);
        if depth <= 0 {
            return;
        }
        let Some(name) = word(tokens, index) else {
            continue;
        };
        let field = name != "pub"
            && mark(tokens, index + 1, ':')
            && !mark(tokens, index + 2, ':')
            && !mark(tokens, index.wrapping_sub(1), ':');
        let variant = variants
            && depth == 1
            && (mark(tokens, index - 1, '{')
                || mark(tokens, index - 1, ',')
                || mark(tokens, index - 1, ']'));
        if field && (depth == 1 || variants) {
            names.push(Named {
                name,
                line: tokens[index].line,
                what: "field",
            });
        } else if variant {
            names.push(Named {
                name,
                line: tokens[index].line,
                what: "variant",
            });
        }
    }
}

fn constant_texts<'a>(tokens: &'a [Located], from: usize, names: &mut Vec<Named<'a>>) {
    let mut depth = 0_isize;
    let mut valued = false;
    for located in tokens.iter().skip(from) {
        let outside = depth <= 0;
        depth += depth_change(&located.token);
        match &located.token {
            Token::Mark('{') if outside && !valued => return,
            Token::Mark('=') if outside => valued = true,
            Token::Mark(';') if depth <= 0 => return,
            Token::Text(text) if named_text(text) => names.push(Named {
                name: text,
                line: located.line,
                what: "constant value",
            }),
            Token::Mark(_) | Token::Word(_) | Token::Text(_) => {}
        }
    }
}

fn named_text(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|character| identifier(character) || matches!(character, '.' | ':' | '-'))
}

fn rust_names(tokens: &[Located]) -> Vec<Named<'_>> {
    let mut names = Vec::new();
    for index in 0..tokens.len() {
        let Some(keyword) = word(tokens, index) else {
            continue;
        };
        let named = |at: usize, what: &'static str| {
            word(tokens, at)
                .filter(|name| !matches!(*name, "fn" | "mut" | "_"))
                .map(|name| Named {
                    name,
                    line: tokens[at].line,
                    what,
                })
        };
        match keyword {
            "struct" | "union" | "enum" => {
                names.extend(named(index + 1, "type"));
                if let Some(open) = body_start(tokens, index + 2) {
                    body_names(tokens, open, keyword == "enum", &mut names);
                }
            }
            "trait" | "type" => names.extend(named(index + 1, "type")),
            "fn" => names.extend(named(index + 1, "function")),
            "const" | "static" => {
                let at = if word(tokens, index + 1) == Some("mut") {
                    index + 2
                } else {
                    index + 1
                };
                if let Some(constant) = named(at, "constant") {
                    names.push(constant);
                    constant_texts(tokens, at, &mut names);
                }
            }
            "section" | "rename" => {
                let text = (index + 1..index + 3).find_map(|at| match &tokens.get(at)?.token {
                    Token::Text(text) => Some((text, tokens[at].line)),
                    Token::Word(_) | Token::Mark(_) => None,
                });
                if let Some((text, line)) = text {
                    names.push(Named {
                        name: text,
                        line,
                        what: "setting name",
                    });
                }
            }
            _ => {}
        }
    }
    names
}

fn wgsl_names(tokens: &[Located]) -> Vec<Named<'_>> {
    tokens
        .iter()
        .filter_map(|located| match &located.token {
            Token::Word(name) => Some(Named {
                name,
                line: located.line,
                what: "WGSL identifier",
            }),
            Token::Text(_) | Token::Mark(_) => None,
        })
        .collect()
}

pub(crate) fn files(folder: &Path, found: &mut Vec<PathBuf>) -> Result<(), String> {
    let fail = |error: std::io::Error| format!("{}: {error}", folder.display());
    let mut paths = fs::read_dir(folder)
        .map_err(fail)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(fail)?;
    paths.sort();
    for path in paths {
        if path.is_dir() {
            files(&path, found)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "rs" || extension == "wgsl")
        {
            found.push(path);
        }
    }
    Ok(())
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
    for path in found {
        let text =
            fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let located = tokens(&text);
        let wgsl = path
            .extension()
            .is_some_and(|extension| extension == "wgsl");
        let names = if wgsl {
            wgsl_names(&located)
        } else {
            rust_names(&located)
        };
        let mut reported: BTreeSet<(&str, &str)> = BTreeSet::new();
        for named in names {
            let Some(word) = technique(named.name) else {
                continue;
            };
            if !reported.insert((named.name, named.what)) {
                continue;
            }
            problems.push(format!(
                "{label}: {}:{} names the technique {word} in the {} {}; the core names no technique",
                shown(&path),
                named.line,
                named.what,
                named.name
            ));
        }
    }
    Ok(())
}
