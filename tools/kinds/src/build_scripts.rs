use std::fs;
use std::path::Path;

const CLIMBS: &[&str] = &["parent", "ancestors"];

pub fn check(
    label: &str,
    script: &Path,
    shown: impl Fn(&Path) -> String,
    problems: &mut Vec<String>,
) -> Result<(), String> {
    let text =
        fs::read_to_string(script).map_err(|error| format!("{}: {error}", script.display()))?;
    let located = tokens(&text);
    for (at, found) in located.iter().enumerate() {
        let reason = match &found.token {
            Token::Text(path) if leaves(path) => {
                format!("names the path {path:?}, which leaves the crate")
            }
            Token::Word(name)
                if CLIMBS.contains(&name.as_str()) && mark(&located, at.wrapping_sub(1), '.') =>
            {
                format!("climbs out of the crate with .{name}()")
            }
            Token::Text(_) | Token::Word(_) | Token::Mark(_) => continue,
        };
        problems.push(format!(
            "{label}: {}:{} {reason}; a core crate's build script reads no file outside its own crate",
            shown(script),
            found.line
        ));
    }
    Ok(())
}

fn leaves(path: &str) -> bool {
    let mut characters = path.chars();
    let drive = matches!(
        (characters.next(), characters.next(), characters.next()),
        (Some(letter), Some(':'), None | Some('/' | '\\')) if letter.is_ascii_alphabetic()
    );
    drive || path.starts_with(['/', '\\']) || path.split(['/', '\\']).any(|part| part == "..")
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Word(String),
    Text(String),
    Mark(char),
}

struct Located {
    token: Token,
    line: usize,
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

fn tokens(source: &str) -> Vec<Located> {
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

fn mark(tokens: &[Located], at: usize, wanted: char) -> bool {
    tokens
        .get(at)
        .is_some_and(|located| located.token == Token::Mark(wanted))
}
