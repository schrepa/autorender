use engine_core::{Error, ErrorCode};
use serde_json::json;

use crate::{Exceptions, Repository};

const MAX_FILE_LINES: usize = 1500;
const MAX_STRUCT_FIELDS: usize = 30;

pub fn check(repository: &Repository, exceptions: &Exceptions) -> Result<(), String> {
    let root = repository.root.as_path();
    let mut problems = Vec::new();
    let mut long = Vec::new();
    let mut wide = Vec::new();
    let games: Vec<String> = repository
        .projects()
        .filter_map(|project| project.path.as_deref())
        .filter(|folder| !folder.is_empty() && !folder.starts_with("crates/"))
        .map(|folder| format!("{folder}/"))
        .collect();
    for path in crate::files::sources(root)? {
        if games.iter().any(|folder| path.starts_with(folder.as_str())) {
            continue;
        }
        let text = std::fs::read_to_string(root.join(&path))
            .map_err(|error| format!("cannot read {path}: {error}"))?;
        let lines = text.lines().count();
        if lines > MAX_FILE_LINES {
            long.push(path.clone());
            let most = match exceptions.long_files.iter().find(|(file, _)| *file == path) {
                Some(&(_, most)) => most,
                None => MAX_FILE_LINES,
            };
            if lines > most {
                problems.push(
                    Error::new(ErrorCode::FileTooLong, path.as_str())
                        .at_most(most as u64)
                        .got(lines),
                );
            }
        }
        if !path.ends_with(".rs") {
            continue;
        }
        for (name, fields) in structs(&code(&text)) {
            if fields <= MAX_STRUCT_FIELDS {
                continue;
            }
            wide.push((path.clone(), name.clone()));
            let most = match exceptions
                .wide_structs
                .iter()
                .find(|(file, wide, _)| *file == path && *wide == name)
            {
                Some(&(_, _, most)) => most,
                None => MAX_STRUCT_FIELDS,
            };
            if fields > most {
                problems.push(
                    Error::new(ErrorCode::StructTooWide, path.as_str())
                        .at_most(most as u64)
                        .got(json!({ name.as_str(): fields })),
                );
            }
        }
    }
    for (path, _) in exceptions.long_files {
        if !long.iter().any(|file| file == path) {
            problems.push(Error::new(ErrorCode::ExceptionUnused, *path).got("long_files"));
        }
    }
    for (path, name, _) in exceptions.wide_structs {
        if !wide.iter().any(|(file, wide)| file == path && wide == name) {
            problems.push(
                Error::new(ErrorCode::ExceptionUnused, *path).got(json!({ "wide_structs": name })),
            );
        }
    }
    if problems.is_empty() {
        println!(
            "No source grew: {} long files and {} wide structs, none over its size.",
            exceptions.long_files.len(),
            exceptions.wide_structs.len()
        );
        return Ok(());
    }
    crate::report(&problems)
}

fn code(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
    let blank = |from: &[char]| -> String {
        from.iter()
            .map(|&character| if character == '\n' { '\n' } else { ' ' })
            .collect()
    };
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while at < characters.len() {
        let rest = &characters[at..];
        let end = match rest {
            ['/', '/', ..] => rest.iter().position(|&c| c == '\n').unwrap_or(rest.len()),
            ['/', '*', ..] => {
                let mut depth = 0;
                let mut index = 0;
                while index < rest.len() {
                    match &rest[index..] {
                        ['/', '*', ..] => {
                            depth += 1;
                            index += 2;
                        }
                        ['*', '/', ..] => {
                            depth -= 1;
                            index += 2;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => index += 1,
                    }
                }
                index
            }
            ['r', '#' | '"', ..] if !word_before(&characters, at) => {
                let hashes = rest[1..].iter().take_while(|&&c| c == '#').count();
                if rest.get(1 + hashes) == Some(&'"') {
                    let close: Vec<char> = std::iter::once('"')
                        .chain(std::iter::repeat_n('#', hashes))
                        .collect();
                    let start = 2 + hashes;
                    rest[start..]
                        .windows(close.len())
                        .position(|window| window == close.as_slice())
                        .map_or(rest.len(), |found| start + found + close.len())
                } else {
                    0
                }
            }
            ['"', ..] => {
                let mut index = 1;
                while index < rest.len() && rest[index] != '"' {
                    index += if rest[index] == '\\' { 2 } else { 1 };
                }
                (index + 1).min(rest.len())
            }
            ['\'', '\\', ..] => rest[2..]
                .iter()
                .position(|&c| c == '\'')
                .map_or(rest.len(), |found| found + 3),
            ['\'', _, '\'', ..] => 3,
            _ => 0,
        };
        if end == 0 {
            out.push(characters[at]);
            at += 1;
        } else {
            out.push_str(&blank(&rest[..end]));
            at += end;
        }
    }
    out
}

fn word_before(characters: &[char], at: usize) -> bool {
    at > 0 && (characters[at - 1].is_alphanumeric() || characters[at - 1] == '_')
}

fn structs(code: &str) -> Vec<(String, usize)> {
    let characters: Vec<char> = code.chars().collect();
    let word_at = |at: usize| -> String {
        characters[at..]
            .iter()
            .take_while(|c| c.is_alphanumeric() || **c == '_')
            .collect()
    };
    let mut found = Vec::new();
    let mut at = 0;
    while at < characters.len() {
        if !(characters[at].is_alphabetic() || characters[at] == '_')
            || word_before(&characters, at)
        {
            at += 1;
            continue;
        }
        let word = word_at(at);
        at += word.chars().count();
        if word != "struct" {
            continue;
        }
        let mut index = at;
        while index < characters.len() && characters[index].is_whitespace() {
            index += 1;
        }
        let name = word_at(index);
        if name.is_empty() {
            continue;
        }
        index += name.chars().count();
        let mut angle = 0_i32;
        while index < characters.len() {
            match characters[index] {
                '-' if characters.get(index + 1) == Some(&'>') => index += 1,
                '<' => angle += 1,
                '>' => angle -= 1,
                '{' | '(' | ';' if angle == 0 => break,
                _ => {}
            }
            index += 1;
        }
        let fields = match characters.get(index) {
            Some('{') => named_fields(&characters, index),
            Some('(') => tuple_fields(&characters, index),
            _ => 0,
        };
        found.push((name, fields));
        at = index;
    }
    found
}

fn named_fields(characters: &[char], open: usize) -> usize {
    let mut depth = 0;
    let mut fields = 0;
    let mut index = open;
    while index < characters.len() {
        let character = characters[index];
        match character {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ if depth == 1
                && (character.is_alphabetic() || character == '_')
                && !word_before(characters, index) =>
            {
                let word: String = characters[index..]
                    .iter()
                    .take_while(|c| c.is_alphanumeric() || **c == '_')
                    .collect();
                let mut after = index + word.chars().count();
                while after < characters.len() && characters[after].is_whitespace() {
                    after += 1;
                }
                if word != "pub"
                    && characters.get(after) == Some(&':')
                    && characters.get(after + 1) != Some(&':')
                {
                    fields += 1;
                }
                index += word.chars().count();
                continue;
            }
            _ => {}
        }
        index += 1;
    }
    fields
}

fn tuple_fields(characters: &[char], open: usize) -> usize {
    let mut depth = 0;
    let mut parts = 0;
    let mut filled = false;
    let mut index = open;
    while index < characters.len() {
        match characters[index] {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            ',' if depth == 1 => {
                parts += usize::from(filled);
                filled = false;
            }
            character if !character.is_whitespace() => filled = true,
            _ => {}
        }
        index += 1;
    }
    parts + usize::from(filled)
}
