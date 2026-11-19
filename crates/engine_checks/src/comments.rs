use engine_core::{Error, ErrorCode};
use serde_json::json;

use crate::Repository;
use crate::policy::rust;

const MARKS: [&str; 1] = ["optics:"];

pub fn covers(path: &str) -> bool {
    [".rs", ".wgsl", ".ts"]
        .iter()
        .any(|extension| path.ends_with(extension))
}

pub fn check(repository: &Repository) -> Result<(), String> {
    let root = repository.root.as_path();
    let mut problems = Vec::new();
    let mut checked = 0;
    for path in crate::files::listed(root, &[])? {
        if !covers(&path) || !crate::files::ours(&path) || !repository.ruled(&path) {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(root.join(&path)) else {
            continue;
        };
        checked += 1;
        let lines = if path.ends_with(".rs") {
            rust_comments(&source)
        } else if path.ends_with(".wgsl") {
            shader_comments(&source)
        } else {
            script_comments(&source)
        };
        problems.extend(
            lines
                .into_iter()
                .map(|line| Error::new(ErrorCode::Comment, crate::at(&path, line))),
        );
    }
    if problems.is_empty() {
        println!("{}", json!({"event": "no_comments", "files": checked}));
        return Ok(());
    }
    crate::report(&problems)
}

fn rust_comments(source: &str) -> Vec<usize> {
    let split = rust::split(source);
    let mut found = Vec::new();
    let mut line = 0;
    while line < split.code.len() {
        let comment = split.comments[line].trim();
        if comment.is_empty() {
            line += 1;
            continue;
        }
        let code = split.code[line].trim();
        if !code.is_empty() {
            if !(comment.starts_with("// SAFETY:") && rust::has_word(code, "unsafe")) {
                found.push(line + 1);
            }
            line += 1;
            continue;
        }
        let start = line;
        while line < split.code.len()
            && split.code[line].trim().is_empty()
            && !split.comments[line].trim().is_empty()
        {
            line += 1;
        }
        if !(comment.starts_with("// SAFETY:") && unsafe_follows(&split.code, line)) {
            found.push(start + 1);
        }
    }
    found
}

fn unsafe_follows(code: &[String], from: usize) -> bool {
    for text in code.iter().skip(from).take(12) {
        let text = text.trim();
        if text.starts_with("#[") && text.ends_with(']') {
            continue;
        }
        if rust::has_word(text, "unsafe") {
            return true;
        }
        if text.is_empty() || text.ends_with([';', '}']) {
            return false;
        }
    }
    false
}

fn line_at(characters: &[char], at: usize) -> usize {
    characters[..at.min(characters.len())]
        .iter()
        .filter(|&&character| character == '\n')
        .count()
        + 1
}

fn shader_comments(source: &str) -> Vec<usize> {
    let characters: Vec<char> = source.chars().collect();
    let mut found = Vec::new();
    let mut at = 0;
    while at < characters.len() {
        let pair = (characters[at], characters.get(at + 1).copied());
        if pair == ('/', Some('/')) {
            let end = (at..characters.len())
                .find(|&index| characters[index] == '\n')
                .unwrap_or(characters.len());
            let text: String = characters[at + 2..end].iter().collect();
            if !MARKS.iter().any(|mark| text.trim_start().starts_with(mark)) {
                found.push(line_at(&characters, at));
            }
            at = end;
        } else if pair == ('/', Some('*')) {
            found.push(line_at(&characters, at));
            let mut depth = 0_usize;
            while at < characters.len() {
                let pair = (characters[at], characters.get(at + 1).copied());
                if pair == ('/', Some('*')) {
                    depth += 1;
                    at += 2;
                } else if pair == ('*', Some('/')) {
                    depth -= 1;
                    at += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    at += 1;
                }
            }
        } else {
            at += 1;
        }
    }
    found
}

fn script_comments(source: &str) -> Vec<usize> {
    let characters: Vec<char> = source.chars().collect();
    let mut found = Vec::new();
    let mut scanner = Scanner {
        characters: &characters,
        at: 0,
        holes: Vec::new(),
        depth: 0,
        last: None,
    };
    while let Some(start) = scanner.next_comment() {
        found.push(line_at(&characters, start));
    }
    found
}

struct Scanner<'a> {
    characters: &'a [char],
    at: usize,
    holes: Vec<usize>,
    depth: usize,
    last: Option<char>,
}

impl Scanner<'_> {
    fn next_comment(&mut self) -> Option<usize> {
        while self.at < self.characters.len() {
            let character = self.characters[self.at];
            let next = self.characters.get(self.at + 1).copied();
            match character {
                '/' if next == Some('/') => {
                    let start = self.at;
                    while self.at < self.characters.len() && self.characters[self.at] != '\n' {
                        self.at += 1;
                    }
                    return Some(start);
                }
                '/' if next == Some('*') => {
                    let start = self.at;
                    self.at += 2;
                    while self.at < self.characters.len()
                        && !(self.characters[self.at] == '*'
                            && self.characters.get(self.at + 1) == Some(&'/'))
                    {
                        self.at += 1;
                    }
                    self.at = (self.at + 2).min(self.characters.len());
                    return Some(start);
                }
                '"' | '\'' => self.quoted(character),
                '`' => {
                    self.at += 1;
                    self.template();
                }
                '/' if self.regex_starts() => self.regex(),
                '{' => {
                    self.depth += 1;
                    self.step(character);
                }
                '}' => {
                    self.depth = self.depth.saturating_sub(1);
                    self.step(character);
                    if self.holes.last() == Some(&self.depth) {
                        self.holes.pop();
                        self.template();
                    }
                }
                _ => self.step(character),
            }
        }
        None
    }

    fn step(&mut self, character: char) {
        if !character.is_whitespace() {
            self.last = Some(character);
        }
        self.at += 1;
    }

    fn quoted(&mut self, quote: char) {
        self.at += 1;
        while self.at < self.characters.len()
            && self.characters[self.at] != quote
            && self.characters[self.at] != '\n'
        {
            self.at += if self.characters[self.at] == '\\' {
                2
            } else {
                1
            };
        }
        self.at = (self.at + 1).min(self.characters.len());
        self.last = Some(quote);
    }

    fn template(&mut self) {
        while self.at < self.characters.len() {
            match self.characters[self.at] {
                '\\' => self.at += 2,
                '`' => {
                    self.at += 1;
                    self.last = Some('`');
                    return;
                }
                '$' if self.characters.get(self.at + 1) == Some(&'{') => {
                    self.at += 2;
                    self.holes.push(self.depth);
                    self.depth += 1;
                    self.last = Some('{');
                    return;
                }
                _ => self.at += 1,
            }
        }
    }

    fn regex_starts(&self) -> bool {
        let Some(last) = self.last else {
            return true;
        };
        if last.is_alphanumeric() || last == '_' || last == '$' {
            let before: String = self.characters[..self.at]
                .iter()
                .rev()
                .skip_while(|character| character.is_whitespace())
                .take_while(|character| character.is_alphanumeric() || **character == '_')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            return matches!(
                before.as_str(),
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

    fn regex(&mut self) {
        self.at += 1;
        let mut class = false;
        while self.at < self.characters.len() && self.characters[self.at] != '\n' {
            let inside = self.characters[self.at];
            self.at += 1;
            match inside {
                '\\' => self.at += 1,
                '/' if !class => break,
                '[' => class = true,
                ']' => class = false,
                _ => {}
            }
        }
        self.last = Some('/');
    }
}
