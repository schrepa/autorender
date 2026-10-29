use serde_json::json;

use crate::Repository;
use crate::policy::rust;

const SCRIPTS: [&str; 4] = [".ts", ".mjs", ".js", ".cjs"];

const SHELLS: [&str; 2] = [".ps1", ".sh"];

const MARKS: [&str; 2] = ["optics:", "filter:"];

pub fn check(repository: &Repository) -> Result<(), String> {
    let root = repository.root.as_path();
    let mut problems = Vec::new();
    let mut checked = 0;
    for path in crate::files::listed(root, &[])? {
        let rust = path.ends_with(".rs");
        let shader = path.ends_with(".wgsl");
        let script = SCRIPTS.iter().any(|extension| path.ends_with(extension));
        let shell = SHELLS.iter().any(|extension| path.ends_with(extension));
        if !(rust || shader || script || shell) || !crate::files::ours(&path) {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(root.join(&path)) else {
            continue;
        };
        checked += 1;
        let lines = if rust {
            rust_comments(&source)
        } else if shader {
            shader_comments(&source)
        } else if script {
            script_comments(&source, path.ends_with(".mjs"))
        } else {
            shell_comments(&source, path.ends_with(".ps1"))
        };
        problems.extend(
            lines
                .into_iter()
                .map(|line| json!({"rule": "comment", "file": path, "line": line}).to_string()),
        );
    }
    if problems.is_empty() {
        println!("{}", json!({"event": "no_comments", "files": checked}));
        return Ok(());
    }
    Err(problems.join("\n"))
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

fn script_comments(source: &str, typed: bool) -> Vec<usize> {
    let characters: Vec<char> = source.chars().collect();
    let mut found = Vec::new();
    let mut scanner = Scanner {
        characters: &characters,
        at: 0,
        holes: Vec::new(),
        depth: 0,
        last: None,
    };
    while let Some((start, end, block)) = scanner.next_comment() {
        let text: String = characters[start..end].iter().collect();
        let allowed = if block {
            let inner = text
                .strip_prefix("/*")
                .and_then(|inner| inner.strip_suffix("*/"))
                .unwrap_or_default();
            inner.trim() == "@vite-ignore"
                || (typed
                    && inner
                        .strip_prefix('*')
                        .is_some_and(|tags| !tags.starts_with('*') && type_tags(tags)))
        } else {
            text.trim_start_matches('/').trim() == "@ts-check"
        };
        if !allowed {
            found.push(line_at(&characters, start));
        }
    }
    found
}

fn type_tags(inner: &str) -> bool {
    let text: String = inner
        .lines()
        .map(|line| line.trim().trim_start_matches('*'))
        .collect::<Vec<_>>()
        .join(" ");
    let characters: Vec<char> = text.chars().collect();
    let skip = |mut at: usize| {
        while at < characters.len() && characters[at].is_whitespace() {
            at += 1;
        }
        at
    };
    let mut at = 0;
    let mut tags = 0;
    loop {
        at = skip(at);
        if at >= characters.len() {
            return tags > 0;
        }
        if characters[at] != '@' {
            return false;
        }
        at += 1;
        while at < characters.len() && (characters[at].is_alphanumeric() || characters[at] == '-') {
            at += 1;
        }
        tags += 1;
        at = skip(at);
        if characters.get(at) == Some(&'{') {
            let mut depth = 0_usize;
            while at < characters.len() {
                match characters[at] {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            at += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                at += 1;
            }
            if depth != 0 {
                return false;
            }
            at = skip(at);
        }
        if at < characters.len() && characters[at] != '@' {
            while at < characters.len() && !characters[at].is_whitespace() {
                at += 1;
            }
        }
    }
}

struct Scanner<'a> {
    characters: &'a [char],
    at: usize,
    holes: Vec<usize>,
    depth: usize,
    last: Option<char>,
}

impl Scanner<'_> {
    fn next_comment(&mut self) -> Option<(usize, usize, bool)> {
        while self.at < self.characters.len() {
            let character = self.characters[self.at];
            let next = self.characters.get(self.at + 1).copied();
            match character {
                '/' if next == Some('/') => {
                    let start = self.at;
                    while self.at < self.characters.len() && self.characters[self.at] != '\n' {
                        self.at += 1;
                    }
                    return Some((start, self.at, false));
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
                    return Some((start, self.at, true));
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

fn shell_comments(source: &str, powershell: bool) -> Vec<usize> {
    let mut found = Vec::new();
    let mut block = false;
    for (index, line) in source.lines().enumerate() {
        if block {
            found.push(index + 1);
            block = !line.contains("#>");
            continue;
        }
        if index == 0 && line.starts_with("#!") {
            continue;
        }
        let mut quote: Option<char> = None;
        let mut previous = ' ';
        for (at, character) in line.char_indices() {
            match quote {
                Some(open) if character == open => quote = None,
                Some(_) => {}
                None if character == '\'' || character == '"' => quote = Some(character),
                None if character == '#' && (previous.is_whitespace() || previous == ';') => {
                    found.push(index + 1);
                    block = powershell && line[..at].ends_with('<') && !line[at..].contains("#>");
                    break;
                }
                None if character == '<' && powershell && line[at..].starts_with("<#") => {
                    found.push(index + 1);
                    block = !line[at..].contains("#>");
                    break;
                }
                None => {}
            }
            previous = character;
        }
    }
    found
}
