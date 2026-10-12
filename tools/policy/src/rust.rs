//! Rust sources as text, per line: the code without comments or literal
//! contents, and the comments. Enough of Rust's lexical grammar for the
//! policy's checks: line, block (nested) and doc comments; string, byte,
//! C and raw string literals; character and byte literals told apart from
//! lifetimes and labels.

/// A Rust file split per line (index 0 is line 1).
pub struct Split {
    /// Each line's code: comments removed, literals kept as empty quotes
    /// (`""`, `''`).
    pub code: Vec<String>,
    /// Each line's comment text, markers included (`// SAFETY: ...`).
    pub comments: Vec<String>,
}

fn identifier(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Whether `word` occurs in `text` whole: not inside a longer identifier.
pub fn has_word(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + word.len()..].chars().next();
        !before.is_some_and(identifier) && !after.is_some_and(identifier)
    })
}

/// Splits `source` into its lines' code and comments.
pub fn split(source: &str) -> Split {
    let chars: Vec<char> = source.chars().collect();
    let mut split = Split {
        code: vec![String::new()],
        comments: vec![String::new()],
    };
    let mut at = 0;
    while at < chars.len() {
        let c = chars[at];
        let next = chars.get(at + 1).copied();
        if c == '\n' {
            split.code.push(String::new());
            split.comments.push(String::new());
            at += 1;
        } else if c == '/' && next == Some('/') {
            while at < chars.len() && chars[at] != '\n' {
                last(&mut split.comments).push(chars[at]);
                at += 1;
            }
        } else if c == '/' && next == Some('*') {
            at = block_comment(&chars, at, &mut split);
        } else if let Some((end, quote)) = literal(&chars, at) {
            // The literal's own newlines still end lines.
            for &inside in &chars[at..end] {
                if inside == '\n' {
                    split.code.push(String::new());
                    split.comments.push(String::new());
                }
            }
            last(&mut split.code).extend([quote, quote]);
            at = end;
        } else {
            last(&mut split.code).push(c);
            at += 1;
        }
    }
    split
}

fn last(lines: &mut [String]) -> &mut String {
    lines
        .last_mut()
        .expect("a split starts with one line and only adds lines")
}

/// Past the block comment opening at `start` (nested comments included),
/// its text recorded line by line.
fn block_comment(chars: &[char], start: usize, split: &mut Split) -> usize {
    let mut depth = 0usize;
    let mut at = start;
    while at < chars.len() {
        let pair = (chars[at], chars.get(at + 1).copied());
        if pair == ('/', Some('*')) {
            depth += 1;
            last(&mut split.comments).push_str("/*");
            at += 2;
        } else if pair == ('*', Some('/')) {
            depth -= 1;
            last(&mut split.comments).push_str("*/");
            at += 2;
            if depth == 0 {
                break;
            }
        } else if chars[at] == '\n' {
            split.code.push(String::new());
            split.comments.push(String::new());
            at += 1;
        } else {
            last(&mut split.comments).push(chars[at]);
            at += 1;
        }
    }
    at
}

/// The end of the literal starting at `start` (prefix included) and its
/// quote character, or `None` where no literal starts.
fn literal(chars: &[char], start: usize) -> Option<(usize, char)> {
    let previous = start.checked_sub(1).map(|at| chars[at]);
    // Prefixes (`b`, `c`, `r`, `br`, `cr`) start a token, not end one.
    let prefix = if previous.is_some_and(identifier) {
        ""
    } else {
        ["br", "cr", "b", "c", "r"]
            .into_iter()
            .find(|prefix| {
                let length = prefix.len();
                chars.get(start..start + length).is_some_and(|text| {
                    text.iter().copied().eq(prefix.chars())
                        && matches!(chars.get(start + length), Some('"' | '\'' | '#'))
                })
            })
            .unwrap_or("")
    };
    let body = start + prefix.len();
    match chars.get(body)? {
        '#' | '"' if prefix.ends_with('r') => raw_string(chars, body),
        '"' if !prefix.ends_with('r') => {
            let mut at = body + 1;
            while at < chars.len() {
                match chars[at] {
                    '\\' => at += 2,
                    '"' => return Some((at + 1, '"')),
                    _ => at += 1,
                }
            }
            Some((chars.len(), '"'))
        }
        '\'' if prefix.is_empty() || prefix == "b" => character(chars, body),
        _ => None,
    }
}

/// A raw string's `#`s and quotes from `body`: its end, or `None` for a
/// raw identifier (`r#match`).
fn raw_string(chars: &[char], body: usize) -> Option<(usize, char)> {
    let hashes = chars[body..].iter().take_while(|&&c| c == '#').count();
    if chars.get(body + hashes) != Some(&'"') {
        return None;
    }
    let mut at = body + hashes + 1;
    while at < chars.len() {
        if chars[at] == '"'
            && chars[at + 1..]
                .iter()
                .take(hashes)
                .filter(|&&c| c == '#')
                .count()
                == hashes
        {
            return Some((at + 1 + hashes, '"'));
        }
        at += 1;
    }
    Some((chars.len(), '"'))
}

/// A character literal at `quote`: its end, or `None` for a lifetime or
/// label (`'a`, `'outer:`).
fn character(chars: &[char], quote: usize) -> Option<(usize, char)> {
    match chars.get(quote + 1)? {
        '\\' => {
            // `'\''` escapes the quote itself; no other escape holds one.
            let mut at = quote + 3;
            while at < chars.len() && chars[at] != '\'' {
                at += 1;
            }
            Some(((at + 1).min(chars.len()), '\''))
        }
        _ if chars.get(quote + 2) == Some(&'\'') => Some((quote + 3, '\'')),
        _ => None,
    }
}
