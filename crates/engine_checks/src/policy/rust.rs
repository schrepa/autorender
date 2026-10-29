pub struct Split {
    pub code: Vec<String>,
    pub comments: Vec<String>,
}

impl Split {
    pub fn safety_comment_above(&self, line: usize) -> bool {
        let mut above = line;
        while above > 0 {
            above -= 1;
            if self.comments[above].contains("SAFETY:") {
                return true;
            }
            let code = self.code[above].trim();
            let attribute = code.starts_with("#[") && code.ends_with(']');
            let comment = code.is_empty() && !self.comments[above].trim().is_empty();
            if !(attribute || comment) {
                return false;
            }
        }
        false
    }

    pub fn statement_start(&self, line: usize) -> usize {
        let mut start = line;
        while start > 0 {
            let above = self.code[start - 1].trim();
            let attribute = above.starts_with("#[") && above.ends_with(']');
            if above.is_empty() || attribute || above.ends_with([';', '{', '}']) {
                break;
            }
            start -= 1;
        }
        start
    }

    pub fn token_after(&self, line: usize, end: usize) -> String {
        let rest = std::iter::once(&self.code[line][end..])
            .chain(self.code[line + 1..].iter().map(String::as_str))
            .flat_map(str::chars)
            .skip_while(|c| c.is_whitespace());
        let mut token = String::new();
        for c in rest {
            if identifier(c) {
                token.push(c);
            } else {
                if token.is_empty() {
                    token.push(c);
                }
                break;
            }
        }
        token
    }
}

fn identifier(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

pub fn has_word(text: &str, word: &str) -> bool {
    words(text, word).next().is_some()
}

pub fn words<'a>(text: &'a str, word: &'a str) -> impl Iterator<Item = usize> + 'a {
    text.match_indices(word).filter_map(move |(at, _)| {
        let before = text[..at].chars().next_back();
        let end = at + word.len();
        let after = text[end..].chars().next();
        (!before.is_some_and(identifier) && !after.is_some_and(identifier)).then_some(end)
    })
}

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

fn literal(chars: &[char], start: usize) -> Option<(usize, char)> {
    let previous = start.checked_sub(1).map(|at| chars[at]);
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

fn character(chars: &[char], quote: usize) -> Option<(usize, char)> {
    match chars.get(quote + 1)? {
        '\\' => {
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
