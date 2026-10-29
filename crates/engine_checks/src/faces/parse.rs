use crate::policy::rust;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Module,
    Function,
    Struct,
    Enum,
    Union,
    Trait,
    Alias,
    Constant,
    Static,
}

impl Kind {
    pub(crate) fn word(self) -> &'static str {
        match self {
            Self::Module => "mod",
            Self::Function => "fn",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Union => "union",
            Self::Trait => "trait",
            Self::Alias => "type",
            Self::Constant => "const",
            Self::Static => "static",
        }
    }
}

pub(crate) struct Member {
    pub word: String,
    pub name: String,
    pub rest: String,
}

pub(crate) struct Item {
    pub kind: Kind,
    pub name: String,
    pub public: bool,
    pub qualifiers: String,
    pub header: String,
    pub members: Vec<Member>,
    pub derives: Vec<String>,
    pub body: Option<String>,
}

pub(crate) struct Use {
    pub public: bool,
    pub path: Vec<String>,
    pub alias: String,
    pub glob: bool,
}

pub(crate) struct Impl {
    pub target: String,
    pub trait_name: Option<String>,
    pub members: Vec<Member>,
}

#[derive(Default)]
pub(crate) struct Parsed {
    pub items: Vec<Item>,
    pub uses: Vec<Use>,
    pub impls: Vec<Impl>,
    pub macros: Vec<String>,
}

struct Attributes {
    derives: Vec<String>,
    exported: bool,
}

pub(crate) fn code(source: &str) -> Vec<char> {
    rust::split(source).code.join("\n").chars().collect()
}

pub(crate) fn parse(code: &[char]) -> Parsed {
    let mut parsed = Parsed::default();
    items(code, 0, code.len(), &mut parsed);
    parsed
}

pub(crate) fn identifier(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

fn space(code: &[char], mut at: usize, end: usize) -> usize {
    while at < end && code[at].is_whitespace() {
        at += 1;
    }
    at
}

fn word(code: &[char], at: usize, end: usize) -> String {
    code[at.min(end)..end]
        .iter()
        .take_while(|&&character| identifier(character))
        .collect()
}

fn after_word(code: &[char], at: usize, end: usize) -> (String, usize) {
    let found = word(code, at, end);
    let next = space(code, at + found.chars().count(), end);
    (found, next)
}

pub(crate) fn closing(code: &[char], open: usize) -> usize {
    let mut depth = 0_usize;
    for (at, &character) in code.iter().enumerate().skip(open) {
        match character {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return at + 1;
                }
            }
            _ => {}
        }
    }
    code.len()
}

fn arrow(code: &[char], at: usize) -> bool {
    at > 0 && matches!(code[at - 1], '-' | '=')
}

fn until(code: &[char], mut at: usize, end: usize, stops: &[char]) -> usize {
    let mut depth = 0_usize;
    let mut angles = 0_usize;
    while at < end {
        let character = code[at];
        if depth == 0 && angles == 0 && stops.contains(&character) {
            return at;
        }
        match character {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            '<' => angles += 1,
            '>' if !arrow(code, at) => angles = angles.saturating_sub(1),
            _ => {}
        }
        at += 1;
    }
    end
}

fn statement_end(code: &[char], at: usize, end: usize) -> usize {
    let stop = until(code, at, end, &[';', '{']);
    match code.get(stop) {
        Some('{') if stop < end => closing(code, stop).min(end),
        _ => (stop + 1).min(end),
    }
}

pub(crate) fn normalize(text: &[char]) -> String {
    let mut out = String::new();
    let mut gap = false;
    for &character in text {
        if character.is_whitespace() {
            gap = true;
            continue;
        }
        match character {
            ')' | ']' | '>' | '}' => {
                let kept = out.trim_end().len();
                out.truncate(kept);
                if out.ends_with(',') {
                    out.pop();
                }
                if character == '}' && !out.ends_with('{') {
                    out.push(' ');
                }
            }
            ',' => {
                let kept = out.trim_end().len();
                out.truncate(kept);
            }
            _ if gap && !out.is_empty() && !out.ends_with(['(', '[', '<']) => out.push(' '),
            _ => {}
        }
        gap = false;
        out.push(character);
    }
    out.trim().trim_end_matches(',').to_owned()
}

fn items(code: &[char], mut at: usize, end: usize, parsed: &mut Parsed) {
    let mut attributes = Attributes {
        derives: Vec::new(),
        exported: false,
    };
    while at < end {
        at = space(code, at, end);
        if at >= end {
            break;
        }
        if code[at] == '#' {
            let open = space(code, at + 1, end);
            let open = if code.get(open) == Some(&'!') {
                space(code, open + 1, end)
            } else {
                open
            };
            if code.get(open) == Some(&'[') {
                let close = closing(code, open).min(end);
                let inner = &code[open + 1..close.saturating_sub(1).max(open + 1)];
                attribute(inner, &mut attributes);
                at = close;
                continue;
            }
        }
        if code[at] == ';' {
            at += 1;
            continue;
        }
        let next = item(code, at, end, &attributes, parsed);
        attributes.derives.clear();
        attributes.exported = false;
        at = next.max(at + 1);
    }
}

fn attribute(text: &[char], attributes: &mut Attributes) {
    let text: String = text.iter().filter(|c| !c.is_whitespace()).collect();
    if text == "macro_export" {
        attributes.exported = true;
    }
    if let Some(listed) = text
        .strip_prefix("derive(")
        .and_then(|listed| listed.strip_suffix(')'))
    {
        attributes.derives.extend(
            listed
                .split(',')
                .filter(|name| !name.is_empty())
                .map(str::to_owned),
        );
    }
}

fn visibility(code: &[char], at: usize, end: usize) -> (bool, usize) {
    let (found, next) = after_word(code, at, end);
    if found != "pub" {
        return (false, at);
    }
    if code.get(next) == Some(&'(') {
        return (false, space(code, closing(code, next), end));
    }
    (true, next)
}

fn qualifiers(code: &[char], mut at: usize, end: usize) -> (String, usize, Option<usize>) {
    let mut qualifiers = String::new();
    loop {
        let (found, next) = after_word(code, at, end);
        match found.as_str() {
            "const" => {
                let following = word(code, next, end);
                if !["fn", "unsafe", "async", "extern"].contains(&following.as_str()) {
                    return (qualifiers, at, None);
                }
                qualifiers.push_str("const ");
                at = next;
            }
            "async" | "unsafe" | "default" => {
                qualifiers.push_str(&found);
                qualifiers.push(' ');
                at = next;
            }
            "extern" => {
                let mut next = next;
                if code.get(next) == Some(&'"') {
                    next = space(code, next + 2, end);
                }
                let following = word(code, next, end);
                if following == "crate" || code.get(next) == Some(&'{') {
                    return (qualifiers, at, Some(statement_end(code, next, end)));
                }
                qualifiers.push_str("extern ");
                at = next;
            }
            _ => return (qualifiers, at, None),
        }
    }
}

fn item(
    code: &[char],
    at: usize,
    end: usize,
    attributes: &Attributes,
    parsed: &mut Parsed,
) -> usize {
    let (public, at) = visibility(code, at, end);
    let (qualifiers, at, skipped) = qualifiers(code, at, end);
    if let Some(next) = skipped {
        return next;
    }
    let (keyword, next) = after_word(code, at, end);
    let declared = Declared {
        public,
        qualifiers,
        derives: attributes.derives.clone(),
    };
    match keyword.as_str() {
        "use" => use_item(code, next, end, public, parsed),
        "mod" => module_item(code, next, end, declared, parsed),
        "fn" => function_item(code, next, end, declared, parsed),
        "struct" => struct_item(code, next, end, Kind::Struct, declared, parsed),
        "union" => struct_item(code, next, end, Kind::Union, declared, parsed),
        "enum" => enum_item(code, next, end, declared, parsed),
        "trait" => trait_item(code, next, end, declared, parsed),
        "type" => alias_item(code, next, end, declared, parsed),
        "const" => value_item(code, next, end, Kind::Constant, declared, parsed),
        "static" => value_item(code, next, end, Kind::Static, declared, parsed),
        "impl" => impl_item(code, next, end, parsed),
        "macro_rules" if code.get(next) == Some(&'!') => {
            let (name, open) = after_word(code, space(code, next + 1, end), end);
            if attributes.exported && !name.is_empty() {
                parsed.macros.push(name);
            }
            statement_end(code, open, end)
        }
        _ => invocation(code, at, end, parsed),
    }
}

struct Declared {
    public: bool,
    qualifiers: String,
    derives: Vec<String>,
}

impl Declared {
    fn item(self, kind: Kind, name: String, header: String, members: Vec<Member>) -> Item {
        Item {
            kind,
            name,
            public: self.public,
            qualifiers: self.qualifiers,
            header,
            members,
            derives: self.derives,
            body: None,
        }
    }
}

fn invocation(code: &[char], at: usize, end: usize, parsed: &mut Parsed) -> usize {
    let mut bang = at;
    while bang < end && (identifier(code[bang]) || code[bang] == ':') {
        bang += 1;
    }
    let bang = space(code, bang, end);
    if bang > at && code.get(bang) == Some(&'!') {
        let open = space(code, bang + 1, end);
        if matches!(code.get(open), Some('(' | '[' | '{')) && open < end {
            let close = closing(code, open).min(end);
            items(
                code,
                open + 1,
                close.saturating_sub(1).max(open + 1),
                parsed,
            );
            let after = space(code, close, end);
            return if code.get(after) == Some(&';') {
                after + 1
            } else {
                close
            };
        }
    }
    statement_end(code, at, end)
}

fn use_item(code: &[char], at: usize, end: usize, public: bool, parsed: &mut Parsed) -> usize {
    let stop = until(code, at, end, &[';']);
    let tree: String = code[at..stop].iter().collect();
    for (path, alias) in expand(&tree) {
        let mut segments: Vec<String> = path
            .split("::")
            .filter(|segment| !segment.is_empty())
            .map(str::to_owned)
            .collect();
        let glob = segments.last().is_some_and(|last| last == "*");
        if glob {
            segments.pop();
        }
        if segments.last().is_some_and(|last| last == "self") {
            segments.pop();
        }
        let alias = alias
            .or_else(|| segments.last().cloned())
            .unwrap_or_default();
        parsed.uses.push(Use {
            public,
            path: segments,
            alias,
            glob,
        });
    }
    (stop + 1).min(end)
}

pub(crate) fn expand(tree: &str) -> Vec<(String, Option<String>)> {
    let normalized: String = tree
        .split_whitespace()
        .map(|part| if part == "as" { "@" } else { part })
        .collect();
    let mut out = Vec::new();
    walk_tree("", &normalized, &mut out);
    out
}

fn walk_tree(prefix: &str, tree: &str, out: &mut Vec<(String, Option<String>)>) {
    if let Some(open) = tree.find('{') {
        let prefix = joined(prefix, tree[..open].trim_end_matches("::"));
        let inner = tree[open + 1..]
            .strip_suffix('}')
            .unwrap_or(&tree[open + 1..]);
        let (mut depth, mut start) = (0_usize, 0);
        for (at, character) in inner.char_indices() {
            match character {
                '{' => depth += 1,
                '}' => depth = depth.saturating_sub(1),
                ',' if depth == 0 => {
                    walk_tree(&prefix, &inner[start..at], out);
                    start = at + 1;
                }
                _ => {}
            }
        }
        walk_tree(&prefix, &inner[start..], out);
        return;
    }
    if tree.is_empty() {
        return;
    }
    let path = joined(prefix, tree);
    match path.split_once('@') {
        Some((path, alias)) => out.push((path.to_owned(), Some(alias.to_owned()))),
        None => out.push((path, None)),
    }
}

pub(crate) fn joined(prefix: &str, path: &str) -> String {
    match (prefix.is_empty(), path.is_empty()) {
        (true, _) => path.to_owned(),
        (false, true) => prefix.to_owned(),
        (false, false) => format!("{prefix}::{path}"),
    }
}

fn module_item(
    code: &[char],
    at: usize,
    end: usize,
    declared: Declared,
    parsed: &mut Parsed,
) -> usize {
    let (name, next) = after_word(code, at, end);
    let mut item = declared.item(Kind::Module, name, String::new(), Vec::new());
    let next = if code.get(next) == Some(&'{') {
        let close = closing(code, next).min(end);
        item.body = Some(
            code[next + 1..close.saturating_sub(1).max(next + 1)]
                .iter()
                .collect(),
        );
        close
    } else {
        statement_end(code, next, end)
    };
    parsed.items.push(item);
    next
}

fn function_item(
    code: &[char],
    at: usize,
    end: usize,
    declared: Declared,
    parsed: &mut Parsed,
) -> usize {
    let (name, next) = after_word(code, at, end);
    let stop = until(code, next, end, &['{', ';']);
    let header = normalize(&code[next..stop]);
    parsed
        .items
        .push(declared.item(Kind::Function, name, header, Vec::new()));
    statement_end(code, stop, end)
}

fn struct_item(
    code: &[char],
    at: usize,
    end: usize,
    kind: Kind,
    declared: Declared,
    parsed: &mut Parsed,
) -> usize {
    let (name, next) = after_word(code, at, end);
    let stop = until(code, next, end, &['{', '(', ';']);
    let mut header = normalize(&code[next..stop]);
    let (members, after) = match code.get(stop) {
        Some('{') if stop < end => {
            let close = closing(code, stop).min(end);
            (fields(code, stop + 1, close.saturating_sub(1)), close)
        }
        Some('(') if stop < end => {
            let close = closing(code, stop).min(end);
            let semicolon = until(code, close, end, &[';']);
            header.push_str(&normalize(&code[stop..semicolon]));
            (Vec::new(), (semicolon + 1).min(end))
        }
        _ => (Vec::new(), (stop + 1).min(end)),
    };
    parsed
        .items
        .push(declared.item(kind, name, header, members));
    after
}

fn parts(code: &[char], from: usize, to: usize) -> Vec<Vec<char>> {
    let mut found = Vec::new();
    let mut start = from;
    while start < to {
        let stop = until(code, start, to, &[',']);
        found.push(code[start..stop].to_vec());
        start = stop + 1;
    }
    found
}

fn unattributed(part: &[char]) -> &[char] {
    let mut at = 0;
    loop {
        while at < part.len() && part[at].is_whitespace() {
            at += 1;
        }
        if part.get(at) == Some(&'#') && part.get(at + 1) == Some(&'[') {
            at = closing(part, at + 1);
            continue;
        }
        return &part[at.min(part.len())..];
    }
}

fn fields(code: &[char], from: usize, to: usize) -> Vec<Member> {
    let mut members = Vec::new();
    for part in parts(code, from, to) {
        let part = unattributed(&part);
        let (public, at) = visibility(part, 0, part.len());
        if !public {
            continue;
        }
        let (name, next) = after_word(part, at, part.len());
        if name.is_empty() {
            continue;
        }
        members.push(Member {
            word: "field".to_owned(),
            name,
            rest: normalize(&part[next..]),
        });
    }
    members
}

fn enum_item(
    code: &[char],
    at: usize,
    end: usize,
    declared: Declared,
    parsed: &mut Parsed,
) -> usize {
    let (name, next) = after_word(code, at, end);
    let stop = until(code, next, end, &['{', ';']);
    let header = normalize(&code[next..stop]);
    let mut members = Vec::new();
    let mut after = (stop + 1).min(end);
    if code.get(stop) == Some(&'{') && stop < end {
        let close = closing(code, stop).min(end);
        for part in parts(code, stop + 1, close.saturating_sub(1)) {
            let part = unattributed(&part);
            let (variant, rest) = after_word(part, 0, part.len());
            if variant.is_empty() {
                continue;
            }
            let rest = normalize(&part[rest..]);
            let rest = if rest.starts_with('{') || rest.starts_with('=') {
                format!(" {rest}")
            } else {
                rest
            };
            members.push(Member {
                word: "variant".to_owned(),
                name: variant,
                rest,
            });
        }
        after = close;
    }
    parsed
        .items
        .push(declared.item(Kind::Enum, name, header, members));
    after
}

fn trait_item(
    code: &[char],
    at: usize,
    end: usize,
    declared: Declared,
    parsed: &mut Parsed,
) -> usize {
    let (name, next) = after_word(code, at, end);
    let stop = until(code, next, end, &['{', ';']);
    let header = normalize(&code[next..stop]);
    let mut members = Vec::new();
    let mut after = (stop + 1).min(end);
    if code.get(stop) == Some(&'{') && stop < end {
        let close = closing(code, stop).min(end);
        let mut inner = Parsed::default();
        items(code, stop + 1, close.saturating_sub(1), &mut inner);
        members = inner.items.into_iter().map(member).collect();
        after = close;
    }
    parsed
        .items
        .push(declared.item(Kind::Trait, name, header, members));
    after
}

fn member(item: Item) -> Member {
    let word = match item.kind {
        Kind::Function => format!("{}fn", item.qualifiers),
        Kind::Static => format!("static {}", item.qualifiers).trim_end().to_owned(),
        Kind::Module
        | Kind::Struct
        | Kind::Enum
        | Kind::Union
        | Kind::Trait
        | Kind::Alias
        | Kind::Constant => item.kind.word().to_owned(),
    };
    Member {
        word,
        name: item.name,
        rest: item.header,
    }
}

fn alias_item(
    code: &[char],
    at: usize,
    end: usize,
    declared: Declared,
    parsed: &mut Parsed,
) -> usize {
    let (name, next) = after_word(code, at, end);
    let stop = until(code, next, end, &[';']);
    let header = normalize(&code[next..stop]);
    parsed
        .items
        .push(declared.item(Kind::Alias, name, header, Vec::new()));
    (stop + 1).min(end)
}

fn value_item(
    code: &[char],
    at: usize,
    end: usize,
    kind: Kind,
    mut declared: Declared,
    parsed: &mut Parsed,
) -> usize {
    let (mut name, mut next) = after_word(code, at, end);
    if name == "mut" {
        declared.qualifiers.push_str("mut ");
        (name, next) = after_word(code, next, end);
    }
    let stop = until(code, next, end, &['=', ';']);
    let header = normalize(&code[next..stop]);
    if name != "_" {
        parsed
            .items
            .push(declared.item(kind, name, header, Vec::new()));
    }
    statement_end(code, stop, end)
}

fn impl_item(code: &[char], at: usize, end: usize, parsed: &mut Parsed) -> usize {
    let mut at = at;
    if code.get(at) == Some(&'<') {
        let mut angles = 0_usize;
        while at < end {
            match code[at] {
                '<' => angles += 1,
                '>' if !arrow(code, at) => {
                    angles = angles.saturating_sub(1);
                    if angles == 0 {
                        at += 1;
                        break;
                    }
                }
                _ => {}
            }
            at += 1;
        }
    }
    let stop = until(code, at, end, &['{', ';']);
    let header = normalize(&code[at..stop]);
    let header = top_level(&header, " where ").map_or(header.as_str(), |(head, _)| head);
    let (trait_name, target) = match top_level(header, " for ") {
        Some((trait_name, target)) => (Some(trait_name.trim().to_owned()), target),
        None => (None, header),
    };
    let target = base_name(target);
    let mut members = Vec::new();
    let mut after = (stop + 1).min(end);
    if code.get(stop) == Some(&'{') && stop < end {
        let close = closing(code, stop).min(end);
        let mut inner = Parsed::default();
        items(code, stop + 1, close.saturating_sub(1), &mut inner);
        if trait_name.is_none() {
            members = inner
                .items
                .into_iter()
                .filter(|item| item.public)
                .map(member)
                .collect();
        }
        after = close;
    }
    if !target.is_empty() {
        parsed.impls.push(Impl {
            target,
            trait_name,
            members,
        });
    }
    after
}

fn top_level<'a>(text: &'a str, separator: &str) -> Option<(&'a str, &'a str)> {
    let mut angles = 0_usize;
    let mut depth = 0_usize;
    let bytes = text.as_bytes();
    for (at, character) in text.char_indices() {
        match character {
            '<' => angles += 1,
            '>' if at == 0 || !matches!(bytes[at - 1], b'-' | b'=') => {
                angles = angles.saturating_sub(1);
            }
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            _ => {}
        }
        if angles == 0 && depth == 0 && text[at..].starts_with(separator) {
            return Some((&text[..at], &text[at + separator.len()..]));
        }
    }
    None
}

fn base_name(target: &str) -> String {
    let mut target = target.trim();
    for prefix in ["&", "mut ", "dyn ", "'static "] {
        target = target.trim_start_matches(prefix).trim_start();
    }
    let path = target.split('<').next().unwrap_or(target);
    path.rsplit("::")
        .next()
        .unwrap_or(path)
        .trim()
        .chars()
        .take_while(|&character| identifier(character))
        .collect()
}
