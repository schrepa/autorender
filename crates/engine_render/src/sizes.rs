use std::collections::BTreeMap;

const DEEPEST: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Token<'t> {
    Word(&'t str),
    Number(&'t str),
    Mark(char),
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    size: u64,
    align: u64,
}

struct Member<'t> {
    align: Option<&'t [Token<'t>]>,
    size: Option<&'t [Token<'t>]>,
    ty: &'t [Token<'t>],
}

#[derive(Default)]
struct Items<'t> {
    structs: BTreeMap<&'t str, Vec<Member<'t>>>,
    aliases: BTreeMap<&'t str, &'t [Token<'t>]>,
    consts: BTreeMap<&'t str, &'t [Token<'t>]>,
}

type Bound<'t> = (&'t [Token<'t>], &'t [Token<'t>], &'t [Token<'t>]);

pub(crate) fn binding_sizes(text: &str) -> Vec<(u32, u32, u64)> {
    let tokens = tokens(text);
    let mut items = Items::default();
    let mut bound: Vec<Bound<'_>> = Vec::new();
    let mut attributes: (Option<&[Token<'_>]>, Option<&[Token<'_>]>) = (None, None);
    let mut at = 0;
    while let Some(token) = tokens.get(at) {
        match *token {
            Token::Mark('@') => {
                let (name, arguments, next) = attribute(&tokens, at);
                match name {
                    Some("group") => attributes.0 = arguments,
                    Some("binding") => attributes.1 = arguments,
                    Some(_) | None => {}
                }
                at = next;
                continue;
            }
            Token::Word("struct") => at = structure(&tokens, at, &mut items),
            Token::Word("alias") => {
                let end = until(&tokens, at, &[';']);
                if let (Some(&Token::Word(name)), Some(Token::Mark('='))) = (tokens.get(at + 1), tokens.get(at + 2))
                    && at + 3 <= end
                {
                    items.aliases.insert(name, &tokens[at + 3..end]);
                }
                at = end + 1;
            }
            Token::Word("const") => {
                let end = until(&tokens, at, &[';']);
                let equals = until(&tokens, at, &['=']).min(end);
                if let Some(&Token::Word(name)) = tokens.get(at + 1)
                    && equals < end
                {
                    items.consts.insert(name, &tokens[equals + 1..end]);
                }
                at = end + 1;
            }
            Token::Word("var") => {
                let (ty, next) = variable(&tokens, at);
                if let (Some(ty), (Some(grouped), Some(numbered))) = (ty, attributes) {
                    bound.push((grouped, numbered, ty));
                }
                at = next;
            }
            Token::Mark('{') => at = closing(&tokens, at, ('{', '}')).map_or(tokens.len(), |end| end + 1),
            Token::Word(_) | Token::Number(_) | Token::Mark(_) => at += 1,
        }
        attributes = (None, None);
    }
    bound
        .into_iter()
        .filter_map(|(group, binding, ty)| {
            let group = u32::try_from(items.number(group, 0)?).ok()?;
            let binding = u32::try_from(items.number(binding, 0)?).ok()?;
            Some((group, binding, items.layout(ty, 0)?.size))
        })
        .collect()
}

fn tokens(text: &str) -> Vec<Token<'_>> {
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(next) = text[at..].chars().next() {
        let rest = &text[at..];
        if next.is_whitespace() {
            at += next.len_utf8();
        } else if rest.starts_with("//") {
            at += rest.find('\n').unwrap_or(rest.len());
        } else if rest.starts_with("/*") {
            at += rest[2..].find("*/").map_or(rest.len(), |end| end + 4);
        } else if next.is_ascii_alphabetic() || next == '_' {
            let end = rest
                .find(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
                .unwrap_or(rest.len());
            found.push(Token::Word(&rest[..end]));
            at += end;
        } else if next.is_ascii_digit() {
            let end = rest
                .find(|character: char| !(character.is_ascii_alphanumeric() || character == '_' || character == '.'))
                .unwrap_or(rest.len());
            found.push(Token::Number(&rest[..end]));
            at += end;
        } else {
            found.push(Token::Mark(next));
            at += next.len_utf8();
        }
    }
    found
}

fn attribute<'t>(tokens: &'t [Token<'t>], at: usize) -> (Option<&'t str>, Option<&'t [Token<'t>]>, usize) {
    let Some(&Token::Word(name)) = tokens.get(at + 1) else {
        return (None, None, at + 1);
    };
    if tokens.get(at + 2) != Some(&Token::Mark('(')) {
        return (Some(name), None, at + 2);
    }
    match closing(tokens, at + 2, ('(', ')')) {
        Some(end) => (Some(name), Some(&tokens[at + 3..end]), end + 1),
        None => (Some(name), None, tokens.len()),
    }
}

fn closing(tokens: &[Token<'_>], at: usize, (open, close): (char, char)) -> Option<usize> {
    let mut depth = 0_u32;
    for (index, token) in tokens.iter().enumerate().skip(at) {
        match *token {
            Token::Mark(mark) if mark == open => depth += 1,
            Token::Mark(mark) if mark == close => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            Token::Word(_) | Token::Number(_) | Token::Mark(_) => {}
        }
    }
    None
}

fn until(tokens: &[Token<'_>], at: usize, stops: &[char]) -> usize {
    let mut depth = 0_i32;
    for (index, token) in tokens.iter().enumerate().skip(at) {
        match *token {
            Token::Mark('(' | '[' | '{') => depth += 1,
            Token::Mark(')' | ']' | '}') => depth -= 1,
            Token::Mark(mark) if depth == 0 && stops.contains(&mark) => return index,
            Token::Word(_) | Token::Number(_) | Token::Mark(_) => {}
        }
    }
    tokens.len()
}

fn split<'t>(tokens: &'t [Token<'t>]) -> Vec<&'t [Token<'t>]> {
    let mut parts = Vec::new();
    let mut depth = 0_i32;
    let mut start = 0;
    for (index, token) in tokens.iter().enumerate() {
        match *token {
            Token::Mark('(' | '[' | '<' | '{') => depth += 1,
            Token::Mark(')' | ']' | '>' | '}') => depth -= 1,
            Token::Mark(',') if depth == 0 => {
                parts.push(&tokens[start..index]);
                start = index + 1;
            }
            Token::Word(_) | Token::Number(_) | Token::Mark(_) => {}
        }
    }
    parts.push(&tokens[start..]);
    parts.retain(|part| !part.is_empty());
    parts
}

fn structure<'t>(tokens: &'t [Token<'t>], at: usize, items: &mut Items<'t>) -> usize {
    let (Some(&Token::Word(name)), Some(Token::Mark('{'))) = (tokens.get(at + 1), tokens.get(at + 2)) else {
        return at + 1;
    };
    let Some(end) = closing(tokens, at + 2, ('{', '}')) else {
        return tokens.len();
    };
    let members: Option<Vec<Member<'t>>> = split(&tokens[at + 3..end]).into_iter().map(member).collect();
    if let Some(members) = members {
        items.structs.insert(name, members);
    }
    end + 1
}

fn member<'t>(tokens: &'t [Token<'t>]) -> Option<Member<'t>> {
    let (mut align, mut size) = (None, None);
    let mut at = 0;
    while tokens.get(at) == Some(&Token::Mark('@')) {
        let (name, arguments, next) = attribute(tokens, at);
        match name {
            Some("align") => align = arguments,
            Some("size") => size = arguments,
            Some(_) | None => {}
        }
        at = next;
    }
    match (tokens.get(at), tokens.get(at + 1), tokens.get(at + 2..)) {
        (Some(Token::Word(_)), Some(Token::Mark(':')), Some(ty)) if !ty.is_empty() => Some(Member { align, size, ty }),
        _ => None,
    }
}

fn variable<'t>(tokens: &'t [Token<'t>], at: usize) -> (Option<&'t [Token<'t>]>, usize) {
    let mut next = at + 1;
    if tokens.get(next) == Some(&Token::Mark('<')) {
        match closing(tokens, next, ('<', '>')) {
            Some(end) => next = end + 1,
            None => return (None, tokens.len()),
        }
    }
    let end = until(tokens, next, &[';', '=']);
    let after = until(tokens, next, &[';']) + 1;
    match (tokens.get(next), tokens.get(next + 1)) {
        (Some(Token::Word(_)), Some(Token::Mark(':'))) if next + 2 < end => (Some(&tokens[next + 2..end]), after),
        _ => (None, after),
    }
}

fn arguments<'t>(rest: &'t [Token<'t>]) -> Option<Vec<&'t [Token<'t>]>> {
    match (rest.first(), rest.last()) {
        (None, _) => Some(Vec::new()),
        (Some(Token::Mark('<')), Some(Token::Mark('>'))) if rest.len() >= 2 => Some(split(&rest[1..rest.len() - 1])),
        _ => None,
    }
}

fn stride(element: Layout) -> u64 {
    element.size.next_multiple_of(element.align.max(1))
}

fn vector(count: u64, scalar: u64) -> Layout {
    Layout {
        size: count * scalar,
        align: if count == 2 { 2 * scalar } else { 4 * scalar },
    }
}

fn scalar(arguments: &[&[Token<'_>]], suffix: &str) -> Option<u64> {
    match (suffix, arguments) {
        ("", [element]) => match *element {
            [Token::Word("f32" | "i32" | "u32")] => Some(4),
            [Token::Word("f16")] => Some(2),
            _ => None,
        },
        ("f" | "i" | "u", []) => Some(4),
        ("h", []) => Some(2),
        _ => None,
    }
}

fn dimension(digit: Option<&str>) -> Option<u64> {
    digit?.parse::<u64>().ok().filter(|count| (2..=4).contains(count))
}

fn shaped(name: &str, arguments: &[&[Token<'_>]]) -> Option<Layout> {
    if let Some(shape) = name.strip_prefix("vec") {
        let count = dimension(shape.get(..1))?;
        return Some(vector(count, scalar(arguments, shape.get(1..)?)?));
    }
    let shape = name.strip_prefix("mat")?;
    if shape.get(1..2) != Some("x") {
        return None;
    }
    let (columns, rows) = (dimension(shape.get(..1))?, dimension(shape.get(2..3))?);
    let column = vector(rows, scalar(arguments, shape.get(3..)?)?);
    Some(Layout {
        size: columns * column.align,
        align: column.align,
    })
}

fn integer(text: &str) -> Option<i64> {
    let digits = text.strip_suffix(['u', 'i']).unwrap_or(text);
    match digits.strip_prefix("0x").or_else(|| digits.strip_prefix("0X")) {
        Some(hex) => i64::from_str_radix(hex, 16).ok(),
        None => digits.parse().ok(),
    }
}

impl Items<'_> {
    fn layout(&self, ty: &[Token<'_>], depth: u32) -> Option<Layout> {
        if depth > DEEPEST {
            return None;
        }
        let (&Token::Word(name), rest) = ty.split_first()? else {
            return None;
        };
        let arguments = arguments(rest)?;
        match (name, arguments.as_slice()) {
            ("f32" | "i32" | "u32" | "bool", []) => Some(Layout { size: 4, align: 4 }),
            ("f16", []) => Some(Layout { size: 2, align: 2 }),
            ("atomic", [_]) => Some(Layout { size: 4, align: 4 }),
            ("array", [element]) => {
                let element = self.layout(element, depth + 1)?;
                Some(Layout {
                    size: stride(element),
                    align: element.align,
                })
            }
            ("array", [element, count]) => {
                let element = self.layout(element, depth + 1)?;
                let count = self.number(count, depth + 1)?;
                Some(Layout {
                    size: stride(element).checked_mul(count)?,
                    align: element.align,
                })
            }
            _ => {
                if let Some(members) = self.structs.get(name) {
                    return self.members(members, depth);
                }
                if let Some(aliased) = self.aliases.get(name) {
                    return self.layout(aliased, depth + 1);
                }
                shaped(name, &arguments)
            }
        }
    }

    fn members(&self, members: &[Member<'_>], depth: u32) -> Option<Layout> {
        let mut end = 0_u64;
        let mut align = 1_u64;
        for member in members {
            let natural = self.layout(member.ty, depth + 1)?;
            let member_align = match member.align {
                Some(value) => self.number(value, depth + 1)?,
                None => natural.align,
            }
            .max(1);
            let member_size = match member.size {
                Some(value) => self.number(value, depth + 1)?,
                None => natural.size,
            };
            end = end.next_multiple_of(member_align).checked_add(member_size)?;
            align = align.max(member_align);
        }
        Some(Layout {
            size: end.next_multiple_of(align),
            align,
        })
    }

    fn number(&self, tokens: &[Token<'_>], depth: u32) -> Option<u64> {
        if depth > DEEPEST {
            return None;
        }
        let mut at = 0;
        let value = self.sum(tokens, &mut at, depth)?;
        if at != tokens.len() {
            return None;
        }
        u64::try_from(value).ok()
    }

    fn sum(&self, tokens: &[Token<'_>], at: &mut usize, depth: u32) -> Option<i64> {
        let mut value = self.product(tokens, at, depth)?;
        loop {
            let adding = match tokens.get(*at) {
                Some(Token::Mark('+')) => true,
                Some(Token::Mark('-')) => false,
                _ => return Some(value),
            };
            *at += 1;
            let right = self.product(tokens, at, depth)?;
            value = if adding { value.checked_add(right)? } else { value.checked_sub(right)? };
        }
    }

    fn product(&self, tokens: &[Token<'_>], at: &mut usize, depth: u32) -> Option<i64> {
        let mut value = self.unary(tokens, at, depth)?;
        loop {
            let operator = match tokens.get(*at) {
                Some(&Token::Mark(mark @ ('*' | '/' | '%'))) => mark,
                _ => return Some(value),
            };
            *at += 1;
            let right = self.unary(tokens, at, depth)?;
            value = match operator {
                '*' => value.checked_mul(right)?,
                '/' => value.checked_div(right)?,
                _ => value.checked_rem(right)?,
            };
        }
    }

    fn unary(&self, tokens: &[Token<'_>], at: &mut usize, depth: u32) -> Option<i64> {
        let token = *tokens.get(*at)?;
        *at += 1;
        match token {
            Token::Mark('-') => self.unary(tokens, at, depth)?.checked_neg(),
            Token::Mark('(') => self.inner(tokens, at, depth),
            Token::Number(text) => integer(text),
            Token::Word("u32" | "i32") if tokens.get(*at) == Some(&Token::Mark('(')) => {
                *at += 1;
                self.inner(tokens, at, depth)
            }
            Token::Word(name) => {
                let value = self.consts.get(name)?;
                i64::try_from(self.number(value, depth + 1)?).ok()
            }
            Token::Mark(_) => None,
        }
    }

    fn inner(&self, tokens: &[Token<'_>], at: &mut usize, depth: u32) -> Option<i64> {
        let value = self.sum(tokens, at, depth)?;
        if tokens.get(*at) != Some(&Token::Mark(')')) {
            return None;
        }
        *at += 1;
        Some(value)
    }
}
