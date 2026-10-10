//! The factory's report formats are JSON Schema (draft 2020-12, the subset
//! its documents use: type, const, enum, pattern, minLength, maxLength,
//! minimum, exclusiveMinimum, minItems, maxItems, items, properties,
//! required, additionalProperties, anyOf, oneOf, $ref and date-time):
//! reports are checked against them before they are sent.

use serde_json::Value;

/// What is wrong with `value` against `schema`, as `path: what`.
pub fn problems(schema: &Value, value: &Value) -> Vec<String> {
    let mut out = Vec::new();
    check(schema, schema, value, "report", &mut out);
    out.truncate(60);
    out
}

fn type_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn check(root: &Value, schema: &Value, value: &Value, at: &str, out: &mut Vec<String>) {
    if let Some(reference) = schema["$ref"].as_str() {
        let name = reference.rsplit('/').next().unwrap_or_default();
        return check(root, &root["$defs"][name], value, at, out);
    }
    for (key, exactly_one) in [("anyOf", false), ("oneOf", true)] {
        if let Some(options) = schema[key].as_array() {
            let results: Vec<Vec<String>> = options
                .iter()
                .map(|option| {
                    let mut found = Vec::new();
                    check(root, option, value, at, &mut found);
                    found
                })
                .collect();
            let passing = results.iter().filter(|found| found.is_empty()).count();
            if (exactly_one && passing != 1) || passing == 0 {
                let closest = results.into_iter().min_by_key(Vec::len).unwrap_or_default();
                if !closest.is_empty() && closest.len() <= 3 {
                    out.extend(closest);
                } else {
                    out.push(format!("{at}: none of its allowed shapes"));
                }
            }
            return;
        }
    }
    if let Some(constant) = schema.get("const") {
        if value != constant {
            out.push(format!("{at}: must be {constant}"));
        }
        return;
    }
    if let Some(allowed) = schema["enum"].as_array() {
        if !allowed.contains(value) {
            let names: Vec<String> = allowed.iter().map(Value::to_string).collect();
            out.push(format!("{at}: must be one of {}", names.join(", ")));
        }
        return;
    }
    if !schema["type"].is_null() {
        let types: Vec<&str> = match &schema["type"] {
            Value::Array(types) => types.iter().filter_map(Value::as_str).collect(),
            kind => kind.as_str().into_iter().collect(),
        };
        let fits = types.iter().any(|kind| match *kind {
            "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
            "number" => value.is_number(),
            kind => kind == type_of(value),
        });
        if !fits {
            out.push(format!("{at}: must be {}", types.join(" or ")));
            return;
        }
    }
    match value {
        Value::String(text) => {
            let length = text.chars().count() as u64;
            if schema["minLength"]
                .as_u64()
                .is_some_and(|minimum| length < minimum)
            {
                out.push(format!("{at}: must not be empty"));
            }
            if schema["maxLength"]
                .as_u64()
                .is_some_and(|maximum| length > maximum)
            {
                out.push(format!("{at}: too long"));
            }
            if let Some(pattern) = schema["pattern"]
                .as_str()
                .filter(|pattern| !matches(pattern, text))
            {
                out.push(format!("{at}: must be like {pattern}"));
            }
            if schema["format"] == "date-time" && !date_time(text) {
                out.push(format!("{at}: must be an ISO date and time"));
            }
        }
        Value::Number(number) => {
            let number = number.as_f64().unwrap_or(f64::NAN);
            if schema["minimum"]
                .as_f64()
                .is_some_and(|minimum| number < minimum)
            {
                out.push(format!("{at}: must be at least {}", schema["minimum"]));
            }
            if schema["exclusiveMinimum"]
                .as_f64()
                .is_some_and(|minimum| number <= minimum)
            {
                out.push(format!(
                    "{at}: must be more than {}",
                    schema["exclusiveMinimum"]
                ));
            }
        }
        Value::Array(items) => {
            let count = items.len() as u64;
            if schema["minItems"]
                .as_u64()
                .is_some_and(|minimum| count < minimum)
            {
                out.push(format!("{at}: at least {} items", schema["minItems"]));
            }
            if schema["maxItems"]
                .as_u64()
                .is_some_and(|maximum| count > maximum)
            {
                out.push(format!("{at}: at most {} items", schema["maxItems"]));
            }
            if schema["items"].is_object() {
                for (index, item) in items.iter().enumerate() {
                    check(root, &schema["items"], item, &format!("{at}[{index}]"), out);
                }
            }
        }
        Value::Object(fields) => {
            for required in schema["required"].as_array().into_iter().flatten() {
                if let Some(name) = required.as_str().filter(|name| !fields.contains_key(*name)) {
                    out.push(format!("{at}.{name} is missing"));
                }
            }
            for (name, field) in fields {
                let path = format!("{at}.{name}");
                if let Some(property) = schema["properties"].get(name) {
                    check(root, property, field, &path, out);
                } else if schema["additionalProperties"] == false {
                    out.push(format!("{path} is not a field"));
                } else if schema["additionalProperties"].is_object() {
                    check(root, &schema["additionalProperties"], field, &path, out);
                }
            }
        }
        _ => {}
    }
}

/// Whether `text` looks like an ISO 8601 date and time.
fn date_time(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() >= 20
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && matches!(bytes[10], b'T' | b't')
        && bytes[13] == b':'
        && bytes[16] == b':'
        && [0..4, 5..7, 8..10, 11..13, 14..16, 17..19]
            .into_iter()
            .all(|range| bytes[range].iter().all(u8::is_ascii_digit))
}

/// A pattern's atom: the character ranges it takes, and at least and at
/// most how many times.
type Atom = (Vec<(char, char)>, usize, usize);

/// Whether `text` matches `pattern`, a regular expression of anchors,
/// literal characters, classes (`[a-z0-9.-]`) and counts (`{n}`, `{m,n}`),
/// which the formats' patterns are.
fn matches(pattern: &str, text: &str) -> bool {
    let mut atoms: Vec<Atom> = Vec::new();
    let mut chars = pattern
        .trim_start_matches('^')
        .trim_end_matches('$')
        .chars()
        .peekable();
    while let Some(c) = chars.next() {
        let ranges = if c == '[' {
            let mut ranges = Vec::new();
            let mut members: Vec<char> = Vec::new();
            for member in chars.by_ref() {
                if member == ']' {
                    break;
                }
                members.push(member);
            }
            let mut index = 0;
            while index < members.len() {
                if index + 2 < members.len() && members[index + 1] == '-' {
                    ranges.push((members[index], members[index + 2]));
                    index += 3;
                } else {
                    ranges.push((members[index], members[index]));
                    index += 1;
                }
            }
            ranges
        } else {
            vec![(c, c)]
        };
        let (mut minimum, mut maximum) = (1, 1);
        if chars.peek() == Some(&'{') {
            chars.next();
            let count: String = chars.by_ref().take_while(|&c| c != '}').collect();
            let (low, high) = count.split_once(',').unwrap_or((&count, &count));
            minimum = low.trim().parse().unwrap_or(0);
            maximum = high.trim().parse().unwrap_or(usize::MAX);
        }
        atoms.push((ranges, minimum, maximum));
    }
    fn fits(atoms: &[Atom], text: &[char]) -> bool {
        let Some(((ranges, minimum, maximum), rest)) = atoms.split_first() else {
            return text.is_empty();
        };
        let taken = text
            .iter()
            .take(*maximum)
            .take_while(|c| ranges.iter().any(|(low, high)| (low..=high).contains(c)))
            .count();
        (*minimum..=taken)
            .rev()
            .any(|count| fits(rest, &text[count..]))
    }
    let text: Vec<char> = text.chars().collect();
    fits(&atoms, &text)
}
