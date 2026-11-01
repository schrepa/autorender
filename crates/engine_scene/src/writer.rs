use engine_core::Error;
use engine_settings::ErrorCode;
use serde_json::{Map, Number, Value};

use crate::{MAX_SOURCE_BYTES, SceneDocument};

const MAX_DEPTH: usize = 64;

impl SceneDocument {
    pub fn to_canonical_json(&self, schema: &Value) -> Result<String, Error> {
        self.structure()?;
        let value =
            serde_json::to_value(self).map_err(|_| Error::new(ErrorCode::InvalidJson, "$"))?;
        let canonical = Canonical { root: schema }.value(&value, Some(schema), 0);
        let mut source = serde_json::to_string_pretty(&canonical)
            .map_err(|_| Error::new(ErrorCode::InvalidJson, "$"))?;
        source.push('\n');
        if source.len() > MAX_SOURCE_BYTES {
            return Err(Error::new(ErrorCode::SourceTooLarge, "$")
                .at_most(MAX_SOURCE_BYTES as u64)
                .got(source.len()));
        }
        Ok(source)
    }
}

struct Canonical<'a> {
    root: &'a Value,
}

impl<'a> Canonical<'a> {
    fn value(&self, value: &Value, schema: Option<&'a Value>, depth: usize) -> Value {
        let parts = self.parts_of(value, schema, depth);
        match value {
            Value::Number(number) if single(&parts) => narrowed(number),
            Value::Array(items) => Value::Array(
                items
                    .iter()
                    .enumerate()
                    .map(|(index, item)| self.value(item, item_schema(&parts, index), depth + 1))
                    .collect(),
            ),
            Value::Object(fields) => Value::Object(self.object(fields, &parts, depth)),
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => value.clone(),
        }
    }

    fn object(
        &self,
        fields: &Map<String, Value>,
        parts: &[&'a Value],
        depth: usize,
    ) -> Map<String, Value> {
        let mut names: Vec<&String> = fields.keys().collect();
        names.sort();
        let mut out = Map::new();
        for name in names {
            let field = &fields[name.as_str()];
            let property = property_schema(parts, name);
            let default = property
                .and_then(|property| property.get("default"))
                .filter(|_| !required(parts, name));
            if let Some(default) = default
                && self.same(field, default, property, depth + 1)
            {
                continue;
            }
            let canonical = self.value(field, property, depth + 1);
            if default.is_some() && empty(&canonical) {
                continue;
            }
            out.insert(name.clone(), canonical);
        }
        out
    }

    fn same(
        &self,
        value: &Value,
        default: &Value,
        schema: Option<&'a Value>,
        depth: usize,
    ) -> bool {
        if value == default {
            return true;
        }
        let parts = self.parts_of(value, schema, depth);
        if let (Value::Number(value), Value::Number(default)) = (value, default) {
            return single(&parts)
                && value.as_f64().map(|value| value as f32)
                    == default.as_f64().map(|default| default as f32);
        }
        if let (Value::Array(items), Value::Array(defaults)) = (value, default) {
            return items.len() == defaults.len()
                && items
                    .iter()
                    .zip(defaults)
                    .enumerate()
                    .all(|(index, (item, default))| {
                        self.same(item, default, item_schema(&parts, index), depth + 1)
                    });
        }
        if let (Value::Object(fields), Value::Object(defaults)) = (value, default) {
            return fields.len() == defaults.len()
                && fields.iter().all(|(name, field)| {
                    defaults.get(name).is_some_and(|default| {
                        self.same(field, default, property_schema(&parts, name), depth + 1)
                    })
                });
        }
        false
    }

    fn parts_of(&self, value: &Value, schema: Option<&'a Value>, depth: usize) -> Vec<&'a Value> {
        let mut parts = Vec::new();
        if let Some(schema) = schema
            && depth <= MAX_DEPTH
            && self.matches(value, schema, 0)
        {
            self.parts(value, schema, &mut parts, 0);
        }
        parts
    }

    fn parts(&self, value: &Value, schema: &'a Value, out: &mut Vec<&'a Value>, depth: usize) {
        if depth > MAX_DEPTH {
            return;
        }
        let schema = self.resolve(schema);
        out.push(schema);
        if let Some(Value::Array(members)) = schema.get("allOf") {
            for member in members {
                self.parts(value, member, out, depth + 1);
            }
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(branches)) = schema.get(key)
                && let Some(branch) = branches
                    .iter()
                    .find(|branch| self.matches(value, branch, depth + 1))
            {
                self.parts(value, branch, out, depth + 1);
            }
        }
    }

    fn matches(&self, value: &Value, schema: &'a Value, depth: usize) -> bool {
        if depth > MAX_DEPTH {
            return false;
        }
        let schema = self.resolve(schema);
        if let Some(constant) = schema.get("const") {
            return value == constant;
        }
        if let Some(Value::Array(options)) = schema.get("enum") {
            return options.contains(value);
        }
        if let Some(kind) = schema.get("type")
            && !typed(value, kind)
        {
            return false;
        }
        if let Value::Object(fields) = value {
            if let Some(Value::Array(names)) = schema.get("required")
                && names
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|name| !fields.contains_key(name))
            {
                return false;
            }
            if let Some(Value::Object(properties)) = schema.get("properties")
                && properties.iter().any(|(name, property)| {
                    let constant = self.resolve(property).get("const");
                    constant.is_some()
                        && fields
                            .get(name)
                            .is_some_and(|field| Some(field) != constant)
                })
            {
                return false;
            }
        }
        if let Some(Value::Array(members)) = schema.get("allOf")
            && !members
                .iter()
                .all(|member| self.matches(value, member, depth + 1))
        {
            return false;
        }
        ["anyOf", "oneOf"]
            .into_iter()
            .all(|key| match schema.get(key) {
                Some(Value::Array(branches)) => branches
                    .iter()
                    .any(|branch| self.matches(value, branch, depth + 1)),
                Some(
                    Value::Null
                    | Value::Bool(_)
                    | Value::Number(_)
                    | Value::String(_)
                    | Value::Object(_),
                )
                | None => true,
            })
    }

    fn resolve(&self, schema: &'a Value) -> &'a Value {
        let mut schema = schema;
        for _ in 0..MAX_DEPTH {
            let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
                return schema;
            };
            let target = match reference.strip_prefix("#/$defs/") {
                Some(name) => self.root.get("$defs").and_then(|defs| defs.get(name)),
                None if reference == "#" => Some(self.root),
                None => None,
            };
            match target {
                Some(target) => schema = target,
                None => return schema,
            }
        }
        schema
    }
}

fn typed(value: &Value, kind: &Value) -> bool {
    let one = |name: &str| match name {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_u64() || value.is_i64(),
        _ => true,
    };
    match kind {
        Value::String(name) => one(name.as_str()),
        Value::Array(names) => names.iter().filter_map(Value::as_str).any(one),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::Object(_) => true,
    }
}

fn single(parts: &[&Value]) -> bool {
    parts
        .iter()
        .any(|part| part.get("format").and_then(Value::as_str) == Some("float"))
}

fn narrowed(number: &Number) -> Value {
    number
        .as_f64()
        .and_then(|wide| (wide as f32).to_string().parse::<f64>().ok())
        .and_then(Number::from_f64)
        .map_or_else(|| Value::Number(number.clone()), Value::Number)
}

fn item_schema<'a>(parts: &[&'a Value], index: usize) -> Option<&'a Value> {
    parts.iter().find_map(|part| {
        part.get("prefixItems")
            .and_then(Value::as_array)
            .and_then(|prefix| prefix.get(index))
            .or_else(|| part.get("items").filter(|items| items.is_object()))
    })
}

fn property_schema<'a>(parts: &[&'a Value], name: &str) -> Option<&'a Value> {
    parts
        .iter()
        .find_map(|part| {
            part.get("properties")
                .and_then(|properties| properties.get(name))
        })
        .or_else(|| {
            parts.iter().find_map(|part| {
                part.get("additionalProperties")
                    .filter(|schema| schema.is_object())
            })
        })
}

fn required(parts: &[&Value], name: &str) -> bool {
    parts.iter().any(|part| {
        part.get("required")
            .and_then(Value::as_array)
            .is_some_and(|names| names.iter().any(|required| required == name))
    })
}

fn empty(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.is_empty(),
        Value::Object(fields) => fields.is_empty(),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => false,
    }
}
