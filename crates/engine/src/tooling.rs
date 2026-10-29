use std::fmt::Write;

use serde_json::Value;

pub fn print_generated(files: &[(&str, String)]) {
    let files: serde_json::Map<String, Value> = files
        .iter()
        .map(|(path, contents)| ((*path).to_owned(), Value::from(contents.as_str())))
        .collect();
    println!("{}", Value::Object(files));
}

pub fn declarations(schema: &Value, root: &str) -> String {
    let mut output = String::new();
    interface(&mut output, root, schema, false);
    for (name, definition) in schema["$defs"].as_object().into_iter().flatten() {
        interface(&mut output, name, definition, false);
    }
    output
}

pub fn message_interfaces(schema: &Value, root: &str, imported: &[(&str, &str)]) -> String {
    let mut output = String::new();
    for (name, module) in imported {
        writeln!(output, "import type {{ {name} }} from \"{module}\";").expect("string_writes");
    }
    interface(&mut output, root, schema, true);
    for (name, definition) in schema["$defs"].as_object().into_iter().flatten() {
        if !imported.iter().any(|(imported, _)| imported == name) {
            interface(&mut output, name, definition, true);
        }
    }
    output
}

pub fn constant(name: &str, value: &Value) -> String {
    format!("\nexport const {name} = {value};\n")
}

fn interface(output: &mut String, name: &str, schema: &Value, optional: bool) {
    output.push('\n');
    if schema.get("properties").is_none() {
        writeln!(output, "export type {name} = {};", type_of(schema)).expect("string_writes");
        return;
    }
    writeln!(output, "export interface {name} {{").expect("string_writes");
    let required = schema["required"].as_array();
    for (field, property) in schema["properties"].as_object().into_iter().flatten() {
        let absent =
            optional && !required.is_some_and(|required| required.iter().any(|name| name == field));
        let mark = if absent { "?" } else { "" };
        writeln!(output, "  {field}{mark}: {};", type_of(property)).expect("string_writes");
    }
    output.push_str("}\n");
}

fn type_of(schema: &Value) -> String {
    let annotation = |key: &String| ["description", "title", "default"].contains(&key.as_str());
    if schema == &Value::Bool(true)
        || schema
            .as_object()
            .is_some_and(|fields| fields.keys().all(annotation))
    {
        return "unknown".to_owned();
    }
    if let Some(reference) = schema["$ref"].as_str() {
        return reference
            .rsplit('/')
            .next()
            .expect("reference_names_definition")
            .to_owned();
    }
    if let Some(value) = schema.get("const").filter(|value| value.is_string()) {
        return value.to_string();
    }
    if let Some(values) = schema["enum"]
        .as_array()
        .filter(|values| values.iter().all(Value::is_string))
    {
        return values
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join(" | ");
    }
    if let Some(members) = schema["oneOf"].as_array().or(schema["anyOf"].as_array()) {
        let mut types: Vec<String> = Vec::new();
        for member in members {
            let member = type_of(member);
            if !types.contains(&member) {
                types.push(member);
            }
        }
        return types.join(" | ");
    }
    if let Some(properties) = schema["properties"].as_object() {
        let required = schema["required"].as_array();
        let fields: Vec<String> = properties
            .iter()
            .map(|(field, property)| {
                let required =
                    required.is_some_and(|required| required.iter().any(|name| name == field));
                let mark = if required { "" } else { "?" };
                format!("{field}{mark}: {}", type_of(property))
            })
            .collect();
        return format!("{{ {} }}", fields.join("; "));
    }
    let types: Vec<&str> = match &schema["type"] {
        Value::String(name) => vec![name.as_str()],
        Value::Array(names) => names.iter().filter_map(Value::as_str).collect(),
        other @ (Value::Null | Value::Bool(_) | Value::Number(_) | Value::Object(_)) => {
            panic!("schema_typescript_type {other} {schema}")
        }
    };
    types
        .into_iter()
        .map(|name| match name {
            "integer" | "number" => "number".to_owned(),
            "boolean" => "boolean".to_owned(),
            "string" => "string".to_owned(),
            "null" => "null".to_owned(),
            "array" => format!("{}[]", type_of(&schema["items"])),
            "object" if schema.get("properties").is_none() => {
                match schema.get("additionalProperties") {
                    Some(values) if values.is_object() => {
                        format!("Record<string, {}>", type_of(values))
                    }
                    _ => "Record<string, unknown>".to_owned(),
                }
            }
            other => panic!("schema_typescript_type {other:?}"),
        })
        .collect::<Vec<_>>()
        .join(" | ")
}
