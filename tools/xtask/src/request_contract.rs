//! Discoverable, validated input contract for local developer operations.

use serde_json::{Value, json};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// JSON Schema; engine workload options remain defined by bench/options.json.
pub fn schema(operation: &str) -> Value {
    let commit = json!({"type":"string", "pattern":"^[0-9a-fA-F]{40}$", "format":"git-commit"});
    let arm = json!({"type":"object", "required":["sha", "options"], "properties":{"sha":commit, "options":{"type":"object"}, "scene_file_hash":commit, "settings_hash":commit}, "additionalProperties":false});
    let check = json!({"type":"object", "required":["name", "sha", "expect"], "properties":{"name":{"type":"string", "minLength":1}, "sha":commit, "scripts_sha":commit, "expect":{"enum":["pass", "fail"]}}, "additionalProperties":false});
    let mut required = vec!["version", "operation", "setup", "environment"];
    if operation == "measure" {
        required.extend(["base_sha", "sha", "workloads"]);
    } else {
        required.push("checks");
    }
    json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema", "title":format!("Engine {operation} request v1"),
        "type":"object", "required":required, "additionalProperties":false,
        "properties":{
            "version":{"const":1}, "operation":{"const":operation},
            "setup":{"type":"object", "required":["runtime"], "properties":{"runtime":{"type":"string"}, "api":{"enum":["d3d12", "vulkan", "metal", "gl"]}}},
            "environment":{"type":"object", "description":"Caller-recorded hardware, driver and runtime identity; exact equality required for continuation."},
            "base_sha":commit, "sha":commit, "main_sha":commit, "tools_sha":commit,
            "pairs":{"type":"integer", "minimum":1},
            "deadline":{"type":"string", "format":"date-time"},
            "result_max_bytes":{"type":"integer", "minimum":1, "description":"Checkpoint admission ceiling, including raw evidence. In-progress writes may exceed it; use an OS quota for a hard bound."},
            "want":{"type":"object", "additionalProperties":false, "properties":{"profiles":{"type":"boolean"}, "shader_report":{"type":"boolean"}}},
            "checks":{"type":"array", "items":check},
            "workloads":{"type":"array", "minItems":1, "items":{"type":"object", "required":["name", "feature", "arms", "workload"], "additionalProperties":false, "properties":{
                "name":{"type":"string", "minLength":1}, "feature":{"type":"string", "minLength":1},
                "arms":{"type":"object", "required":["base", "candidate"], "properties":{"base":arm, "candidate":arm}, "additionalProperties":false},
                "workload":{"type":"object"}
            }}}
        }
    })
}

/// Enforce the emitted schema's constraints before creating an attempt.
pub fn validate(value: &Value, operation: &str) -> Result<(), String> {
    if !["measure", "checks"].contains(&operation) {
        return Err("unknown operation".into());
    }
    validate_at(value, &schema(operation), "request")
}

fn validate_at(value: &Value, shape: &Value, at: &str) -> Result<(), String> {
    let valid_type = match shape["type"].as_str() {
        Some("object") => value.is_object(),
        Some("array") => value.is_array(),
        Some("string") => value.is_string(),
        Some("boolean") => value.is_boolean(),
        Some("integer") => value.as_u64().is_some(),
        _ => true,
    };
    if !valid_type
        || shape.get("const").is_some_and(|expected| expected != value)
        || shape["enum"]
            .as_array()
            .is_some_and(|values| !values.contains(value))
    {
        return Err(format!("{at}: value does not match the operation schema"));
    }
    if let Some(minimum) = shape["minimum"].as_u64()
        && value.as_u64().is_none_or(|value| value < minimum)
    {
        return Err(format!("{at}: must be at least {minimum}"));
    }
    if let Some(text) = value.as_str() {
        if shape["minLength"]
            .as_u64()
            .is_some_and(|min| text.chars().count() < min as usize)
        {
            return Err(format!("{at}: string is too short"));
        }
        if shape["format"] == "git-commit"
            && (text.len() != 40 || !text.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Err(format!("{at}: expected a full commit id"));
        }
        if shape["format"] == "date-time" && deadline(text).is_none() {
            return Err(format!("{at}: invalid ISO-8601 timestamp"));
        }
    }
    if let Some(values) = value.as_array() {
        if shape["minItems"]
            .as_u64()
            .is_some_and(|min| values.len() < min as usize)
        {
            return Err(format!("{at}: array is empty"));
        }
        for (index, child) in values.iter().enumerate() {
            validate_at(child, &shape["items"], &format!("{at}[{index}]"))?;
        }
    }
    if let Some(values) = value.as_object() {
        for required in shape["required"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if !values.contains_key(required) {
                return Err(format!("{at}.{required}: required"));
            }
        }
        for (key, child) in values {
            let member = &shape["properties"][key];
            if member.is_null() && shape["additionalProperties"] == false {
                return Err(format!("{at}.{key}: unknown engine operation input"));
            }
            validate_at(child, member, &format!("{at}.{key}"))?;
        }
    }
    Ok(())
}

/// Strict calendar and timezone parsing; malformed deadlines never become an
/// unlimited operation. Fractional seconds are retained to nanosecond precision.
pub fn deadline(text: &str) -> Option<SystemTime> {
    if text.get(4..5)? != "-"
        || text.get(7..8)? != "-"
        || !matches!(text.get(10..11)?, "T" | "t")
        || text.get(13..14)? != ":"
        || text.get(16..17)? != ":"
    {
        return None;
    }
    let number = |range: std::ops::Range<usize>| text.get(range)?.parse::<i64>().ok();
    let (year, month, day) = (number(0..4)?, number(5..7)?, number(8..10)?);
    let (hour, minute, second) = (number(11..13)?, number(14..16)?, number(17..19)?);
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=12).contains(&month)
        || !(1..=days[(month - 1) as usize]).contains(&day)
        || !(0..24).contains(&hour)
        || !(0..60).contains(&minute)
        || !(0..60).contains(&second)
    {
        return None;
    }
    let mut rest = text.get(19..)?;
    let mut nanos = 0;
    if let Some(fraction) = rest.strip_prefix('.') {
        let digits = fraction.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return None;
        }
        let count = digits.min(9);
        nanos = fraction.get(..count)?.parse::<u32>().ok()? * 10u32.pow(9 - count as u32);
        rest = fraction.get(digits..)?;
    }
    let offset = match rest {
        "Z" | "z" => 0,
        _ => {
            if rest.len() != 6 || rest.get(3..4)? != ":" {
                return None;
            }
            let sign = match rest.get(..1)? {
                "+" => 1,
                "-" => -1,
                _ => return None,
            };
            let hours = rest.get(1..3)?.parse::<i64>().ok()?;
            let minutes = rest.get(4..6)?.parse::<i64>().ok()?;
            if !(0..24).contains(&hours) || !(0..60).contains(&minutes) {
                return None;
            }
            sign * (hours * 3600 + minutes * 60)
        }
    };
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    let seconds = days * 86_400 + hour * 3600 + minute * 60 + second - offset;
    UNIX_EPOCH.checked_add(Duration::new(u64::try_from(seconds).ok()?, nanos))
}
