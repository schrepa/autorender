use gltf::accessor::Dimensions;
use serde_json::Value;

use crate::{ImportError, MAX_INPUT_BYTES, unsupported};

pub(crate) fn container(bytes: &[u8]) -> Result<Value, ImportError> {
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(ImportError::new(
            "input_too_large",
            "$",
            "GLB source exceeds 16 MiB",
        ));
    }
    let invalid = || {
        ImportError::new(
            "invalid_glb",
            "$",
            "expected an exact GLB 2 container with JSON followed by one BIN chunk",
        )
    };
    if bytes.len() < 28
        || &bytes[..4] != b"glTF"
        || number(bytes, 4) != Some(2)
        || number(bytes, 8) != Some(bytes.len() as u32)
    {
        return Err(invalid());
    }
    let json_len = number(bytes, 12).ok_or_else(invalid)? as usize;
    let json_end = 20usize
        .checked_add(json_len)
        .filter(|end| *end + 8 <= bytes.len())
        .ok_or_else(invalid)?;
    if !json_len.is_multiple_of(4)
        || &bytes[16..20] != b"JSON"
        || &bytes[json_end + 4..json_end + 8] != b"BIN\0"
    {
        return Err(invalid());
    }
    let binary_len = number(bytes, json_end).ok_or_else(invalid)? as usize;
    if !binary_len.is_multiple_of(4) || json_end + 8 + binary_len != bytes.len() {
        return Err(invalid());
    }
    serde_json::from_slice(&bytes[20..json_end])
        .map_err(|error| ImportError::new("invalid_json", "$", error.to_string()))
}

fn number(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

pub(crate) fn structure(json: &Value) -> Result<(), ImportError> {
    if json["asset"]["version"] != "2.0"
        || json["asset"]
            .get("minVersion")
            .is_some_and(|version| version != "2.0")
    {
        return Err(unsupported(
            "$.asset",
            "asset.version and any minVersion must be exactly 2.0",
        ));
    }
    for (field, maximum) in [
        ("buffers", 1),
        ("bufferViews", 512),
        ("accessors", 512),
        ("nodes", 1024),
        ("meshes", 128),
        ("materials", 256),
        ("images", 64),
        ("textures", 64),
        ("samplers", 64),
        ("scenes", 1),
    ] {
        if json[field]
            .as_array()
            .is_some_and(|items| items.len() > maximum)
        {
            return Err(ImportError::new(
                "count_limit",
                format!("$.{field}"),
                format!("at most {maximum} entries are supported"),
            ));
        }
    }
    for field in [
        "animations",
        "skins",
        "cameras",
        "extensionsUsed",
        "extensionsRequired",
    ] {
        if json[field]
            .as_array()
            .is_some_and(|items| !items.is_empty())
        {
            return Err(unsupported(
                format!("$.{field}"),
                "animations, skins, cameras and extensions are outside the static diffuse import profile",
            ));
        }
    }
    reject_extensions(json, "$", 0)?;
    let scenes = json["scenes"]
        .as_array()
        .filter(|scenes| scenes.len() == 1)
        .ok_or_else(|| unsupported("$.scenes", "exactly one scene is required"))?;
    let nodes = json["nodes"]
        .as_array()
        .ok_or_else(|| ImportError::new("invalid_nodes", "$.nodes", "a node array is required"))?;
    let roots = scenes[0]["nodes"].as_array().ok_or_else(|| {
        ImportError::new(
            "invalid_nodes",
            "$.scenes[0].nodes",
            "scene roots are required",
        )
    })?;
    let mut visited = vec![false; nodes.len()];
    for root in roots {
        visit(root, nodes, &mut visited, 0)?;
    }
    if visited.iter().any(|seen| !seen) {
        return Err(unsupported(
            "$.nodes",
            "every node must belong to the single scene; disconnected nodes or cycles are rejected",
        ));
    }
    for (index, mesh) in json["meshes"].as_array().into_iter().flatten().enumerate() {
        if mesh.get("weights").is_some() {
            return Err(unsupported(
                format!("$.meshes[{index}].weights"),
                "morph weights are unsupported",
            ));
        }
        if mesh["primitives"]
            .as_array()
            .is_some_and(|primitives| primitives.len() > 128)
        {
            return Err(ImportError::new(
                "count_limit",
                format!("$.meshes[{index}].primitives"),
                "at most 128 primitives per source mesh",
            ));
        }
    }
    Ok(())
}

fn reject_extensions(value: &Value, path: &str, depth: usize) -> Result<(), ImportError> {
    if depth > 64 {
        return Err(ImportError::new(
            "depth_limit",
            path,
            "JSON nesting exceeds 64",
        ));
    }
    match value {
        Value::Object(fields) => {
            for (key, child) in fields {
                if key == "extensions" && child.as_object().is_none_or(|object| !object.is_empty())
                {
                    return Err(unsupported(
                        format!("{path}.extensions"),
                        "all glTF extensions are unsupported, including compression and texture transforms",
                    ));
                }
                if key != "extras" {
                    reject_extensions(child, &format!("{path}.{key}"), depth + 1)?;
                }
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                reject_extensions(child, &format!("{path}[{index}]"), depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn visit(
    value: &Value,
    nodes: &[Value],
    visited: &mut [bool],
    depth: usize,
) -> Result<(), ImportError> {
    if depth > 64 {
        return Err(ImportError::new(
            "depth_limit",
            "$.nodes",
            "node hierarchy exceeds 64 levels",
        ));
    }
    let index = value
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .filter(|index| *index < nodes.len())
        .ok_or_else(|| {
            ImportError::new(
                "invalid_node",
                "$.nodes",
                "node reference is outside the node array",
            )
        })?;
    if std::mem::replace(&mut visited[index], true) {
        return Err(ImportError::new(
            "invalid_node_tree",
            format!("$.nodes[{index}]"),
            "cycles, shared children and repeated roots are unsupported",
        ));
    }
    let node = &nodes[index];
    if node.get("matrix").is_some()
        && ["translation", "rotation", "scale"]
            .iter()
            .any(|field| node.get(field).is_some())
    {
        return Err(ImportError::new(
            "invalid_transform",
            format!("$.nodes[{index}]"),
            "matrix and translation/rotation/scale cannot be combined",
        ));
    }
    if let Some(rotation) = node.get("rotation") {
        let unit = rotation
            .as_array()
            .filter(|items| items.len() == 4)
            .and_then(|items| items.iter().map(Value::as_f64).collect::<Option<Vec<_>>>())
            .is_some_and(|items| {
                (items.iter().map(|value| value * value).sum::<f64>() - 1.0).abs() <= 0.00001
            });
        if !unit {
            return Err(ImportError::new(
                "invalid_transform",
                format!("$.nodes[{index}].rotation"),
                "rotation must be a unit quaternion",
            ));
        }
    }
    for field in ["skin", "weights", "camera"] {
        if node.get(field).is_some() {
            return Err(unsupported(
                format!("$.nodes[{index}].{field}"),
                "only static mesh nodes and transform groups are supported",
            ));
        }
    }
    for child in node["children"].as_array().into_iter().flatten() {
        visit(child, nodes, visited, depth + 1)?;
    }
    Ok(())
}

pub(crate) fn buffers(document: &gltf::Document, binary: &[u8]) -> Result<(), ImportError> {
    let buffers: Vec<_> = document.buffers().collect();
    if buffers.len() != 1
        || !matches!(buffers[0].source(), gltf::buffer::Source::Bin)
        || buffers[0].length() > binary.len()
        || binary.len() - buffers[0].length() > 3
    {
        return Err(unsupported(
            "$.buffers",
            "exactly one embedded buffer matching the BIN chunk is required; external and data URIs are rejected",
        ));
    }
    for view in document.views() {
        let path = format!("$.bufferViews[{}]", view.index());
        if view.buffer().index() != 0
            || view
                .offset()
                .checked_add(view.length())
                .is_none_or(|end| end > buffers[0].length())
        {
            return Err(ImportError::new(
                "buffer_bounds",
                path,
                "buffer view exceeds the declared embedded buffer",
            ));
        }
        if view
            .stride()
            .is_some_and(|stride| !(4..=252).contains(&stride) || stride % 4 != 0)
        {
            return Err(ImportError::new(
                "invalid_stride",
                path,
                "vertex stride must be a multiple of four from 4 through 252",
            ));
        }
    }
    for accessor in document.accessors() {
        let path = format!("$.accessors[{}]", accessor.index());
        if accessor.sparse().is_some()
            || matches!(
                accessor.dimensions(),
                Dimensions::Mat2 | Dimensions::Mat3 | Dimensions::Mat4
            )
        {
            return Err(unsupported(
                path,
                "sparse and matrix accessors are unsupported",
            ));
        }
        let view = accessor
            .view()
            .ok_or_else(|| unsupported(&path, "accessors must have a buffer view"))?;
        if accessor.count() == 0 || accessor.count() > 98_304 {
            return Err(ImportError::new(
                "count_limit",
                path,
                "accessor count must be from 1 through 98304",
            ));
        }
        let size = accessor.size();
        let stride = view.stride().unwrap_or(size);
        let component_size = accessor.data_type().size();
        let end = (accessor.count() - 1)
            .checked_mul(stride)
            .and_then(|tail| tail.checked_add(size))
            .and_then(|length| accessor.offset().checked_add(length));
        let absolute = view.offset().checked_add(accessor.offset());
        if stride < size
            || accessor.offset() % component_size != 0
            || absolute.is_none_or(|offset| offset % component_size != 0)
            || end.is_none_or(|end| end > view.length())
        {
            return Err(ImportError::new(
                "accessor_bounds",
                path,
                "accessor alignment, stride or byte range is invalid",
            ));
        }
    }
    Ok(())
}
