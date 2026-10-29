use gltf::accessor::Dimensions;
use serde_json::Value;

use engine_core::{Error, ErrorCode, Expected};

use crate::{MAX_INPUT_BYTES, gltf_error, unsupported};

pub(crate) fn container(bytes: &[u8]) -> Result<Value, Error> {
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(Error::new(ErrorCode::SourceTooLarge, "$")
            .at_most(MAX_INPUT_BYTES as u64)
            .got(bytes.len()));
    }
    let invalid = || gltf_error("$");
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
    serde_json::from_slice(&bytes[20..json_end]).map_err(|_| Error::new(ErrorCode::InvalidJson, "$"))
}

fn number(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

const MAX_DEPTH: usize = 64;

pub(crate) const SUPPORTED_EXTENSIONS: [&str; 2] =
    ["KHR_texture_transform", "KHR_materials_emissive_strength"];

pub(crate) fn structure(json: &Value) -> Result<Vec<String>, Error> {
    if json["asset"]["version"] != "2.0" {
        return Err(unsupported("$.asset.version")
            .got(&json["asset"]["version"])
            .variants(["2.0"]));
    }
    if let Some(version) = json["asset"]
        .get("minVersion")
        .filter(|version| *version != "2.0")
    {
        return Err(unsupported("$.asset.minVersion")
            .got(version)
            .variants(["2.0"]));
    }
    if let Some(buffers) = json["buffers"].as_array().filter(|items| items.len() > 1) {
        return Err(unsupported("$.buffers").at_most(1).got(buffers.len()));
    }
    let names = |field: &str| -> Vec<&str> {
        json[field]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect()
    };
    if let Some(required) = names("extensionsRequired")
        .into_iter()
        .find(|name| !SUPPORTED_EXTENSIONS.contains(name))
    {
        return Err(unsupported("$.extensionsRequired")
            .got(required)
            .variants(SUPPORTED_EXTENSIONS));
    }
    let ignored = names("extensionsUsed")
        .into_iter()
        .filter(|name| !SUPPORTED_EXTENSIONS.contains(name))
        .map(str::to_owned)
        .collect();
    let scenes = json["scenes"]
        .as_array()
        .filter(|scenes| scenes.len() == 1)
        .ok_or_else(|| {
            unsupported("$.scenes")
                .range(1.0, 1.0)
                .got(json["scenes"].as_array().map_or(0, Vec::len))
        })?;
    let nodes = json["nodes"]
        .as_array()
        .ok_or_else(|| gltf_error("$.nodes").expected(Expected::Array))?;
    let roots = scenes[0]["nodes"]
        .as_array()
        .ok_or_else(|| gltf_error("$.scenes[0].nodes").expected(Expected::Array))?;
    let mut visited = vec![false; nodes.len()];
    for root in roots {
        visit(root, nodes, &mut visited, 0)?;
    }
    if let Some(index) = visited.iter().position(|seen| !seen) {
        return Err(unsupported(format!("$.nodes[{index}]")));
    }
    for (index, mesh) in json["meshes"].as_array().into_iter().flatten().enumerate() {
        if mesh.get("weights").is_some() {
            return Err(unsupported(format!("$.meshes[{index}].weights")).expected(Expected::Absent));
        }
    }
    Ok(ignored)
}

fn visit(
    value: &Value,
    nodes: &[Value],
    visited: &mut [bool],
    depth: usize,
) -> Result<(), Error> {
    if depth > MAX_DEPTH {
        return Err(Error::new(ErrorCode::HierarchyDepth, "$.nodes")
            .at_most(MAX_DEPTH as u64)
            .got(depth));
    }
    let index = value
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .filter(|index| *index < nodes.len())
        .ok_or_else(|| {
            gltf_error("$.nodes")
                .at_most(nodes.len().saturating_sub(1) as u64)
                .got(value)
        })?;
    if std::mem::replace(&mut visited[index], true) {
        return Err(Error::new(
            ErrorCode::HierarchyCycle,
            format!("$.nodes[{index}]"),
        ));
    }
    let node = &nodes[index];
    if node.get("matrix").is_some()
        && ["translation", "rotation", "scale"]
            .iter()
            .any(|field| node.get(field).is_some())
    {
        return Err(gltf_error(format!("$.nodes[{index}].matrix")).expected(Expected::Absent));
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
            return Err(
                Error::new(ErrorCode::InvalidRotation, format!("$.nodes[{index}].rotation"))
                    .expected(Expected::Normalized)
                    .got(rotation),
            );
        }
    }
    if node.get("weights").is_some() {
        return Err(unsupported(format!("$.nodes[{index}].weights")).expected(Expected::Absent));
    }
    for child in node["children"].as_array().into_iter().flatten() {
        visit(child, nodes, visited, depth + 1)?;
    }
    Ok(())
}

pub(crate) fn buffers(document: &gltf::Document, binary: &[u8]) -> Result<(), Error> {
    let buffers: Vec<_> = document.buffers().collect();
    if buffers.len() != 1
        || !matches!(buffers[0].source(), gltf::buffer::Source::Bin)
        || buffers[0].length() > binary.len()
        || binary.len() - buffers[0].length() > 3
    {
        return Err(unsupported("$.buffers"));
    }
    for view in document.views() {
        let path = format!("$.bufferViews[{}]", view.index());
        if view.buffer().index() != 0
            || view
                .offset()
                .checked_add(view.length())
                .is_none_or(|end| end > buffers[0].length())
        {
            return Err(gltf_error(path));
        }
        if let Some(stride) = view
            .stride()
            .filter(|stride| !(4..=252).contains(stride) || stride % 4 != 0)
        {
            return Err(gltf_error(format!("{path}.byteStride"))
                .range(4.0, 252.0)
                .got(stride));
        }
    }
    for accessor in document.accessors() {
        let path = format!("$.accessors[{}]", accessor.index());
        if accessor.sparse().is_some()
            || matches!(accessor.dimensions(), Dimensions::Mat2 | Dimensions::Mat3)
        {
            return Err(unsupported(path));
        }
        let view = accessor
            .view()
            .ok_or_else(|| unsupported(format!("{path}.bufferView")).expected(Expected::Present))?;
        if accessor.count() == 0 {
            return Err(gltf_error(format!("{path}.count")).at_least(1).got(0));
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
            return Err(gltf_error(path));
        }
    }
    Ok(())
}
