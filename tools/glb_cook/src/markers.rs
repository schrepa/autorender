use glam::Mat4;
use serde_json::{Value, json};

use crate::source::Source;

pub fn markers(source: &Source, space: Mat4, joint_of: impl Fn(usize) -> Option<usize>) -> Value {
    let found: Vec<Value> = source
        .gltf
        .nodes()
        .filter(|node| node.mesh().is_none())
        .map(|node| {
            let placed = space * source.world(node.index());
            let (scale, rotation, position) = placed.to_scale_rotation_translation();
            json!({
                "node": node.name(),
                "position": position.to_array(),
                "rotation": rotation.normalize().to_array(),
                "scale": scale.to_array(),
                "joint": joint_of(node.index()),
                "extras": source.extras("nodes", node.index()),
            })
        })
        .collect();
    json!({ "markers": found })
}
