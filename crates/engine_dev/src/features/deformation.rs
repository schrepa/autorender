//! Deformation's development tools: the inspectors of deformers and the
//! tracks they press.

use serde_json::{Value, json};

use crate::{DevTools, InspectorDescriptor};

/// Registers deformation's inspectors.
pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools.register_component::<engine_core::Deformer>(
        InspectorDescriptor::generated(schemas, "Deformer")
            .with_default(json!(engine_core::Deformer::default())),
        |deformer| json!(deformer),
    )
    .expect("built-in descriptor is bounded");
    tools.register_resource::<engine_core::Tracks>(
        InspectorDescriptor::generated(schemas, "Tracks"),
        |tracks| {
            json!(crate::views::TracksView {
                stamps: tracks.len(),
                pressed: tracks.pressed(),
                dropped: tracks.dropped(),
                latest: tracks.latest().copied(),
            })
        },
    )
    .expect("built-in descriptor is bounded");
}

/// Adds the schemas of deformation's inspectors.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert("Deformer".into(), super::schema::<engine_core::Deformer>());
    schemas.insert("Tracks".into(), super::schema::<crate::views::TracksView>());
}
