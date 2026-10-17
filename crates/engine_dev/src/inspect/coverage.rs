//! Coverage inspection uses the renderer's prepared state, then accounts
//! for the entity's tint and vertex opacity without rebuilding histograms.

use super::*;
use crate::views::AlphaCoverageView;
use engine_core::AlphaCoverageState;

pub(super) fn register(tools: &mut DevTools, schemas: &Value) {
    tools
        .insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "AlphaCoverage"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Mesh3d>(entity).is_some())
            }),
            read: Box::new(|world, entity, context| {
                let entity = entity?;
                let mesh = world.get::<Mesh3d>(entity)?;
                let assets = world.get_resource::<AssetLibrary>()?;
                view(assets, mesh, world.get::<Tint>(entity), context).map(|view| json!(view))
            }),
        })
        .expect("built-in descriptor is bounded");
}

pub(super) fn view(
    assets: &AssetLibrary,
    mesh: &Mesh3d,
    tint: Option<&Tint>,
    context: &Context<'_>,
) -> Option<AlphaCoverageView> {
    let material = assets.material(mesh.material)?;
    let prepared = context
        .renderer
        .and_then(|renderer| renderer.alpha_coverage(material));
    let mut view = AlphaCoverageView {
        requested: material.alpha_coverage().into(),
        prepared,
        effective: "unknown",
        fallback_reason: Some(if context.renderer.is_none() {
            "renderer_unavailable"
        } else {
            "not_uploaded"
        }),
    };
    match prepared {
        Some(AlphaCoverageState::Legacy { reason }) => {
            view.effective = "legacy";
            view.fallback_reason = Some(reason);
        }
        Some(AlphaCoverageState::Corrected { .. }) => {
            view.effective = "corrected";
            view.fallback_reason = None;
            if tint.is_some_and(|tint| !opaque(tint.apply([1.0; 4])[3])) {
                view.effective = "legacy";
                view.fallback_reason = Some("instance_opacity");
            } else if let Some(colors) = assets.mesh(mesh.mesh).and_then(|mesh| mesh.colors())
                && colors.iter().any(|color| !opaque(color[3]))
            {
                view.effective = if colors.iter().any(|color| opaque(color[3])) {
                    "conditional"
                } else {
                    "legacy"
                };
                view.fallback_reason = Some("vertex_opacity");
            }
        }
        None => {}
    }
    Some(view)
}

// Instance and vertex colors reach the shader as 16-bit UNORM. A value
// just below one may round to one and must inspect as the shader sees it.
fn opaque(alpha: f32) -> bool {
    (engine_core::math::clamp(alpha, 0.0, 1.0) * 65535.0).round() == 65535.0
}
