use engine_core::DebugViewMode;

pub(crate) fn shows_water(view: DebugViewMode) -> bool {
    matches!(
        view,
        DebugViewMode::WaterDepth
            | DebugViewMode::WaterSpeed
            | DebugViewMode::WaterFroude
            | DebugViewMode::WaterTurbulence
            | DebugViewMode::WaterWhite
            | DebugViewMode::WaterReach
            | DebugViewMode::WaterSpan
            | DebugViewMode::WaterEddies
            | DebugViewMode::WaterRoughness
    )
}

pub(crate) fn water_constants(water_views: bool) -> [(&'static str, f64); 1] {
    [("DEBUG_WATER_VIEWS", if water_views { 1.0 } else { 0.0 })]
}
