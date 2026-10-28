macro_rules! error_codes {
    ($($variant:ident => $code:literal,)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum ErrorCode {
            $($variant,)*
        }

        impl ErrorCode {
            pub const ALL: &'static [(ErrorCode, &'static str)] = &[
                $((ErrorCode::$variant, $code),)*
            ];

            pub const fn as_str(self) -> &'static str {
                match self {
                    $(ErrorCode::$variant => $code,)*
                }
            }

            pub fn named(name: &str) -> Option<ErrorCode> {
                Self::ALL
                    .iter()
                    .find(|(_, code)| *code == name)
                    .map(|(code, _)| *code)
            }
        }
    };
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl serde::Serialize for ErrorCode {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for ErrorCode {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ErrorCode".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let codes: Vec<&str> = Self::ALL.iter().map(|&(_, code)| code).collect();
        schemars::json_schema!({
            "description": "An error's code, as replies name it.",
            "type": "string",
            "enum": codes,
        })
    }
}

error_codes! {
    InvalidJson => "invalid_json",
    UnknownField => "unknown_field",
    MissingField => "missing_field",
    InvalidType => "invalid_type",
    UnknownVariant => "unknown_variant",
    InvalidValue => "invalid_value",
    OutOfRange => "out_of_range",
    InvalidRequest => "invalid_request",
    RequestTooLarge => "request_too_large",
    ResponseTooLarge => "response_too_large",
    UnsupportedVersion => "unsupported_version",
    UnsupportedCommand => "unsupported_command",
    InvalidUtf8 => "invalid_utf8",
    InvalidName => "invalid_name",
    InvalidPage => "invalid_page",
    InvalidPath => "invalid_path",
    InvalidPosition => "invalid_position",
    InvalidRay => "invalid_ray",
    InvalidRotation => "invalid_rotation",
    InvalidCamera => "invalid_camera",
    InvalidInput => "invalid_input",
    InvalidStep => "invalid_step",
    InvalidSelector => "invalid_selector",
    InvalidRevision => "invalid_revision",
    InvalidArguments => "invalid_arguments",
    InvalidBenchmark => "invalid_benchmark",
    InvalidPresentationAlpha => "invalid_presentation_alpha",
    InvalidShape => "invalid_shape",

    GameRule => "game_rule",
    NoScene => "no_scene",
    UnknownTerrain => "unknown_terrain",
    UnknownEntity => "unknown_entity",
    EntityRemoved => "entity_removed",
    WrongWorld => "wrong_world",
    MissingComponent => "missing_component",
    MissingResource => "missing_resource",
    UnknownInspector => "unknown_inspector",
    UnknownDefinition => "unknown_definition",
    DuplicateInspector => "duplicate_inspector",
    ReservedInspector => "reserved_inspector",
    InspectorLimit => "inspector_limit",
    InspectorTooLarge => "inspector_too_large",
    ScanLimit => "scan_limit",
    RevisionConflict => "revision_conflict",
    RevisionExhausted => "revision_exhausted",
    SceneIdMismatch => "scene_id_mismatch",
    SourceTooLarge => "source_too_large",
    SourceChanged => "source_changed",
    SourceIo => "source_io",
    SourceLocked => "source_locked",
    RenderFailed => "render_failed",
    DuplicateId => "duplicate_id",
    DuplicateTag => "duplicate_tag",
    InvalidId => "invalid_id",
    InvalidNumber => "invalid_number",
    InvalidUri => "invalid_uri",
    MissingFile => "missing_file",
    UnknownReference => "unknown_reference",
    UnknownParent => "unknown_parent",
    HierarchyCycle => "hierarchy_cycle",
    HierarchyDepth => "hierarchy_depth",
    HierarchyInstallFailed => "hierarchy_install_failed",
    NestedPrefab => "nested_prefab",
    MissingAssetLibrary => "missing_asset_library",
    AssetBudgetExceeded => "asset_budget_exceeded",
    InvalidMesh => "invalid_mesh",
    InvalidMaterial => "invalid_material",
    InvalidTexture => "invalid_texture",
    InvalidRenderable => "invalid_renderable",
    InvalidCollider => "invalid_collider",
    InvalidColliderTransform => "invalid_collider_transform",
    InvalidWorldTransform => "invalid_world_transform",
    InvalidSpline => "invalid_spline",
    InvalidTerrain => "invalid_terrain",
    InvalidNavAgent => "invalid_nav_agent",
    InvalidStreamed => "invalid_streamed",
    InvalidGrounded => "invalid_grounded",
    InvalidRigidBody => "invalid_rigid_body",
    InvalidEnvironment => "invalid_environment",

    MissingPlugin => "missing_plugin",

    CollisionInvalidShape => "collision_invalid_shape",
    CollisionInvalidTransform => "collision_invalid_transform",
    CollisionInvalidRay => "collision_invalid_ray",
    CollisionInvalidCast => "collision_invalid_cast",
    CollisionInvalidConfig => "collision_invalid_config",
    CollisionIncompleteWorld => "collision_incomplete_world",
    CollisionCandidateBudget => "collision_candidate_budget",
    CollisionPairBudget => "collision_pair_budget",
    CollisionResultBudget => "collision_result_budget",
    CollisionNumericalFailure => "collision_numerical_failure",

    PauseRequired => "pause_required",
    CommandBusy => "command_busy",
    BrowserCommandFailed => "browser_command_failed",
    RuntimePanicked => "runtime_panicked",
    CaptureRequiresPause => "capture_requires_pause",
    CaptureFailed => "capture_failed",
    BenchmarkInterrupted => "benchmark_interrupted",
    BenchmarkTimeout => "benchmark_timeout",
    BenchmarkFailed => "benchmark_failed",
    InvalidMeasurement => "invalid_measurement",
    ProfilingUnavailable => "profiling_unavailable",
    PixUnavailable => "pix_unavailable",
    ProfileTooLarge => "profile_too_large",
    BridgeUnreachable => "bridge_unreachable",
    BridgeError => "bridge_error",
    OriginRejected => "origin_rejected",
    ContentType => "content_type",
    RouteNotFound => "route_not_found",
    InvalidCommand => "invalid_command",
    InvalidRequestId => "invalid_request_id",
    DuplicateRequest => "duplicate_request",
    InvalidUrl => "invalid_url",
    QueueFull => "queue_full",
    QueueTimeout => "queue_timeout",
    CommandTimeout => "command_timeout",
    InvalidBrowserResponse => "invalid_browser_response",
    RequestOutcomeUnknown => "request_outcome_unknown",
    InvalidSession => "invalid_session",
    SessionMismatch => "session_mismatch",
    SessionUnavailable => "session_unavailable",
    SessionClosed => "session_closed",
    SessionCancelled => "session_cancelled",
    SessionReplacing => "session_replacing",
    SessionRequestLimit => "session_request_limit",
    RuntimeBusy => "runtime_busy",
    RuntimeNotPaused => "runtime_not_paused",
    RuntimeReplaced => "runtime_replaced",
    RuntimeUnavailable => "runtime_unavailable",
    MissingRuntimeIdentity => "missing_runtime_identity",
    BuildMismatch => "build_mismatch",

    WatchLocked => "watch_locked",
    IterationStopped => "iteration_stopped",
    IterationError => "iteration_error",
    BuildSuperseded => "build_superseded",
    BuildWorkerUnavailable => "build_worker_unavailable",
    BuildWorkerFailed => "build_worker_failed",
    InvalidBuildResult => "invalid_build_result",
    BuildResultTooLarge => "build_result_too_large",
    InvalidCandidate => "invalid_candidate",
    InvalidFingerprint => "invalid_fingerprint",
    UnsafeOutputPath => "unsafe_output_path",
    ActivationFailed => "activation_failed",
    ActivationCommitted => "activation_committed",
    ActivationNotCommitted => "activation_not_committed",
    ActivationExpired => "activation_expired",
    SourceBudget => "source_budget",
    SourceSymlink => "source_symlink",
    SourceType => "source_type",
}
