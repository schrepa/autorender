//! Every error code the engine, its scenes, the development protocol, the
//! browser bridge and the watch tools answer with: one list, which
//! `ErrorCode` is made of and `protocol.json` carries for agents, and which
//! the web host's check holds the bridge's codes to.

/// Declares [`ErrorCode`] from `Variant => "code"` entries, each described
/// by its doc comment.
macro_rules! error_codes {
    ($($(#[doc = $doc:literal])+ $variant:ident => $code:literal,)*) => {
        /// An error's code, as replies, scene errors and the catalog in
        /// `protocol.json` name it.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum ErrorCode {
            $($(#[doc = $doc])+ $variant,)*
        }

        impl ErrorCode {
            /// Every code with what it means, in catalog order.
            pub const ALL: &'static [(ErrorCode, &'static str, &'static str)] = &[
                $((ErrorCode::$variant, $code, concat!($($doc),+)),)*
            ];

            /// The code as replies name it.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(ErrorCode::$variant => $code,)*
                }
            }

            /// What the code means.
            pub fn description(self) -> &'static str {
                Self::ALL
                    .iter()
                    .find(|(code, _, _)| *code == self)
                    .map_or("", |(_, _, description)| description.trim())
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

error_codes! {
    // Requests and sources.
    /// The request or source is not JSON.
    InvalidJson => "invalid_json",
    /// An object has a field its type does not.
    UnknownField => "unknown_field",
    /// An object lacks a field its type requires.
    MissingField => "missing_field",
    /// A value has the wrong JSON type.
    InvalidType => "invalid_type",
    /// A tag, such as an op, names no variant its type has.
    UnknownVariant => "unknown_variant",
    /// A value has the right type but no meaning, such as a malformed number.
    InvalidValue => "invalid_value",
    /// A number lies outside its bounds.
    OutOfRange => "out_of_range",
    /// A request breaks a rule of its own; the message names it.
    InvalidRequest => "invalid_request",
    /// The request exceeds the request size limit.
    RequestTooLarge => "request_too_large",
    /// The reply would exceed the reply size limit; ask for less.
    ResponseTooLarge => "response_too_large",
    /// The protocol or scene version is not one this build reads.
    UnsupportedVersion => "unsupported_version",
    /// The field needs a newer scene version than the source declares.
    VersionRequired => "version_required",
    /// The game or host runs no such command.
    UnsupportedCommand => "unsupported_command",
    /// Text is not UTF-8.
    InvalidUtf8 => "invalid_utf8",
    /// A name is empty, too long or has characters names may not.
    InvalidName => "invalid_name",
    /// A paging limit or offset is out of bounds.
    InvalidPage => "invalid_page",
    /// A path does not name a file.
    InvalidPath => "invalid_path",
    /// A position is not finite or out of bounds.
    InvalidPosition => "invalid_position",
    /// A ray's direction is zero or not finite, or its length out of bounds.
    InvalidRay => "invalid_ray",
    /// A rotation is not a finite unit quaternion.
    InvalidRotation => "invalid_rotation",
    /// A camera is not finite or its ranges are unusable.
    InvalidCamera => "invalid_camera",
    /// An input axis or turn is out of bounds.
    InvalidInput => "invalid_input",
    /// A step count lies outside 1 to the most ticks a step advances.
    InvalidStep => "invalid_step",
    /// A selector names no entity, component or resource the way selectors do.
    InvalidSelector => "invalid_selector",
    /// A revision is not an unsigned decimal string.
    InvalidRevision => "invalid_revision",
    /// A command line's arguments are not ones the command takes.
    InvalidArguments => "invalid_arguments",
    /// A benchmark's mode, warmup or samples are out of bounds.
    InvalidBenchmark => "invalid_benchmark",
    /// A presentation alpha is not finite or outside 0 to 1.
    InvalidPresentationAlpha => "invalid_presentation_alpha",
    /// A query shape's dimensions are not finite or out of bounds.
    InvalidShape => "invalid_shape",

    // The engine and its scenes.
    /// The game refused a scene or request by a rule of its own; the message
    /// says which.
    GameRule => "game_rule",
    /// No authored scene is loaded.
    NoScene => "no_scene",
    /// No terrain answers to that name.
    UnknownTerrain => "unknown_terrain",
    /// Only terrains generated from layers take edits.
    NotGenerated => "not_generated",
    /// No entity answers to that selector.
    UnknownEntity => "unknown_entity",
    /// The authored entity was removed since.
    EntityRemoved => "entity_removed",
    /// The scene instance belongs to another engine's world.
    WrongWorld => "wrong_world",
    /// The entity lacks the component asked for.
    MissingComponent => "missing_component",
    /// The world lacks the resource asked for.
    MissingResource => "missing_resource",
    /// No inspector has that name; describe lists them.
    UnknownInspector => "unknown_inspector",
    /// The scene schema has no definition of that name; describe
    /// SceneDocument lists them.
    UnknownDefinition => "unknown_definition",
    /// Two inspectors share a name.
    DuplicateInspector => "duplicate_inspector",
    /// The development tools hold their most inspectors.
    InspectorLimit => "inspector_limit",
    /// An inspector's description or schema exceeds its bounds.
    InspectorTooLarge => "inspector_too_large",
    /// A lookup would scan past its bound.
    ScanLimit => "scan_limit",
    /// The saved source's revision is not the one expected.
    RevisionConflict => "revision_conflict",
    /// The scene revision cannot advance further.
    RevisionExhausted => "revision_exhausted",
    /// The candidate source changes the saved scene's ID.
    SceneIdMismatch => "scene_id_mismatch",
    /// The source exceeds the source size limit.
    SourceTooLarge => "source_too_large",
    /// The source changed on disk since it was read.
    SourceChanged => "source_changed",
    /// The source could not be read or written.
    SourceIo => "source_io",
    /// Another save holds the source.
    SourceLocked => "source_locked",
    /// Rendering after the command failed; simulation may have advanced.
    RenderFailed => "render_failed",
    /// Two scene records share an ID.
    DuplicateId => "duplicate_id",
    /// An entity repeats a tag.
    DuplicateTag => "duplicate_tag",
    /// An ID is empty or has characters IDs may not.
    InvalidId => "invalid_id",
    /// A number is not finite or out of its bounds.
    InvalidNumber => "invalid_number",
    /// A file URI is not one scenes may reference.
    InvalidUri => "invalid_uri",
    /// The scene references a file the host did not supply.
    MissingFile => "missing_file",
    /// A reference names no record of its kind.
    UnknownReference => "unknown_reference",
    /// A parent names no entity.
    UnknownParent => "unknown_parent",
    /// Parents form a cycle.
    HierarchyCycle => "hierarchy_cycle",
    /// A hierarchy is deeper than its bound.
    HierarchyDepth => "hierarchy_depth",
    /// The checked parents could not be installed.
    HierarchyInstallFailed => "hierarchy_install_failed",
    /// A prefab places another prefab.
    NestedPrefab => "nested_prefab",
    /// The world lacks the asset library scenes instantiate into.
    MissingAssetLibrary => "missing_asset_library",
    /// The scene's assets exceed the asset library's budget.
    AssetBudgetExceeded => "asset_budget_exceeded",
    /// A mesh is invalid.
    InvalidMesh => "invalid_mesh",
    /// A material is invalid.
    InvalidMaterial => "invalid_material",
    /// A texture is invalid.
    InvalidTexture => "invalid_texture",
    /// An animation or animation player is invalid.
    InvalidAnimation => "invalid_animation",
    /// A skin is invalid.
    InvalidSkin => "invalid_skin",
    /// A tint is invalid.
    InvalidTint => "invalid_tint",
    /// A renderable is invalid.
    InvalidRenderable => "invalid_renderable",
    /// A collider is invalid.
    InvalidCollider => "invalid_collider",
    /// A collider's world transform is unusable for collision.
    InvalidColliderTransform => "invalid_collider_transform",
    /// A world transform is not finite or degenerate.
    InvalidWorldTransform => "invalid_world_transform",
    /// A local transform is not finite or its rotation not a unit quaternion.
    InvalidTransform => "invalid_transform",
    /// A local light is invalid.
    InvalidLight => "invalid_light",
    /// A particle emitter is invalid.
    InvalidParticles => "invalid_particles",
    /// A reflection probe is invalid.
    InvalidReflectionProbe => "invalid_reflection_probe",
    /// A decal is invalid.
    InvalidDecal => "invalid_decal",
    /// A spline mesh is invalid.
    InvalidSpline => "invalid_spline",
    /// A terrain is invalid.
    InvalidTerrain => "invalid_terrain",
    /// A navigation agent is invalid.
    InvalidNavAgent => "invalid_nav_agent",
    /// A streamed placement is invalid.
    InvalidStreamed => "invalid_streamed",
    /// Only roots stand on the ground.
    InvalidGrounded => "invalid_grounded",
    /// An environment section is invalid.
    InvalidEnvironment => "invalid_environment",

    /// A request or scene needs a feature whose plugin the engine lacks:
    /// its message names the plugin to add.
    MissingPlugin => "missing_plugin",

    // Collision.
    /// A collider's dimensions are not finite or out of bounds.
    CollisionInvalidShape => "collision_invalid_shape",
    /// A collider's world transform is unusable for collision.
    CollisionInvalidTransform => "collision_invalid_transform",
    /// A ray is not finite or its direction is zero.
    CollisionInvalidRay => "collision_invalid_ray",
    /// A shape cast's shape, pose, motion or options are unusable.
    CollisionInvalidCast => "collision_invalid_cast",
    /// Collision budgets exceed their bounds.
    CollisionInvalidConfig => "collision_invalid_config",
    /// Not every collider entered the collision world.
    CollisionIncompleteWorld => "collision_incomplete_world",
    /// A query met more candidates than its budget.
    CollisionCandidateBudget => "collision_candidate_budget",
    /// Sensors met more pairs than their budget.
    CollisionPairBudget => "collision_pair_budget",
    /// A query found more hits than its budget.
    CollisionResultBudget => "collision_result_budget",
    /// A narrow-phase test did not converge.
    CollisionNumericalFailure => "collision_numerical_failure",

    // The browser host and bridge.
    /// The command needs the host paused.
    PauseRequired => "pause_required",
    /// Another development command is running; retry.
    CommandBusy => "command_busy",
    /// The command failed in the browser; diagnostics hold the cause.
    BrowserCommandFailed => "browser_command_failed",
    /// The game's module panicked; the message holds the panic. The page
    /// must reload.
    RuntimePanicked => "runtime_panicked",
    /// A capture needs the host paused.
    CaptureRequiresPause => "capture_requires_pause",
    /// The capture could not be taken.
    CaptureFailed => "capture_failed",
    /// The benchmark's canvas, visibility or pause changed while it ran.
    BenchmarkInterrupted => "benchmark_interrupted",
    /// The benchmark ran past its time.
    BenchmarkTimeout => "benchmark_timeout",
    /// A benchmark measured no usable sample.
    InvalidMeasurement => "invalid_measurement",
    /// The build lacks the profiling instrumentation.
    ProfilingUnavailable => "profiling_unavailable",
    /// The profile exceeds its size limit.
    ProfileTooLarge => "profile_too_large",
    /// The bridge did not answer.
    BridgeUnreachable => "bridge_unreachable",
    /// The request came from an origin the bridge refuses.
    OriginRejected => "origin_rejected",
    /// The request's content type is not JSON.
    ContentType => "content_type",
    /// No route answers to that path.
    RouteNotFound => "route_not_found",
    /// The command is not one the bridge runs.
    InvalidCommand => "invalid_command",
    /// A request ID is missing or malformed.
    InvalidRequestId => "invalid_request_id",
    /// A request with that ID already ran.
    DuplicateRequest => "duplicate_request",
    /// A URL is not one sessions may open.
    InvalidUrl => "invalid_url",
    /// The bridge's queue is full; retry later.
    QueueFull => "queue_full",
    /// The request waited too long in the bridge's queue.
    QueueTimeout => "queue_timeout",
    /// The browser did not answer in time; the command's outcome is unknown.
    CommandTimeout => "command_timeout",
    /// The browser answered with something not a reply.
    InvalidBrowserResponse => "invalid_browser_response",
    /// Whether the request ran is unknown; inspect before retrying.
    RequestOutcomeUnknown => "request_outcome_unknown",
    /// No session answers to that ID.
    InvalidSession => "invalid_session",
    /// The session belongs to another bridge or build.
    SessionMismatch => "session_mismatch",
    /// The session is not available.
    SessionUnavailable => "session_unavailable",
    /// The session closed.
    SessionClosed => "session_closed",
    /// The session was cancelled.
    SessionCancelled => "session_cancelled",
    /// A new build is replacing the session.
    SessionReplacing => "session_replacing",
    /// The session ran its most requests.
    SessionRequestLimit => "session_request_limit",
    /// The browser runtime is busy.
    RuntimeBusy => "runtime_busy",
    /// The runtime is not paused.
    RuntimeNotPaused => "runtime_not_paused",
    /// The browser runtime changed; start a new session.
    RuntimeReplaced => "runtime_replaced",
    /// The browser runtime is not available.
    RuntimeUnavailable => "runtime_unavailable",
    /// The page reports no runtime identity.
    MissingRuntimeIdentity => "missing_runtime_identity",
    /// The page runs another build than the session's.
    BuildMismatch => "build_mismatch",

    // The watch tools.
    /// The watch is already running.
    WatchLocked => "watch_locked",
    /// The watch stopped.
    IterationStopped => "iteration_stopped",
    /// A newer build replaced this one.
    BuildSuperseded => "build_superseded",
    /// The build worker is not available.
    BuildWorkerUnavailable => "build_worker_unavailable",
    /// The build worker failed.
    BuildWorkerFailed => "build_worker_failed",
    /// The build's result is malformed.
    InvalidBuildResult => "invalid_build_result",
    /// The build's result exceeds its size limit.
    BuildResultTooLarge => "build_result_too_large",
    /// A build candidate is malformed.
    InvalidCandidate => "invalid_candidate",
    /// A source fingerprint is malformed.
    InvalidFingerprint => "invalid_fingerprint",
    /// An output path lies outside the build's folder.
    UnsafeOutputPath => "unsafe_output_path",
    /// Activating the build failed.
    ActivationFailed => "activation_failed",
    /// The build was already activated.
    ActivationCommitted => "activation_committed",
    /// The activation was not committed.
    ActivationNotCommitted => "activation_not_committed",
    /// The activation expired.
    ActivationExpired => "activation_expired",
    /// The watched sources exceed their budget.
    SourceBudget => "source_budget",
    /// A watched source is a symbolic link.
    SourceSymlink => "source_symlink",
    /// A watched source is not a file the watch reads.
    SourceType => "source_type",
}
