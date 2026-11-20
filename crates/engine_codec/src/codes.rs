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

            pub const NAMES: &'static [&'static str] = &[$($code,)*];

            pub const fn as_str(&self) -> &str {
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

impl<'de> serde::Deserialize<'de> for ErrorCode {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::named(&name)
            .ok_or_else(|| <D::Error as serde::de::Error>::unknown_variant(&name, Self::NAMES))
    }
}

#[cfg(feature = "dev")]
impl schemars::JsonSchema for ErrorCode {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ErrorCode".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
        })
    }
}

error_codes! {
    InvalidJson => "invalid_json",
    UnknownField => "unknown_field",
    MissingField => "missing_field",
    InvalidType => "invalid_type",
    UnknownVariant => "unknown_variant",
    UnknownValue => "unknown_value",
    InvalidValue => "invalid_value",
    OutOfRange => "out_of_range",
    InvalidRequest => "invalid_request",
    UnsupportedCommand => "unsupported_command",
    InvalidUtf8 => "invalid_utf8",
    InvalidName => "invalid_name",
    InvalidPage => "invalid_page",
    InvalidPath => "invalid_path",
    InvalidCamera => "invalid_camera",
    InvalidInput => "invalid_input",
    InvalidRevision => "invalid_revision",
    InvalidArguments => "invalid_arguments",
    InvalidPresentationAlpha => "invalid_presentation_alpha",

    UnknownEntity => "unknown_entity",
    EntityRemoved => "entity_removed",
    MissingComponent => "missing_component",
    MissingResource => "missing_resource",
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
    InvalidMesh => "invalid_mesh",
    InvalidMaterial => "invalid_material",
    InvalidTexture => "invalid_texture",
    InvalidRenderable => "invalid_renderable",
    InvalidWorldTransform => "invalid_world_transform",
    InvalidEnvironment => "invalid_environment",

    BrowserCommandFailed => "browser_command_failed",
    RuntimePanicked => "runtime_panicked",
    ProfilingUnavailable => "profiling_unavailable",
    SessionClosed => "session_closed",
    RuntimeUnavailable => "runtime_unavailable",

    InvalidRecord => "invalid_record",
    UnsupportedFeature => "unsupported_feature",
    SectionUnlisted => "section_unlisted",

    PlaceNotFilled => "place_not_filled",
    NameTwice => "name_twice",
    PlaceCycle => "place_cycle",
    AmbiguousPasses => "ambiguous_passes",
    ReadBeforeWrite => "read_before_write",
    UsageConflict => "usage_conflict",
    BindingLimit => "binding_limit",
    UnknownVariable => "unknown_variable",
    PassMismatch => "pass_mismatch",
    ShaderLink => "shader_link",
    UnknownModule => "unknown_module",
    WgslUnused => "wgsl_unused",

    Comment => "comment",
    Marker => "marker",
    LintAllowed => "lint_allowed",
    LintMissing => "lint_missing",
    UnsafeUnlisted => "unsafe_unlisted",
    SafetyMissing => "safety_missing",
    ErrorType => "error_type",
    VersionUnpinned => "version_unpinned",
    NotInherited => "not_inherited",
    ExceptionUnused => "exception_unused",
    FileTooLong => "file_too_long",
    StructTooWide => "struct_too_wide",
}
