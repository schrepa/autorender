//! One error for games: every engine error converts to it with `?`, keeping
//! a code from the catalog (`ErrorCode`), the JSON path or field at fault
//! where there is one, and a message for people.
//!
//! # When the engine panics, refuses, or degrades
//!
//! - **Panic** only on a broken invariant of the engine's own code, with an
//!   `expect` or `unreachable!` message that names the invariant. Nothing a
//!   game, scene, request, file, worker, device or budget supplies may
//!   panic: shipped builds abort, and Wasm cannot unwind, so
//!   `catch_unwind` would catch nothing.
//! - **Return an [`Error`]** when a caller asked for something that cannot
//!   be done. Check before changing anything, so a failure changes nothing.
//!   A library crate's own error converts into [`Error`] here, through an
//!   exhaustive match with no `_` arm, so a new variant chooses its code.
//!   `Result<_, String>` is for tools, binaries and job payloads only.
//! - **Degrade and report** when engine-scheduled work fails or is refused
//!   during a frame: skip the work and keep the frame, and count what was
//!   left out as `dropped_*` in the frame's statistics, or as a failure
//!   with its latest message in the feature's inspector. Never `.ok()` an
//!   engine error away uncounted. Only the failure path allocates.
//!
//! # Codes
//!
//! Every code is declared once, in the catalog ([`ErrorCode`],
//! `engine_settings::codes`); Rust never spells one as a string. A code
//! names the caller's remedy. [`ErrorCode::GameRule`] is only for a game's
//! own refusals, made with [`Error::game_rule`]. Lint fails at a public
//! engine error type without a conversion here.

use std::fmt;

use engine_core::terrain_data::{FormatError, codec::CodecError};
use engine_core::{
    AssetError, BiomeError, ErrorCode, HierarchyError, InterpolationError, InvalidSetting,
    LayerError, PyramidError, SimplifyError, TransformEditError,
};

/// An engine or game error: its code, where, and what is wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub code: ErrorCode,
    /// The JSON path or field at fault, `$` when none applies.
    pub path: String,
    pub message: String,
}

/// A result whose error is the engine's.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The name of the JavaScript `Error`s the engine throws on a page, which
/// carry its `code` and `path` (the page's generated `engineError` tells
/// them apart).
pub const JS_ERROR_NAME: &str = "EngineError";

impl Error {
    pub fn new(code: ErrorCode, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
        }
    }

    /// A game's own refusal (`game_rule`) with its message: a request the
    /// game's rules refuse, where the engine's would take it.
    pub fn game_rule(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::GameRule, "$", message)
    }

    /// The error with `what` it was doing before its message (`what:
    /// message`); its code and path stay.
    #[must_use]
    pub fn context(mut self, what: impl fmt::Display) -> Self {
        self.message = format!("{what}: {}", self.message);
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at {}: {}",
            self.code, self.path, self.message
        )
    }
}

impl std::error::Error for Error {}

impl From<engine_scene::SceneError> for Error {
    fn from(error: engine_scene::SceneError) -> Self {
        Self::new(error.code, error.path, error.message)
    }
}

#[cfg(feature = "dev")]
impl From<engine_dev::DevError> for Error {
    fn from(error: engine_dev::DevError) -> Self {
        Self::new(error.code, error.path, error.message)
    }
}

impl From<engine_collision::CollisionError> for Error {
    fn from(error: engine_collision::CollisionError) -> Self {
        Self::new(error.code(), "$", error.to_string())
    }
}

impl From<InvalidSetting> for Error {
    fn from(invalid: InvalidSetting) -> Self {
        Self::new(
            ErrorCode::OutOfRange,
            format!("$.{}", invalid.field),
            invalid.rule.into_owned(),
        )
    }
}

impl From<AssetError> for Error {
    fn from(error: AssetError) -> Self {
        let code = match error {
            AssetError::InvalidColor { .. }
            | AssetError::UnsupportedAlpha
            | AssetError::InvalidOpacity
            | AssetError::InvalidSurface { .. }
            | AssetError::InvalidMaterialMap { .. }
            | AssetError::InvalidMaterialHandle => ErrorCode::InvalidMaterial,
            AssetError::InvalidTextureDimensions { .. }
            | AssetError::InvalidTextureLength { .. }
            | AssetError::UnsupportedTextureAlpha { .. }
            | AssetError::InvalidTextureHandle
            | AssetError::InvalidCompressedTexture { .. } => ErrorCode::InvalidTexture,
            AssetError::InvalidAnimation { .. } => ErrorCode::InvalidAnimation,
            AssetError::MeshCapacity { .. }
            | AssetError::MaterialCapacity { .. }
            | AssetError::TextureCapacity { .. }
            | AssetError::CpuBudget { .. } => ErrorCode::AssetBudgetExceeded,
            AssetError::EmptyMesh
            | AssetError::InvalidTriangleCount { .. }
            | AssetError::InvalidVertex { .. }
            | AssetError::InvalidNormal { .. }
            | AssetError::InvalidUv { .. }
            | AssetError::AttributeCount { .. }
            | AssetError::InvalidAttribute { .. }
            | AssetError::IndexOutOfBounds { .. }
            | AssetError::InvalidVirtualMesh { .. } => ErrorCode::InvalidMesh,
        };
        Self::new(code, "$", error.to_string())
    }
}

impl From<HierarchyError> for Error {
    fn from(error: HierarchyError) -> Self {
        let code = match error {
            HierarchyError::MissingEntity(_) => ErrorCode::UnknownEntity,
            HierarchyError::MissingTransform(_) => ErrorCode::MissingComponent,
            HierarchyError::Cycle(_) | HierarchyError::DuplicateChild(_) => {
                ErrorCode::HierarchyCycle
            }
            HierarchyError::DepthExceeded | HierarchyError::CapacityExceeded => {
                ErrorCode::HierarchyDepth
            }
        };
        Self::new(code, "$", error.to_string())
    }
}

impl From<TransformEditError> for Error {
    fn from(error: TransformEditError) -> Self {
        let code = match error {
            TransformEditError::MissingEntity(_) => ErrorCode::UnknownEntity,
            TransformEditError::MissingTransform(_) => ErrorCode::MissingComponent,
        };
        Self::new(code, "$", error.to_string())
    }
}

impl From<SimplifyError> for Error {
    fn from(error: SimplifyError) -> Self {
        let code = match error {
            SimplifyError::TriangleCount { .. }
            | SimplifyError::IndexOutOfBounds { .. }
            | SimplifyError::TooLarge { .. }
            | SimplifyError::AttributeLayout { .. }
            | SimplifyError::VertexFlagCount { .. } => ErrorCode::InvalidMesh,
            SimplifyError::InvalidWeight { .. } | SimplifyError::InvalidTargetError => {
                ErrorCode::InvalidValue
            }
        };
        Self::new(code, "$", error.to_string())
    }
}

#[cfg(feature = "dev")]
impl From<engine_core::DevStepError> for Error {
    fn from(error: engine_core::DevStepError) -> Self {
        Self::new(ErrorCode::InvalidStep, "$.ticks", error.to_string())
    }
}

impl From<engine_render::RenderError> for Error {
    fn from(error: engine_render::RenderError) -> Self {
        Self::new(ErrorCode::RenderFailed, "$", error.to_string())
    }
}

impl From<CodecError> for Error {
    fn from(error: CodecError) -> Self {
        Self::new(ErrorCode::InvalidTerrain, "$", error.to_string())
    }
}

impl From<FormatError> for Error {
    fn from(error: FormatError) -> Self {
        let code = match error {
            FormatError::Magic
            | FormatError::Version(_)
            | FormatError::Pyramid(_)
            | FormatError::Step
            | FormatError::Truncated
            | FormatError::Corrupt(_) => ErrorCode::InvalidTerrain,
        };
        Self::new(code, "$", error.to_string())
    }
}

impl From<LayerError> for Error {
    fn from(error: LayerError) -> Self {
        Self::new(
            ErrorCode::InvalidTerrain,
            format!("$.layers[{}].{}", error.layer, error.field),
            error.rule.into_owned(),
        )
    }
}

impl From<PyramidError> for Error {
    fn from(error: PyramidError) -> Self {
        let path = match error {
            PyramidError::Tiles(_) => "$.tiles",
            PyramidError::Spacing(_) => "$.spacing",
        };
        Self::new(ErrorCode::InvalidTerrain, path, error.to_string())
    }
}

impl From<BiomeError> for Error {
    fn from(error: BiomeError) -> Self {
        Self::new(
            ErrorCode::InvalidTerrain,
            format!("$.{}", error.path),
            error.rule,
        )
    }
}

impl From<InterpolationError> for Error {
    fn from(error: InterpolationError) -> Self {
        Self::new(
            ErrorCode::InvalidPresentationAlpha,
            "$.alpha",
            error.to_string(),
        )
    }
}

/// What the page receives of a failure: an `Error` named
/// [`JS_ERROR_NAME`], whose `code` and `path` say what failed and where,
/// and whose message says both and what is wrong.
#[cfg(target_arch = "wasm32")]
impl From<Error> for wasm_bindgen::JsValue {
    fn from(error: Error) -> Self {
        let thrown = js_sys::Error::new(&error.to_string());
        thrown.set_name(JS_ERROR_NAME);
        let object: &js_sys::Object = thrown.as_ref();
        for (field, value) in [("code", error.code.as_str()), ("path", error.path.as_str())] {
            // Setting a field of a new error object cannot fail.
            let _ = js_sys::Reflect::set(object, &field.into(), &value.into());
        }
        thrown.into()
    }
}
