//! One error for games: every engine error converts to it with `?`, keeping
//! a code from the catalog (`ErrorCode`), the JSON path or field at fault
//! where there is one, and a message for people.

use std::fmt;

use engine_core::{
    AssetError, ErrorCode, HierarchyError, InterpolationError, InvalidSetting, TransformEditError,
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

/// A game's own refusal, with its message.
impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::new(ErrorCode::GameRule, "$", message)
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::from(message.to_owned())
    }
}

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
