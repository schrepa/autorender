use std::fmt;

use engine_core::terrain_data::codec::CodecError;
use engine_core::{
    AssetError, BiomeError, ErrorCode, HierarchyError, InterpolationError, InvalidSetting,
    LayerError, PyramidError, TransformEditError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub code: ErrorCode,
    pub path: String,
    pub message: String,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub const JS_ERROR_NAME: &str = "EngineError";

impl Error {
    pub fn new(code: ErrorCode, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
        }
    }

    pub fn game_rule(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::GameRule, "$", message)
    }

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
            | AssetError::InvalidMeshHandle
            | AssetError::InvalidLevels { .. }
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

#[cfg(target_arch = "wasm32")]
impl From<Error> for wasm_bindgen::JsValue {
    fn from(error: Error) -> Self {
        let thrown = js_sys::Error::new(&error.to_string());
        thrown.set_name(JS_ERROR_NAME);
        let object: &js_sys::Object = thrown.as_ref();
        for (field, value) in [("code", error.code.as_str()), ("path", error.path.as_str())] {
            let _ = js_sys::Reflect::set(object, &field.into(), &value.into());
        }
        thrown.into()
    }
}
