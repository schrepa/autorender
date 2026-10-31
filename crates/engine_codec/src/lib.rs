use serde::{Serialize, de::DeserializeOwned};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecError(String);

impl std::fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CodecError {}

pub fn encode<T: Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec(value).unwrap_or_default()
}

pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CodecError> {
    serde_json::from_slice(bytes).map_err(|error| CodecError(error.to_string()))
}
