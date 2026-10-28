#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AdapterStatus {
    pub name: String,
    pub vendor: String,
    pub architecture: String,
    pub device_type: String,
    pub driver: String,
    pub backend: String,
}

impl AdapterStatus {
    pub fn of(adapter: &crate::AdapterInfo) -> Self {
        let driver = [adapter.driver.as_str(), adapter.driver_info.as_str()]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        Self {
            name: adapter.name.clone(),
            vendor: vendor(adapter).to_owned(),
            architecture: String::new(),
            device_type: adapter.device_type.clone(),
            driver,
            backend: adapter.backend.clone(),
        }
    }
}

pub fn vendor(adapter: &crate::AdapterInfo) -> &'static str {
    match adapter.vendor {
        0x10de => "nvidia",
        0x1002 | 0x1022 => "amd",
        0x8086 => "intel",
        0x106b => "apple",
        0x5143 => "qualcomm",
        0x13b5 => "arm",
        _ if adapter.name.starts_with("Apple") => "apple",
        0 => "",
        _ => "other",
    }
}
