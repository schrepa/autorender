//! The GPU a host draws on, as every host's development status names it.

/// The GPU a host draws on, in the shape every host's development status
/// names it: what the renderer knows of it, which a page adds the browser's
/// vendor and architecture to.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AdapterStatus {
    /// Its name: the driver's natively; on a page the browser's
    /// description, which browsers may leave empty.
    pub name: String,
    /// Its vendor, in lowercase: natively by its PCI vendor ID, on a page
    /// as the browser names it.
    pub vendor: String,
    /// Its architecture, where the browser names it; empty natively.
    pub architecture: String,
    /// Its kind: `IntegratedGpu`, `DiscreteGpu`, `VirtualGpu`, `Cpu` or
    /// `Other`.
    pub device_type: String,
    /// Its driver and the driver's version, natively; empty on a page.
    pub driver: String,
    /// The GPU interface: `Dx12`, `Vulkan`, `Metal`, `Gl`, or
    /// `BrowserWebGpu` on a page.
    pub backend: String,
}

impl AdapterStatus {
    /// The status of the GPU `adapter` describes, without what only a
    /// page's browser knows.
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

/// The GPU's vendor as browsers name it, by its PCI vendor ID or name;
/// empty where neither tells, as on a page.
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
