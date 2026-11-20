use std::sync::Arc;

use engine_core::{Entry, Error, ErrorCode, Source, Var};
use serde::{Deserialize, Serialize};

#[cfg(feature = "dev")]
pub(crate) mod counting;
#[cfg(feature = "dev")]
mod driver;
#[cfg(feature = "dev")]
mod facts;
#[cfg(feature = "dev")]
mod link;

#[cfg(feature = "dev")]
pub(crate) use link::ShaderInputs;

pub const CORE_PACKAGE: &str = "engine_render";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Sample {
    Float,
    Depth,
    Uint,
    Sint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Ty {
    Uniform,
    Storage {
        write: bool,
    },
    Texture {
        sample: Sample,
        view: wgpu::TextureViewDimension,
    },
    StorageTexture {
        format: wgpu::TextureFormat,
        access: wgpu::StorageTextureAccess,
        view: wgpu::TextureViewDimension,
    },
    Sampler,
    Comparison,
}

impl Ty {
    pub(crate) fn binding(self, filterable: bool) -> wgpu::BindingType {
        let buffer = |ty| wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        match self {
            Self::Uniform => buffer(wgpu::BufferBindingType::Uniform),
            Self::Storage { write } => {
                buffer(wgpu::BufferBindingType::Storage { read_only: !write })
            }
            Self::Texture { sample, view } => wgpu::BindingType::Texture {
                sample_type: match sample {
                    Sample::Float => wgpu::TextureSampleType::Float { filterable },
                    Sample::Depth => wgpu::TextureSampleType::Depth,
                    Sample::Uint => wgpu::TextureSampleType::Uint,
                    Sample::Sint => wgpu::TextureSampleType::Sint,
                },
                view_dimension: view,
                multisampled: false,
            },
            Self::StorageTexture {
                format,
                access,
                view,
            } => wgpu::BindingType::StorageTexture {
                access,
                format,
                view_dimension: view,
            },
            Self::Sampler => wgpu::BindingType::Sampler(if filterable {
                wgpu::SamplerBindingType::Filtering
            } else {
                wgpu::SamplerBindingType::NonFiltering
            }),
            Self::Comparison => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
        }
    }

    pub(crate) fn writes(self) -> bool {
        match self {
            Self::Storage { write } => write,
            Self::StorageTexture { access, .. } => access != wgpu::StorageTextureAccess::ReadOnly,
            Self::Uniform | Self::Texture { .. } | Self::Sampler | Self::Comparison => false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Declaration {
    pub(crate) group: u32,
    pub(crate) binding: u32,
    pub(crate) name: String,
    pub(crate) ty: Ty,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct Fragment {
    pub(crate) outputs: Vec<(u32, wgpu::TextureSampleType)>,
    pub(crate) inputs: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Composed {
    pub(crate) module: String,
    pub(crate) entry: String,
    pub(crate) stage: wgpu::ShaderStages,
    pub(crate) text: String,
    pub(crate) overrides: Vec<(String, String)>,
    pub(crate) fragment: Fragment,
    pub(crate) reached: Vec<(String, Declaration)>,
}

impl Composed {
    pub(crate) fn fragment(&self, entry: &str) -> Option<&Fragment> {
        (self.entry == entry).then_some(&self.fragment)
    }

    pub(crate) fn constants<'a>(&'a self, values: &[(&str, f64)]) -> Vec<(&'a str, f64)> {
        values
            .iter()
            .filter_map(|&(name, value)| {
                self.overrides
                    .iter()
                    .find(|(plain, _)| plain == name)
                    .map(|(_, mangled)| (mangled.as_str(), value))
            })
            .collect()
    }
}

#[derive(bevy_ecs::resource::Resource, Clone, Copy)]
pub struct LinkedShaders(pub &'static str);

#[derive(Serialize, Deserialize)]
pub(crate) struct Shaders {
    sources: Vec<(String, u64)>,
    generated: Vec<(String, u64)>,
    modules: Vec<ShaderModule>,
}

#[derive(Serialize, Deserialize)]
struct ShaderModule {
    path: String,
    core: bool,
    declared: Vec<Declaration>,
    entries: Vec<Arc<Composed>>,
}

pub(crate) fn path_of(var_or_entry: (engine_core::Module, &str)) -> (&'static str, String) {
    let (module, name) = var_or_entry;
    let path = module
        .source()
        .modules()
        .get(module.index() as usize)
        .map_or("", |text| text.path());
    (path, name.to_owned())
}

pub(crate) fn sources(registry: &engine_registry::Registry) -> Vec<&'static Source> {
    let mut found: Vec<&'static Source> = Vec::new();
    for (_, record) in registry.iter() {
        for &source in record.render_entries::<&'static Source>() {
            if !found.iter().any(|known| std::ptr::eq(*known, source)) {
                found.push(source);
            }
        }
    }
    found
}

impl Shaders {
    #[cfg(not(feature = "dev"))]
    pub(crate) fn embedded(
        text: &str,
        registry: &engine_registry::Registry,
    ) -> Result<Self, Error> {
        let shaders: Self = serde_json::from_str(text).map_err(|error| {
            Error::new(ErrorCode::ShaderLink, "$.shaders").got(error.to_string())
        })?;
        crate::shaders::unchanged(registry, &shaders.sources, &shaders.generated)?;
        Ok(shaders)
    }

    fn module(&self, path: &str) -> Option<&ShaderModule> {
        self.modules.iter().find(|module| module.path == path)
    }

    fn composed(&self, entry: Entry) -> Option<&Arc<Composed>> {
        let (path, name) = path_of((entry.module(), entry.name()));
        (self.module(path)?.entries.iter()).find(|composed| composed.entry == name)
    }

    pub(crate) fn declared(&self, var: Var) -> Option<&Declaration> {
        let (path, name) = path_of((var.module(), var.name()));
        (self.module(path)?.declared.iter()).find(|declared| declared.name == name)
    }

    pub(crate) fn declarations(&self, path: &str) -> &[Declaration] {
        self.module(path).map_or(&[], |module| &module.declared)
    }

    pub(crate) fn buffer_usage(&self, var: Var) -> wgpu::BufferUsages {
        match self.declared(var).map(|declared| declared.ty) {
            Some(Ty::Uniform) => wgpu::BufferUsages::UNIFORM,
            Some(Ty::Storage { .. }) => wgpu::BufferUsages::STORAGE,
            Some(_) | None => wgpu::BufferUsages::empty(),
        }
    }

    pub(crate) fn has_entry(&self, entry: Entry) -> bool {
        self.composed(entry).is_some()
    }

    pub(crate) fn entry(&self, entry: Entry) -> Result<Arc<Composed>, Error> {
        self.composed(entry).map(Arc::clone).ok_or_else(|| {
            let (path, name) = path_of((entry.module(), entry.name()));
            Error::new(ErrorCode::UnknownModule, format!("$.shaders.{path}.{name}")).got(name)
        })
    }

    pub(crate) fn reaches(&self, entry: Entry, var: Var) -> bool {
        let (path, name) = path_of((var.module(), var.name()));
        self.composed(entry).is_some_and(|composed| {
            (composed.reached.iter()).any(|(module, found)| module == path && found.name == name)
        })
    }

    pub(crate) fn is_core(&self, path: &str) -> bool {
        self.module(path).is_some_and(|module| module.core)
    }

    pub(crate) fn frame(&self) -> Vec<(String, String, Ty, u32)> {
        let mut frame: Vec<(String, String, Ty, u32)> = (self.modules.iter())
            .flat_map(|module| {
                (module.declared.iter())
                    .filter(|declared| declared.group == 0)
                    .map(|declared| {
                        let name = declared.name.clone();
                        (module.path.clone(), name, declared.ty, declared.binding)
                    })
            })
            .collect();
        frame.sort_by_key(|(.., number)| *number);
        frame
    }
}
