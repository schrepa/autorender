use std::borrow::Cow;

use engine_core::{GpuStruct, Module, ShaderText, Source};
use engine_registry::Registry;

use crate::drawn::{DRAWN_MODULE, DRAWN_POSE_MODULE, POSES_MODULE};
use crate::linker::CORE_PACKAGE;

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0100_0000_01b3;
const VIEW_MODULE: &str = "engine::view";

macro_rules! shader_modules {
    ($($path:literal = $file:literal;)*) => {
        const TEXTS: &[ShaderText] = &[
            $(ShaderText::generated(
                $path,
                include_str!($file),
                concat!(env!("CARGO_MANIFEST_DIR"), "/src/", $file),
            ),)*
            crate::shared::generated(DRAWN_MODULE),
            crate::shared::generated(POSES_MODULE),
            crate::shared::generated(DRAWN_POSE_MODULE),
        ];
        const HASH: u64 = 0_u64 $(.wrapping_add(entry_hash($file, include_str!($file))))*;
    };
}

shader_modules! {
    "engine::fullscreen" = "fullscreen.wgsl";
    "engine::color" = "color.wgsl";
    "engine::math" = "math.wgsl";
    "engine::view" = "frame/view.wgsl";
    "engine::post::frame" = "effects.wgsl";
}

pub static CORE: Source = Source::generated(CORE_PACKAGE, false, HASH, TEXTS);

pub const FULLSCREEN: engine_core::Entry =
    engine_core::Entry::generated(Module::generated(&CORE, 0), "vs_fullscreen");

pub(crate) type Handed = Vec<(String, &'static str, Vec<Cow<'static, str>>)>;

pub(crate) fn core_module(path: &str) -> Module {
    let index = TEXTS.iter().position(|text| text.path() == path);
    Module::generated(
        &CORE,
        index.unwrap_or_else(|| panic!("core_module_listed {path}")) as u32,
    )
}

const fn fold(mut hash: u64, bytes: &[u8]) -> u64 {
    let mut at = 0;
    while at < bytes.len() {
        hash = (hash ^ bytes[at] as u64).wrapping_mul(PRIME);
        at += 1;
    }
    hash
}

const fn entry_hash(key: &str, text: &str) -> u64 {
    fold(fold(fold(OFFSET, key.as_bytes()), &[0]), text.as_bytes())
}

pub(crate) fn handed(registry: &Registry) -> Handed {
    let core = [
        (VIEW_MODULE, Cow::Borrowed(<crate::CameraUniform as GpuStruct>::WGSL)),
        (DRAWN_MODULE, Cow::Owned(crate::drawn::source())),
        (POSES_MODULE, Cow::Owned(engine_place::poses_wgsl())),
        (DRAWN_POSE_MODULE, Cow::Owned(crate::drawn::pose_source())),
    ];
    let core = (core.into_iter())
        .map(|(path, text)| (path.to_owned(), CORE_PACKAGE, vec![text]));
    let bricks = registry.iter().filter_map(|(_, record)| {
        let parts: Vec<Cow<'static, str>> = record
            .render_entries::<crate::registration::Generated>()
            .flat_map(|entry| entry.0.iter().map(|part| Cow::Borrowed(*part)))
            .collect();
        let prefix = record.prefix().filter(|_| !parts.is_empty())?;
        Some((format!("generated::{prefix}"), record.package, parts))
    });
    core.chain(bricks).collect()
}

pub(crate) fn hashed(handed: &Handed) -> Vec<(String, u64)> {
    (handed.iter())
        .map(|(path, _, parts)| {
            let start = fold(OFFSET, path.as_bytes());
            let hash = (parts.iter())
                .fold(start, |hash, part| fold(fold(hash, &[0]), part.as_bytes()));
            (path.clone(), hash)
        })
        .collect()
}

#[cfg(not(feature = "dev"))]
pub(crate) fn unchanged(
    registry: &Registry,
    sources: &[(String, u64)],
    generated: &[(String, u64)],
) -> Result<(), engine_core::Error> {
    let current: Vec<(String, u64)> = std::iter::once(&CORE)
        .chain(crate::linker::sources(registry))
        .map(|source| (source.package().to_owned(), source.hash()))
        .collect();
    let parts = hashed(&handed(registry));
    let changed = (current.iter().find(|source| !sources.contains(source)))
        .or_else(|| parts.iter().find(|part| !generated.contains(part)));
    match changed {
        Some((name, _)) => Err(engine_core::Error::new(
            engine_core::ErrorCode::SourceChanged,
            format!("$.shaders.{name}"),
        )),
        None => Ok(()),
    }
}
