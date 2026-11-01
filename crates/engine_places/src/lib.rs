#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate engine_ecs as bevy_ecs;

macro_rules! contract {
    ($file:literal $(, $name:ident)* $(,)?) => {{
        const CONSTANTS: &[(&str, u32)] = &[$((::core::stringify!($name), $name)),*];
        const TEXT: &str = ::core::include_str!($file);
        const BYTES: [u8; crate::contract::length(CONSTANTS, TEXT)] =
            crate::contract::written(CONSTANTS, TEXT);
        crate::contract::text(&BYTES)
    }};
}

pub mod bloom;
pub mod dither;
pub mod exposure;
pub mod ground;
pub mod lighting;
pub mod lights;
pub mod meshes;
pub mod navigation;
pub mod particles;
pub mod physics;
pub mod probes;
pub mod reflections;
pub mod shadows;
pub mod sky;
mod smokes;
pub mod solids;
pub mod splines;
pub mod streaming;
pub mod sun;
pub mod terrain;
pub mod tonemap;
pub mod water;
pub mod wind;

use engine_place::PlaceDecl;

mod contract {
    const LEAD: &[u8] = b"const ";
    const TYPED: &[u8] = b": u32 = ";
    const END: &[u8] = b"u;\n";
    const IMPORT: &[u8] = b"import ";

    const fn digits(value: u32) -> usize {
        let mut count = 1;
        let mut rest = value / 10;
        while rest > 0 {
            count += 1;
            rest /= 10;
        }
        count
    }

    pub(crate) const fn length(constants: &[(&str, u32)], text: &str) -> usize {
        let mut total = text.len();
        let mut at = 0;
        while at < constants.len() {
            let (name, value) = constants[at];
            total += LEAD.len() + name.len() + TYPED.len() + digits(value) + END.len();
            at += 1;
        }
        total
    }

    const fn opens(text: &[u8], at: usize) -> bool {
        let mut index = 0;
        while index < IMPORT.len() {
            if at + index >= text.len() || text[at + index] != IMPORT[index] {
                return false;
            }
            index += 1;
        }
        true
    }

    const fn imports_end(text: &[u8]) -> usize {
        let mut at = 0;
        loop {
            let mut next = at;
            while next < text.len() && text[next].is_ascii_whitespace() {
                next += 1;
            }
            if !opens(text, next) {
                return at;
            }
            at = next;
            while at < text.len() && text[at] != b';' {
                at += 1;
            }
            while at < text.len() && text[at] != b'\n' {
                at += 1;
            }
            if at < text.len() {
                at += 1;
            }
        }
    }

    const fn put<const N: usize>(out: &mut [u8; N], mut end: usize, bytes: &[u8]) -> usize {
        let mut at = 0;
        while at < bytes.len() {
            out[end] = bytes[at];
            end += 1;
            at += 1;
        }
        end
    }

    pub(crate) const fn written<const N: usize>(constants: &[(&str, u32)], text: &str) -> [u8; N] {
        let (imports, rest) = text.as_bytes().split_at(imports_end(text.as_bytes()));
        let mut out = [0; N];
        let mut end = put(&mut out, 0, imports);
        let mut at = 0;
        while at < constants.len() {
            let (name, value) = constants[at];
            end = put(&mut out, end, LEAD);
            end = put(&mut out, end, name.as_bytes());
            end = put(&mut out, end, TYPED);
            let mut place = digits(value);
            let mut rest = value;
            while place > 0 {
                place -= 1;
                out[end + place] = b'0' + (rest % 10) as u8;
                rest /= 10;
            }
            end += digits(value);
            end = put(&mut out, end, END);
            at += 1;
        }
        put(&mut out, end, rest);
        out
    }

    pub(crate) const fn text(bytes: &'static [u8]) -> &'static str {
        match core::str::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => panic!("contract_text_is_utf8"),
        }
    }
}

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

pub const PLACES: &[PlaceDecl] = &[
    PlaceDecl::slot::<sun::SunPath>(),
    PlaceDecl::slot::<sun::Sun>(),
    PlaceDecl::slot::<sky::SkyLight>(),
    PlaceDecl::slot::<exposure::Exposure>(),
    PlaceDecl::slot::<tonemap::Display>(),
    PlaceDecl::slot::<lighting::Lighting>(),
    PlaceDecl::slot::<lights::Lights>(),
    PlaceDecl::component::<lights::LocalLight>("local_light"),
    PlaceDecl::slot::<shadows::ShadowMaps>(),
    PlaceDecl::slot::<shadows::CasterRoutes>(),
    PlaceDecl::slot::<shadows::ShadowCasters>(),
    PlaceDecl::component::<shadows::ShadowChanges>("shadow_changes"),
    PlaceDecl::slot::<probes::ProbeViews>(),
    PlaceDecl::slot::<probes::ProbeCaptures>(),
    PlaceDecl::slot::<probes::ProbeLight>(),
    PlaceDecl::component::<probes::ReflectionProbe>("reflection_probe"),
    PlaceDecl::slot::<bloom::Glow>(),
    PlaceDecl::slot::<reflections::Reflected>(),
    PlaceDecl::slot::<dither::DitherPlace>(),
    PlaceDecl::slot::<meshes::Instances>(),
    PlaceDecl::component::<meshes::Cube>("cube"),
    PlaceDecl::component::<meshes::Tint>("tint"),
    PlaceDecl::slot::<particles::Transparency>(),
    PlaceDecl::component::<particles::ParticleEmitter>("particle_emitter"),
    PlaceDecl::slot::<ground::Ground>(),
    PlaceDecl::slot::<ground::TerrainTiles>(),
    PlaceDecl::slot::<ground::GroundSurface>(),
    PlaceDecl::slot::<ground::Heightfield>(),
    PlaceDecl::slot::<ground::GroundWork>(),
    PlaceDecl::slot::<ground::GroundTiles>(),
    PlaceDecl::slot::<ground::Horizons>(),
    PlaceDecl::slot::<ground::Bounce>(),
    PlaceDecl::slot::<ground::PlacedTiles>(),
    PlaceDecl::slot::<ground::Cover>(),
    PlaceDecl::list::<ground::FarCover>(),
    PlaceDecl::slot::<ground::Occupancy>(),
    PlaceDecl::slot::<ground::Foliage>(),
    PlaceDecl::component::<ground::HeightHeld>("height_held"),
    PlaceDecl::component::<ground::Grounded>("grounded"),
    PlaceDecl::component::<ground::GroundedAt>("grounded_at"),
    PlaceDecl::component::<navigation::NavAgent>("nav_agent"),
    PlaceDecl::component::<navigation::NavWalk>("nav_walk"),
    PlaceDecl::slot::<water::Waterline>(),
    PlaceDecl::slot::<water::WaterObstacles>(),
    PlaceDecl::slot::<water::WaterFlow>(),
    PlaceDecl::slot::<water::Hydrology>(),
    PlaceDecl::component::<water::WaterBody>("water_body"),
    PlaceDecl::slot::<wind::Wind>(),
    PlaceDecl::slot::<physics::Collision>(),
    PlaceDecl::component::<physics::Collider>("collider"),
    PlaceDecl::component::<physics::RigidBody>("rigid_body"),
    PlaceDecl::component::<solids::SolidSource>("solid_source"),
    PlaceDecl::component::<splines::SplineMesh>("spline_mesh"),
    PlaceDecl::component::<streaming::StreamingFocus>("streaming_focus"),
    PlaceDecl::list::<terrain::TerrainLayers>(),
];
