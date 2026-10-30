pub(crate) use crate::gpu_layouts::constants::PYRAMID_EXTENT_BYTES;
use crate::gpu_layouts::impostors::{ImpostorBakeUniform, ImpostorRecord};
use crate::gpu_layouts::scene::{ColorData, GroundData, MaterialUniform, PoseData};

mod groups;
mod table;

pub use groups::{
    BindingDecl, Contribution, GroupRef, ResolvedBinding, ResolvedLayout, SharedBinding,
    SharedGroups, StageCounts,
};
pub use table::{
    Binding, Format, Kind, Sample, Sampling, Stages, Table, View, bind, dynamic_uniform, read,
    read_of, sampler, texture, uniform, uniform_of, write,
};

use Format::R32Float;
use Sample::{Depth, Float, Unfilterable};
use Sampling::Filtering;
use View::{D2, D2Array};

const C: Stages = Stages::COMPUTE;
const F: Stages = Stages::FRAGMENT;
const V: Stages = Stages::VERTEX;
const VF: Stages = Stages::VERTEX_FRAGMENT;
const POSE: usize = size_of::<PoseData>();
const COLOR: usize = size_of::<ColorData>();
const GROUND: usize = size_of::<GroundData>();

macro_rules! tables {
    ($(
        $(#[$meta:meta])*
        const $name:ident: Table = Table {
            label: $label:literal,
            bindings: &[$($binding:expr),* $(,)?] $(,)?
        };
    )*) => {
        $(
            $(#[$meta])*
            pub const $name: Table = Table {
                label: $label,
                bindings: &[$($binding),*],
            };
        )*

        pub const TABLES: &[(&str, &Table)] = &[$((stringify!($name), &$name)),*];
    };
}

tables! {
    const EMPTY: Table = Table {
        label: "engine/empty-layout",
        bindings: &[],
    };

    const MATERIAL: Table = Table {
        label: "engine/material-layout",
        bindings: &[
            texture("base_color_texture", Float, D2, F),
            sampler("base_color_sampler", Filtering, F),
            texture("metallic_roughness_texture", Float, D2, F),
            sampler("metallic_roughness_sampler", Filtering, F),
            texture("normal_texture", Float, D2, F),
            sampler("normal_sampler", Filtering, F),
            texture("occlusion_texture", Float, D2, F),
            sampler("occlusion_sampler", Filtering, F),
            texture("emissive_texture", Float, D2, F),
            sampler("emissive_sampler", Filtering, F),
            uniform_of("material_maps", size_of::<MaterialUniform>(), F),
        ],
    };

    const RETAINED_POSES: Table = Table {
        label: "engine/retained-pose-layout",
        bindings: &[
            read_of("poses", POSE, VF),
            read_of("colors", COLOR, V),
            read_of("grounds", GROUND, F),
        ],
    };

    const TEMPORAL_POSES: Table = Table {
        label: "engine/temporal-pose-layout",
        bindings: &[
            read_of("poses", POSE, VF),
            read_of("colors", COLOR, V),
            read_of("previous_poses", POSE, V),
            read_of("grounds", GROUND, F),
        ],
    };

    const IMPOSTOR_DRAW: Table = Table {
        label: "engine/impostor-layout",
        bindings: &[
            read_of("impostor_records", size_of::<ImpostorRecord>(), VF),
            texture("impostor_albedo", Float, D2Array, F),
            texture("impostor_emissive", Float, D2Array, F),
            texture("impostor_geometry", Float, D2Array, F),
            texture("impostor_surface", Float, D2Array, F),
            sampler("impostor_sampler", Filtering, F),
        ],
    };

    const IMPOSTOR_BAKE: Table = Table {
        label: "engine/impostor-bake-layout",
        bindings: &[
            uniform_of("impostor_bake", size_of::<ImpostorBakeUniform>(), VF),
            texture("impostor_shade", Depth, D2, F),
        ],
    };

    const PRESENTATION_COMPOSE: Table = Table {
        label: "engine/presentation-compose-layout",
        bindings: &[
            uniform("frame", C),
            read("locals", C),
            read("info", C),
            read("authority", C),
            write("composed", C),
            read("parents", C),
        ],
    };

    const PRESENTATION_LEVEL: Table = Table {
        label: "engine/presentation-level-layout",
        bindings: &[uniform("level", C)],
    };

    const PRESENTATION_CULL: Table = Table {
        label: "engine/presentation-cull-layout",
        bindings: &[
            uniform("frame", C),
            read("locals", C),
            read("info", C),
            read("authority", C),
            write("composed", C),
            read("batches", C),
            write("poses", C),
            write("scan", C),
            read("slot_candidates", C),
        ],
    };

    const PRESENTATION_TESTS: Table = Table {
        label: "engine/presentation-test-layout",
        bindings: &[
            uniform("frame", C),
            read("info", C),
            read("batches", C),
            write("poses", C),
            write("scan", C),
            write("locals_shift", C),
            read("slot_candidates", C),
            texture("depth_pyramid", Unfilterable, D2, C),
            write("occlusion_counts", C),
            write("still_frames", C),
        ],
    };

    const PRESENTATION_COMPACT: Table = Table {
        label: "engine/presentation-compact-layout",
        bindings: &[
            uniform("frame", C),
            read("candidates", C),
            read("batches", C),
            write("scan", C),
            write("batch_bases", C),
            write("visible_ids", C),
            write("draws", C),
            write("occlusion_counts", C),
            write("impostor_ids", C),
        ],
    };

    const PRESENTATION_FADES: Table = Table {
        label: "engine/presentation-fades-layout",
        bindings: &[
            uniform("frame", C),
            read("batches", C),
            write("poses", C),
            write("scan", C),
            read("slot_candidates", C),
            write("lod_states", C),
            write("fade_cards", C),
        ],
    };

    const POST: Table = Table {
        label: "engine/post-layout",
        bindings: &[
            uniform("post", F),
            texture("scene", Float, D2, F),
            texture("occlusion", Float, D2, F),
            sampler("linear", Filtering, F),
        ],
    };

    const TRANSPARENT_COMPOSITE: Table = Table {
        label: "engine/transparent-composite-layout",
        bindings: &[
            texture("accumulated", Unfilterable, D2, F),
            texture("revealed", Unfilterable, D2, F),
        ],
    };

    const MIPMAPS: Table = Table {
        label: "engine/texture-mip-layout",
        bindings: &[
            texture("source", Float, D2, F),
            sampler("source_sampler", Filtering, F),
        ],
    };

    const PYRAMID_FIRST: Table = Table {
        label: "engine/depth-pyramid-first-layout",
        bindings: &[
            texture("source_depth", Depth, D2, C),
            bind("target_level", Kind::Storage(R32Float), C),
            uniform_of("extent", PYRAMID_EXTENT_BYTES, C),
        ],
    };

    const PYRAMID_DOWN: Table = Table {
        label: "engine/depth-pyramid-down-layout",
        bindings: &[
            texture("source_level", Unfilterable, D2, C),
            bind("target_level", Kind::Storage(R32Float), C),
            uniform_of("extent", PYRAMID_EXTENT_BYTES, C),
        ],
    };
}
