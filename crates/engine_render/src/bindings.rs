use crate::gpu_layouts::impostors::{ImpostorBakeUniform, ImpostorRecord};
use crate::gpu_layouts::scene::MaterialUniform;

mod groups;
mod table;

pub use groups::{BindingDecl, ResolvedBinding, ResolvedLayout, StageCounts};
pub use table::{
    Binding, Format, Kind, Sample, Sampling, Stages, Table, View, bind, dynamic_uniform, read,
    read_of, sampler, texture, uniform, uniform_of, write,
};

use Sample::{Depth, Float};
use Sampling::Filtering;
use View::{D2, D2Array};

const F: Stages = Stages::FRAGMENT;
const VF: Stages = Stages::VERTEX_FRAGMENT;

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

    const MIPMAPS: Table = Table {
        label: "engine/texture-mip-layout",
        bindings: &[
            texture("source", Float, D2, F),
            sampler("source_sampler", Filtering, F),
        ],
    };

}
