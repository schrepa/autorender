//! Every struct the renderer uploads to the GPU, declared once with the WGSL
//! structs mirroring it, and every constant Rust and WGSL share
//! ([`constants`]). `cargo xtask lint` checks each mirror in every
//! shader module holding it against the Rust layout, field by field: names,
//! offsets and sizes, a mirror holding the leading fields alone where its
//! shaders read no more; and each constant's value. Upload structs live here
//! and nowhere else (the lint refuses `Pod` derives outside this module), so
//! none goes unchecked.
// Native builds read only the layouts; Wasm builds upload the structs.

/// A WGSL struct mirroring a Rust one: its name, in one shader module or in
/// every module declaring it.
#[derive(Debug)]
pub struct Mirror {
    pub name: &'static str,
    pub module: Option<&'static str>,
}

/// One field of a Rust layout.
#[derive(Debug)]
pub struct Field {
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
}

/// A struct's layout as Rust lays it out, and its WGSL mirrors.
#[derive(Debug)]
pub struct Layout {
    pub rust: &'static str,
    pub mirrors: &'static [Mirror],
    pub size: usize,
    pub fields: &'static [Field],
}

impl Layout {
    /// The field named `name`; a name the struct lacks fails to evaluate.
    pub const fn field(&self, name: &str) -> &Field {
        let mut index = 0;
        while index < self.fields.len() {
            if same(self.fields[index].name, name) {
                return &self.fields[index];
            }
            index += 1;
        }
        panic!("no field of that name")
    }
}

const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut index = 0;
    while index < a.len() {
        if a[index] != b[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// Declares upload structs (`#[repr(C)]`, fields visible to the crate) and
/// their [`Layout`]s: `struct Name as ["WgslName", "Other" in MODULE] {..}`,
/// `as [...]` omitted for structs no WGSL struct mirrors (vertex streams).
macro_rules! gpu_structs {
    (@module) => { None };
    (@module $module:ident) => { Some(stringify!($module)) };
    ($(
        $(#[$meta:meta])*
        $vis:vis struct $name:ident $(as [$($mirror:literal $(in $module:ident)?),* $(,)?])? {
            $($(#[$field_meta:meta])* $field_vis:vis $field:ident: $ty:ty),* $(,)?
        }
    )*) => {
        $(
            $(#[$meta])*
            #[repr(C)]
            pub(crate) struct $name {
                $($(#[$field_meta])* pub $field: $ty,)*
            }

            impl $name {
                /// This struct's layout and mirrors.
                pub(crate) const LAYOUT: $crate::gpu_layouts::Layout = $crate::gpu_layouts::Layout {
                    rust: stringify!($name),
                    mirrors: &[$($($crate::gpu_layouts::Mirror {
                        name: $mirror,
                        module: gpu_structs!(@module $($module)?),
                    }),*)?],
                    size: ::core::mem::size_of::<$name>(),
                    fields: &[$($crate::gpu_layouts::Field {
                        name: stringify!($field),
                        offset: ::core::mem::offset_of!($name, $field),
                        size: ::core::mem::size_of::<$ty>(),
                    }),*],
                };
            }
        )*

        /// Every struct's layout and mirrors.
        pub const LAYOUTS: &[$crate::gpu_layouts::Layout] = &[$($name::LAYOUT),*];
    };
}

/// A value Rust and WGSL share: its Rust name, its value and the WGSL
/// constants mirroring it.
#[derive(Debug)]
pub struct Constant {
    pub rust: &'static str,
    pub value: f64,
    pub mirrors: &'static [Mirror],
}

/// Declares constants WGSL mirrors, and [`Constant`]s for them (`DECLARED`):
/// `const NAME: Type = value => ["WGSL_NAME", "Other" in MODULE];`, a mirror
/// without a module standing in every module declaring it.
macro_rules! gpu_consts {
    ($(
        $(#[$meta:meta])*
        $vis:vis const $name:ident: $ty:ty = $value:expr => [$($mirror:literal $(in $module:ident)?),+ $(,)?];
    )*) => {
        $($(#[$meta])* $vis const $name: $ty = $value;)*

        /// Every constant declared here, with its mirrors.
        pub const DECLARED: &[$crate::gpu_layouts::Constant] = &[$(
            $crate::gpu_layouts::Constant {
                rust: stringify!($name),
                value: $name as f64,
                mirrors: &[$($crate::gpu_layouts::Mirror {
                    name: $mirror,
                    module: gpu_structs!(@module $($module)?),
                }),+],
            },
        )*];
    };
}

/// Registers constants other crates declare and WGSL mirrors (`MIRRORED`):
/// `path::NAME => ["WGSL_NAME" in MODULE];`.
macro_rules! gpu_mirrors {
    ($($path:path => [$($mirror:literal $(in $module:ident)?),+ $(,)?];)*) => {
        /// Every constant of other crates WGSL mirrors.
        pub const MIRRORED: &[$crate::gpu_layouts::Constant] = &[$(
            $crate::gpu_layouts::Constant {
                rust: stringify!($path),
                value: $path as f64,
                mirrors: &[$($crate::gpu_layouts::Mirror {
                    name: $mirror,
                    module: gpu_structs!(@module $($module)?),
                }),+],
            },
        )*];
    };
}

/// Every constant Rust and WGSL share.
pub fn constants() -> impl Iterator<Item = &'static Constant> {
    constants::DECLARED.iter().chain(constants::MIRRORED)
}

pub mod constants;

/// Declares the modules of upload structs and [`layouts`] from the same
/// list.
macro_rules! layout_modules {
    ($($module:ident),* $(,)?) => {
        $(pub mod $module;)*

        /// Every upload struct's layout.
        pub fn layouts() -> Vec<&'static Layout> {
            let mut layouts = Vec::new();
            $(layouts.extend($module::LAYOUTS);)*
            layouts
        }
    };
}

layout_modules! {
    scene,
    clouds,
    deformation,
    decals,
    ground_cover,
    impostors,
    lights,
    occupancy,
    particles,
    scatter,
    sky,
    terrain,
    terrain_horizons,
    terrain_sky,
    terrain_vt,
    virtual_geometry,
    water,
}
