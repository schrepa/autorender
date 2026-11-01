use std::any::TypeId;

use bevy_ecs::{component::Component, resource::Resource, world::World};
use serde_json::Value;

pub use engine_assets::{MaterialHandle, MeshHandle, TextureHandle};
pub use engine_settings::{Error, ErrorCode};

pub trait Slot: Resource + Default {
    const NAME: &'static str;
    const WGSL: Option<&'static str> = None;
    const SMOKES: &'static [Smoke] = &[];
}

pub trait List: Send + Sync + 'static {
    const NAME: &'static str;
    type Member: Send + Sync + 'static;
    const WGSL: Option<&'static str> = None;
    const SMOKES: &'static [Smoke] = &[];
}

pub trait DrawPoint: Slot {
    type Layer: Copy + Ord + Send + Sync + 'static;
    type Args: Copy + Send + Sync + 'static;
}

#[derive(Clone, Copy)]
pub struct PlaceDecl {
    name: &'static str,
    kind: PlaceKind,
    wgsl: Option<&'static str>,
    smokes: &'static [Smoke],
    insert: fn(&mut World),
    ty: fn() -> TypeId,
}

impl PlaceDecl {
    pub const fn slot<S: Slot>() -> Self {
        Self {
            name: S::NAME,
            kind: PlaceKind::Slot,
            wgsl: S::WGSL,
            smokes: S::SMOKES,
            insert: insert_slot::<S>,
            ty: TypeId::of::<S>,
        }
    }

    pub const fn list<L: List>() -> Self {
        Self {
            name: L::NAME,
            kind: PlaceKind::List,
            wgsl: L::WGSL,
            smokes: L::SMOKES,
            insert: insert_nothing,
            ty: TypeId::of::<L>,
        }
    }

    pub const fn component<C: Component>(name: &'static str) -> Self {
        Self {
            name,
            kind: PlaceKind::Component,
            wgsl: None,
            smokes: &[],
            insert: insert_component::<C>,
            ty: TypeId::of::<C>,
        }
    }

    #[doc(hidden)]
    pub fn name(&self) -> &'static str {
        self.name
    }

    #[doc(hidden)]
    pub fn kind(&self) -> PlaceKind {
        self.kind
    }

    #[doc(hidden)]
    pub fn wgsl(&self) -> Option<&'static str> {
        self.wgsl
    }

    #[doc(hidden)]
    pub fn smokes(&self) -> &'static [Smoke] {
        self.smokes
    }

    #[doc(hidden)]
    pub fn insert(&self, world: &mut World) {
        (self.insert)(world)
    }

    #[doc(hidden)]
    pub fn type_id(&self) -> TypeId {
        (self.ty)()
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaceKind {
    Slot,
    List,
    Component,
}

fn insert_slot<S: Slot>(world: &mut World) {
    world.init_resource::<S>();
}

fn insert_component<C: Component>(world: &mut World) {
    world.register_component::<C>();
}

fn insert_nothing(_: &mut World) {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Stamp(u64);

impl Stamp {
    pub fn next() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct JobRef {
    name: Box<str>,
    source: [u8; 32],
}

impl JobRef {
    #[doc(hidden)]
    pub fn named(name: &str, source: [u8; 32]) -> Self {
        Self {
            name: name.into(),
            source,
        }
    }

    #[doc(hidden)]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[doc(hidden)]
    pub fn source(&self) -> [u8; 32] {
        self.source
    }
}

impl engine_codec::Codec for JobRef {
    fn write(&self, out: &mut engine_codec::Writer) {
        out.put(&self.name.to_string()).put(&self.source);
    }

    fn read(input: &mut engine_codec::Reader<'_>) -> Result<Self, Error> {
        let name: String = input.get()?;
        let source: [u8; 32] = input.get()?;
        Ok(Self {
            name: name.into_boxed_str(),
            source,
        })
    }
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct Mesh3d {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InterpolateTransform {
    pub resets: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Runs {
    Headless,
    Gpu,
}

#[derive(Clone, Copy)]
pub struct Smoke {
    pub name: &'static str,
    pub runs: Runs,
    pub run: fn(&mut dyn SmokeSession) -> Result<(), SmokeFailure>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct SmokeFailure {
    pub smoke: &'static str,
    pub check: &'static str,
    pub got: Value,
    pub expected: Value,
    pub at: &'static str,
}

impl SmokeFailure {
    #[doc(hidden)]
    pub fn new(
        check: &'static str,
        got: &impl serde::Serialize,
        expected: &impl serde::Serialize,
        at: &'static str,
    ) -> Self {
        Self {
            smoke: "",
            check,
            got: serde_json::to_value(got).unwrap_or(Value::Null),
            expected: serde_json::to_value(expected).unwrap_or(Value::Null),
            at,
        }
    }
}

pub trait SmokeSession {
    fn request(&mut self, verb: &str, body: Value) -> Result<Value, Value>;
    fn step(&mut self, ticks: u32);
    fn frame(&mut self) -> bool;
    fn source(&mut self, document: &str);
    fn file(&mut self, name: &str, bytes: Vec<u8>);
}

#[macro_export]
macro_rules! smoke_eq {
    ($got:expr, $expected:expr, $check:expr $(,)?) => {{
        let (got, expected) = (&$got, &$expected);
        if *got != *expected {
            return ::core::result::Result::Err($crate::SmokeFailure::new(
                $check,
                got,
                expected,
                ::core::concat!(::core::file!(), ":", ::core::line!()),
            ));
        }
    }};
}

pub trait GpuStruct: bytemuck::Pod + Send + Sync + 'static {
    const WGSL_NAME: &'static str;
    const WGSL: &'static str;
}

#[macro_export]
macro_rules! gpu_struct {
    (@wgsl f32) => { "f32" };
    (@wgsl u32) => { "u32" };
    (@wgsl i32) => { "i32" };
    (@wgsl [f32; 2]) => { "vec2<f32>" };
    (@wgsl [u32; 2]) => { "vec2<u32>" };
    (@wgsl [i32; 2]) => { "vec2<i32>" };
    (@wgsl [f32; 3]) => { "vec3<f32>" };
    (@wgsl [u32; 3]) => { "vec3<u32>" };
    (@wgsl [i32; 3]) => { "vec3<i32>" };
    (@wgsl [f32; 4]) => { "vec4<f32>" };
    (@wgsl [u32; 4]) => { "vec4<u32>" };
    (@wgsl [i32; 4]) => { "vec4<i32>" };
    (@wgsl [[f32; 4]; 4]) => { "mat4x4<f32>" };
    (@wgsl [[[f32; 4]; 4]; $count:tt]) => { ::core::concat!("array<mat4x4<f32>, ", ::core::stringify!($count), ">") };
    (@wgsl [[f32; 4]; $count:tt]) => { ::core::concat!("array<vec4<f32>, ", ::core::stringify!($count), ">") };
    (@wgsl [[u32; 4]; $count:tt]) => { ::core::concat!("array<vec4<u32>, ", ::core::stringify!($count), ">") };
    (@wgsl [[i32; 4]; $count:tt]) => { ::core::concat!("array<vec4<i32>, ", ::core::stringify!($count), ">") };
    (@wgsl [$inner:ident; $count:tt]) => { ::core::concat!("array<", ::core::stringify!($inner), ", ", ::core::stringify!($count), ">") };
    (@wgsl $inner:ident) => { ::core::stringify!($inner) };
    (@align f32) => { 4 };
    (@align u32) => { 4 };
    (@align i32) => { 4 };
    (@align [f32; 2]) => { 8 };
    (@align [u32; 2]) => { 8 };
    (@align [i32; 2]) => { 8 };
    (@align [f32; 3]) => { 16 };
    (@align [u32; 3]) => { 16 };
    (@align [i32; 3]) => { 16 };
    (@align [f32; 4]) => { 16 };
    (@align [u32; 4]) => { 16 };
    (@align [i32; 4]) => { 16 };
    (@align [[[f32; 4]; 4]; $count:tt]) => { 16 };
    (@align [[f32; 4]; $count:tt]) => { 16 };
    (@align [[u32; 4]; $count:tt]) => { 16 };
    (@align [[i32; 4]; $count:tt]) => { 16 };
    (@align [$inner:ident; $count:tt]) => { 16 };
    (@align $inner:ident) => { 16 };
    (
        @crate $bytemuck:literal;
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $($field_vis:vis $field:ident: $ty:tt),* $(,)?
        }
    ) => {
        #[derive(
            ::core::clone::Clone,
            ::core::marker::Copy,
            ::core::fmt::Debug,
            ::core::cmp::PartialEq,
            $crate::__private::bytemuck::Pod,
            $crate::__private::bytemuck::Zeroable
        )]
        #[bytemuck(crate = $bytemuck)]
        #[repr(C)]
        $(#[$meta])*
        $vis struct $name {
            $($field_vis $field: $ty,)*
        }

        impl ::core::default::Default for $name {
            fn default() -> Self {
                <Self as $crate::__private::bytemuck::Zeroable>::zeroed()
            }
        }

        const _: () = {
            $(::core::assert!(
                ::core::mem::offset_of!($name, $field) % $crate::gpu_struct!(@align $ty) == 0,
                "a gpu_struct field is off its WGSL alignment"
            );)*
            ::core::assert!(::core::mem::size_of::<$name>() % 16 == 0, "a gpu_struct is not a multiple of 16 bytes");
        };

        impl $crate::GpuStruct for $name {
            const WGSL_NAME: &'static str = ::core::stringify!($name);
            const WGSL: &'static str = ::core::concat!(
                "struct ", ::core::stringify!($name), " {",
                $(" ", ::core::stringify!($field), ": ", $crate::gpu_struct!(@wgsl $ty), ",",)*
                " }\n"
            );
        }
    };
    ($($body:tt)*) => {
        $crate::gpu_struct! { @crate "::engine_place::__private::bytemuck"; $($body)* }
    };
}

#[macro_export]
macro_rules! gpu_consts {
    ($($vis:vis const $name:ident: $ty:ident = $value:expr;)*) => {
        $($vis const $name: $ty = $value;)*
        pub const WGSL: &str = ::core::concat!($(
            "const ", ::core::stringify!($name), ": ", ::core::stringify!($ty), " = ", ::core::stringify!($value), ";\n",
        )*);
    };
}

#[doc(hidden)]
pub mod __private {
    pub use ::bytemuck;
}
