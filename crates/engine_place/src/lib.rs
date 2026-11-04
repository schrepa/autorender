mod screens;

use std::any::{Any, TypeId};

use bevy_ecs::{component::Component, resource::Resource, world::World};
use serde_json::Value;

pub use engine_assets::{MaterialHandle, MeshHandle, TextureHandle};
pub use engine_settings::{Error, ErrorCode};
pub use screens::{LevelProgress, LevelRequest, Screens};

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
    const UBER_MATERIALS: bool = false;
}

type ReadMembers = fn(&World, &mut dyn FnMut(&dyn Any));

#[derive(Clone, Copy)]
pub struct PlaceDecl {
    name: &'static str,
    kind: PlaceKind,
    wgsl: Option<&'static str>,
    smokes: &'static [Smoke],
    insert: fn(&mut World),
    read: ReadMembers,
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
            read: read_slot::<S>,
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
            read: read_nothing,
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
            read: read_nothing,
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
    pub fn read(&self, world: &World, each: &mut dyn FnMut(&dyn Any)) {
        (self.read)(world, each)
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

fn read_slot<S: Slot>(world: &World, each: &mut dyn FnMut(&dyn Any)) {
    if let Some(value) = world.get_resource::<S>() {
        each(value);
    }
}

fn read_nothing(_: &World, _: &mut dyn FnMut(&dyn Any)) {}

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
    fn file(&mut self, name: &str, bytes: &[u8]);
    fn idle(&mut self, milliseconds: f64);
    fn joined(&mut self, list: TypeId, each: &mut dyn FnMut(&dyn Any));
    fn members(&mut self, _place: &str) -> Vec<&'static str> {
        Vec::new()
    }
    fn slot(&mut self, _slot: TypeId, _each: &mut dyn FnMut(&dyn Any)) {}
    fn without(&mut self, _bricks: &[&'static str]) {}
    fn artifact(&mut self, _path: &str) -> Option<Vec<u8>> {
        None
    }
    fn decode_image(&mut self, _path: &str, _bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), Value> {
        Err(Value::Null)
    }
}

#[macro_export]
macro_rules! smoke_check {
    ($condition:expr, $check:expr, $got:expr, $expected:expr $(,)?) => {
        if !$condition {
            return ::core::result::Result::Err($crate::SmokeFailure::new(
                $check,
                &$got,
                &$expected,
                ::core::concat!(::core::file!(), ":", ::core::line!()),
            ));
        }
    };
}

pub fn ask(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Result<Value, SmokeFailure> {
    let reply = session.request(verb, body);
    smoke_check!(
        reply.is_ok(),
        "request_answered",
        &reply.as_ref().err().map(|refused| &refused["error"]),
        &verb
    );
    Ok(reply
        .map(|mut answered| answered["result"].take())
        .unwrap_or_default())
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
            let aligns: &[usize] = &[$($crate::gpu_struct!(@align $ty)),*];
            let mut align = 1;
            let mut at = 0;
            while at < aligns.len() {
                if aligns[at] > align {
                    align = aligns[at];
                }
                at += 1;
            }
            ::core::assert!(::core::mem::size_of::<$name>() % align == 0, "a gpu_struct is not a multiple of its WGSL alignment");
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
    (@munch [$($pieces:tt)*]) => {
        pub const WGSL: &str = {
            const __GPU_CONSTS_PIECES: &[$crate::__private::wgsl::Piece] = &[$($pieces)*];
            const __GPU_CONSTS_LEN: usize =
                $crate::__private::wgsl::write::<0>(__GPU_CONSTS_PIECES).len;
            const __GPU_CONSTS_BYTES: [u8; __GPU_CONSTS_LEN] =
                $crate::__private::wgsl::write::<__GPU_CONSTS_LEN>(__GPU_CONSTS_PIECES).bytes;
            $crate::__private::wgsl::text(&__GPU_CONSTS_BYTES)
        };
    };
    (@munch [$($pieces:tt)*] $vis:vis const $name:ident: $ty:ident = $value:literal; $($rest:tt)*) => {
        $vis const $name: $ty = $value;
        $crate::gpu_consts! {
            @munch [
                $($pieces)*
                $crate::__private::wgsl::Piece::Text(::core::concat!(
                    "const ", ::core::stringify!($name), ": ", ::core::stringify!($ty), " = ",
                    ::core::stringify!($value), ";\n",
                )),
            ]
            $($rest)*
        }
    };
    (@munch [$($pieces:tt)*] $vis:vis const $name:ident: $ty:ident = $value:expr; $($rest:tt)*) => {
        $vis const $name: $ty = $value;
        $crate::gpu_consts! {
            @munch [
                $($pieces)*
                $crate::__private::wgsl::Piece::Text(::core::concat!(
                    "const ", ::core::stringify!($name), ": ", ::core::stringify!($ty), " = ",
                )),
                $crate::gpu_consts!(@value $ty $name),
                $crate::__private::wgsl::Piece::Text(";\n"),
            ]
            $($rest)*
        }
    };
    (@value u32 $name:ident) => { $crate::__private::wgsl::Piece::U32($name) };
    (@value i32 $name:ident) => { $crate::__private::wgsl::Piece::I32($name) };
    (@value f32 $name:ident) => { $crate::__private::wgsl::Piece::F32($name) };
    (@value $ty:ident $name:ident) => {
        ::core::compile_error!(::core::concat!(
            "gpu_consts! writes WGSL for u32, i32 and f32, not ",
            ::core::stringify!($ty)
        ))
    };
    ($($body:tt)*) => {
        $crate::gpu_consts! { @munch [] $($body)* }
    };
}

#[doc(hidden)]
pub mod __private {
    pub use ::bytemuck;

    pub mod wgsl {
        #[derive(Clone, Copy)]
        pub enum Piece {
            Text(&'static str),
            U32(u32),
            I32(i32),
            F32(f32),
        }

        pub struct Written<const N: usize> {
            pub bytes: [u8; N],
            pub len: usize,
        }

        pub const fn write<const N: usize>(pieces: &[Piece]) -> Written<N> {
            let mut out = Written {
                bytes: [0; N],
                len: 0,
            };
            let mut at = 0;
            while at < pieces.len() {
                match pieces[at] {
                    Piece::Text(text) => out.text(text),
                    Piece::U32(value) => out.whole(value as u128),
                    Piece::I32(value) => {
                        if value < 0 {
                            out.push(b'-');
                        }
                        out.whole(value.unsigned_abs() as u128);
                    }
                    Piece::F32(value) => out.float(value),
                }
                at += 1;
            }
            out
        }

        pub const fn text(bytes: &'static [u8]) -> &'static str {
            match core::str::from_utf8(bytes) {
                Ok(text) => text,
                Err(_) => panic!("gpu_consts WGSL is UTF-8"),
            }
        }

        const LIMBS: usize = 6;

        impl<const N: usize> Written<N> {
            const fn push(&mut self, byte: u8) {
                if self.len < N {
                    self.bytes[self.len] = byte;
                }
                self.len += 1;
            }

            const fn text(&mut self, text: &str) {
                let bytes = text.as_bytes();
                let mut at = 0;
                while at < bytes.len() {
                    self.push(bytes[at]);
                    at += 1;
                }
            }

            const fn whole(&mut self, value: u128) {
                let mut digits = [0_u8; 39];
                let mut count = 0;
                let mut rest = value;
                loop {
                    digits[count] = b'0' + (rest % 10) as u8;
                    count += 1;
                    rest /= 10;
                    if rest == 0 {
                        break;
                    }
                }
                while count > 0 {
                    count -= 1;
                    self.push(digits[count]);
                }
            }

            const fn float(&mut self, value: f32) {
                let bits = value.to_bits();
                let exponent = (bits >> 23) & 0xff;
                assert!(exponent != 0xff, "a gpu_consts f32 is finite");
                if bits >> 31 != 0 {
                    self.push(b'-');
                }
                let fraction = bits & 0x7f_ffff;
                let (mantissa, shift) = if exponent == 0 {
                    (fraction, -149_i32)
                } else {
                    (fraction | 0x80_0000, exponent as i32 - 150)
                };
                if shift >= 0 {
                    self.whole((mantissa as u128) << shift as u32);
                    self.text(".0");
                    return;
                }
                let below = shift.unsigned_abs();
                let (whole, rest) = if below < 24 {
                    (mantissa >> below, mantissa & ((1 << below) - 1))
                } else {
                    (0, mantissa)
                };
                self.whole(whole as u128);
                self.push(b'.');
                if rest == 0 {
                    self.push(b'0');
                    return;
                }
                let mut limbs = [0_u32; LIMBS];
                limbs[0] = rest;
                let word = (below / 32) as usize;
                let bit = below % 32;
                while !zero(&limbs) {
                    let mut carry = 0_u64;
                    let mut at = 0;
                    while at < LIMBS {
                        let wide = limbs[at] as u64 * 10 + carry;
                        limbs[at] = wide as u32;
                        carry = wide >> 32;
                        at += 1;
                    }
                    let mut digit = limbs[word] >> bit;
                    if bit > 28 {
                        digit |= limbs[word + 1] << (32 - bit);
                    }
                    limbs[word] &= if bit == 0 { 0 } else { (1 << bit) - 1 };
                    let mut above = word + 1;
                    while above < LIMBS {
                        limbs[above] = 0;
                        above += 1;
                    }
                    self.push(b'0' + (digit & 0xf) as u8);
                }
            }
        }

        const fn zero(limbs: &[u32; LIMBS]) -> bool {
            let mut at = 0;
            while at < LIMBS {
                if limbs[at] != 0 {
                    return false;
                }
                at += 1;
            }
            true
        }
    }
}
