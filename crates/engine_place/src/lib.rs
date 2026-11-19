#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod asks;
mod calls;
mod drawn;
mod feeds;
mod names;
mod nodes;
mod passes;
mod poses;
mod readback;
mod readiness;
mod requests;
mod views;

use bevy_ecs::world::World;

pub use asks::Asked;
pub use calls::{SCENE_SCHEMA, schema_of};
pub use drawn::{DrawKind, Drawn, InstanceValues, KindDecl, Layers, NodeId};
pub use engine_assets::{
    AttributeFormat, GeometryHandle, MaterialHandle, StreamShape, TextureHandle,
};
pub use engine_settings::{Error, ErrorCode};
pub use feeds::{Changed, Changes, Cursor, Events, Feed, FeedView, Ring, Since, StoreId};
pub use names::{AuthoredId, OperationScene, SceneEntity, entity_removed};
pub use nodes::{Clock, Mobility, Mover, MoverId, Node, NodeWord, Pace};
pub use passes::{
    KindCell, KindJoin, Pass, PassBit, PassEntry, PassMask, PassName, PassViews, ScenePass,
    ScenePasses, TargetName, Writes,
};
pub use poses::{
    COEFFICIENT_LIMIT, COMMON_DETERMINANT_MARGIN, DrawnPoses, MIN_LINEAR_SCALE, NodePose,
    PoseEntry, PoseFrame, PoseNodes, PoseRows, Poses, poses_wgsl, rows_matrix, same_pose_bits,
    same_rows_bits,
};
pub use readback::{FrameId, Readable, Readback};
pub use readiness::{CORE_ROW, Readiness, Unsettled};
pub use requests::{Board, Census, Interest, Progress, Request, Requests, Settling, Ticket};
pub use views::{Clear, Rect, Target, View, ViewSkip};

pub trait List: Send + Sync + 'static {
    const NAME: &'static str;
    type Member: Send + Sync + 'static;

    fn dropped(_world: &mut World, _member: &Self::Member) {}
}

pub const UNIFORM_BINDING_BYTES: u64 = 16_384;

pub trait GpuStruct: bytemuck::Pod + Send + Sync + 'static {
    const WGSL_NAME: &'static str;
    const WGSL: &'static str;
}

pub use engine_gpu_bits::gpu_bits;

pub trait BitValue: Copy + Sized {
    const WIDTH: u32;
    const WGSL: &'static str = "";
    fn bits(self) -> u32;
}

impl BitValue for bool {
    const WIDTH: u32 = 1;

    fn bits(self) -> u32 {
        u32::from(self)
    }
}

impl BitValue for u8 {
    const WIDTH: u32 = 8;

    fn bits(self) -> u32 {
        u32::from(self)
    }
}

impl BitValue for u16 {
    const WIDTH: u32 = 16;

    fn bits(self) -> u32 {
        u32::from(self)
    }
}

impl BitValue for u32 {
    const WIDTH: u32 = 32;

    fn bits(self) -> u32 {
        self
    }
}

pub trait GpuBits: Copy + Sized + Send + Sync + 'static {
    const WGSL: &'static str;
    fn pack(self) -> u32;
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

#[doc(hidden)]
pub mod __private {
    pub use ::bytemuck;
}
