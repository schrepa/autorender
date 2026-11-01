#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod error;

pub use engine_codec::{
    Allowed, Detail, Error, ErrorCode, Expected, ExtraRules, ID_PATTERN, InvalidSetting,
    MAX_GOT_BYTES, MAX_ID_BYTES, Rule, Setting, closest, fields, is_id, rule, state, stated,
    valid_uri,
};
#[cfg(target_arch = "wasm32")]
pub use error::js_error;
pub use error::{JS_ERROR_NAME, parse_json, parse_value, parse_value_at};

#[doc(hidden)]
pub mod __private {
    pub use ::schemars;
}

#[doc(hidden)]
pub fn refused(checked: Result<(), InvalidSetting>) -> Result<(), Error> {
    checked.map_err(|invalid| invalid.error(ErrorCode::InvalidEnvironment, "$"))
}

#[macro_export]
macro_rules! settings {
    (
        @section $key:literal [$($check:path)?]
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                $field_vis:vis $field:ident: $ty:ty = $default:expr
                    => $rule:ident $(($($arg:expr),* $(,)?))?
            ),* $(,)?
        }
    ) => {
        #[derive(
            ::core::clone::Clone,
            ::core::fmt::Debug,
            ::core::cmp::PartialEq,
            bevy_ecs::resource::Resource,
            ::engine_core::__private::serde::Serialize,
            ::engine_core::__private::serde::Deserialize,
            ::engine_core::__private::schemars::JsonSchema
        )]
        #[component(immutable)]
        #[serde(crate = "::engine_core::__private::serde", default)]
        #[schemars(
            crate = "::engine_core::__private::schemars",
            transform = Self::state_rules
        )]
        $(#[$meta])*
        $vis struct $name {
            $($(#[$field_meta])* $field_vis $field: $ty,)*
        }

        impl ::core::default::Default for $name {
            fn default() -> Self {
                Self { $($field: $default,)* }
            }
        }

        impl $name {
            #[doc(hidden)]
            pub fn state_rules(schema: &mut $crate::__private::schemars::Schema) {
                $($crate::state(
                    schema,
                    ::core::stringify!($field),
                    $crate::rule::$rule($($($arg),*)?),
                );)*
            }
        }

        impl ::engine_core::Section for $name {
            const KEY: &'static str = $key;

            fn check(&self) -> ::core::result::Result<(), $crate::Error> {
                $($crate::__checked!(
                    self.$field,
                    ::core::stringify!($field);
                    $rule $(($($arg),*))?
                );)*
                $($crate::refused($check(self))?;)?
                ::core::result::Result::Ok(())
            }
        }
    };
    (
        #[section($key:literal)]
        #[check($check:path)]
        $($rest:tt)*
    ) => {
        $crate::settings! { @section $key [$check] $($rest)* }
    };
    (
        #[section($key:literal)]
        $($rest:tt)*
    ) => {
        $crate::settings! { @section $key [] $($rest)* }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __checked {
    ($value:expr, $field:expr; any) => {};
    ($value:expr, $field:expr; $rule:ident $(($($arg:expr),*))?) => {
        $crate::refused($crate::Setting::check(
            &$value,
            $field,
            $crate::rule::$rule($($($arg),*)?),
        ))?
    };
}

#[macro_export]
macro_rules! stats {
    (@pending $this:tt $field:ident pending) => { $this.$field };
    (@pending $this:tt $field:ident) => { 0u32 };
    (
        #[section($key:literal)]
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $($(#[$mark:ident])? $field_vis:vis $field:ident: $ty:ty),* $(,)?
        }
    ) => {
        #[derive(
            ::core::clone::Clone,
            ::core::fmt::Debug,
            ::core::default::Default,
            ::core::cmp::PartialEq,
            bevy_ecs::resource::Resource,
            ::engine_core::__private::serde::Serialize,
            ::engine_core::__private::serde::Deserialize,
            ::engine_core::__private::schemars::JsonSchema
        )]
        #[serde(crate = "::engine_core::__private::serde", default)]
        #[schemars(crate = "::engine_core::__private::schemars")]
        $(#[$meta])*
        $vis struct $name {
            $($field_vis $field: $ty,)*
        }

        impl ::engine_core::Section for $name {
            const KEY: &'static str = $key;

            fn pending(&self) -> u32 {
                0u32 $(.saturating_add($crate::stats!(@pending self $field $($mark)?)))*
            }
        }
    };
}
