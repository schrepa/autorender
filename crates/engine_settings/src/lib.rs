#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod error;
pub mod fields;

pub use engine_codec::{Allowed, Detail, Error, ErrorCode, Expected, MAX_GOT_BYTES, closest};
#[cfg(target_arch = "wasm32")]
pub use error::js_error;
pub use error::{JS_ERROR_NAME, parse_json, parse_value, parse_value_at};
pub use fields::{ExtraRules, ID_PATTERN, InvalidSetting, MAX_ID_BYTES, Rule, Setting, is_id};
#[cfg(feature = "schema")]
pub use fields::{state, stated};

pub mod rule {
    use crate::{Error, ErrorCode};

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Rule {
        low: f64,
        high: f64,
    }

    pub trait Setting {
        fn measure(&self) -> Option<f64>;

        #[doc(hidden)]
        fn kept_by(&self, rule: Rule) -> bool {
            self.measure()
                .is_none_or(|measured| measured >= rule.low && measured <= rule.high)
        }
    }

    impl Setting for bool {
        fn measure(&self) -> Option<f64> {
            None
        }
    }

    macro_rules! measured {
        ($($ty:ty),*) => {$(
            impl Setting for $ty {
                fn measure(&self) -> Option<f64> {
                    Some(*self as f64)
                }
            }
        )*};
    }
    measured!(f32, f64, u8, u16, u32, u64, i32, i64, usize);

    impl<T: Setting, const N: usize> Setting for [T; N] {
        fn measure(&self) -> Option<f64> {
            None
        }

        fn kept_by(&self, rule: Rule) -> bool {
            self.iter().all(|value| value.kept_by(rule))
        }
    }

    impl<T: Setting> Setting for Option<T> {
        fn measure(&self) -> Option<f64> {
            None
        }

        fn kept_by(&self, rule: Rule) -> bool {
            self.as_ref().is_none_or(|value| value.kept_by(rule))
        }
    }

    impl<T: Setting> Setting for Vec<T> {
        fn measure(&self) -> Option<f64> {
            None
        }

        fn kept_by(&self, rule: Rule) -> bool {
            self.iter().all(|value| value.kept_by(rule))
        }
    }

    pub fn any() -> Rule {
        Rule {
            low: f64::NEG_INFINITY,
            high: f64::INFINITY,
        }
    }

    pub fn at_least(min: impl Into<f64>) -> Rule {
        Rule {
            low: min.into(),
            high: f64::INFINITY,
        }
    }

    pub fn at_most(max: impl Into<f64>) -> Rule {
        Rule {
            low: f64::NEG_INFINITY,
            high: max.into(),
        }
    }

    pub fn range(min: impl Into<f64>, max: impl Into<f64>) -> Rule {
        Rule {
            low: min.into(),
            high: max.into(),
        }
    }

    impl Rule {
        pub fn check(self, value: &impl Setting, path: &'static str) -> Result<(), Error> {
            if value.kept_by(self) {
                return Ok(());
            }
            let refused = Error::new(ErrorCode::InvalidValue, path);
            Err(match (self.low.is_finite(), self.high.is_finite()) {
                (true, true) => refused.range(self.low, self.high),
                (true, false) => refused.range(self.low, f64::MAX),
                (false, true) => refused.range(f64::MIN, self.high),
                (false, false) => refused,
            })
        }
    }
}

#[macro_export]
macro_rules! settings {
    (
        #[section($key:literal)]
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $(
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
        #[schemars(crate = "::engine_core::__private::schemars")]
        $(#[$meta])*
        $vis struct $name {
            $($field_vis $field: $ty,)*
        }

        impl ::core::default::Default for $name {
            fn default() -> Self {
                Self { $($field: $default,)* }
            }
        }

        impl ::engine_core::Section for $name {
            const KEY: &'static str = $key;

            fn check(&self) -> ::core::result::Result<(), $crate::Error> {
                $($crate::__checked!(
                    self.$field,
                    ::core::concat!($key, ".", ::core::stringify!($field));
                    $rule $(($($arg),*))?
                );)*
                ::core::result::Result::Ok(())
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __checked {
    ($value:expr, $path:expr; any) => {};
    ($value:expr, $path:expr; $rule:ident $(($($arg:expr),*))?) => {
        $crate::rule::$rule($($($arg),*)?).check(&$value, $path)?
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
