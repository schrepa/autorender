use std::borrow::Cow;

use glam::Vec3;

use crate::{Allowed, Detail, Error, ErrorCode, Expected};

#[derive(Clone, Debug, PartialEq)]
pub struct InvalidSetting {
    pub field: Cow<'static, str>,
    pub detail: Detail,
}

impl InvalidSetting {
    pub fn new(field: &'static str, expected: Expected) -> Self {
        Self {
            field: Cow::Borrowed(field),
            detail: Detail {
                expected: Some(expected),
                ..Detail::default()
            },
        }
    }

    pub fn broken(field: &'static str, rule: Rule) -> Self {
        Self {
            field: Cow::Borrowed(field),
            detail: rule.detail(),
        }
    }

    #[must_use]
    pub fn allowed(mut self, allowed: Allowed) -> Self {
        self.detail.allowed = Some(allowed);
        self
    }

    #[must_use]
    pub fn got(mut self, value: impl serde::Serialize) -> Self {
        self.detail.got = Error::new(ErrorCode::OutOfRange, "$").got(value).detail.got;
        self
    }

    #[must_use]
    pub fn nearest(mut self, value: impl serde::Serialize) -> Self {
        self.detail.nearest = Error::new(ErrorCode::OutOfRange, "$")
            .nearest(value)
            .detail
            .nearest;
        self
    }

    #[must_use]
    pub fn within(self, parent: &'static str) -> Self {
        Self {
            field: Cow::Owned(format!("{parent}.{}", self.field)),
            detail: self.detail,
        }
    }

    #[must_use]
    pub fn indexed(self, field: &str, index: usize) -> Self {
        let rest = self.field.strip_prefix(field).unwrap_or_default();
        Self {
            field: Cow::Owned(format!("{field}[{index}]{rest}")),
            detail: self.detail,
        }
    }

    pub fn error(self, code: ErrorCode, path: &str) -> Error {
        Error {
            code,
            path: format!("{path}.{}", self.field),
            detail: Box::new(self.detail),
        }
    }
}

impl From<InvalidSetting> for Error {
    fn from(invalid: InvalidSetting) -> Self {
        invalid.error(ErrorCode::OutOfRange, "$")
    }
}

impl std::fmt::Display for InvalidSetting {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Error::from(self.clone()).fmt(formatter)
    }
}

impl std::error::Error for InvalidSetting {}

pub const MAX_ID_BYTES: usize = 64;

pub const ID_PATTERN: &str = "^[A-Za-z0-9_.-]+$";

pub fn is_id(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= MAX_ID_BYTES
        && text
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

const MAX_URI_BYTES: usize = 256;

pub fn valid_uri(uri: &str, path: &str) -> Result<(), Error> {
    let valid = !uri.is_empty()
        && uri.len() <= MAX_URI_BYTES
        && uri
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_./-".contains(&byte))
        && !uri.starts_with(['/', '.'])
        && uri
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..");
    if valid {
        Ok(())
    } else {
        Err(Error::new(ErrorCode::InvalidUri, path)
            .expected(Expected::Uri)
            .at_most(MAX_URI_BYTES as u64)
            .got(uri))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    Any,
    Finite,
    Range(f64, f64),
    AboveTo(f64, f64),
    Above(f64),
    AtLeast(f64),
    AtMost(f64),
    Budget,
    PowerOfTwo(f64, f64),
    Multiple(f64, f64, f64),
    Id,
    Count(f64, f64),
}

impl Rule {
    pub const fn any() -> Self {
        Self::Any
    }

    pub const fn finite() -> Self {
        Self::Finite
    }

    pub const fn range(min: f64, max: f64) -> Self {
        Self::Range(min, max)
    }

    pub const fn above_to(min: f64, max: f64) -> Self {
        Self::AboveTo(min, max)
    }

    pub const fn above(min: f64) -> Self {
        Self::Above(min)
    }

    pub const fn at_least(min: f64) -> Self {
        Self::AtLeast(min)
    }

    pub const fn at_most(max: f64) -> Self {
        Self::AtMost(max)
    }

    pub const fn budget() -> Self {
        Self::Budget
    }

    pub const fn power_of_two(min: f64, max: f64) -> Self {
        Self::PowerOfTwo(min, max)
    }

    pub const fn multiple(step: f64, min: f64, max: f64) -> Self {
        Self::Multiple(step, min, max)
    }

    pub const fn id() -> Self {
        Self::Id
    }

    pub const fn count(min: f64, max: f64) -> Self {
        Self::Count(min, max)
    }

    pub fn detail(self) -> Detail {
        let (expected, allowed) = match self {
            Self::Any => (None, None),
            Self::Finite => (Some(Expected::Finite), None),
            Self::Range(min, max) => (None, Some(Allowed::Range { min, max })),
            Self::AboveTo(min, max) => (Some(Expected::Above), Some(Allowed::Range { min, max })),
            Self::Above(min) => (
                Some(Expected::Above),
                Some(Allowed::Range { min, max: f64::MAX }),
            ),
            Self::AtLeast(min) if min >= 0.0 && min.fract() == 0.0 => (
                Some(Expected::Finite),
                Some(Allowed::AtLeast { min: min as u64 }),
            ),
            Self::AtLeast(min) => (
                Some(Expected::Finite),
                Some(Allowed::Range { min, max: f64::MAX }),
            ),
            Self::AtMost(max) if max >= 0.0 && max.fract() == 0.0 => (
                Some(Expected::Finite),
                Some(Allowed::AtMost { max: max as u64 }),
            ),
            Self::AtMost(max) => (
                Some(Expected::Finite),
                Some(Allowed::Range { min: f64::MIN, max }),
            ),
            Self::Budget => (Some(Expected::Integer), Some(Allowed::AtLeast { min: 1 })),
            Self::PowerOfTwo(min, max) => (
                Some(Expected::PowerOfTwo),
                Some(Allowed::Range { min, max }),
            ),
            Self::Multiple(_, min, max) => {
                (Some(Expected::Multiple), Some(Allowed::Range { min, max }))
            }
            Self::Id => (
                Some(Expected::Id),
                Some(Allowed::AtMost {
                    max: MAX_ID_BYTES as u64,
                }),
            ),
            Self::Count(min, max) if min > 0.0 => {
                (Some(Expected::Array), Some(Allowed::Range { min, max }))
            }
            Self::Count(_, max) => (
                Some(Expected::Array),
                Some(Allowed::AtMost { max: max as u64 }),
            ),
        };
        Detail {
            expected,
            allowed,
            ..Detail::default()
        }
    }

    fn keywords(self, whole: bool) -> serde_json::Map<String, serde_json::Value> {
        use serde_json::{Value, json};
        let number = |value: f64| -> Value {
            if whole {
                json!(value as i64)
            } else {
                json!(value)
            }
        };
        let mut keywords = serde_json::Map::new();
        match self {
            Self::Any | Self::Finite => {}
            Self::Range(min, max) => {
                keywords.insert("minimum".into(), number(min));
                keywords.insert("maximum".into(), number(max));
            }
            Self::AboveTo(min, max) => {
                keywords.insert("exclusiveMinimum".into(), number(min));
                keywords.insert("maximum".into(), number(max));
            }
            Self::Above(min) => {
                keywords.insert("exclusiveMinimum".into(), number(min));
            }
            Self::AtLeast(min) => {
                keywords.insert("minimum".into(), number(min));
            }
            Self::AtMost(max) => {
                keywords.insert("maximum".into(), number(max));
            }
            Self::Budget => {
                keywords.insert("minimum".into(), json!(1));
            }
            Self::PowerOfTwo(min, max) => {
                let powers: Vec<Value> = (0..63)
                    .map(|bit| 1_u64 << bit)
                    .filter(|&power| (min..=max).contains(&(power as f64)))
                    .map(Value::from)
                    .collect();
                keywords.insert("enum".into(), Value::Array(powers));
            }
            Self::Multiple(step, min, max) => {
                keywords.insert("multipleOf".into(), number(step));
                keywords.insert("minimum".into(), number(min));
                keywords.insert("maximum".into(), number(max));
            }
            Self::Id => {
                keywords.insert("maxLength".into(), json!(MAX_ID_BYTES));
                keywords.insert("minLength".into(), json!(1));
                keywords.insert("pattern".into(), json!(ID_PATTERN));
            }
            Self::Count(min, max) => {
                keywords.insert("maxItems".into(), json!(max as u64));
                if min > 0.0 {
                    keywords.insert("minItems".into(), json!(min as u64));
                }
            }
        }
        keywords
    }
}

pub trait Bound: Copy {
    fn bound(self) -> f64;
}

macro_rules! bounds {
    ($($ty:ty),*) => {$(
        impl Bound for $ty {
            fn bound(self) -> f64 {
                self as f64
            }
        }
    )*};
}

bounds!(f64, i32, u8, u16, u32, u64, usize);

impl Bound for f32 {
    fn bound(self) -> f64 {
        self.to_string().parse().unwrap_or(f64::from(self))
    }
}

pub fn any() -> Rule {
    Rule::Any
}

pub fn finite() -> Rule {
    Rule::Finite
}

pub fn range(min: impl Bound, max: impl Bound) -> Rule {
    Rule::Range(min.bound(), max.bound())
}

pub fn above_to(min: impl Bound, max: impl Bound) -> Rule {
    Rule::AboveTo(min.bound(), max.bound())
}

pub fn above(min: impl Bound) -> Rule {
    Rule::Above(min.bound())
}

pub fn at_least(min: impl Bound) -> Rule {
    Rule::AtLeast(min.bound())
}

pub fn at_most(max: impl Bound) -> Rule {
    Rule::AtMost(max.bound())
}

pub fn budget() -> Rule {
    Rule::Budget
}

pub fn power_of_two(min: impl Bound, max: impl Bound) -> Rule {
    Rule::PowerOfTwo(min.bound(), max.bound())
}

pub fn multiple(step: impl Bound, min: impl Bound, max: impl Bound) -> Rule {
    Rule::Multiple(step.bound(), min.bound(), max.bound())
}

pub fn same(rule: Option<Rule>) -> Rule {
    rule.expect("mirrored_rule_declared")
}

pub fn id() -> Rule {
    Rule::Id
}

pub fn count(min: impl Bound, max: impl Bound) -> Rule {
    Rule::Count(min.bound(), max.bound())
}

pub fn state(schema: &mut schemars::Schema, field: &str, rule: Rule) {
    use serde_json::{Map, Value};
    fn has_type(schema: &Map<String, Value>, wanted: &str) -> bool {
        schema.get("type").is_some_and(|kind| {
            kind == wanted
                || kind
                    .as_array()
                    .is_some_and(|kinds| kinds.iter().any(|k| k == wanted))
        })
    }
    fn numbers<'a>(schema: &'a mut Map<String, Value>, field: &str) -> &'a mut Map<String, Value> {
        if has_type(schema, "array") {
            let items = schema
                .get_mut("items")
                .and_then(Value::as_object_mut)
                .unwrap_or_else(|| panic!("array_schema_has_items {field}"));
            return numbers(items, field);
        }
        assert!(
            has_type(schema, "number") || has_type(schema, "integer"),
            "rule_needs_number {field}"
        );
        schema
    }
    if matches!(rule, Rule::Any | Rule::Finite) {
        return;
    }
    let Some(property) = schema
        .get_mut("properties")
        .and_then(|properties| properties.get_mut(field))
        .and_then(Value::as_object_mut)
    else {
        panic!("field_in_schema {field}");
    };
    let target = match rule {
        Rule::Id => {
            assert!(has_type(property, "string"), "rule_needs_string {field}");
            property
        }
        Rule::Count(..) => {
            assert!(has_type(property, "array"), "rule_needs_list {field}");
            property
        }
        Rule::Any
        | Rule::Finite
        | Rule::Range(..)
        | Rule::AboveTo(..)
        | Rule::Above(_)
        | Rule::AtLeast(_)
        | Rule::AtMost(_)
        | Rule::Budget
        | Rule::PowerOfTwo(..)
        | Rule::Multiple(..) => numbers(property, field),
    };
    let whole = has_type(target, "integer");
    target.extend(rule.keywords(whole));
}

pub fn stated(rule: Rule) -> impl Fn(&mut schemars::Schema) {
    move |schema| {
        let mut object = schemars::Schema::default();
        object.ensure_object().insert(
            "properties".into(),
            serde_json::json!({ "value": schema.clone() }),
        );
        state(&mut object, "value", rule);
        *schema = object
            .get("properties")
            .and_then(|properties| properties.get("value"))
            .cloned()
            .and_then(|value| schemars::Schema::try_from(value).ok())
            .expect("stated_schema_valid");
    }
}

pub trait ExtraRules: Sized {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        Ok(())
    }

    #[must_use]
    fn sanitize_extra(self) -> Self {
        self
    }
}

pub trait Setting: Clone {
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        assert_eq!(rule, Rule::Any, "type_takes_any_only {field}");
        Ok(())
    }

    #[must_use]
    fn kept(self, rule: Rule, default: Self) -> Self {
        let _ = (rule, default);
        self
    }

    fn usable(&self) -> bool {
        true
    }
}

impl Setting for f32 {
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        let value = *self;
        let keeps = match rule {
            Rule::Any => true,
            Rule::Finite => value.is_finite(),
            Rule::Range(min, max) => (min as f32..=max as f32).contains(&value),
            Rule::AboveTo(min, max) => value > min as f32 && value <= max as f32,
            Rule::Above(min) => value.is_finite() && value > min as f32,
            Rule::AtLeast(min) => value.is_finite() && value >= min as f32,
            Rule::AtMost(max) => value.is_finite() && value <= max as f32,
            Rule::Budget | Rule::PowerOfTwo(..) | Rule::Multiple(..) => {
                panic!("rule_needs_whole_number {field}")
            }
            Rule::Id | Rule::Count(..) => panic!("rule_needs_id_or_list {field}"),
        };
        if keeps {
            Ok(())
        } else {
            Err(InvalidSetting::broken(field, rule).got(Bound::bound(value)))
        }
    }

    fn kept(self, rule: Rule, default: Self) -> Self {
        if !self.is_finite() && rule != Rule::Any {
            return default;
        }
        match rule {
            Rule::Range(min, max) => engine_math::clamp(self, min as f32, max as f32),
            Rule::AboveTo(min, max) if self > min as f32 => engine_math::min(self, max as f32),
            Rule::Above(min) | Rule::AboveTo(min, _) if self <= min as f32 => default,
            Rule::AtLeast(min) => engine_math::max(self, min as f32),
            Rule::AtMost(max) => engine_math::min(self, max as f32),
            Rule::Any
            | Rule::Finite
            | Rule::AboveTo(..)
            | Rule::Above(_)
            | Rule::Budget
            | Rule::PowerOfTwo(..)
            | Rule::Multiple(..)
            | Rule::Id
            | Rule::Count(..) => self,
        }
    }

    fn usable(&self) -> bool {
        self.is_finite()
    }
}

fn whole_keeps(value: u64, rule: Rule) -> bool {
    let within = |min: f64, max: f64| (min as u64..=max as u64).contains(&value);
    match rule {
        Rule::Any | Rule::Finite => true,
        Rule::Range(min, max) => within(min, max),
        Rule::AboveTo(min, max) => value > min as u64 && value <= max as u64,
        Rule::Above(min) => value > min as u64,
        Rule::AtLeast(min) => value >= min as u64,
        Rule::AtMost(max) => value <= max as u64,
        Rule::Budget => value >= 1,
        Rule::PowerOfTwo(min, max) => within(min, max) && value.is_power_of_two(),
        Rule::Multiple(step, min, max) => within(min, max) && value.is_multiple_of(step as u64),
        Rule::Id | Rule::Count(..) => panic!("rule_needs_id_or_list"),
    }
}

fn whole_kept(value: u64, rule: Rule) -> u64 {
    let clamp = |min: f64, max: f64| value.clamp(min as u64, max as u64);
    match rule {
        Rule::Any | Rule::Finite => value,
        Rule::Range(min, max) => clamp(min, max),
        Rule::AboveTo(min, max) => value.clamp(min as u64 + 1, max as u64),
        Rule::Above(min) => value.max(min as u64 + 1),
        Rule::AtLeast(min) => value.max(min as u64),
        Rule::AtMost(max) => value.min(max as u64),
        Rule::Budget => value.max(1),
        Rule::PowerOfTwo(min, max) => clamp(min, max).next_power_of_two().min(max as u64),
        Rule::Multiple(step, min, max) => {
            let step = step as u64;
            (clamp(min, max) / step * step).max(min as u64)
        }
        Rule::Id | Rule::Count(..) => value,
    }
}

macro_rules! whole_settings {
    ($($ty:ty),*) => {$(
        impl Setting for $ty {
            fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
                let value = u64::from(*self);
                if whole_keeps(value, rule) {
                    Ok(())
                } else {
                    Err(InvalidSetting::broken(field, rule)
                        .got(value)
                        .nearest(whole_kept(value, rule)))
                }
            }

            fn kept(self, rule: Rule, _: Self) -> Self {
                whole_kept(u64::from(self), rule).min(u64::from(<$ty>::MAX)) as $ty
            }
        }
    )*};
}

whole_settings!(u8, u16, u32, u64);

impl Setting for usize {
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        (*self as u64).check(field, rule)
    }

    fn kept(self, rule: Rule, _: Self) -> Self {
        whole_kept(self as u64, rule) as usize
    }
}

impl Setting for f64 {
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        let value = *self;
        let keeps = match rule {
            Rule::Any => true,
            Rule::Finite => value.is_finite(),
            Rule::Range(min, max) => (min..=max).contains(&value),
            Rule::AboveTo(min, max) => value > min && value <= max,
            Rule::Above(min) => value.is_finite() && value > min,
            Rule::AtLeast(min) => value.is_finite() && value >= min,
            Rule::AtMost(max) => value.is_finite() && value <= max,
            Rule::Budget | Rule::PowerOfTwo(..) | Rule::Multiple(..) => {
                panic!("rule_needs_whole_number {field}")
            }
            Rule::Id | Rule::Count(..) => panic!("rule_needs_id_or_list {field}"),
        };
        if keeps {
            Ok(())
        } else {
            Err(InvalidSetting::broken(field, rule).got(value))
        }
    }

    fn usable(&self) -> bool {
        self.is_finite()
    }
}

impl Setting for bool {}

impl Setting for String {
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        match rule {
            Rule::Any => Ok(()),
            Rule::Id if is_id(self) => Ok(()),
            Rule::Id => Err(InvalidSetting::broken(field, rule).got(self)),
            Rule::Finite
            | Rule::Range(..)
            | Rule::AboveTo(..)
            | Rule::Above(_)
            | Rule::AtLeast(_)
            | Rule::AtMost(_)
            | Rule::Budget
            | Rule::PowerOfTwo(..)
            | Rule::Multiple(..)
            | Rule::Count(..) => panic!("rule_needs_any_or_id {field}"),
        }
    }
}
impl Setting for (Vec3, Vec3) {}

impl<T: Setting> Setting for Option<T> {
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        self.as_ref()
            .map_or(Ok(()), |value| value.check(field, rule))
    }

    fn kept(self, rule: Rule, default: Self) -> Self {
        match (self, default) {
            (Some(value), Some(default)) => Some(value.kept(rule, default)),
            (Some(value), None) if value.usable() => Some(value.clone().kept(rule, value)),
            (Some(_), None) => None,
            (None, _) => None,
        }
    }

    fn usable(&self) -> bool {
        self.as_ref().is_none_or(Setting::usable)
    }
}

impl<T: Setting, const N: usize> Setting for [T; N] {
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        for (index, value) in self.iter().enumerate() {
            value
                .check(field, rule)
                .map_err(|invalid| invalid.indexed(field, index))?;
        }
        Ok(())
    }

    fn kept(self, rule: Rule, default: Self) -> Self {
        let mut defaults = default.into_iter();
        self.map(|value| {
            let default = defaults.next().expect("arrays_same_length");
            value.kept(rule, default)
        })
    }

    fn usable(&self) -> bool {
        self.iter().all(Setting::usable)
    }
}

impl<T: Setting> Setting for Vec<T> {
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        if let Rule::Count(min, max) = rule {
            return if (min as usize..=max as usize).contains(&self.len()) {
                Ok(())
            } else {
                Err(InvalidSetting::broken(field, rule).got(self.len()))
            };
        }
        for (index, value) in self.iter().enumerate() {
            value
                .check(field, rule)
                .map_err(|invalid| invalid.indexed(field, index))?;
        }
        Ok(())
    }

    fn kept(self, rule: Rule, _: Self) -> Self {
        if let Rule::Count(_, max) = rule {
            let mut kept = self;
            kept.truncate(max as usize);
            return kept;
        }
        self.into_iter()
            .filter(Setting::usable)
            .map(|value| value.clone().kept(rule, value))
            .collect()
    }

    fn usable(&self) -> bool {
        self.iter().all(Setting::usable)
    }
}

impl Setting for Vec3 {
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        self.to_array().check(field, rule)
    }

    fn kept(self, rule: Rule, default: Self) -> Self {
        Self::from_array(self.to_array().kept(rule, default.to_array()))
    }

    fn usable(&self) -> bool {
        self.is_finite()
    }
}

#[macro_export]
macro_rules! fields {
    (
        @declare [$($schema:tt)*] $rules:ident
        $(#[$meta:meta])*
        pub struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                pub $field:ident: $ty:ty = $default:expr => $rule:ident $(($($arg:expr),* $(,)?))?
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $($schema)*
        pub struct $name {
            $($(#[$field_meta])* pub $field: $ty,)*
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    $($field: $default,)*
                }
            }
        }

        $crate::fields!(@rules $rules $name);

        impl $name {
            pub fn validate(&self) -> Result<(), $crate::fields::InvalidSetting> {
                $($crate::fields::Setting::check(
                    &self.$field,
                    stringify!($field),
                    $crate::fields::$rule($($($arg),*)?),
                )?;)*
                $crate::fields::ExtraRules::check_extra(self)
            }

            pub fn sanitized(self) -> Self {
                let defaults = Self::default();
                $crate::fields::ExtraRules::sanitize_extra(Self {
                    $($field: $crate::fields::Setting::kept(
                        self.$field,
                        $crate::fields::$rule($($($arg),*)?),
                        defaults.$field,
                    ),)*
                })
            }

            pub fn rule_of(field: &str) -> Option<$crate::fields::Rule> {
                match field {
                    $(stringify!($field) => Some($crate::fields::$rule($($($arg),*)?)),)*
                    _ => None,
                }
            }

            pub fn state_rules(schema: &mut $crate::__private::schemars::Schema) {
                $($crate::fields::state(
                    schema,
                    stringify!($field),
                    $crate::fields::$rule($($($arg),*)?),
                );)*
            }
        }

        impl $crate::fields::Setting for $name {
            fn check(
                &self,
                field: &'static str,
                _: $crate::fields::Rule,
            ) -> Result<(), $crate::fields::InvalidSetting> {
                self.validate().map_err(|invalid| invalid.within(field))
            }

            fn kept(self, _: $crate::fields::Rule, _: Self) -> Self {
                self.sanitized()
            }
        }
    };
    (@rules plain $name:ident) => {
        impl $crate::fields::ExtraRules for $name {}
    };
    (@rules extra $name:ident) => {};
    (unauthored; extra_rules; $($rest:tt)*) => {
        $crate::fields!(@declare [] extra $($rest)*);
    };
    (unauthored; $($rest:tt)*) => {
        $crate::fields!(@declare [] plain $($rest)*);
    };
    (extra_rules; $($rest:tt)*) => {
        $crate::fields!(
            @declare [#[schemars(transform = Self::state_rules)]] extra
            $($rest)*
        );
    };
    ($($rest:tt)*) => {
        $crate::fields!(
            @declare [#[schemars(transform = Self::state_rules)]] plain
            $($rest)*
        );
    };
}
