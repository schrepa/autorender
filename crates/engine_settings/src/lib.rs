//! Settings and authored values declared once per field. [`settings!`] takes
//! each field's default and the [`Rule`] its values keep, and from that one
//! declaration gives the type its `Default`, `validate` (the first field
//! breaking its rule, then [`ExtraRules::check_extra`]), `sanitized` (every
//! field made to keep its rule, then [`ExtraRules::sanitize_extra`]) and the
//! bounds its JSON schema states. Every field names a rule, `any` where none
//! applies, so none goes unchecked by omission.

use std::borrow::Cow;

use glam::Vec3;

/// A field outside its valid values: its name and the rule it breaks, such
/// as `must be in [0, 1]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidSetting {
    /// The field's name, with its parents' (`motion_detail.speed`).
    pub field: Cow<'static, str>,
    pub rule: Cow<'static, str>,
}

impl InvalidSetting {
    pub fn new(field: &'static str, rule: impl Into<Cow<'static, str>>) -> Self {
        Self {
            field: Cow::Borrowed(field),
            rule: rule.into(),
        }
    }

    /// The error of a field nested in `parent`.
    pub fn within(self, parent: &'static str) -> Self {
        Self {
            field: Cow::Owned(format!("{parent}.{}", self.field)),
            rule: self.rule,
        }
    }
}

impl std::fmt::Display for InvalidSetting {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} {}", self.field, self.rule)
    }
}

impl std::error::Error for InvalidSetting {}

/// What values a setting accepts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    /// Any value of its type: flags, names, seeds, and parts that keep rules
    /// of their own.
    Any,
    /// Any finite value.
    Finite,
    /// From the first bound to the second, both included.
    Range(f64, f64),
    /// Above the first bound and at most the second.
    AboveTo(f64, f64),
    /// Above the bound.
    Above(f64),
    /// At least the bound.
    AtLeast(f64),
    /// An optional budget: omitted for none, otherwise at least 1.
    Budget,
    /// A power of two from the first bound to the second; sanitizing rounds
    /// up to one.
    PowerOfTwo(f64, f64),
    /// A multiple of the first value from the second to the third;
    /// sanitizing rounds down to one.
    Multiple(f64, f64, f64),
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

    pub const fn budget() -> Self {
        Self::Budget
    }

    pub const fn power_of_two(min: f64, max: f64) -> Self {
        Self::PowerOfTwo(min, max)
    }

    pub const fn multiple(step: f64, min: f64, max: f64) -> Self {
        Self::Multiple(step, min, max)
    }

    /// The rule as an error states it.
    pub fn describe(self) -> String {
        match self {
            Self::Any => "may be anything".to_owned(),
            Self::Finite => "must be finite".to_owned(),
            Self::Range(min, max) => format!("must be in [{min}, {max}]"),
            Self::AboveTo(min, max) => format!("must be in ({min}, {max}]"),
            Self::Above(min) => format!("must be finite and above {min}"),
            Self::AtLeast(min) => format!("must be at least {min}"),
            Self::Budget => "must be at least 1, or omitted for no budget".to_owned(),
            Self::PowerOfTwo(min, max) => format!("must be a power of two in [{min}, {max}]"),
            Self::Multiple(step, min, max) => {
                format!("must be a multiple of {step} in [{min}, {max}]")
            }
        }
    }

    /// The JSON schema keywords stating the rule, its bounds whole numbers
    /// for `whole` values.
    #[cfg(feature = "schema")]
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
        }
        keywords
    }
}

/// The rules as declarations name them: `range(min, max)` and the like,
/// their bounds any number, and `same(...)` for the rule another type
/// declares for a field it mirrors.
pub mod rule {
    use super::Rule;

    /// A rule's bound: any number.
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

    /// Through its shortest decimal form, so 0.1 states 0.1.
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

    pub fn budget() -> Rule {
        Rule::Budget
    }

    pub fn power_of_two(min: impl Bound, max: impl Bound) -> Rule {
        Rule::PowerOfTwo(min.bound(), max.bound())
    }

    pub fn multiple(step: impl Bound, min: impl Bound, max: impl Bound) -> Rule {
        Rule::Multiple(step.bound(), min.bound(), max.bound())
    }

    /// The rule another type declares for a field this one mirrors, from
    /// its `rule_of`.
    pub fn same(rule: Option<Rule>) -> Rule {
        rule.expect("the mirrored type declares the field")
    }
}

/// States `rule` in the schema of `field`, a property of `schema`: on each
/// number of an array (of arrays), otherwise on the value. A bound on
/// anything but numbers fails the schema's build.
#[cfg(feature = "schema")]
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
                .unwrap_or_else(|| panic!("{field}: array schemas describe their items"));
            return numbers(items, field);
        }
        assert!(
            has_type(schema, "number") || has_type(schema, "integer"),
            "{field}: rules with bounds apply to numbers"
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
        panic!("settings field {field} has no schema property");
    };
    let target = numbers(property, field);
    let whole = has_type(target, "integer");
    target.extend(rule.keywords(whole));
}

/// States `rule` on `schema` itself, such as a request field's: a
/// transform `stated(rule)` for `#[schemars(transform = ...)]`.
#[cfg(feature = "schema")]
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
            .expect("the stated schema stays a schema");
    }
}

/// What a declared type checks beyond each field's own rule: rules relating
/// fields, and what a [`Rule`] cannot state. Every type says, if only with an
/// empty `impl`.
pub trait ExtraRules: Sized {
    /// Ok when the extra rules hold; runs after every field kept its rule.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        Ok(())
    }

    /// The value made to keep the extra rules, after every field was made to
    /// keep its own.
    fn sanitize_extra(self) -> Self {
        self
    }
}

/// A field type rules apply to. Types with nothing to check (flags, names,
/// enums) implement it empty and take `any`; [`settings!`] implements it for
/// the types it declares, which then nest.
pub trait Setting: Clone {
    /// Ok when the value keeps `rule`; otherwise the error names `field`.
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        assert_eq!(rule, Rule::Any, "{field}: this type takes no rule but any");
        Ok(())
    }

    /// The value made to keep `rule`, `default` where it is unusable.
    fn kept(self, rule: Rule, default: Self) -> Self {
        let _ = (rule, default);
        self
    }

    /// Whether the value can keep a rule by clamping: not NaN or infinite.
    fn usable(&self) -> bool {
        true
    }
}

fn broken(field: &'static str, rule: Rule) -> InvalidSetting {
    InvalidSetting::new(field, rule.describe())
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
            Rule::Budget | Rule::PowerOfTwo(..) | Rule::Multiple(..) => {
                panic!("{field}: whole-number rules apply to whole numbers")
            }
        };
        if keeps {
            Ok(())
        } else {
            Err(broken(field, rule))
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
            _ => self,
        }
    }

    fn usable(&self) -> bool {
        self.is_finite()
    }
}

/// Integer rules, in `u64`.
fn whole_keeps(value: u64, rule: Rule) -> bool {
    let within = |min: f64, max: f64| (min as u64..=max as u64).contains(&value);
    match rule {
        Rule::Any | Rule::Finite => true,
        Rule::Range(min, max) => within(min, max),
        Rule::AboveTo(min, max) => value > min as u64 && value <= max as u64,
        Rule::Above(min) => value > min as u64,
        Rule::AtLeast(min) => value >= min as u64,
        Rule::Budget => value >= 1,
        Rule::PowerOfTwo(min, max) => within(min, max) && value.is_power_of_two(),
        Rule::Multiple(step, min, max) => within(min, max) && value.is_multiple_of(step as u64),
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
        Rule::Budget => value.max(1),
        Rule::PowerOfTwo(min, max) => clamp(min, max).next_power_of_two().min(max as u64),
        Rule::Multiple(step, min, max) => {
            let step = step as u64;
            (clamp(min, max) / step * step).max(min as u64)
        }
    }
}

macro_rules! whole_settings {
    ($($ty:ty),*) => {$(
        impl Setting for $ty {
            fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
                if whole_keeps(u64::from(*self), rule) {
                    Ok(())
                } else {
                    Err(broken(field, rule))
                }
            }

            fn kept(self, rule: Rule, _: Self) -> Self {
                whole_kept(u64::from(self), rule).min(u64::from(<$ty>::MAX)) as $ty
            }
        }
    )*};
}

whole_settings!(u8, u16, u32, u64);

impl Setting for bool {}
impl Setting for String {}
/// A world box, which its type's extra rules check (`any` here).
impl Setting for (Vec3, Vec3) {}

/// Omitted, or keeping the rule; an unusable value reads as the default.
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

/// Each value of an array keeps the rule.
impl<T: Setting, const N: usize> Setting for [T; N] {
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        if self.iter().all(|value| value.check(field, rule).is_ok()) {
            Ok(())
        } else {
            Err(InvalidSetting::new(
                field,
                format!("each value {}", rule.describe()),
            ))
        }
    }

    fn kept(self, rule: Rule, default: Self) -> Self {
        let mut defaults = default.into_iter();
        self.map(|value| {
            let default = defaults.next().expect("arrays of one length");
            value.kept(rule, default)
        })
    }

    fn usable(&self) -> bool {
        self.iter().all(Setting::usable)
    }
}

/// Each value of a list keeps the rule; an error names the value by index.
impl<T: Setting> Setting for Vec<T> {
    fn check(&self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        for (index, value) in self.iter().enumerate() {
            value.check(field, rule).map_err(|invalid| {
                let rest = invalid.field.strip_prefix(field).unwrap_or_default();
                InvalidSetting {
                    field: format!("{field}[{index}]{rest}").into(),
                    rule: invalid.rule,
                }
            })?;
        }
        Ok(())
    }

    fn kept(self, rule: Rule, _: Self) -> Self {
        self.into_iter()
            .filter(Setting::usable)
            .map(|value| value.clone().kept(rule, value))
            .collect()
    }

    fn usable(&self) -> bool {
        self.iter().all(Setting::usable)
    }
}

/// Each channel of a vector keeps the rule.
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

/// Declares a type once per field: `pub name: Type = default => rule(...)`,
/// where the rule is one of [`rule`]'s (`any`, `finite`, `range(min, max)`,
/// `above_to(min, max)`, `above(min)`, `at_least(min)`, `budget`,
/// `power_of_two(min, max)`, `multiple(step, min, max)`, or
/// `same(Other::rule_of("field"))` for a field mirroring another type's). The
/// caller derives what the type needs (the JSON schema included, which states
/// the rules when the caller's `schema` feature is on) and implements
/// [`ExtraRules`] for it. A declared type nests in others as a field of rule
/// `any`, checked by its own rules.
///
/// A type with no JSON schema of its own, such as a runtime type an authored
/// one mirrors, starts with `unauthored;`; its mirror's schema states the
/// same rules with `#[schemars(transform = Type::state_rules)]`.
#[macro_export]
macro_rules! settings {
    (
        @declare [$($schema:tt)*]
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

        impl $name {
            /// Ok when every field keeps its rule, as the schema states it,
            /// and the type's extra rules hold; otherwise the first field
            /// breaking one.
            pub fn validate(&self) -> Result<(), $crate::InvalidSetting> {
                $($crate::Setting::check(
                    &self.$field,
                    stringify!($field),
                    $crate::rule::$rule($($($arg),*)?),
                )?;)*
                $crate::ExtraRules::check_extra(self)
            }

            /// The value with every field made to keep its rule (clamped,
            /// or its default where unusable), then the type's extra rules.
            pub fn sanitized(self) -> Self {
                let defaults = Self::default();
                $crate::ExtraRules::sanitize_extra(Self {
                    $($field: $crate::Setting::kept(
                        self.$field,
                        $crate::rule::$rule($($($arg),*)?),
                        defaults.$field,
                    ),)*
                })
            }

            /// The rule of field `field`, for authored types mirroring this
            /// one under other names.
            pub fn rule_of(field: &str) -> Option<$crate::Rule> {
                match field {
                    $(stringify!($field) => Some($crate::rule::$rule($($($arg),*)?)),)*
                    _ => None,
                }
            }

            /// States each field's rule in the JSON schema of the type, or
            /// of an authored type mirroring it field for field.
            #[cfg(feature = "schema")]
            pub fn state_rules(schema: &mut schemars::Schema) {
                $($crate::state(
                    schema,
                    stringify!($field),
                    $crate::rule::$rule($($($arg),*)?),
                );)*
            }
        }

        /// Nested in another declared type, it keeps its own rules.
        impl $crate::Setting for $name {
            fn check(
                &self,
                field: &'static str,
                _: $crate::Rule,
            ) -> Result<(), $crate::InvalidSetting> {
                self.validate().map_err(|invalid| invalid.within(field))
            }

            fn kept(self, _: $crate::Rule, _: Self) -> Self {
                self.sanitized()
            }
        }
    };
    (unauthored; $($rest:tt)*) => {
        $crate::settings!(@declare [] $($rest)*);
    };
    ($($rest:tt)*) => {
        $crate::settings!(
            @declare [#[cfg_attr(feature = "schema", schemars(transform = Self::state_rules))]]
            $($rest)*
        );
    };
}
