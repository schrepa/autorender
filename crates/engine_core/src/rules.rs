//! Settings declared once per field. [`settings!`] takes each field's default
//! and the [`Rule`] its values keep, and from that one declaration gives the
//! type its `Default`, `validate` (the first field breaking its rule, then
//! [`ExtraRules::check_extra`]), `sanitized` (every field made to keep its
//! rule, then [`ExtraRules::sanitize_extra`]) and the bounds its JSON schema
//! states. Every field names a rule, `any` where none applies, so none goes
//! unchecked by omission.

use std::borrow::Cow;

use glam::Vec3;

use crate::environment::InvalidSetting;

/// What values a setting accepts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    /// Any value of its type: flags, seeds, and parts that keep rules of
    /// their own.
    Any,
    /// From the first bound to the second, both included.
    Range(f64, f64),
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

    pub const fn range(min: f64, max: f64) -> Self {
        Self::Range(min, max)
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
            Self::Range(min, max) => format!("must be in [{min}, {max}]"),
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
            Self::Any => {}
            Self::Range(min, max) => {
                keywords.insert("minimum".into(), number(min));
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

/// States `rule` in the schema of `field`, a property of `schema`: on each
/// item of an array, otherwise on the value.
#[cfg(feature = "schema")]
pub fn state(schema: &mut schemars::Schema, field: &str, rule: Rule) {
    let Some(property) = schema
        .get_mut("properties")
        .and_then(|properties| properties.get_mut(field))
        .and_then(serde_json::Value::as_object_mut)
    else {
        panic!("settings field {field} has no schema property");
    };
    let is_array = property.get("type").is_some_and(|kind| {
        kind == "array"
            || kind
                .as_array()
                .is_some_and(|kinds| kinds.iter().any(|k| k == "array"))
    });
    let target = if is_array {
        property
            .get_mut("items")
            .and_then(serde_json::Value::as_object_mut)
            .expect("array schemas describe their items")
    } else {
        property
    };
    let whole = target.get("type").is_some_and(|kind| {
        kind == "integer"
            || kind
                .as_array()
                .is_some_and(|kinds| kinds.iter().any(|k| k == "integer"))
    });
    target.extend(rule.keywords(whole));
}

/// A settings type [`settings!`] declared.
pub trait Settings: Copy {
    /// Ok when every field keeps its rule and the type's extra rules hold.
    fn check(&self) -> Result<(), InvalidSetting>;
    /// The settings made to keep every rule.
    fn kept(self) -> Self;
}

/// What a settings type checks beyond each field's own rule: rules relating
/// fields, and what a [`Rule`] cannot state. Every type says, if only with an
/// empty `impl`.
pub trait ExtraRules: Sized {
    /// Ok when the extra rules hold; runs after every field kept its rule.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        Ok(())
    }

    /// The settings made to keep the extra rules, after every field was made
    /// to keep its own.
    fn sanitize_extra(self) -> Self {
        self
    }
}

/// A field type rules apply to.
pub trait Setting: Copy {
    /// Ok when the value keeps `rule`; otherwise the error names `field`.
    fn check(self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting>;
    /// The value made to keep `rule`, `default` where it is unusable.
    fn kept(self, rule: Rule, default: Self) -> Self;
}

fn broken(field: &'static str, rule: Rule) -> InvalidSetting {
    InvalidSetting::new(field, rule.describe())
}

impl Setting for f32 {
    fn check(self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        let keeps = match rule {
            Rule::Any => true,
            Rule::Range(min, max) => (min as f32..=max as f32).contains(&self),
            Rule::Above(min) => self.is_finite() && self > min as f32,
            Rule::AtLeast(min) => self.is_finite() && self >= min as f32,
            Rule::Budget | Rule::PowerOfTwo(..) | Rule::Multiple(..) => {
                unreachable!("{field}: whole-number rules apply to whole numbers")
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
            Rule::Range(min, max) => crate::math::clamp(self, min as f32, max as f32),
            Rule::Above(min) if self <= min as f32 => default,
            Rule::AtLeast(min) => crate::math::max(self, min as f32),
            _ => self,
        }
    }
}

/// Integer rules, in `u64`.
fn whole_keeps(value: u64, rule: Rule) -> bool {
    let within = |min: f64, max: f64| (min as u64..=max as u64).contains(&value);
    match rule {
        Rule::Any => true,
        Rule::Range(min, max) => within(min, max),
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
        Rule::Any => value,
        Rule::Range(min, max) => clamp(min, max),
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

impl Setting for u32 {
    fn check(self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        if whole_keeps(self.into(), rule) {
            Ok(())
        } else {
            Err(broken(field, rule))
        }
    }

    fn kept(self, rule: Rule, _: Self) -> Self {
        whole_kept(self.into(), rule).min(u32::MAX.into()) as u32
    }
}

impl Setting for u64 {
    fn check(self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        if whole_keeps(self, rule) {
            Ok(())
        } else {
            Err(broken(field, rule))
        }
    }

    fn kept(self, rule: Rule, _: Self) -> Self {
        whole_kept(self, rule)
    }
}

impl Setting for bool {
    fn check(self, _: &'static str, _: Rule) -> Result<(), InvalidSetting> {
        Ok(())
    }

    fn kept(self, _: Rule, _: Self) -> Self {
        self
    }
}

/// Budgets and other optional numbers: omitted, or keeping the rule.
impl Setting for Option<u32> {
    fn check(self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        self.map_or(Ok(()), |value| value.check(field, rule))
    }

    fn kept(self, rule: Rule, _: Self) -> Self {
        self.map(|value| value.kept(rule, value))
    }
}

/// Each value of an array keeps the rule.
impl<const N: usize> Setting for [f32; N] {
    fn check(self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
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
        std::array::from_fn(|index| self[index].kept(rule, default[index]))
    }
}

/// Each channel of a vector keeps the rule.
impl Setting for Vec3 {
    fn check(self, field: &'static str, rule: Rule) -> Result<(), InvalidSetting> {
        self.to_array().check(field, rule)
    }

    fn kept(self, rule: Rule, default: Self) -> Self {
        Self::from_array(self.to_array().kept(rule, default.to_array()))
    }
}

/// Optional nested settings keep their own rules (`any` here).
impl<T: Settings> Setting for Option<T> {
    fn check(self, field: &'static str, _: Rule) -> Result<(), InvalidSetting> {
        self.map_or(Ok(()), |settings| {
            settings.check().map_err(|invalid| invalid.within(field))
        })
    }

    fn kept(self, _: Rule, _: Self) -> Self {
        self.map(Settings::kept)
    }
}

/// A world box, which its type's extra rules check (`any` here).
impl Setting for Option<(Vec3, Vec3)> {
    fn check(self, _: &'static str, _: Rule) -> Result<(), InvalidSetting> {
        Ok(())
    }

    fn kept(self, _: Rule, _: Self) -> Self {
        self
    }
}

impl InvalidSetting {
    /// The error of a field nested in `parent`.
    pub(crate) fn within(self, parent: &'static str) -> Self {
        Self {
            field: Cow::Owned(format!("{parent}.{}", self.field)),
            rule: self.rule,
        }
    }
}

/// Declares a settings type once per field: `pub name: Type = default =>
/// rule(...)`, where the rule is one of [`Rule`]'s constructors (`any`,
/// `range(min, max)`, `above(min)`, `at_least(min)`, `budget`,
/// `power_of_two(min, max)`, `multiple(step, min, max)`). The caller derives
/// what the type needs (the JSON schema included) and implements
/// [`ExtraRules`] for it.
macro_rules! settings {
    (
        $(#[$meta:meta])*
        pub struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                pub $field:ident: $ty:ty = $default:expr => $rule:ident $(($($arg:expr),* $(,)?))?
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[cfg_attr(feature = "schema", schemars(transform = $name::state_rules))]
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
            pub fn validate(&self) -> Result<(), $crate::environment::InvalidSetting> {
                $($crate::rules::Setting::check(
                    self.$field,
                    stringify!($field),
                    $crate::rules::Rule::$rule($($($arg as f64),*)?),
                )?;)*
                $crate::rules::ExtraRules::check_extra(self)
            }

            /// The settings with every field made to keep its rule (clamped,
            /// or its default where unusable), then the type's extra rules.
            pub fn sanitized(self) -> Self {
                let defaults = Self::default();
                $crate::rules::ExtraRules::sanitize_extra(Self {
                    $($field: $crate::rules::Setting::kept(
                        self.$field,
                        $crate::rules::Rule::$rule($($($arg as f64),*)?),
                        defaults.$field,
                    ),)*
                })
            }

            /// States each field's rule in the type's JSON schema.
            #[cfg(feature = "schema")]
            fn state_rules(schema: &mut schemars::Schema) {
                $($crate::rules::state(
                    schema,
                    stringify!($field),
                    $crate::rules::Rule::$rule($($($arg as f64),*)?),
                );)*
            }
        }

        impl $crate::rules::Settings for $name {
            fn check(&self) -> Result<(), $crate::environment::InvalidSetting> {
                self.validate()
            }

            fn kept(self) -> Self {
                self.sanitized()
            }
        }
    };
}

pub(crate) use settings;
