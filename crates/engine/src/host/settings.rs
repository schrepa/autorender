use bevy_ecs::prelude::World;
use engine_core::{Engine, Error, ErrorCode, SceneContext, Section};
use engine_registry::SettingsEntry;
use engine_settings::Expected;
use serde_json::Value;

use super::JobClock;
use crate::{Game, GameHost};

#[derive(Clone, Copy, Debug)]
pub struct Setting {
    pub name: &'static str,
    pub values: &'static [&'static str],
}

impl Setting {
    pub fn accepted(&self) -> String {
        self.values.join("|")
    }

    pub fn default_value(&self) -> &'static str {
        self.values[0]
    }
}

pub const OVERLAY: &str = "set";

const JOB_CLOCK: Setting = Setting {
    name: "job-clock",
    values: &["wall", "frames"],
};

pub const SETTINGS: &[Setting] = &[JOB_CLOCK];

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HostSettings {
    pub job_clock: Option<JobClock>,
    pub overlay: Vec<(String, Value)>,
}

impl HostSettings {
    pub fn takes(name: &str) -> bool {
        name == OVERLAY || SETTINGS.iter().any(|setting| setting.name == name)
    }

    pub fn set(&mut self, name: &str, value: &str) -> Result<bool, Error> {
        if name == OVERLAY {
            self.overlay.push(overlaid(value)?);
        } else if name == JOB_CLOCK.name {
            self.job_clock = Some(job_clock(value)?);
        } else {
            return Ok(false);
        }
        Ok(true)
    }

    pub fn apply<G: Game>(&self, host: &mut GameHost<G>) -> Result<(), Error> {
        if let Some(clock) = self.job_clock {
            host.set_job_clock(clock);
        }
        host.set_settings(&self.overlay)
            .map_err(|error| error.within(OVERLAY))
    }
}

fn job_clock(value: &str) -> Result<JobClock, Error> {
    match value {
        "wall" => Ok(JobClock::Wall),
        "frames" => Ok(JobClock::Frames),
        _ => Err(Error::new(ErrorCode::InvalidValue, "$")
            .got(value)
            .variants(JOB_CLOCK.values.iter().copied())),
    }
}

fn overlaid(entry: &str) -> Result<(String, Value), Error> {
    let Some((key, value)) = entry.split_once('=').filter(|(key, _)| !key.is_empty()) else {
        return Err(Error::new(ErrorCode::InvalidValue, "$").got(entry));
    };
    let value = engine_settings::parse_json::<Value>(value).map_err(|error| error.within("$"))?;
    Ok((key.to_owned(), value))
}

fn scoped<R>(world: &mut World, run: impl FnOnce(&mut World) -> R) -> R {
    match world.get_resource::<SceneContext>().cloned() {
        Some(context) => context.scope(|| run(world)),
        None => run(world),
    }
}

fn split(key: &str) -> (&str, Option<&str>) {
    match key.split_once('.') {
        Some((section, field)) => (section, Some(field)),
        None => (key, None),
    }
}

fn not_object(section: &str) -> Error {
    Error::new(ErrorCode::InvalidType, format!("$.{section}")).expected(Expected::Object)
}

fn known_field(entry: &SettingsEntry, key: &str, field: &str) -> Result<(), Error> {
    let schema = entry.schema();
    let fields = schema.get("properties").and_then(Value::as_object);
    if fields.is_some_and(|fields| fields.contains_key(field)) {
        return Ok(());
    }
    Err(Error::new(ErrorCode::UnknownField, format!("$.{key}"))
        .got(field)
        .variants(fields.into_iter().flatten().map(|(name, _)| name.as_str())))
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SettingsLog {
    entries: Vec<(String, Value)>,
}

impl SettingsLog {
    pub fn set(&mut self, world: &mut World, entries: &[(String, Value)]) -> Result<(), Error> {
        apply_settings(world, entries)?;
        for (key, value) in entries {
            self.entries.retain(|(held, _)| !replaces(key, held));
            self.entries.push((key.clone(), value.clone()));
        }
        Ok(())
    }

    pub fn replay(&self, world: &mut World) -> Result<(), Error> {
        apply_settings(world, &self.entries)
    }
}

fn replaces(key: &str, held: &str) -> bool {
    held.strip_prefix(key)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
}

pub struct Settings<'w, S> {
    world: &'w mut World,
    section: std::marker::PhantomData<fn() -> S>,
}

impl<S: Section> Settings<'_, S> {
    pub fn get(&self) -> S {
        self.world.get_resource::<S>().cloned().unwrap_or_default()
    }

    pub fn set(self, change: impl FnOnce(&mut S)) -> Result<(), Error> {
        let entry = changed_section(self.world, change)?;
        apply_settings(self.world, &[entry])
    }
}

pub trait TypedSettings {
    fn section<S: Section>(&mut self) -> Settings<'_, S>;
}

impl TypedSettings for World {
    fn section<S: Section>(&mut self) -> Settings<'_, S> {
        Settings {
            world: self,
            section: std::marker::PhantomData,
        }
    }
}

impl TypedSettings for Engine {
    fn section<S: Section>(&mut self) -> Settings<'_, S> {
        self.app.world_mut().section()
    }
}

pub fn changed_section<S: Section>(
    world: &World,
    change: impl FnOnce(&mut S),
) -> Result<(String, Value), Error> {
    let mut section = world.get_resource::<S>().cloned().unwrap_or_default();
    change(&mut section);
    let value = serde_json::to_value(&section).map_err(|error| {
        Error::new(ErrorCode::InvalidValue, format!("$.{}", S::KEY)).got(error.to_string())
    })?;
    Ok((S::KEY.to_owned(), value))
}

pub fn has_setting(world: &World, key: &str) -> bool {
    let (section, _) = split(key);
    engine_registry::settings(world, section).is_ok()
}

pub fn apply_setting(world: &mut World, key: &str, value: &Value) -> Result<(), Error> {
    apply_settings(world, &[(key.to_owned(), value.clone())])
}

pub fn apply_settings(world: &mut World, entries: &[(String, Value)]) -> Result<(), Error> {
    if entries.is_empty() {
        return Ok(());
    }
    let mut touched: Vec<SettingsEntry> = Vec::with_capacity(entries.len());
    for (key, _) in entries {
        let (section, _) = split(key);
        let entry = engine_registry::settings(world, section)?;
        if !touched.iter().any(|known| known.key == entry.key) {
            touched.push(entry);
        }
    }
    scoped(world, |world| {
        let before: Vec<(SettingsEntry, Value)> = touched
            .iter()
            .map(|entry| entry.current(world).map(|value| (*entry, value)))
            .collect::<Result<_, _>>()?;
        let applied = entries
            .iter()
            .try_for_each(|(key, value)| set_one(world, key, value))
            .and_then(|()| engine_registry::check_sections(world));
        if applied.is_err() {
            for (entry, value) in &before {
                if entry.current(world).ok().as_ref() != Some(value) {
                    let _ = entry.set(world, value);
                }
            }
        }
        applied
    })
}

fn set_one(world: &mut World, key: &str, value: &Value) -> Result<(), Error> {
    let (section, field) = split(key);
    let entry = engine_registry::settings(world, section)?;
    let Some(field) = field else {
        return entry.set(world, value);
    };
    known_field(&entry, key, field)?;
    let mut current = entry.current(world)?;
    let Some(fields) = current.as_object_mut() else {
        return Err(not_object(section));
    };
    fields.insert(field.to_owned(), value.clone());
    entry.set(world, &current)
}

pub fn reset_setting(world: &mut World, key: &str) -> Result<(), Error> {
    let (section, field) = split(key);
    let entry = engine_registry::settings(world, section)?;
    let Some(field) = field else {
        entry.reset(world);
        return Ok(());
    };
    scoped(world, |world| {
        let defaults = entry.default_value();
        let mut current = entry.current(world)?;
        let Some(fields) = current.as_object_mut() else {
            return Err(not_object(section));
        };
        known_field(&entry, key, field)?;
        match defaults.get(field) {
            Some(default) => {
                fields.insert(field.to_owned(), default.clone());
            }
            None => {
                fields.remove(field);
            }
        }
        entry.set(world, &current)
    })
}
