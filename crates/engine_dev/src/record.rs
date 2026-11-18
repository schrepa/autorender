use std::sync::Arc;

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_registry::{Answer, NameEntry, Reach, ReachMut, Registry, register, unavailable};
use engine_settings::{Error, ErrorCode, Expected, closest, parse_value_at};
use schemars::JsonSchema;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value, json};

pub trait RunSink: Send + Sync {
    fn line(&mut self, line: String) -> bool;
}

#[derive(Resource)]
pub struct RecordHost {
    pub about: fn(&World) -> Map<String, Value>,
    pub open: Open,
    pub instruments: &'static [&'static str],
    pub instrument: Switch,
}

pub type Open = fn(&str, usize) -> Result<Box<dyn RunSink>, Error>;

pub type Switch = fn(&mut ReachMut<'_>, &str, bool) -> Result<&'static str, &'static str>;

pub struct MarkPicture {
    pub run: String,
    pub file: String,
    pub size: [u32; 2],
}

#[derive(Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum Every {
    Frame,
    Mark,
}

#[derive(JsonSchema)]
#[schemars(untagged)]
enum RecordArgs {
    Start(Start),
    Stop(Stop),
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Stop {
    #[schemars(extend("const" = true))]
    stop: bool,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Start {
    name: String,
    every: Every,
    names: Vec<String>,
    instruments: Vec<String>,
    queue: usize,
    events: usize,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Mark {
    name: String,
    picture: Option<[u32; 2]>,
}

#[derive(Resource, Clone, Copy)]
struct Moment {
    frame: u64,
    tick: u64,
    seconds: f64,
}

#[derive(Clone, Copy, Default, Serialize)]
struct Dropped {
    entries: u64,
    events: u64,
}

struct Entry {
    at: Moment,
    mark: Option<String>,
    values: Vec<Value>,
    events: Value,
    events_dropped: u64,
    picture: Option<Value>,
}

#[derive(Serialize)]
struct Line<'a> {
    frame: u64,
    tick: u64,
    t: f64,
    mark: Option<&'a str>,
    values: Values<'a>,
    events: &'a Value,
    dropped: Dropped,
    #[serde(skip_serializing_if = "Option::is_none")]
    picture: Option<&'a Value>,
}

struct Values<'a> {
    names: &'a [Arc<NameEntry>],
    values: &'a [Value],
}

impl Serialize for Values<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.values.len()))?;
        for (entry, value) in self.names.iter().zip(self.values) {
            map.serialize_entry(entry.name.as_ref(), value)?;
        }
        map.end()
    }
}

#[derive(Resource)]
struct Recording {
    name: String,
    every: Every,
    names: Vec<Arc<NameEntry>>,
    instruments: Vec<&'static str>,
    switch: Switch,
    sink: Box<dyn RunSink>,
    events: Vec<engine_profile::Event>,
    held: Vec<(String, Entry)>,
    written: u64,
    unsent: Dropped,
    lost: Dropped,
}

impl Recording {
    fn entry(&mut self, reach: &ReachMut<'_>, at: Moment, mark: Option<&str>) -> Entry {
        let values = reach.read(|reach| self.names.iter().map(|entry| read(reach, entry)).collect());
        let events_dropped = engine_profile::drain(&mut self.events);
        self.lost.events += events_dropped;
        let events = serde_json::to_value(&self.events).unwrap_or_else(unavailable);
        self.events.clear();
        Entry {
            at,
            mark: mark.map(str::to_owned),
            values,
            events,
            events_dropped,
            picture: None,
        }
    }

    fn send(&mut self, entry: Entry) {
        let line = Line {
            frame: entry.at.frame,
            tick: entry.at.tick,
            t: entry.at.seconds,
            mark: entry.mark.as_deref(),
            values: Values {
                names: &self.names,
                values: &entry.values,
            },
            events: &entry.events,
            dropped: Dropped {
                entries: self.unsent.entries,
                events: self.unsent.events + entry.events_dropped,
            },
            picture: entry.picture.as_ref(),
        };
        let sent = serde_json::to_string(&line).is_ok_and(|line| self.sink.line(line));
        if sent {
            self.written += 1;
            self.unsent = Dropped::default();
        } else {
            self.unsent.entries += 1;
            self.unsent.events += entry.events_dropped;
            self.lost.entries += 1;
        }
    }
}

pub(crate) fn install(world: &mut World) {
    register(
        world,
        NameEntry::call("record", |reach, args| record(reach, args).map(Answer::Value)).args(
            shape::<RecordArgs>(),
            json!({"name": "cam-sweep", "every": "frame", "names": ["frame"], "instruments": [], "queue": 1024, "events": 4096}),
        ),
    );
    register(
        world,
        NameEntry::call("mark", mark).args(
            shape::<Mark>(),
            json!({"name": "after-turn", "picture": [1280, 720]}),
        ),
    );
}

pub fn record(reach: &mut ReachMut<'_>, args: &Value) -> Result<Value, Error> {
    let parsed = match args.get("stop") {
        Some(_) => RecordArgs::Stop(parse_value_at(args.clone(), "$")?),
        None => RecordArgs::Start(parse_value_at(args.clone(), "$")?),
    };
    match parsed {
        RecordArgs::Start(args) => start(reach, args),
        RecordArgs::Stop(Stop { stop: true }) => stop(reach),
        RecordArgs::Stop(Stop { stop: false }) => Err(Error::new(ErrorCode::InvalidValue, "$.stop")
            .expected(Expected::Present)
            .got(false)),
    }
}

fn shape<T: JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or_else(unavailable)
}

pub fn frame_end(reach: &mut ReachMut<'_>, frame: u64, tick: u64, seconds: f64) {
    let at = Moment {
        frame,
        tick,
        seconds,
    };
    reach.world.insert_resource(at);
    let Some(mut recording) = reach.world.remove_resource::<Recording>() else {
        return;
    };
    if recording.every == Every::Frame {
        let entry = recording.entry(reach, at, None);
        recording.send(entry);
    }
    reach.world.insert_resource(recording);
}

pub fn pictured(world: &mut World, mark: &MarkPicture, picture: Value) -> Value {
    let Some(mut recording) = world.get_resource_mut::<Recording>() else {
        return json!({"picture": picture, "record": unavailable("record_stopped")});
    };
    let Some(index) = (recording.name == mark.run)
        .then(|| recording.held.iter().position(|(held, _)| *held == mark.file))
        .flatten()
    else {
        return json!({"picture": picture, "record": unavailable("record_stopped")});
    };
    let (_, mut entry) = recording.held.remove(index);
    let reply = json!({"name": entry.mark, "frame": entry.at.frame, "picture": picture});
    entry.picture = Some(picture);
    recording.send(entry);
    reply
}

fn start(reach: &mut ReachMut<'_>, start: Start) -> Result<Value, Error> {
    if let Some(recording) = reach.world.get_resource::<Recording>() {
        return Err(Error::new(ErrorCode::CommandBusy, "$").got(&recording.name));
    }
    if !file_safe(&start.name) {
        return Err(Error::new(ErrorCode::InvalidName, "$.name").got(&start.name));
    }
    if start.queue == 0 {
        return Err(Error::new(ErrorCode::OutOfRange, "$.queue")
            .at_least(1)
            .got(start.queue));
    }
    let (names, links) = resolve(reach.world, &start.names)?;
    let facts = names
        .iter()
        .map(|entry| json!({"name": entry.name, "kind": entry.kind, "unit": entry.unit}))
        .collect();
    let host = reach
        .world
        .get_resource::<RecordHost>()
        .ok_or_else(|| Error::new(ErrorCode::MissingResource, "$").expected(Expected::Supported))?;
    let (about, open, known, instrument) = (host.about, host.open, host.instruments, host.instrument);
    let chosen = chosen(known, &start.instruments)?;
    let mut header = about(reach.world);
    engine_profile::begin(start.events)
        .map_err(|reason| Error::new(ErrorCode::InvalidValue, "$.events").got(reason))?;
    let sink = open(&start.name, start.queue).inspect_err(|_| engine_profile::end())?;
    let mut on = Vec::with_capacity(chosen.len());
    let mut instruments = Vec::with_capacity(chosen.len());
    for name in chosen {
        match instrument(reach, name, true) {
            Ok(changes) => {
                on.push(name);
                instruments.push(json!({"name": name, "changes": changes}));
            }
            Err(reason) => instruments.push(json!({"name": name, "unavailable": reason})),
        }
    }
    let links = match links {
        Some(links) => reach.read(|reach| read(reach, &links)),
        None => unavailable("unregistered"),
    };
    header.insert("name".into(), start.name.clone().into());
    header.insert(
        "every".into(),
        match start.every {
            Every::Frame => "frame",
            Every::Mark => "mark",
        }
        .into(),
    );
    header.insert("instruments".into(), Value::Array(instruments));
    header.insert("facts".into(), facts);
    header.insert("links".into(), links);
    header.insert("queue".into(), start.queue.into());
    header.insert("events".into(), start.events.into());
    let mut recording = Recording {
        name: start.name,
        every: start.every,
        names,
        instruments: on,
        switch: instrument,
        sink,
        events: Vec::new(),
        held: Vec::new(),
        written: 0,
        unsent: Dropped::default(),
        lost: Dropped::default(),
    };
    if !recording.sink.line(Value::Object(header).to_string()) {
        switch_off(reach, instrument, &recording.instruments);
        engine_profile::end();
        return Err(Error::new(ErrorCode::QueueFull, "$.queue").got(start.queue));
    }
    let reply = json!({"name": recording.name});
    reach.world.insert_resource(recording);
    Ok(reply)
}

fn stop(reach: &mut ReachMut<'_>) -> Result<Value, Error> {
    let mut recording = reach
        .world
        .remove_resource::<Recording>()
        .ok_or_else(not_recording)?;
    for (_, mut entry) in std::mem::take(&mut recording.held) {
        entry.picture = Some(unavailable("record_stopped"));
        recording.send(entry);
    }
    let refused = switch_off(reach, recording.switch, &recording.instruments);
    let dropped = engine_profile::drain(&mut recording.events);
    engine_profile::end();
    recording.lost.events += dropped + recording.events.len() as u64;
    let mut reply = json!({
        "name": recording.name,
        "entries": recording.written,
        "dropped": {"entries": recording.lost.entries, "events": recording.lost.events},
    });
    if !refused.is_empty() {
        reply["refused"] = Value::Array(refused);
    }
    Ok(reply)
}

fn mark(reach: &mut ReachMut<'_>, args: &Value) -> Result<Answer, Error> {
    let mark: Mark = parse_value_at(args.clone(), "$")?;
    if !file_safe(&mark.name) {
        return Err(Error::new(ErrorCode::InvalidName, "$.name").got(&mark.name));
    }
    let at = reach
        .world
        .get_resource::<Moment>()
        .copied()
        .ok_or_else(|| Error::new(ErrorCode::InvalidRequest, "$").expected(Expected::Running))?;
    let mut recording = reach
        .world
        .remove_resource::<Recording>()
        .ok_or_else(not_recording)?;
    let entry = recording.entry(reach, at, Some(&mark.name));
    let answer = match mark.picture {
        None => {
            recording.send(entry);
            Answer::Value(json!({"name": mark.name, "frame": at.frame}))
        }
        Some(size) => {
            let file = format!("{}-{}.png", at.frame, mark.name);
            recording.held.push((file.clone(), entry));
            Answer::Loop(Box::new(MarkPicture {
                run: recording.name.clone(),
                file,
                size,
            }))
        }
    };
    reach.world.insert_resource(recording);
    Ok(answer)
}

fn read(reach: &Reach<'_>, entry: &NameEntry) -> Value {
    match entry.read_value(reach, &Value::Object(Map::new())) {
        Some(Ok(value)) => value,
        Some(Err(error)) => unavailable(error.code),
        None => unavailable("unreadable"),
    }
}

type Resolved = (Vec<Arc<NameEntry>>, Option<Arc<NameEntry>>);

fn resolve(world: &World, names: &[String]) -> Result<Resolved, Error> {
    let registry = world
        .get_resource::<Registry>()
        .ok_or_else(|| Error::new(ErrorCode::MissingResource, "$.names"))?;
    let links = registry
        .entry("links")
        .filter(|entry| entry.readable())
        .cloned();
    let names = names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let path = format!("$.names[{index}]");
            match registry.entry(name) {
                Some(entry) if entry.readable() => Ok(Arc::clone(entry)),
                Some(_) => Err(Error::new(ErrorCode::InvalidValue, path)
                    .expected(Expected::Readable)
                    .got(name)),
                None => Err(unknown(
                    path,
                    name,
                    registry.table().map(|entry| entry.name.to_string()).collect(),
                )),
            }
        })
        .collect::<Result<_, _>>()?;
    Ok((names, links))
}

fn chosen(known: &'static [&'static str], names: &[String]) -> Result<Vec<&'static str>, Error> {
    names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            known
                .iter()
                .find(|known| **known == name.as_str())
                .copied()
                .ok_or_else(|| {
                    unknown(
                        format!("$.instruments[{index}]"),
                        name,
                        known.iter().map(|known| (*known).to_owned()).collect(),
                    )
                })
        })
        .collect()
}

fn switch_off(reach: &mut ReachMut<'_>, instrument: Switch, on: &[&'static str]) -> Vec<Value> {
    on.iter()
        .filter_map(|name| {
            instrument(reach, name, false)
                .err()
                .map(|reason| json!({"name": name, "unavailable": reason}))
        })
        .collect()
}

fn unknown(path: String, got: &str, known: Vec<String>) -> Error {
    let error = Error::new(ErrorCode::UnknownValue, path).got(got);
    match closest(got, &known) {
        Some(nearest) => error.nearest(nearest),
        None => error,
    }
}

fn not_recording() -> Error {
    Error::new(ErrorCode::MissingResource, "record").expected(Expected::Live)
}

fn file_safe(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && !name.contains("..")
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}
