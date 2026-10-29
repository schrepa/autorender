use engine::{Error, ErrorCode};

use crate::scene;

const MAX_BODIES: u32 = 64;

const SWITCH: &str = "on|off";

const STAND: &str = "x,z,yaw,pitch";

pub(crate) const WATCH_BODIES: &str = "watch-bodies";

pub(crate) const CONTROLS: [&str; 5] = ["effects", "stand", WATCH_BODIES, "lod", "impostors"];

#[derive(Clone, Debug)]
pub struct Options {
    pub effects: String,
    pub stand: Option<String>,
    pub watch_bodies: bool,
    pub lod: bool,
    pub impostors: bool,
    pub rocks: bool,
    pub bodies: u32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            effects: "all".into(),
            stand: None,
            watch_bodies: false,
            lod: true,
            impostors: true,
            rocks: true,
            bodies: 0,
        }
    }
}

impl Options {
    pub fn value(&self, name: &str) -> Option<String> {
        let switch = |on: bool| String::from(if on { "on" } else { "off" });
        match name {
            "effects" => Some(self.effects.clone()),
            "stand" => self.stand.clone(),
            WATCH_BODIES => Some(switch(self.watch_bodies)),
            "lod" => Some(switch(self.lod)),
            "impostors" => Some(switch(self.impostors)),
            "rocks" => Some(switch(self.rocks)),
            "bodies" => Some(self.bodies.to_string()),
            _ => None,
        }
    }

    pub fn set(&mut self, name: &str, value: &str) -> engine::Result<()> {
        match name {
            "effects" => self.effects = value.to_owned(),
            "stand" => self.stand = Some(value.to_owned()),
            WATCH_BODIES => self.watch_bodies = switch(name, value)?,
            "lod" => self.lod = switch(name, value)?,
            "impostors" => self.impostors = switch(name, value)?,
            "rocks" => self.rocks = switch(name, value)?,
            "bodies" => {
                self.bodies = value
                    .parse::<u32>()
                    .ok()
                    .filter(|bodies| *bodies <= MAX_BODIES)
                    .ok_or_else(|| {
                        Error::new(ErrorCode::OutOfRange, "$.bodies")
                            .range(0.0, f64::from(MAX_BODIES))
                            .got(value)
                    })?;
            }
            _ => {
                let known = parameters().into_iter().map(|(known, _)| known);
                return Err(Error::new(ErrorCode::InvalidArguments, format!("$.{name}"))
                    .got(name)
                    .variants(known));
            }
        }
        Ok(())
    }
}

fn switch(name: &str, value: &str) -> engine::Result<bool> {
    match value {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err(Error::new(ErrorCode::InvalidValue, format!("$.{name}"))
            .got(value)
            .variants(SWITCH.split('|'))),
    }
}

pub fn parameters() -> Vec<(&'static str, String)> {
    vec![
        ("effects", scene::effects()),
        ("stand", STAND.to_owned()),
        (WATCH_BODIES, SWITCH.to_owned()),
        ("lod", SWITCH.to_owned()),
        ("impostors", SWITCH.to_owned()),
        ("rocks", SWITCH.to_owned()),
        ("bodies", format!("0-{MAX_BODIES}")),
    ]
}

pub fn controls() -> Vec<(&'static str, String)> {
    parameters()
        .into_iter()
        .filter(|(name, _)| CONTROLS.contains(name))
        .collect()
}
