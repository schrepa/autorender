use engine::{Error, ErrorCode};

use crate::{scene, views};

const MAX_BODIES: u32 = 64;

const SWITCH: &str = "on|off";

pub(crate) const CONTROLS: [&str; 5] = ["look", "effects", "view", "lod", "impostors"];

#[derive(Clone, Debug)]
pub struct Options {
    pub look: String,
    pub effects: String,
    pub view: Option<String>,
    pub lod: bool,
    pub impostors: bool,
    pub rocks: bool,
    pub bodies: u32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            look: "scene".into(),
            effects: "all".into(),
            view: None,
            lod: true,
            impostors: true,
            rocks: true,
            bodies: 0,
        }
    }
}

impl Options {
    pub fn set(&mut self, name: &str, value: &str) -> engine::Result<()> {
        match name {
            "look" => self.look = value.to_owned(),
            "effects" => self.effects = value.to_owned(),
            "view" => self.view = Some(value.to_owned()),
            "lod" => self.lod = switch(name, value)?,
            "impostors" => self.impostors = switch(name, value)?,
            "rocks" => self.rocks = switch(name, value)?,
            "bodies" => {
                self.bodies = value
                    .parse::<u32>()
                    .ok()
                    .filter(|bodies| *bodies <= MAX_BODIES)
                    .ok_or_else(|| {
                        Error::new(ErrorCode::OutOfRange, "$.bodies", format!("0-{MAX_BODIES}"))
                    })?;
            }
            _ => {
                let known: Vec<&str> = parameters().into_iter().map(|(known, _)| known).collect();
                return Err(Error::new(
                    ErrorCode::InvalidArguments,
                    format!("$.{name}"),
                    known.join("|"),
                ));
            }
        }
        Ok(())
    }
}

fn switch(name: &str, value: &str) -> engine::Result<bool> {
    match value {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err(Error::new(
            ErrorCode::InvalidValue,
            format!("$.{name}"),
            SWITCH,
        )),
    }
}

pub fn parameters() -> Vec<(&'static str, String)> {
    vec![
        ("look", scene::looks()),
        ("effects", scene::effects()),
        ("view", views::names()),
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
        .map(|(name, values)| (name, format!("{{name}}: {values}")))
        .collect()
}
