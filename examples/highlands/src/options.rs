use engine::{Error, ErrorCode};

const SWITCH: &str = "on|off";

pub(crate) const WATCH_BODIES: &str = "watch-bodies";

pub(crate) const CONTROLS: [&str; 1] = [WATCH_BODIES];

#[derive(Clone, Debug)]
pub struct Options {
    pub watch_bodies: bool,
    pub rocks: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            watch_bodies: false,
            rocks: true,
        }
    }
}

impl Options {
    pub fn value(&self, name: &str) -> Option<String> {
        let switch = |on: bool| String::from(if on { "on" } else { "off" });
        match name {
            "rocks" => Some(switch(self.rocks)),
            _ => None,
        }
    }

    pub fn set(&mut self, name: &str, value: &str) -> engine::Result<()> {
        match name {
            WATCH_BODIES => self.watch_bodies = switch(name, value)?,
            "rocks" => self.rocks = switch(name, value)?,
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
    vec![("rocks", SWITCH.to_owned())]
}

pub fn controls() -> Vec<(&'static str, String)> {
    CONTROLS
        .into_iter()
        .map(|name| (name, SWITCH.to_owned()))
        .collect()
}
