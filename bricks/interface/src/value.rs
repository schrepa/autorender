#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Val<'a> {
    Unknown,
    Missing,
    Bool(bool),
    Num(f32),
    Text(&'a str),
}

impl<'a> Val<'a> {
    pub(super) fn truthy(self) -> bool {
        match self {
            Val::Unknown | Val::Missing => false,
            Val::Bool(value) => value,
            Val::Num(value) => value != 0.0,
            Val::Text(text) => !text.is_empty(),
        }
    }

    pub(super) fn number(self) -> Option<f32> {
        match self {
            Val::Num(value) => Some(value),
            Val::Bool(value) => Some(f32::from(u8::from(value))),
            Val::Unknown | Val::Missing | Val::Text(_) => None,
        }
    }

    pub(super) fn text(self) -> Option<&'a str> {
        match self {
            Val::Text(text) => Some(text),
            Val::Unknown | Val::Missing | Val::Bool(_) | Val::Num(_) => None,
        }
    }
}
