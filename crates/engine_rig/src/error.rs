use std::fmt;

use engine_codec::{Error, ErrorCode, Expected};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Fault {
    Truncated {
        item: &'static str,
        needed: usize,
        held: usize,
    },
    Magic {
        item: &'static str,
        got: Vec<u8>,
    },
    TrailingBytes {
        item: &'static str,
        bytes: usize,
    },
    EmptyName {
        item: &'static str,
        index: usize,
    },
    NameNotUtf8 {
        item: &'static str,
        at: usize,
    },
    NameTwice {
        item: &'static str,
        name: String,
    },
    ParentAfterChild {
        joint: String,
        parent: usize,
    },
    InverseBindNotInvertible {
        joint: String,
    },
    KeysCode {
        bone: String,
        code: u32,
    },
    RootFlag {
        code: u32,
    },
    Duration {
        duration: f32,
    },
    NoKeys,
    KeyTimes {
        at: usize,
        time: f32,
    },
    TrackLength {
        bone: String,
        keys: usize,
        values: usize,
    },
    NotFinite {
        bone: String,
        key: usize,
    },
    Seconds {
        seconds: f32,
    },
    Length {
        item: &'static str,
        expected: usize,
        got: usize,
    },
    TrackMissing {
        track: usize,
        tracks: usize,
    },
}

impl fmt::Display for Fault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { item, needed, held } => write!(
                formatter,
                "Truncated: {item} needs {needed} bytes, holds {held}"
            ),
            Self::Magic { item, got } => {
                write!(formatter, "Magic: {item} starts with {got:?}")
            }
            Self::TrailingBytes { item, bytes } => {
                write!(
                    formatter,
                    "TrailingBytes: {item} has {bytes} bytes after its end"
                )
            }
            Self::EmptyName { item, index } => {
                write!(formatter, "EmptyName: {item} {index} has an empty name")
            }
            Self::NameNotUtf8 { item, at } => {
                write!(formatter, "NameNotUtf8: {item} name at byte {at}")
            }
            Self::NameTwice { item, name } => {
                write!(formatter, "NameTwice: {item} `{name}` appears twice")
            }
            Self::ParentAfterChild { joint, parent } => write!(
                formatter,
                "ParentAfterChild: joint `{joint}` names parent {parent}, not an earlier joint"
            ),
            Self::InverseBindNotInvertible { joint } => write!(
                formatter,
                "InverseBindNotInvertible: joint `{joint}` has a non-finite or singular inverse bind"
            ),
            Self::KeysCode { bone, code } => {
                write!(formatter, "KeysCode: track `{bone}` has key code {code}")
            }
            Self::RootFlag { code } => write!(formatter, "RootFlag: root track flag {code}"),
            Self::Duration { duration } => {
                write!(formatter, "Duration: clip duration {duration}")
            }
            Self::NoKeys => write!(formatter, "NoKeys: clip has no key times"),
            Self::KeyTimes { at, time } => write!(
                formatter,
                "KeyTimes: key {at} at {time} s is not finite, increasing and within the duration"
            ),
            Self::TrackLength { bone, keys, values } => write!(
                formatter,
                "TrackLength: track `{bone}` has {values} values for {keys} keys"
            ),
            Self::NotFinite { bone, key } => {
                write!(formatter, "NotFinite: track `{bone}` key {key}")
            }
            Self::Seconds { seconds } => write!(formatter, "Seconds: sample time {seconds}"),
            Self::Length {
                item,
                expected,
                got,
            } => write!(formatter, "Length: {item} holds {got}, expected {expected}"),
            Self::TrackMissing { track, tracks } => write!(
                formatter,
                "TrackMissing: bound track {track} but the clip has {tracks}; sample with the clip given to bind"
            ),
        }
    }
}

impl From<Fault> for Error {
    fn from(fault: Fault) -> Self {
        let (code, path, expected) = match &fault {
            Fault::Truncated { item, .. }
            | Fault::Magic { item, .. }
            | Fault::TrailingBytes { item, .. }
            | Fault::EmptyName { item, .. }
            | Fault::NameNotUtf8 { item, .. }
            | Fault::NameTwice { item, .. } => {
                (ErrorCode::InvalidRecord, format!("$.{item}"), None)
            }
            Fault::ParentAfterChild { .. } => (
                ErrorCode::InvalidRecord,
                "$.skeleton.parent".to_owned(),
                Some(Expected::Sorted),
            ),
            Fault::InverseBindNotInvertible { .. } => (
                ErrorCode::InvalidRecord,
                "$.skeleton.inverse_bind".to_owned(),
                Some(Expected::Invertible),
            ),
            Fault::KeysCode { .. } | Fault::RootFlag { .. } | Fault::TrackLength { .. } => {
                (ErrorCode::InvalidRecord, "$.clip.tracks".to_owned(), None)
            }
            Fault::NotFinite { .. } => (
                ErrorCode::InvalidRecord,
                "$.clip.tracks".to_owned(),
                Some(Expected::Finite),
            ),
            Fault::Duration { .. } => (
                ErrorCode::InvalidRecord,
                "$.clip.duration".to_owned(),
                Some(Expected::Finite),
            ),
            Fault::NoKeys => (
                ErrorCode::InvalidRecord,
                "$.clip.keys".to_owned(),
                Some(Expected::NonEmpty),
            ),
            Fault::KeyTimes { .. } => (
                ErrorCode::InvalidRecord,
                "$.clip.keys".to_owned(),
                Some(Expected::Sorted),
            ),
            Fault::Seconds { .. } => (
                ErrorCode::OutOfRange,
                "$.seconds".to_owned(),
                Some(Expected::Finite),
            ),
            Fault::Length { item, .. } => (
                ErrorCode::InvalidInput,
                format!("$.{item}"),
                Some(Expected::Matching),
            ),
            Fault::TrackMissing { .. } => (
                ErrorCode::InvalidInput,
                "$.bound".to_owned(),
                Some(Expected::Matching),
            ),
        };
        let error = Error::new(code, path).got(fault.to_string());
        match expected {
            Some(expected) => error.expected(expected),
            None => error,
        }
    }
}
