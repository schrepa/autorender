use glam::{Quat, Vec3};

use engine_codec::Error;

use crate::bytes::{Reader, put_count, put_floats, put_name, put_u32};
use crate::error::Fault;

const MAGIC: [u8; 4] = *b"CLIP";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keys {
    Linear,
    Step,
}

impl Keys {
    const fn code(self) -> u32 {
        match self {
            Self::Linear => 0,
            Self::Step => 1,
        }
    }

    fn of_code(code: u32, bone: &str) -> Result<Self, Error> {
        match code {
            0 => Ok(Self::Linear),
            1 => Ok(Self::Step),
            code => Err(Error::from(Fault::KeysCode {
                bone: bone.to_owned(),
                code,
            })),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RotationTrack {
    pub bone: String,
    pub keys: Keys,
    pub values: Vec<Quat>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TranslationTrack {
    pub bone: String,
    pub keys: Keys,
    pub values: Vec<Vec3>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RootTrack {
    pub bone: String,
    pub keys: Keys,
    pub values: Vec<Vec3>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    duration: f32,
    times: Vec<f32>,
    rotations: Vec<RotationTrack>,
    translations: Vec<TranslationTrack>,
    root: Option<RootTrack>,
}

fn track_shape(
    bone: &str,
    index: usize,
    keys: usize,
    values: usize,
    finite: impl Iterator<Item = bool>,
) -> Result<(), Error> {
    if bone.is_empty() {
        return Err(Error::from(Fault::EmptyName {
            item: "track",
            index,
        }));
    }
    if values != keys {
        return Err(Error::from(Fault::TrackLength {
            bone: bone.to_owned(),
            keys,
            values,
        }));
    }
    let mut finite = finite;
    match finite.position(|held| !held) {
        Some(key) => Err(Error::from(Fault::NotFinite {
            bone: bone.to_owned(),
            key,
        })),
        None => Ok(()),
    }
}

fn unique<'a>(item: &'static str, bones: impl Iterator<Item = &'a str>) -> Result<(), Error> {
    let bones: Vec<&str> = bones.collect();
    for (index, bone) in bones.iter().enumerate() {
        if bones[..index].contains(bone) {
            return Err(Error::from(Fault::NameTwice {
                item,
                name: (*bone).to_owned(),
            }));
        }
    }
    Ok(())
}

fn put_vectors(bytes: &mut Vec<u8>, bone: &str, keys: Keys, values: &[Vec3]) {
    put_name(bytes, bone);
    put_u32(bytes, keys.code());
    for value in values {
        put_floats(bytes, &value.to_array());
    }
}

fn read_vectors(
    reader: &mut Reader<'_>,
    keys: usize,
    held: usize,
) -> Result<(String, Keys, Vec<Vec3>), Error> {
    let bone = reader.name()?;
    let kind = Keys::of_code(reader.u32()?, &bone)?;
    let mut values = Vec::with_capacity(keys.min(held));
    for _ in 0..keys {
        values.push(Vec3::from_array(reader.floats::<3>()?));
    }
    Ok((bone, kind, values))
}

impl Clip {
    pub fn new(
        duration: f32,
        times: Vec<f32>,
        rotations: Vec<RotationTrack>,
        translations: Vec<TranslationTrack>,
        root: Option<RootTrack>,
    ) -> Result<Self, Error> {
        if !duration.is_finite() || duration < 0.0 {
            return Err(Error::from(Fault::Duration { duration }));
        }
        if times.is_empty() {
            return Err(Error::from(Fault::NoKeys));
        }
        for (at, &time) in times.iter().enumerate() {
            let after_previous = at == 0 || time > times[at - 1];
            if !time.is_finite() || time < 0.0 || time > duration || !after_previous {
                return Err(Error::from(Fault::KeyTimes { at, time }));
            }
        }
        unique(
            "rotation track",
            rotations.iter().map(|track| track.bone.as_str()),
        )?;
        unique(
            "translation track",
            translations.iter().map(|track| track.bone.as_str()),
        )?;
        for (index, track) in rotations.iter().enumerate() {
            track_shape(
                &track.bone,
                index,
                times.len(),
                track.values.len(),
                track.values.iter().map(|value| value.is_finite()),
            )?;
        }
        let vectors = translations
            .iter()
            .map(|track| (&track.bone, &track.values))
            .chain(root.iter().map(|track| (&track.bone, &track.values)));
        for (index, (bone, values)) in vectors.enumerate() {
            track_shape(
                bone,
                rotations.len() + index,
                times.len(),
                values.len(),
                values.iter().map(|value| value.is_finite()),
            )?;
        }
        Ok(Self {
            duration,
            times,
            rotations,
            translations,
            root,
        })
    }

    pub fn duration(&self) -> f32 {
        self.duration
    }

    pub fn times(&self) -> &[f32] {
        &self.times
    }

    pub fn rotations(&self) -> &[RotationTrack] {
        &self.rotations
    }

    pub fn translations(&self) -> &[TranslationTrack] {
        &self.translations
    }

    pub fn root(&self) -> Option<&RootTrack> {
        self.root.as_ref()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let keys = self.times.len();
        let tracks = 4 * self.rotations.len() + 3 * (self.translations.len() + 1);
        let mut bytes = Vec::with_capacity(16 + keys * 4 * (1 + tracks));
        bytes.extend_from_slice(&MAGIC);
        put_floats(&mut bytes, &[self.duration]);
        put_count(&mut bytes, keys);
        put_floats(&mut bytes, &self.times);
        put_count(&mut bytes, self.rotations.len());
        for track in &self.rotations {
            put_name(&mut bytes, &track.bone);
            put_u32(&mut bytes, track.keys.code());
            for value in &track.values {
                put_floats(&mut bytes, &value.to_array());
            }
        }
        put_count(&mut bytes, self.translations.len());
        for track in &self.translations {
            put_vectors(&mut bytes, &track.bone, track.keys, &track.values);
        }
        match &self.root {
            None => put_u32(&mut bytes, 0),
            Some(track) => {
                put_u32(&mut bytes, 1);
                put_vectors(&mut bytes, &track.bone, track.keys, &track.values);
            }
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut reader = Reader::new(bytes, "clip", MAGIC)?;
        let duration = reader.f32()?;
        let keys = reader.count()?;
        let held = bytes.len() / 4;
        let mut times = Vec::with_capacity(keys.min(held));
        for _ in 0..keys {
            times.push(reader.f32()?);
        }
        let tracks = reader.count()?;
        let mut rotations = Vec::with_capacity(tracks.min(held));
        for _ in 0..tracks {
            let bone = reader.name()?;
            let kind = Keys::of_code(reader.u32()?, &bone)?;
            let mut values = Vec::with_capacity(keys.min(held));
            for _ in 0..keys {
                values.push(Quat::from_array(reader.floats::<4>()?));
            }
            rotations.push(RotationTrack {
                bone,
                keys: kind,
                values,
            });
        }
        let moved = reader.count()?;
        let mut translations = Vec::with_capacity(moved.min(held));
        for _ in 0..moved {
            let (bone, keys, values) = read_vectors(&mut reader, keys, held)?;
            translations.push(TranslationTrack { bone, keys, values });
        }
        let root = match reader.u32()? {
            0 => None,
            1 => {
                let (bone, keys, values) = read_vectors(&mut reader, keys, held)?;
                Some(RootTrack { bone, keys, values })
            }
            code => return Err(Error::from(Fault::RootFlag { code })),
        };
        reader.finish()?;
        Self::new(duration, times, rotations, translations, root)
    }
}
