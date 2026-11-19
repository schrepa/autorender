use glam::{Mat4, Quat, Vec3};

use engine_codec::Error;

use crate::error::Fault;
use crate::{Clip, Keys, Posed, Skeleton};

pub type PaletteRow = [f32; 12];

#[derive(Clone, Debug, PartialEq)]
pub struct Bound {
    tracks: Vec<Option<usize>>,
    moves: Vec<Option<usize>>,
    root: Option<(usize, f32)>,
    skipped: usize,
}

pub fn bind(skeleton: &Skeleton, clip: &Clip) -> Bound {
    let mut tracks = vec![None; skeleton.len()];
    let mut skipped = 0;
    for (index, track) in clip.rotations().iter().enumerate() {
        match skeleton.find(&track.bone) {
            Some(joint) => tracks[joint] = Some(index),
            None => skipped += 1,
        }
    }
    let mut moves = vec![None; skeleton.len()];
    for (index, track) in clip.translations().iter().enumerate() {
        match skeleton.find(&track.bone) {
            Some(joint) => moves[joint] = Some(index),
            None => skipped += 1,
        }
    }
    let root = clip.root().and_then(|track| {
        let joint = skeleton.find(&track.bone);
        if joint.is_none() {
            skipped += 1;
        }
        joint.map(|joint| (joint, skeleton.rest(joint).position.y))
    });
    Bound {
        tracks,
        moves,
        root,
        skipped,
    }
}

struct Key {
    first: usize,
    second: usize,
    fraction: f32,
}

fn key(times: &[f32], seconds: f32) -> Key {
    let first = times
        .partition_point(|&time| time <= seconds)
        .saturating_sub(1);
    let second = first + 1;
    match times.get(second) {
        Some(&next) if seconds > times[first] => Key {
            first,
            second,
            fraction: (seconds - times[first]) / (next - times[first]),
        },
        Some(_) | None => Key {
            first,
            second: first,
            fraction: 0.0,
        },
    }
}

pub(crate) fn nlerp(from: Quat, to: Quat, fraction: f32) -> Quat {
    let to = if from.dot(to) < 0.0 { -to } else { to };
    (from + (to - from) * fraction).normalize()
}

fn vector(keys: Keys, values: &[Vec3], at: &Key) -> Vec3 {
    let (from, to) = (values[at.first], values[at.second]);
    match keys {
        Keys::Linear => from + (to - from) * at.fraction,
        Keys::Step => from,
    }
}

impl Bound {
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    pub fn root_joint(&self) -> Option<usize> {
        self.root.map(|(joint, _)| joint)
    }

    fn rotation(&self, clip: &Clip, track: usize, at: &Key) -> Result<Quat, Error> {
        let held = clip
            .rotations()
            .get(track)
            .ok_or(Error::from(Fault::TrackMissing {
                track,
                tracks: clip.rotations().len(),
            }))?;
        let (from, to) = (held.values[at.first], held.values[at.second]);
        Ok(match held.keys {
            Keys::Linear => nlerp(from, to, at.fraction),
            Keys::Step => from,
        })
    }

    fn hip(&self, clip: &Clip, at: &Key) -> Result<Vec3, Error> {
        let held = clip.root().ok_or(Error::from(Fault::TrackMissing {
            track: clip.rotations().len(),
            tracks: clip.rotations().len(),
        }))?;
        Ok(vector(held.keys, &held.values, at))
    }

    fn offset(&self, clip: &Clip, track: usize, at: &Key) -> Result<Vec3, Error> {
        let held = clip
            .translations()
            .get(track)
            .ok_or(Error::from(Fault::TrackMissing {
                track,
                tracks: clip.translations().len(),
            }))?;
        Ok(vector(held.keys, &held.values, at))
    }

    pub fn sample(
        &self,
        clip: &Clip,
        seconds: f32,
        skeleton: &Skeleton,
        out: &mut [Posed],
    ) -> Result<(), Error> {
        for (item, got) in [
            ("posed joints", out.len()),
            ("bound joints", self.tracks.len()),
        ] {
            if got != skeleton.len() {
                return Err(Error::from(Fault::Length {
                    item,
                    expected: skeleton.len(),
                    got,
                }));
            }
        }
        if !seconds.is_finite() {
            return Err(Error::from(Fault::Seconds { seconds }));
        }
        let held = if seconds < 0.0 {
            0.0
        } else if seconds > clip.duration() {
            clip.duration()
        } else {
            seconds
        };
        let at = key(clip.times(), held);
        for joint in 0..skeleton.len() {
            let rest = skeleton.rest(joint);
            let parent = skeleton.joints()[joint].parent;
            let rotation = match (self.tracks[joint], parent) {
                (Some(track), _) => self.rotation(clip, track, &at)? * rest.rotation,
                (None, Some(parent)) => {
                    out[parent].rotation * skeleton.inverse_rest(parent) * rest.rotation
                }
                (None, None) => rest.rotation,
            }
            .normalize();
            let moved = match self.moves[joint] {
                Some(track) => self.offset(clip, track, &at)?,
                None => Vec3::ZERO,
            };
            let position = match (self.root, parent) {
                (Some((hip, height)), _) if hip == joint => {
                    rest.position + self.hip(clip, &at)? * height
                }
                (_, Some(parent)) => {
                    out[parent].position + out[parent].rotation * (skeleton.offset(joint) + moved)
                }
                (_, None) => rest.position + moved,
            };
            out[joint] = Posed { rotation, position };
        }
        Ok(())
    }
}

pub fn model_matrices(posed: &[Posed], out: &mut [Mat4]) -> Result<(), Error> {
    if out.len() != posed.len() {
        return Err(Error::from(Fault::Length {
            item: "model matrices",
            expected: posed.len(),
            got: out.len(),
        }));
    }
    for (matrix, joint) in out.iter_mut().zip(posed) {
        *matrix = Mat4::from_rotation_translation(joint.rotation, joint.position);
    }
    Ok(())
}

pub fn palette(skeleton: &Skeleton, model: &[Mat4], out: &mut [PaletteRow]) -> Result<(), Error> {
    for (item, got) in [("model matrices", model.len()), ("palette rows", out.len())] {
        if got != skeleton.len() {
            return Err(Error::from(Fault::Length {
                item,
                expected: skeleton.len(),
                got,
            }));
        }
    }
    for ((row, matrix), joint) in out.iter_mut().zip(model).zip(skeleton.joints()) {
        let skin = *matrix * joint.inverse_bind;
        for (index, lane) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *lane = skin.row(index).to_array();
        }
    }
    Ok(())
}
