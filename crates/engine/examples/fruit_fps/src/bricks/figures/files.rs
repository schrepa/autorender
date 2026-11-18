use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::{Error, ErrorCode, Expected};
use engine_rig::{Bound, Clip, Posed, Skeleton};

use crate::names::HEAD;
use crate::play::figures::{Figure, FigureBytes, HIP, LEFT_ARM, LEFT_LEG, RIGHT_ARM, RIGHT_LEG};

pub(super) fn unreadable(path: String, error: &Error) -> Error {
    Error::new(ErrorCode::InvalidRecord, path).got(error.to_string())
}

pub(super) struct Played {
    pub(super) clip: Arc<Clip>,
    pub(super) bound: Arc<Bound>,
    pub(super) stride_mps: f32,
}

impl Played {
    pub(super) fn duration(&self) -> f32 {
        self.clip.duration()
    }
}

pub(super) struct Limbs {
    pub(super) hip: Option<usize>,
    pub(super) right_arm: Option<[usize; 3]>,
    pub(super) left_arm: Option<[usize; 3]>,
    pub(super) legs: Option<[[usize; 3]; 2]>,
}

fn chain(skeleton: &Skeleton, names: [&str; 3]) -> Option<[usize; 3]> {
    let [upper, middle, end] = names.map(|name| skeleton.find(name));
    Some([upper?, middle?, end?])
}

impl Limbs {
    fn of(skeleton: &Skeleton) -> Self {
        Self {
            hip: skeleton.find(HIP),
            right_arm: chain(skeleton, RIGHT_ARM),
            left_arm: chain(skeleton, LEFT_ARM),
            legs: chain(skeleton, LEFT_LEG)
                .zip(chain(skeleton, RIGHT_LEG))
                .map(|(left, right)| [left, right]),
        }
    }
}

pub(super) struct Rig {
    pub(super) skeleton_uri: String,
    pub(super) skeleton: Arc<Skeleton>,
    pub(super) clips: BTreeMap<String, Played>,
    pub(super) head: Option<usize>,
    pub(super) limbs: Limbs,
}

impl Rig {
    pub(super) fn duration(&self, clip: &str) -> f32 {
        self.clips.get(clip).map_or(0.0, Played::duration)
    }
}

const STRIDE_SAMPLES: usize = 48;

fn stride(skeleton: &Skeleton, clip: &Clip, bound: &Bound, limbs: &Limbs) -> f32 {
    let (Some(hip), Some(legs)) = (limbs.hip, limbs.legs) else {
        return 0.0;
    };
    let duration = clip.duration();
    if duration <= 0.0 {
        return 0.0;
    }
    let mut posed: Vec<Posed> = (0..skeleton.len())
        .map(|joint| skeleton.rest(joint))
        .collect();
    let mut before: Option<[Vec3; 2]> = None;
    let mut travel = 0.0;
    for sample in 0..=STRIDE_SAMPLES {
        let seconds = duration * sample as f32 / STRIDE_SAMPLES as f32;
        if bound.sample(clip, seconds, skeleton, &mut posed).is_err() {
            return 0.0;
        }
        let feet = legs.map(|leg| posed[leg[2]].position - posed[hip].position);
        if let Some(was) = before {
            let planted = usize::from(posed[legs[1][2]].position.y < posed[legs[0][2]].position.y);
            let back = was[planted].z - feet[planted].z;
            if back > 0.0 {
                travel += back;
            }
        }
        before = Some(feet);
    }
    travel / duration
}

type MaskKey = (String, String, u32);

type Masked = (Arc<Skeleton>, Result<Arc<[f32]>, Error>);

type Binding = (Arc<Skeleton>, Arc<Clip>, Arc<Bound>, f32);

type Parsed<T> = (Arc<[u8]>, Result<Arc<T>, Error>);

#[derive(Resource, Default)]
pub struct FigureFiles {
    skeletons: BTreeMap<String, Parsed<Skeleton>>,
    clips: BTreeMap<String, Parsed<Clip>>,
    bound: BTreeMap<(String, String), Binding>,
    masks: BTreeMap<MaskKey, Masked>,
}

fn missing(uri: &str, path: &str) -> Error {
    Error::new(ErrorCode::MissingFile, path)
        .expected(Expected::File)
        .got(uri)
}

fn parsed<T>(held: &BTreeMap<String, Parsed<T>>) -> u32 {
    held.values().filter(|(_, found)| found.is_ok()).count() as u32
}

fn read<T>(
    held: &mut BTreeMap<String, Parsed<T>>,
    (bytes, uri, path): (&FigureBytes, &str, &str),
    parse: impl FnOnce(&[u8]) -> Result<T, Error>,
) -> Result<Arc<T>, Error> {
    let bytes = bytes.get(uri).ok_or_else(|| missing(uri, path))?;
    if let Some((known, found)) = held.get(uri)
        && Arc::ptr_eq(known, bytes)
    {
        return found.clone();
    }
    let found = parse(bytes)
        .map(Arc::new)
        .map_err(|error| unreadable(path.to_owned(), &error));
    held.insert(uri.to_owned(), (Arc::clone(bytes), found.clone()));
    found
}

impl FigureFiles {
    pub(super) fn counts(&self) -> (u32, u32) {
        (parsed(&self.skeletons), parsed(&self.clips))
    }

    pub(super) fn rig(&mut self, figure: &Figure) -> Result<Rig, Error> {
        let skeleton = read(
            &mut self.skeletons,
            (&figure.bytes, &figure.skeleton, "$.figure.skeleton"),
            Skeleton::from_bytes,
        )?;
        let limbs = Limbs::of(&skeleton);
        let mut clips = BTreeMap::new();
        for (name, uri) in &figure.clips {
            let path = format!("$.figure.clips.{name}");
            let clip = read(
                &mut self.clips,
                (&figure.bytes, uri, &path),
                Clip::from_bytes,
            )?;
            let key = (figure.skeleton.clone(), uri.clone());
            let (bound, stride_mps) = match self.bound.get(&key) {
                Some((bones, motion, bound, stride))
                    if Arc::ptr_eq(bones, &skeleton) && Arc::ptr_eq(motion, &clip) =>
                {
                    (Arc::clone(bound), *stride)
                }
                Some(_) | None => {
                    let bound = Arc::new(engine_rig::bind(&skeleton, &clip));
                    let stride = stride(&skeleton, &clip, &bound, &limbs);
                    self.bound.insert(
                        key,
                        (
                            Arc::clone(&skeleton),
                            Arc::clone(&clip),
                            Arc::clone(&bound),
                            stride,
                        ),
                    );
                    (bound, stride)
                }
            };
            clips.insert(
                name.clone(),
                Played {
                    clip,
                    bound,
                    stride_mps,
                },
            );
        }
        Ok(Rig {
            skeleton_uri: figure.skeleton.clone(),
            head: skeleton.find(HEAD),
            limbs,
            skeleton,
            clips,
        })
    }

    pub(super) fn mask(
        &mut self,
        rig: &Rig,
        from: &str,
        feather: u32,
    ) -> Result<Arc<[f32]>, Error> {
        let key = (rig.skeleton_uri.clone(), from.to_owned(), feather);
        if let Some((bones, found)) = self.masks.get(&key)
            && Arc::ptr_eq(bones, &rig.skeleton)
        {
            return found.clone();
        }
        let found = rig
            .skeleton
            .mask(from, feather as usize)
            .map(Arc::from)
            .map_err(|error| unreadable("$.figure_play.upper.from".to_owned(), &error));
        self.masks
            .insert(key, (Arc::clone(&rig.skeleton), found.clone()));
        found
    }
}
