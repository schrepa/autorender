use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::math::max;
use engine_core::{AssetLibrary, Error, ErrorCode, Expected, GeometryHandle, Mesh};
use engine_rig::{Bound, Clip, Posed, Skeleton};

use super::JOINTS;
use crate::names::HEAD;
use crate::play::figures::{
    Figure, FigureBytes, HIP, LEFT_ARM, LEFT_EYE, LEFT_LEG, RIGHT_ARM, RIGHT_EYE, RIGHT_LEG,
};
use crate::play::puppet::HeadCheck;

pub(super) fn unreadable(path: String, error: &Error) -> Error {
    Error::new(ErrorCode::InvalidRecord, path).got(error.to_string())
}

pub(super) struct Played {
    pub(super) clip: Arc<Clip>,
    pub(super) bound: Arc<Bound>,
    pub(super) motion: Motion,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Motion {
    pub(super) stride_mps: f32,
    pub(super) drift_m: f32,
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

fn flat(value: Vec3) -> Vec3 {
    Vec3::new(value.x, 0.0, value.z)
}

fn stride(skeleton: &Skeleton, clip: &Clip, bound: &Bound, limbs: &Limbs) -> Motion {
    let still = Motion::default();
    let (Some(hip), Some(legs)) = (limbs.hip, limbs.legs) else {
        return still;
    };
    let duration = clip.duration();
    if duration <= 0.0 {
        return still;
    }
    let mut posed: Vec<Posed> = (0..skeleton.len())
        .map(|joint| skeleton.rest(joint))
        .collect();
    let mut before: Option<[Vec3; 2]> = None;
    let mut hips = [Vec3::ZERO; 2];
    let mut moves = Vec::with_capacity(STRIDE_SAMPLES);
    for sample in 0..=STRIDE_SAMPLES {
        let seconds = duration * sample as f32 / STRIDE_SAMPLES as f32;
        if bound.sample(clip, seconds, skeleton, &mut posed).is_err() {
            return still;
        }
        let feet = legs.map(|leg| posed[leg[2]].position - posed[hip].position);
        if let Some(was) = before {
            let planted = usize::from(posed[legs[1][2]].position.y < posed[legs[0][2]].position.y);
            moves.push(flat(was[planted] - feet[planted]));
        }
        before = Some(feet);
        match sample {
            0 => hips[0] = posed[hip].position,
            STRIDE_SAMPLES => hips[1] = posed[hip].position,
            _ => {}
        }
    }
    let way = moves
        .iter()
        .fold(Vec3::ZERO, |sum, moved| sum + *moved)
        .normalize_or_zero();
    let travel: f32 = moves.iter().map(|moved| max(moved.dot(way), 0.0)).sum();
    Motion {
        stride_mps: travel / duration,
        drift_m: flat(hips[1] - hips[0]).length(),
    }
}

type MaskKey = (String, String, u32);

type Masked = (Arc<Skeleton>, Result<Arc<[f32]>, Error>);

type Binding = (Arc<Skeleton>, Arc<Clip>, Arc<Bound>, Motion);

type Parsed<T> = (Arc<[u8]>, Result<Arc<T>, Error>);

#[derive(Resource, Default)]
pub struct FigureFiles {
    skeletons: BTreeMap<String, Parsed<Skeleton>>,
    clips: BTreeMap<String, Parsed<Clip>>,
    bound: BTreeMap<(String, String), Binding>,
    masks: BTreeMap<MaskKey, Masked>,
    pub(super) crowns: BTreeMap<GeometryHandle, Crown>,
    eyes: BTreeMap<(String, String), Option<f32>>,
    fits: BTreeMap<(String, GeometryHandle), Result<(), Error>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Crown {
    pub(super) top: f32,
    pub(super) crown: f32,
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

fn crowned(rig: &Rig, mesh: &Mesh, check: HeadCheck) -> Option<Crown> {
    let head = rig.head?;
    let bound = rig
        .skeleton
        .joints()
        .get(head)?
        .inverse_bind
        .inverse()
        .w_axis;
    let above: Vec<[f32; 3]> = mesh
        .vertices()
        .iter()
        .map(|vertex| vertex.position)
        .filter(|at| at[1] > bound.y)
        .collect();
    let top = above.iter().map(|at| at[1]).reduce(max)?;
    let slice = max(check.slice_m, 1.0e-3);
    let count = ((top - bound.y) / slice) as usize + 1;
    let mut widths = vec![0.0_f32; count];
    for at in &above {
        let band = (((top - at[1]) / slice) as usize).min(count - 1);
        let reach = Vec3::new(at[0] - bound.x, 0.0, at[2] - bound.z).length();
        widths[band] = max(widths[band], reach);
    }
    let widest = widths.iter().copied().reduce(max)?;
    let band = widths
        .iter()
        .position(|width| *width >= widest * check.solid_share)?;
    Some(Crown {
        top,
        crown: top - band as f32 * slice,
    })
}

impl FigureFiles {
    pub(super) fn crown(
        &mut self,
        rig: &Rig,
        (library, mesh): (&AssetLibrary, GeometryHandle),
        check: HeadCheck,
    ) -> Option<Crown> {
        if let Some(known) = self.crowns.get(&mesh) {
            return Some(*known);
        }
        let found = crowned(rig, library.mesh(mesh)?, check)?;
        self.crowns.insert(mesh, found);
        Some(found)
    }

    pub(super) fn rest_eye(&mut self, rig: &Rig, reference: &str) -> Option<f32> {
        let key = (rig.skeleton_uri.clone(), reference.to_owned());
        if let Some(known) = self.eyes.get(&key) {
            return *known;
        }
        let found = rig.clips.get(reference).and_then(|played| {
            let skeleton = &*rig.skeleton;
            let mut pose: Vec<Posed> = (0..skeleton.len()).map(|at| skeleton.rest(at)).collect();
            played
                .bound
                .sample(&played.clip, 0.0, skeleton, &mut pose)
                .ok()?;
            let left = pose.get(skeleton.find(LEFT_EYE)?)?.position;
            let right = pose.get(skeleton.find(RIGHT_EYE)?)?.position;
            Some((left.y + right.y) * 0.5)
        });
        self.eyes.insert(key, found);
        found
    }

    pub(super) fn form_fits(
        &mut self,
        rig: &Rig,
        (library, mesh): (&AssetLibrary, GeometryHandle),
        path: &str,
    ) -> Result<(), Error> {
        let key = (rig.skeleton_uri.clone(), mesh);
        if let Some(known) = self.fits.get(&key) {
            return known.clone();
        }
        let joints = rig.skeleton.len() as u32;
        let found = match library.mesh(mesh) {
            None => Err(Error::new(ErrorCode::MissingFile, path.to_owned())),
            Some(form) => {
                let highest = form
                    .stream(JOINTS)
                    .map(|stream| stream.words().map(u32::from_le_bytes).max().unwrap_or(0));
                match highest {
                    Some(top) if top < joints => Ok(()),
                    Some(top) => Err(Error::new(ErrorCode::OutOfRange, path.to_owned())
                        .at_most(u64::from(joints.saturating_sub(1)))
                        .got(top)),
                    None => Err(Error::new(ErrorCode::InvalidRenderable, path.to_owned())
                        .got("no joints stream")),
                }
            }
        };
        self.fits.insert(key, found.clone());
        found
    }

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
            let (bound, motion) = match self.bound.get(&key) {
                Some((bones, held, bound, motion))
                    if Arc::ptr_eq(bones, &skeleton) && Arc::ptr_eq(held, &clip) =>
                {
                    (Arc::clone(bound), *motion)
                }
                Some(_) | None => {
                    let bound = Arc::new(engine_rig::bind(&skeleton, &clip));
                    let motion = stride(&skeleton, &clip, &bound, &limbs);
                    self.bound.insert(
                        key,
                        (
                            Arc::clone(&skeleton),
                            Arc::clone(&clip),
                            Arc::clone(&bound),
                            motion,
                        ),
                    );
                    (bound, motion)
                }
            };
            clips.insert(
                name.clone(),
                Played {
                    clip,
                    bound,
                    motion,
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
        path: &str,
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
            .map_err(|error| unreadable(path.to_owned(), &error));
        self.masks
            .insert(key, (Arc::clone(&rig.skeleton), found.clone()));
        found
    }
}
