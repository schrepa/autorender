use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::math::{max, min};
use engine_core::{Asset, AssetId, Assets, Error, ErrorCode, MeshFile};
use engine_meshpack::POSITION;
use engine_rig::{Bound, Clip, Posed, Skeleton};

use super::pose::Tree;
use crate::play::skinned::{Figure, FigureFit, FigureLimbs};

pub(super) fn unreadable(path: String, error: &Error) -> Error {
    Error::new(ErrorCode::InvalidRecord, path).got(error.to_string())
}

pub(super) fn skeleton(bytes: Arc<[u8]>, uri: &str) -> Result<Arc<Skeleton>, Error> {
    Skeleton::from_bytes(&bytes)
        .map(Arc::new)
        .map_err(|error| unreadable(uri.to_owned(), &error))
}

pub(super) fn clip(bytes: Arc<[u8]>, uri: &str) -> Result<Arc<Clip>, Error> {
    Clip::from_bytes(&bytes)
        .map(Arc::new)
        .map_err(|error| unreadable(uri.to_owned(), &error))
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Extent {
    pub(super) radius: f32,
    pub(super) floor: f32,
    pub(super) ceiling: f32,
}

impl Extent {
    fn of(posed: &[Posed]) -> Self {
        let mut extent = Self {
            radius: 0.0,
            floor: f32::INFINITY,
            ceiling: f32::NEG_INFINITY,
        };
        extent.take(posed);
        extent
    }

    fn take(&mut self, posed: &[Posed]) {
        for joint in posed {
            let at = joint.position;
            self.radius = max(self.radius, Vec3::new(at.x, 0.0, at.z).length());
            self.floor = min(self.floor, at.y);
            self.ceiling = max(self.ceiling, at.y);
        }
    }

    fn joined(self, other: Self) -> Self {
        Self {
            radius: max(self.radius, other.radius),
            floor: min(self.floor, other.floor),
            ceiling: max(self.ceiling, other.ceiling),
        }
    }
}

#[derive(Default)]
pub(super) struct Limbs {
    pub(super) head: Option<usize>,
    pub(super) hip: Option<usize>,
    pub(super) right_arm: Option<[usize; 3]>,
    pub(super) left_arm: Option<[usize; 3]>,
    pub(super) legs: Option<[[usize; 3]; 2]>,
}

fn joint(skeleton: &Skeleton, name: &str) -> Result<usize, Error> {
    skeleton.find(name).ok_or_else(|| {
        Error::new(ErrorCode::UnknownReference, "$.figure_limbs".to_owned()).got(name)
    })
}

fn chain(skeleton: &Skeleton, names: &[String; 3]) -> Result<[usize; 3], Error> {
    Ok([
        joint(skeleton, &names[0])?,
        joint(skeleton, &names[1])?,
        joint(skeleton, &names[2])?,
    ])
}

impl Limbs {
    fn of(skeleton: &Skeleton, limbs: Option<&FigureLimbs>) -> Result<Self, Error> {
        let Some(limbs) = limbs else {
            return Ok(Self::default());
        };
        let [left, right] = &limbs.legs;
        Ok(Self {
            head: Some(joint(skeleton, &limbs.head)?),
            hip: Some(joint(skeleton, &limbs.hip)?),
            right_arm: Some(chain(skeleton, &limbs.right_arm)?),
            left_arm: Some(chain(skeleton, &limbs.left_arm)?),
            legs: Some([chain(skeleton, left)?, chain(skeleton, right)?]),
        })
    }
}

pub(super) struct Rig {
    pub(super) skeleton_file: SkeletonFile,
    pub(super) skeleton: Arc<Skeleton>,
    pub(super) tree: Tree,
    pub(super) names: Vec<String>,
    pub(super) clips: Vec<Played>,
    pub(super) limbs: Limbs,
    pub(super) extent: Extent,
}

impl Rig {
    pub(super) fn duration(&self, clip: usize) -> f32 {
        self.clips.get(clip).map_or(0.0, Played::duration)
    }
}

const STRIDE_SAMPLES: usize = 48;

type Legs = (usize, [[usize; 3]; 2]);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Measured {
    motion: Motion,
    extent: Extent,
}

fn strode(moves: &[Vec3], hips: [Vec3; 2], duration: f32) -> Motion {
    let way = moves
        .iter()
        .fold(Vec3::ZERO, |sum, moved| sum + *moved)
        .normalize_or_zero();
    let travel: f32 = moves.iter().map(|moved| max(moved.dot(way), 0.0)).sum();
    Motion {
        stride_mps: travel / duration,
        drift_m: (hips[1] - hips[0]).with_y(0.0).length(),
    }
}

fn measure(skeleton: &Skeleton, clip: &Clip, bound: &Bound, legs: Option<Legs>) -> Measured {
    let mut posed: Vec<Posed> = (0..skeleton.len())
        .map(|joint| skeleton.rest(joint))
        .collect();
    let mut extent = Extent::of(&posed);
    let duration = clip.duration();
    let samples = if duration > 0.0 { STRIDE_SAMPLES } else { 0 };
    let mut before: Option<[Vec3; 2]> = None;
    let mut hips = [Vec3::ZERO; 2];
    let mut moves = Vec::with_capacity(samples);
    for sample in 0..=samples {
        let seconds = duration * sample as f32 / STRIDE_SAMPLES as f32;
        if bound.sample(clip, seconds, skeleton, &mut posed).is_err() {
            return Measured {
                motion: Motion::default(),
                extent,
            };
        }
        extent.take(&posed);
        let Some((hip, legs)) = legs else {
            continue;
        };
        let feet = legs.map(|leg| posed[leg[2]].position - posed[hip].position);
        if let Some(was) = before {
            let planted = usize::from(posed[legs[1][2]].position.y < posed[legs[0][2]].position.y);
            moves.push((was[planted] - feet[planted]).with_y(0.0));
        }
        before = Some(feet);
        match sample {
            0 => hips[0] = posed[hip].position,
            STRIDE_SAMPLES => hips[1] = posed[hip].position,
            _ => {}
        }
    }
    let motion = if samples > 0 && legs.is_some() {
        strode(&moves, hips, duration)
    } else {
        Motion::default()
    };
    Measured { motion, extent }
}

type MaskKey = (SkeletonFile, usize, u32);

type SkeletonFile = Asset<Arc<Skeleton>>;

type Masked = (Arc<Skeleton>, Result<Arc<[f32]>, Error>);

type Binding = (
    Arc<Skeleton>,
    Arc<Clip>,
    Arc<Bound>,
    BTreeMap<Option<Legs>, Measured>,
);

type FitKey = (SkeletonFile, String, Vec<String>);

#[derive(Resource, Default)]
pub struct FigureFiles {
    bound: BTreeMap<(SkeletonFile, Asset<Arc<Clip>>), Binding>,
    masks: BTreeMap<MaskKey, Masked>,
    tops: BTreeMap<Asset<MeshFile>, f32>,
    rests: BTreeMap<FitKey, Option<f32>>,
}

fn topped(rig: &Rig, mesh: &MeshFile) -> Option<f32> {
    let head = rig.limbs.head?;
    let bound = rig
        .skeleton
        .joints()
        .get(head)?
        .inverse_bind
        .inverse()
        .w_axis;
    mesh.floats3(POSITION, "$")
        .ok()?
        .into_iter()
        .filter(|at| at[1] > bound.y)
        .map(|at| at[1])
        .reduce(max)
}

impl FigureFiles {
    pub(super) fn keep(&mut self, assets: &Assets) {
        let live = |id: AssetId| assets.revision_of(id).is_some();
        self.bound.retain(|(skeleton, clip), _| live(skeleton.id()) && live(clip.id()));
        self.masks.retain(|(skeleton, ..), _| live(skeleton.id()));
        self.tops.retain(|mesh, _| live(mesh.id()));
        self.rests.retain(|(skeleton, ..), _| live(skeleton.id()));
    }

    pub(super) fn top(&mut self, rig: &Rig, (assets, mesh): (&Assets, Asset<MeshFile>)) -> Option<f32> {
        if let Some(known) = self.tops.get(&mesh) {
            return Some(*known);
        }
        let found = topped(rig, assets.get(mesh).ok()?)?;
        self.tops.insert(mesh, found);
        Some(found)
    }

    pub(super) fn fit_rest(&mut self, rig: &Rig, fit: &FigureFit) -> Option<f32> {
        let key = (rig.skeleton_file, fit.clip.clone(), fit.joints.clone());
        if let Some(known) = self.rests.get(&key) {
            return *known;
        }
        let played = (rig.names.binary_search(&fit.clip).ok()).and_then(|at| rig.clips.get(at));
        let found = played.and_then(|played| {
            let skeleton = &*rig.skeleton;
            let mut pose: Vec<Posed> = (0..skeleton.len()).map(|at| skeleton.rest(at)).collect();
            played
                .bound
                .sample(&played.clip, 0.0, skeleton, &mut pose)
                .ok()?;
            let mut heights = Vec::with_capacity(fit.joints.len());
            for name in &fit.joints {
                heights.push(pose.get(skeleton.find(name)?)?.position.y);
            }
            let sum = heights.iter().copied().reduce(|sum, height| sum + height)?;
            Some(sum / heights.len() as f32)
        });
        self.rests.insert(key, found);
        found
    }

    pub(super) fn rig(
        &mut self,
        (figure, limbs): (&Figure, Option<&FigureLimbs>),
        assets: &Assets,
    ) -> Result<Rig, Error> {
        let skeleton = Arc::clone(assets.get(figure.skeleton)?);
        let limbs = Limbs::of(&skeleton, limbs)?;
        let legs = limbs.hip.zip(limbs.legs);
        let rest: Vec<Posed> = (0..skeleton.len()).map(|at| skeleton.rest(at)).collect();
        let mut extent = Extent::of(&rest);
        let mut clips = Vec::with_capacity(figure.clips.len());
        for file in figure.clips.values() {
            let clip = Arc::clone(assets.get(*file)?);
            let made = || -> Binding {
                let bound = Arc::new(engine_rig::bind(&skeleton, &clip));
                (Arc::clone(&skeleton), Arc::clone(&clip), bound, BTreeMap::new())
            };
            let (_, _, bound, measures) = match self.bound.entry((figure.skeleton, *file)) {
                Entry::Occupied(held)
                    if Arc::ptr_eq(&held.get().0, &skeleton)
                        && Arc::ptr_eq(&held.get().1, &clip) =>
                {
                    held.into_mut()
                }
                Entry::Occupied(mut held) => {
                    held.insert(made());
                    held.into_mut()
                }
                Entry::Vacant(free) => free.insert(made()),
            };
            let measured = *measures
                .entry(legs)
                .or_insert_with(|| measure(&skeleton, &clip, bound, legs));
            extent = extent.joined(measured.extent);
            clips.push(Played {
                clip,
                bound: Arc::clone(bound),
                motion: measured.motion,
            });
        }
        Ok(Rig {
            skeleton_file: figure.skeleton,
            tree: Tree::of(&skeleton),
            names: figure.clips.keys().cloned().collect(),
            limbs,
            skeleton,
            clips,
            extent,
        })
    }

    pub(super) fn mask(
        &mut self,
        rig: &Rig,
        from: usize,
        feather: u32,
        path: &str,
    ) -> Result<Arc<[f32]>, Error> {
        let key = (rig.skeleton_file, from, feather);
        if let Some((bones, found)) = self.masks.get(&key)
            && Arc::ptr_eq(bones, &rig.skeleton)
        {
            return found.clone();
        }
        let found = rig
            .skeleton
            .mask(from, feather as usize)
            .map(Arc::from)
            .ok_or_else(|| Error::new(ErrorCode::UnknownReference, path.to_owned()).got(from));
        self.masks
            .insert(key, (Arc::clone(&rig.skeleton), found.clone()));
        found
    }
}
