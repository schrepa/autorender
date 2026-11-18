use glam::{Mat4, Quat, Vec3};

use crate::RigError;
use crate::bytes::{Reader, put_count, put_floats, put_name, put_u32};

const MAGIC: [u8; 4] = *b"SKEL";
const NO_PARENT: u32 = u32::MAX;

#[derive(Clone, Debug, PartialEq)]
pub struct Joint {
    pub name: String,
    pub parent: Option<usize>,
    pub inverse_bind: Mat4,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Posed {
    pub rotation: Quat,
    pub position: Vec3,
}

impl Posed {
    pub fn matrix(self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.position)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Skeleton {
    joints: Vec<Joint>,
    rest: Vec<Posed>,
    inverse_rest: Vec<Quat>,
    offsets: Vec<Vec3>,
}

impl Skeleton {
    pub fn new(joints: Vec<Joint>) -> Result<Self, RigError> {
        let mut rest = Vec::with_capacity(joints.len());
        for (index, joint) in joints.iter().enumerate() {
            if joint.name.is_empty() {
                return Err(RigError::EmptyName {
                    item: "joint",
                    index,
                });
            }
            if joints[..index]
                .iter()
                .any(|earlier| earlier.name == joint.name)
            {
                return Err(RigError::NameTwice {
                    item: "joint",
                    name: joint.name.clone(),
                });
            }
            if let Some(parent) = joint.parent
                && parent >= index
            {
                return Err(RigError::ParentAfterChild {
                    joint: joint.name.clone(),
                    parent,
                });
            }
            let singular = || RigError::InverseBindNotInvertible {
                joint: joint.name.clone(),
            };
            if !joint.inverse_bind.is_finite() || joint.inverse_bind.determinant() == 0.0 {
                return Err(singular());
            }
            let (_, rotation, position) =
                joint.inverse_bind.inverse().to_scale_rotation_translation();
            if !rotation.is_finite() || !position.is_finite() {
                return Err(singular());
            }
            rest.push(Posed {
                rotation: rotation.normalize(),
                position,
            });
        }
        let inverse_rest: Vec<Quat> = rest
            .iter()
            .map(|posed| posed.rotation.conjugate())
            .collect();
        let offsets = joints
            .iter()
            .zip(&rest)
            .map(|(joint, posed)| match joint.parent {
                Some(parent) => inverse_rest[parent] * (posed.position - rest[parent].position),
                None => Vec3::ZERO,
            })
            .collect();
        Ok(Self {
            joints,
            rest,
            inverse_rest,
            offsets,
        })
    }

    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    pub fn len(&self) -> usize {
        self.joints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|joint| joint.name == name)
    }

    pub fn rest(&self, joint: usize) -> Posed {
        self.rest[joint]
    }

    pub fn mask(&self, joint: &str, feather_levels: usize) -> Result<Vec<f32>, RigError> {
        let top = self.find(joint).ok_or_else(|| RigError::JointMissing {
            name: joint.to_owned(),
        })?;
        let mut depths: Vec<Option<usize>> = vec![None; self.joints.len()];
        depths[top] = Some(0);
        for index in top + 1..self.joints.len() {
            depths[index] = self.joints[index]
                .parent
                .and_then(|parent| depths[parent])
                .map(|depth| depth + 1);
        }
        Ok(depths
            .into_iter()
            .map(|depth| match depth {
                None => 0.0,
                Some(depth) if depth < feather_levels => {
                    (depth + 1) as f32 / (feather_levels + 1) as f32
                }
                Some(_) => 1.0,
            })
            .collect())
    }

    pub(crate) fn below(&self, top: usize, joint: usize) -> bool {
        let mut held = Some(joint);
        while let Some(at) = held {
            if at == top {
                return true;
            }
            if at < top {
                return false;
            }
            held = self.joints[at].parent;
        }
        false
    }

    pub(crate) fn inverse_rest(&self, joint: usize) -> Quat {
        self.inverse_rest[joint]
    }

    pub(crate) fn offset(&self, joint: usize) -> Vec3 {
        self.offsets[joint]
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + self.joints.len() * 96);
        bytes.extend_from_slice(&MAGIC);
        put_count(&mut bytes, self.joints.len());
        for joint in &self.joints {
            put_name(&mut bytes, &joint.name);
            put_u32(
                &mut bytes,
                joint.parent.map_or(NO_PARENT, |parent| {
                    u32::try_from(parent).expect("parent_fits_u32")
                }),
            );
            put_floats(&mut bytes, &joint.inverse_bind.to_cols_array());
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RigError> {
        let mut reader = Reader::new(bytes, "skeleton", MAGIC)?;
        let count = reader.count()?;
        let mut joints = Vec::with_capacity(count.min(bytes.len() / 72));
        for _ in 0..count {
            let name = reader.name()?;
            let parent = match reader.u32()? {
                NO_PARENT => None,
                parent => Some(parent as usize),
            };
            let inverse_bind = Mat4::from_cols_array(&reader.floats::<16>()?);
            joints.push(Joint {
                name,
                parent,
                inverse_bind,
            });
        }
        reader.finish()?;
        Self::new(joints)
    }
}
