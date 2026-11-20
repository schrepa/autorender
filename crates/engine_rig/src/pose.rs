use crate::bound::nlerp;
use engine_codec::Error;

use crate::error::Fault;
use crate::{Posed, Skeleton};

fn lengths(skeleton: &Skeleton, held: &[(&'static str, usize)]) -> Result<(), Error> {
    for &(item, got) in held {
        if got != skeleton.len() {
            return Err(Error::from(Fault::Length {
                item,
                expected: skeleton.len(),
                got,
            }));
        }
    }
    Ok(())
}

pub fn blend(
    skeleton: &Skeleton,
    from: &[Posed],
    to: &[Posed],
    weights: &[f32],
    root_motion: Option<usize>,
    out: &mut [Posed],
) -> Result<(), Error> {
    lengths(
        skeleton,
        &[
            ("blend from", from.len()),
            ("blend to", to.len()),
            ("blend weights", weights.len()),
            ("blended joints", out.len()),
        ],
    )?;
    for joint in 0..skeleton.len() {
        let weight = weights[joint];
        let (a, b) = (from[joint], to[joint]);
        let rotation = nlerp(a.rotation, b.rotation, weight);
        let parent = skeleton.joints()[joint].parent;
        let position = match parent {
            Some(parent) if root_motion != Some(joint) => {
                let offset = |pose: &[Posed]| {
                    pose[parent].rotation.conjugate()
                        * (pose[joint].position - pose[parent].position)
                };
                let (near, far) = (offset(from), offset(to));
                out[parent].position + out[parent].rotation * (near + (far - near) * weight)
            }
            Some(_) | None => a.position + (b.position - a.position) * weight,
        };
        out[joint] = Posed { rotation, position };
    }
    Ok(())
}
