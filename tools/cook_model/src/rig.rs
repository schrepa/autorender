use cook::Refusal;
use engine_rig::{Joint, Skeleton};
use glam::{Mat4, Quat, Vec3};

use crate::source::{Source, local, node_label};

pub struct Rig {
    pub skeleton: Skeleton,
    pub nodes: Vec<usize>,
    pub of_skin: Vec<u32>,
    pub space: Mat4,
    pub scale: f32,
    pub bind_gap: f32,
    pub frames: Vec<Quat>,
}

impl Rig {
    pub fn joint_of_node(&self, node: usize) -> Option<usize> {
        self.nodes.iter().position(|&held| held == node)
    }

    pub fn model_of(&self, world: Mat4) -> Mat4 {
        Mat4::from_scale(Vec3::splat(self.scale)) * self.space.inverse() * world
    }
}

fn unit_scale(matrix: Mat4, tolerance: f32) -> bool {
    let (scale, _, _) = matrix.to_scale_rotation_translation();
    (scale - Vec3::ONE).abs().max_element() <= tolerance
}

fn order(source: &Source, joints: &[usize]) -> Vec<usize> {
    let mut ordered = Vec::with_capacity(joints.len());
    let mut pending: Vec<usize> = joints
        .iter()
        .copied()
        .filter(|&node| {
            source
                .parent(node)
                .is_none_or(|parent| !joints.contains(&parent))
        })
        .rev()
        .collect();
    while let Some(node) = pending.pop() {
        ordered.push(node);
        let children: Vec<usize> = source
            .node(node)
            .children()
            .map(|child| child.index())
            .filter(|child| joints.contains(child))
            .collect();
        pending.extend(children.into_iter().rev());
    }
    ordered
}

fn root_space(
    source: &Source,
    ordered: &[usize],
    joints: &[usize],
    tolerance: f32,
) -> Result<Mat4, Refusal> {
    let roots: Vec<usize> = ordered
        .iter()
        .copied()
        .filter(|&node| {
            source
                .parent(node)
                .is_none_or(|parent| !joints.contains(&parent))
        })
        .collect();
    let parent = source.parent(roots[0]);
    if let Some(&other) = roots.iter().find(|&&root| source.parent(root) != parent) {
        return Err(source.refuse(
            "RootParents",
            node_label(&source.node(other)),
            "skin roots hang under different parents; one armature node is required",
        ));
    }
    let space = parent.map_or(Mat4::IDENTITY, |parent| source.world(parent));
    if !unit_scale(space, tolerance) || space.determinant() < 0.0 {
        return Err(source.refuse(
            "RootParentScale",
            parent
                .map(|node| node_label(&source.node(node)))
                .unwrap_or_default(),
            "the armature node scales; it must be rigid",
        ));
    }
    Ok(space)
}

pub fn rig(source: &Source, scale: f32, tolerance: f32) -> Result<Rig, Refusal> {
    let skins: Vec<gltf::Skin<'_>> = source.gltf.skins().collect();
    let [skin] = skins.as_slice() else {
        return Err(source.refuse(
            "SkinCount",
            "skins",
            format!("{} skins, one is required", skins.len()),
        ));
    };
    let joints: Vec<usize> = skin.joints().map(|node| node.index()).collect();
    if joints.is_empty() {
        return Err(source.refuse("SkinCount", "skins[0]", "the skin has no joints"));
    }
    let inverse_binds: Vec<Mat4> = skin
        .reader(|held| source.data(&held))
        .read_inverse_bind_matrices()
        .ok_or_else(|| source.refuse("MissingInverseBinds", "skins[0]", "no inverseBindMatrices"))?
        .map(|columns| Mat4::from_cols_array_2d(&columns))
        .collect();
    if inverse_binds.len() != joints.len() {
        return Err(source.refuse(
            "MissingInverseBinds",
            "skins[0]",
            format!(
                "{} matrices for {} joints",
                inverse_binds.len(),
                joints.len()
            ),
        ));
    }
    let ordered = order(source, &joints);
    let space = root_space(source, &ordered, &joints, tolerance)?;
    let scaling = Mat4::from_scale(Vec3::splat(scale));
    let mut built = Vec::with_capacity(ordered.len());
    for &node in &ordered {
        let held = source.node(node);
        let name = held.name().ok_or_else(|| {
            source.refuse("JointUnnamed", node_label(&held), "a joint needs a name")
        })?;
        if !unit_scale(local(&held), tolerance) {
            return Err(source.refuse(
                "JointScale",
                node_label(&held),
                "rest scale is not 1 within the tolerance",
            ));
        }
        let skin_index = joints
            .iter()
            .position(|&joint| joint == node)
            .expect("ordered_from_joints");
        let model = inverse_binds[skin_index] * space;
        built.push(Joint {
            name: name.to_owned(),
            parent: source
                .parent(node)
                .and_then(|parent| ordered.iter().position(|&held| held == parent)),
            inverse_bind: scaling * model * scaling.inverse(),
        });
    }
    let skeleton =
        Skeleton::new(built).map_err(|error| source.refuse("Skeleton", "skins[0]", error))?;
    let of_skin = joints
        .iter()
        .map(|node| {
            let at = ordered
                .iter()
                .position(|held| held == node)
                .expect("joint_ordered");
            u32::try_from(at).expect("joint_count_fits_u32")
        })
        .collect();
    let rig = Rig {
        frames: vec![Quat::IDENTITY; ordered.len()],
        skeleton,
        nodes: ordered,
        of_skin,
        space,
        scale,
        bind_gap: 0.0,
    };
    Ok(with_gap(source, rig))
}

fn with_gap(source: &Source, mut rig: Rig) -> Rig {
    rig.bind_gap = (0..rig.nodes.len())
        .map(|joint| {
            let rest = rig
                .model_of(source.world(rig.nodes[joint]))
                .w_axis
                .truncate();
            (rest - rig.skeleton.rest(joint).position).length()
        })
        .fold(
            0.0,
            |widest: f32, gap| if gap > widest { gap } else { widest },
        );
    rig
}

pub fn rig_by_names(source: &Source, from: &Rig, tolerance: f32) -> Result<Rig, Refusal> {
    let mut nodes = Vec::with_capacity(from.skeleton.len());
    for joint in from.skeleton.joints() {
        let mut named = source
            .gltf
            .nodes()
            .filter(|node| node.name() == Some(joint.name.as_str()));
        let node = named.next().ok_or_else(|| {
            source.refuse(
                "JointNodeMissing",
                format!("joint `{}`", joint.name),
                "no node of that name",
            )
        })?;
        if named.next().is_some() {
            return Err(source.refuse(
                "JointNodeTwice",
                format!("joint `{}`", joint.name),
                "two nodes of that name",
            ));
        }
        nodes.push(node.index());
    }
    for (index, joint) in from.skeleton.joints().iter().enumerate() {
        let parent = source.parent(nodes[index]);
        let expected = joint.parent.map(|parent| nodes[parent]);
        let fits = match expected {
            Some(_) => parent == expected,
            None => parent.is_none_or(|parent| !nodes.contains(&parent)),
        };
        if !fits {
            return Err(source.refuse(
                "JointParent",
                format!("joint `{}`", joint.name),
                "its node's parent differs from the skeleton's",
            ));
        }
    }
    let space = root_space(source, &nodes, &nodes, tolerance)?;
    let mut turned: Vec<Quat> = Vec::with_capacity(nodes.len());
    for (index, joint) in from.skeleton.joints().iter().enumerate() {
        let (_, rotation, _) = source.node(nodes[index]).transform().decomposed();
        let own = Quat::from_array(rotation).normalize();
        let held = joint.parent.map_or(own, |parent| turned[parent] * own);
        turned.push(held.normalize());
    }
    let frames = turned
        .iter()
        .enumerate()
        .map(|(index, held)| (held.conjugate() * from.skeleton.rest(index).rotation).normalize())
        .collect();
    let rig = Rig {
        skeleton: from.skeleton.clone(),
        nodes,
        of_skin: Vec::new(),
        space,
        scale: from.scale,
        bind_gap: 0.0,
        frames,
    };
    Ok(with_gap(source, rig))
}
