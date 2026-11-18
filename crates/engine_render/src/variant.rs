use std::hash::{DefaultHasher, Hash, Hasher};

use crate::install::RenderDecl;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PerBatch {
    pub state: bool,
    pub cull: bool,
    pub fragment: bool,
}

impl PerBatch {
    pub const NONE: Self = Self {
        state: false,
        cull: false,
        fragment: false,
    };
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Variant {
    pub constants: Vec<(String, f64)>,
    pub targets: Option<Vec<(Option<wgpu::BlendState>, wgpu::ColorWrites)>>,
    pub depth_write: Option<bool>,
    pub depth_compare: Option<wgpu::CompareFunction>,
    pub depth_bias: Option<wgpu::DepthBiasState>,
    pub cull: Option<wgpu::Face>,
    pub fragment: bool,
    pub stand_in: Option<Box<Variant>>,
}

impl Variant {
    pub(crate) fn cut(&self, desc: &RenderDecl, declared: impl Fn(&str) -> bool) -> Option<Self> {
        let taken = desc.per_batch;
        let cut = Self {
            constants: (self.constants.iter())
                .filter(|(name, _)| declared(name))
                .cloned()
                .collect(),
            targets: self.targets.clone().filter(|_| taken.state),
            depth_write: self.depth_write.filter(|_| taken.state),
            depth_compare: self.depth_compare.filter(|_| taken.state),
            depth_bias: self.depth_bias.filter(|_| taken.state),
            cull: if taken.cull {
                self.cull
            } else {
                desc.primitive.cull_mode
            },
            fragment: taken.fragment && self.fragment,
            stand_in: None,
        };
        let plain = cut.constants.is_empty()
            && cut.targets.is_none()
            && cut.depth_write.is_none()
            && cut.depth_compare.is_none()
            && cut.depth_bias.is_none()
            && cut.cull == desc.primitive.cull_mode
            && !cut.fragment;
        (!plain).then_some(cut)
    }

    pub(crate) fn key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        for (name, value) in &self.constants {
            (name, value.to_bits()).hash(&mut hasher);
        }
        (self.targets.as_ref(), self.depth_write, self.depth_compare).hash(&mut hasher);
        self.depth_bias
            .map(|bias| (bias.constant, bias.slope_scale.to_bits(), bias.clamp.to_bits()))
            .hash(&mut hasher);
        (self.cull, self.fragment).hash(&mut hasher);
        hasher.finish()
    }
}
