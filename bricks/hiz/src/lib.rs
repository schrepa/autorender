#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

engine::brick!();

use engine::prelude::*;
use engine_core::{Error, GpuStruct, Source, TargetName, gpu_struct};
use engine_render::{
    Allocation, Bind, Buffer, ComputePipeline, Encode, GpuPart, Install, Lifetime, Pass, Prepare,
    RenderBricks, Texture, wgpu,
};

pub const MAX_HIZ_MIPS: usize = 11;
pub const MAX_HIZ_DIM: u32 = 1024;

const WORKGROUP_SIZE: u32 = 8;

gpu_struct! {
    pub struct HizInfo {
        pub src_size: [u32; 2],
        pub mip0_size: [u32; 2],
        pub mip0_size_f: [f32; 2],
        pub mip_count: u32,
        pub pad: u32,
    }
}

gpu_struct! {
    pub struct HizCullParams {
        pub view: [[f32; 4]; 4],
        pub proj: [[f32; 4]; 4],
        pub near_z: f32,
        pub count: u32,
        pub pad0: u32,
        pub pad1: u32,
    }
}

gpu_struct! {
    pub struct HizSphere {
        pub center_radius: [f32; 4],
    }
}

const GENERATED: &[&str] = &[
    <HizInfo as GpuStruct>::WGSL,
    <HizCullParams as GpuStruct>::WGSL,
    <HizSphere as GpuStruct>::WGSL,
];

pub const fn prev_pow2(value: u32) -> u32 {
    if value <= 1 {
        1
    } else {
        1_u32 << (31 - value.leading_zeros())
    }
}

pub const fn mip_extent(mip0_w: u32, mip0_h: u32, mip: u32) -> (u32, u32) {
    let shift = if mip > 31 { 31 } else { mip };
    let w = mip0_w >> shift;
    let h = mip0_h >> shift;
    (if w > 0 { w } else { 1 }, if h > 0 { h } else { 1 })
}

impl HizInfo {
    #[must_use]
    pub const fn for_size(src_w: u32, src_h: u32) -> Self {
        let sw = if src_w > 0 { src_w } else { 1 };
        let sh = if src_h > 0 { src_h } else { 1 };
        let mut mip0_w = prev_pow2(sw);
        let mut mip0_h = prev_pow2(sh);
        while mip0_w > MAX_HIZ_DIM || mip0_h > MAX_HIZ_DIM {
            if mip0_w > 1 {
                mip0_w >>= 1;
            }
            if mip0_h > 1 {
                mip0_h >>= 1;
            }
        }
        let max_dim = if mip0_w >= mip0_h { mip0_w } else { mip0_h };
        let raw_mips = max_dim.ilog2() + 1;
        let max_mips = MAX_HIZ_MIPS as u32;
        let mip_count = if raw_mips > max_mips {
            max_mips
        } else {
            raw_mips
        };
        Self {
            src_size: [sw, sh],
            mip0_size: [mip0_w, mip0_h],
            mip0_size_f: [mip0_w as f32, mip0_h as f32],
            mip_count,
            pad: 0,
        }
    }
}

pub struct HizBrick;

impl Brick for HizBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks.generated(GENERATED);
    }
}

pub struct HizPass<P> {
    build_mip0: ComputePipeline<P>,
    clear_mip0: ComputePipeline<P>,
    downsamples: Vec<ComputePipeline<P>>,
}

pub struct HizPyramid<P> {
    texture: Texture<P>,
    uniform: Buffer<P>,
    primary: HizPass<P>,
    info: HizInfo,
    uploaded: Option<HizInfo>,
}

fn install_downsamples<P: GpuPart>(
    cx: &mut Install<'_, P>,
    pass: Pass<P>,
    texture: Texture<P>,
) -> Result<Vec<ComputePipeline<P>>, Error> {
    use shaders::hiz::pyramid;
    let mut downsamples = Vec::with_capacity(MAX_HIZ_MIPS - 1);
    for mip in 0..(MAX_HIZ_MIPS - 1) as u32 {
        let pipeline = cx.compute_pipeline(
            pass,
            pyramid::CS_HIZ_DOWNSAMPLE,
            &[],
            &[
                (pyramid::HIZ_SRC, Bind::TextureMip(texture, mip)),
                (pyramid::HIZ_DST, Bind::TextureMip(texture, mip + 1)),
            ],
        )?;
        downsamples.push(pipeline);
    }
    Ok(downsamples)
}

fn install_clear<P: GpuPart>(
    cx: &mut Install<'_, P>,
    pass: Pass<P>,
    texture: Texture<P>,
) -> Result<ComputePipeline<P>, Error> {
    use shaders::hiz::pyramid;
    cx.compute_pipeline(
        pass,
        pyramid::CS_HIZ_CLEAR,
        &[],
        &[(pyramid::HIZ_DST, Bind::TextureMip(texture, 0))],
    )
}

impl<P: GpuPart> HizPyramid<P> {
    pub fn install_from_depth(
        cx: &mut Install<'_, P>,
        pass: Pass<P>,
        depth_target: TargetName,
    ) -> Result<Self, Error> {
        let texture = cx.texture("hiz_pyramid", wgpu::TextureFormat::R32Float, Lifetime::Kept);
        let uniform = cx.buffer("hiz_info", Lifetime::Kept);
        let primary = Self::build_pass_from_depth(cx, pass, texture, depth_target)?;
        Ok(Self {
            texture,
            uniform,
            primary,
            info: HizInfo::default(),
            uploaded: None,
        })
    }

    pub fn install_from_depth_and_sw(
        cx: &mut Install<'_, P>,
        pass: Pass<P>,
        depth_target: TargetName,
        sw_depth_buf: Buffer<P>,
    ) -> Result<Self, Error> {
        let texture = cx.texture("hiz_pyramid", wgpu::TextureFormat::R32Float, Lifetime::Kept);
        let uniform = cx.buffer("hiz_info", Lifetime::Kept);
        let primary =
            Self::build_pass_from_depth_and_sw(cx, pass, texture, uniform, depth_target, sw_depth_buf)?;
        Ok(Self {
            texture,
            uniform,
            primary,
            info: HizInfo::default(),
            uploaded: None,
        })
    }

    pub fn install_pass_from_depth(
        &self,
        cx: &mut Install<'_, P>,
        pass: Pass<P>,
        depth_target: TargetName,
    ) -> Result<HizPass<P>, Error> {
        Self::build_pass_from_depth(cx, pass, self.texture, depth_target)
    }

    pub fn install_pass_from_depth_and_sw(
        &self,
        cx: &mut Install<'_, P>,
        pass: Pass<P>,
        depth_target: TargetName,
        sw_depth_buf: Buffer<P>,
    ) -> Result<HizPass<P>, Error> {
        Self::build_pass_from_depth_and_sw(
            cx,
            pass,
            self.texture,
            self.uniform,
            depth_target,
            sw_depth_buf,
        )
    }

    fn build_pass_from_depth(
        cx: &mut Install<'_, P>,
        pass: Pass<P>,
        texture: Texture<P>,
        depth_target: TargetName,
    ) -> Result<HizPass<P>, Error> {
        use shaders::hiz::pyramid;
        let build_mip0 = cx.compute_pipeline(
            pass,
            pyramid::CS_HIZ_FROM_DEPTH,
            &[],
            &[
                (pyramid::HW_DEPTH, Bind::Target(depth_target)),
                (pyramid::HIZ_DST, Bind::TextureMip(texture, 0)),
            ],
        )?;
        let clear_mip0 = install_clear(cx, pass, texture)?;
        let downsamples = install_downsamples(cx, pass, texture)?;
        Ok(HizPass {
            build_mip0,
            clear_mip0,
            downsamples,
        })
    }

    fn build_pass_from_depth_and_sw(
        cx: &mut Install<'_, P>,
        pass: Pass<P>,
        texture: Texture<P>,
        uniform: Buffer<P>,
        depth_target: TargetName,
        sw_depth_buf: Buffer<P>,
    ) -> Result<HizPass<P>, Error> {
        use shaders::hiz::pyramid;
        let build_mip0 = cx.compute_pipeline(
            pass,
            pyramid::CS_HIZ_FROM_DEPTH_AND_SW,
            &[],
            &[
                (pyramid::HIZ_INFO, Bind::Buffer(uniform)),
                (pyramid::HW_DEPTH, Bind::Target(depth_target)),
                (pyramid::SW_DEPTH_U32, Bind::Buffer(sw_depth_buf)),
                (pyramid::HIZ_DST, Bind::TextureMip(texture, 0)),
            ],
        )?;
        let clear_mip0 = install_clear(cx, pass, texture)?;
        let downsamples = install_downsamples(cx, pass, texture)?;
        Ok(HizPass {
            build_mip0,
            clear_mip0,
            downsamples,
        })
    }

    pub fn prepare(&mut self, cx: &mut Prepare<'_, P>) -> Allocation {
        let size = cx.canvas_size();
        self.prepare_sized(cx, size)
    }

    pub fn prepare_sized(&mut self, cx: &mut Prepare<'_, P>, src_size: (u32, u32)) -> Allocation {
        let info = HizInfo::for_size(src_size.0, src_size.1);
        let tex_alloc = cx.allocate_mips(
            self.texture,
            wgpu::Extent3d {
                width: info.mip0_size[0],
                height: info.mip0_size[1],
                depth_or_array_layers: 1,
            },
            info.mip_count,
        );
        let buf_alloc = cx.allocate_buffer(self.uniform, size_of::<HizInfo>() as u64);
        if buf_alloc == Allocation::Fresh || self.uploaded != Some(info) {
            cx.write(self.uniform, 0, &[info]);
            self.uploaded = Some(info);
        }
        self.info = info;
        tex_alloc
    }

    pub fn encode(&self, cx: &mut Encode<'_, P>) {
        self.encode_pass(cx, &self.primary);
    }

    pub fn encode_pass(&self, cx: &mut Encode<'_, P>, pass: &HizPass<P>) {
        let mip0_w = self.info.mip0_size[0];
        let mip0_h = self.info.mip0_size[1];
        if mip0_w == 0 || mip0_h == 0 || self.info.mip_count == 0 {
            return;
        }
        cx.dispatch(
            &pass.build_mip0,
            [
                mip0_w.div_ceil(WORKGROUP_SIZE),
                mip0_h.div_ceil(WORKGROUP_SIZE),
                1,
            ],
        );
        self.encode_downsamples(cx, pass);
    }

    pub fn clear_empty(&self, cx: &mut Encode<'_, P>) {
        self.clear_pass(cx, &self.primary);
    }

    pub fn clear_pass(&self, cx: &mut Encode<'_, P>, pass: &HizPass<P>) {
        let mip0_w = self.info.mip0_size[0];
        let mip0_h = self.info.mip0_size[1];
        if mip0_w == 0 || mip0_h == 0 || self.info.mip_count == 0 {
            return;
        }
        cx.dispatch(
            &pass.clear_mip0,
            [
                mip0_w.div_ceil(WORKGROUP_SIZE),
                mip0_h.div_ceil(WORKGROUP_SIZE),
                1,
            ],
        );
        self.encode_downsamples(cx, pass);
    }

    fn encode_downsamples(&self, cx: &mut Encode<'_, P>, pass: &HizPass<P>) {
        let mip0_w = self.info.mip0_size[0];
        let mip0_h = self.info.mip0_size[1];
        let count = (self.info.mip_count.saturating_sub(1) as usize).min(pass.downsamples.len());
        for (mip_idx, pipeline) in pass.downsamples.iter().take(count).enumerate() {
            let dst_mip = (mip_idx + 1) as u32;
            let (dst_w, dst_h) = mip_extent(mip0_w, mip0_h, dst_mip);
            cx.dispatch(
                pipeline,
                [
                    dst_w.div_ceil(WORKGROUP_SIZE),
                    dst_h.div_ceil(WORKGROUP_SIZE),
                    1,
                ],
            );
        }
    }

    #[must_use]
    pub fn texture(&self) -> Texture<P> {
        self.texture
    }

    #[must_use]
    pub fn uniform(&self) -> Buffer<P> {
        self.uniform
    }

    #[must_use]
    pub fn info(&self) -> HizInfo {
        self.info
    }

    #[must_use]
    pub fn mip0_size(&self) -> (u32, u32) {
        (self.info.mip0_size[0], self.info.mip0_size[1])
    }

    #[must_use]
    pub fn mip_count(&self) -> u32 {
        self.info.mip_count
    }
}
