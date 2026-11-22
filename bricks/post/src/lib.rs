#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

engine::brick!();

use engine::prelude::*;
use engine_core::{Error, Filter, GpuStruct, Sampler, Source, TargetName, Wrap, gpu_struct};
use engine_render::{
    Allocation, Bind, Buffer, ComputePipeline, DepthUse, Draw, Encode, GpuPart, Install, Lifetime,
    Own, Pass, PerBatch, Pipeline, Prepare, Recorded, RenderBricks, RenderPipeline, Shape, Texture,
    wgpu,
};

pub const MAX_BLOOM_MIPS: usize = 5;
pub const MAX_BLOOM_DIM: u32 = 1024;

pub const POST_FLAG_ACES: u32 = 1 << 0;
pub const POST_FLAG_LINEAR_OUT: u32 = 1 << 1;
pub const POST_FLAG_DISABLE_BLOOM: u32 = 1 << 2;
pub const POST_FLAG_AGX_PUNCH: u32 = 1 << 3;
pub const POST_FLAG_DISABLE_SHAFTS: u32 = 1 << 4;
pub const POST_FLAG_DISABLE_DITHER: u32 = 1 << 5;
pub const POST_FLAG_STANDARD_Z: u32 = 1 << 6;

const WORKGROUP_SIZE: u32 = 8;

const LINEAR_CLAMP: Sampler = Sampler {
    wrap_u: Wrap::ClampToEdge,
    wrap_v: Wrap::ClampToEdge,
    wrap_w: Wrap::ClampToEdge,
    mag_filter: Filter::Linear,
    min_filter: Filter::Linear,
    mipmap_filter: None,
    compare: None,
    anisotropy: 1,
};

gpu_struct! {
    pub struct PostParams {
        pub exposure: f32,
        pub bloom_intensity: f32,
        pub bloom_threshold: f32,
        pub bloom_knee: f32,
        pub shaft_color: [f32; 3],
        pub shaft_intensity: f32,
        pub sun_uv: [f32; 2],
        pub sun_visibility: f32,
        pub vignette: f32,
        pub saturation: f32,
        pub grain_intensity: f32,
        pub time: f32,
        pub flags: u32,
    }
}

const GENERATED: &[&str] = &[<PostParams as GpuStruct>::WGSL];

impl PostParams {
    #[must_use]
    pub const fn cinematic() -> Self {
        Self {
            exposure: 1.0,
            bloom_intensity: 0.18,
            bloom_threshold: 1.15,
            bloom_knee: 0.5,
            shaft_color: [1.0, 0.85, 0.62],
            shaft_intensity: 0.0,
            sun_uv: [0.5, 0.5],
            sun_visibility: 1.0,
            vignette: 0.18,
            saturation: 1.08,
            grain_intensity: 0.008,
            time: 0.0,
            flags: 0,
        }
    }

    #[must_use]
    pub fn with_sun_projection(
        mut self,
        view_projection: [[f32; 4]; 4],
        camera_forward: [f32; 3],
        sun_dir: [f32; 3],
    ) -> Self {
        let forward_dot = camera_forward[0] * sun_dir[0]
            + camera_forward[1] * sun_dir[1]
            + camera_forward[2] * sun_dir[2];
        let clip_x = view_projection[0][0] * sun_dir[0]
            + view_projection[1][0] * sun_dir[1]
            + view_projection[2][0] * sun_dir[2];
        let clip_y = view_projection[0][1] * sun_dir[0]
            + view_projection[1][1] * sun_dir[1]
            + view_projection[2][1] * sun_dir[2];
        let clip_w = view_projection[0][3] * sun_dir[0]
            + view_projection[1][3] * sun_dir[1]
            + view_projection[2][3] * sun_dir[2];
        if clip_w > 0.01 && forward_dot > 0.05 {
            let ux = (clip_x / clip_w) * 0.5 + 0.5;
            let uy = 0.5 - (clip_y / clip_w) * 0.5;
            let edge = engine::math::clamp(1.65 - (ux - 0.5).abs() - (uy - 0.5).abs(), 0.0, 1.0);
            let align = engine::math::clamp((forward_dot - 0.05) * 2.2, 0.0, 1.0);
            self.sun_uv = [ux, uy];
            self.sun_visibility = edge * align;
        } else {
            self.sun_uv = [0.5, 0.5];
            self.sun_visibility = 0.0;
        }
        self
    }
}

pub const fn bloom_mip0_size(src_w: u32, src_h: u32) -> (u32, u32) {
    let sw = if src_w > 0 { src_w } else { 1 };
    let sh = if src_h > 0 { src_h } else { 1 };
    let half_w = sw.div_ceil(2);
    let half_h = sh.div_ceil(2);
    let (capped_w, capped_h) = if half_w >= half_h && half_w > MAX_BLOOM_DIM {
        let scaled_h = (half_h as u64 * MAX_BLOOM_DIM as u64).div_ceil(half_w as u64) as u32;
        (MAX_BLOOM_DIM, scaled_h)
    } else if half_h > MAX_BLOOM_DIM {
        let scaled_w = (half_w as u64 * MAX_BLOOM_DIM as u64).div_ceil(half_h as u64) as u32;
        (scaled_w, MAX_BLOOM_DIM)
    } else {
        (half_w, half_h)
    };
    let w = if capped_w < 2 { 2 } else { capped_w };
    let h = if capped_h < 2 { 2 } else { capped_h };
    (w, h)
}

pub const fn bloom_mip_count(mip0_w: u32, mip0_h: u32) -> u32 {
    let max_dim = if mip0_w >= mip0_h { mip0_w } else { mip0_h };
    let raw = if max_dim <= 1 { 1 } else { max_dim.ilog2() + 1 };
    let cap = MAX_BLOOM_MIPS as u32;
    if raw > cap { cap } else { raw }
}

pub const fn mip_extent(mip0_w: u32, mip0_h: u32, mip: u32) -> (u32, u32) {
    let shift = if mip > 31 { 31 } else { mip };
    let w = mip0_w >> shift;
    let h = mip0_h >> shift;
    (if w > 0 { w } else { 1 }, if h > 0 { h } else { 1 })
}

pub struct PostBrick;

impl Brick for PostBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks.generated(GENERATED);
    }
}

pub struct PostPipeline<P> {
    bloom_down: Texture<P>,
    bloom_up: Texture<P>,
    uniform: Buffer<P>,
    sampler: engine_render::Sampler<P>,
    prefilter: ComputePipeline<P>,
    downsamples: Vec<ComputePipeline<P>>,
    upsamples_first: Vec<ComputePipeline<P>>,
    upsamples_chain: Vec<ComputePipeline<P>>,
    composite: Pipeline<P, Own>,
    has_depth: bool,
    bloom_size: (u32, u32),
    mip_count: u32,
    params: PostParams,
    uploaded: Option<PostParams>,
}

fn composite_desc() -> RenderPipeline<'static> {
    use shaders::post::composite;
    RenderPipeline {
        vertex: composite::VS_FULLSCREEN,
        fragment: Some(composite::FS_COMPOSITE),
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets: &[(None, wgpu::ColorWrites::ALL)],
        constants: &[],
        per_batch: PerBatch::NONE,
    }
}

fn install_prefilter<P: GpuPart>(
    cx: &mut Install<'_, P>,
    pass: Pass<P>,
    uniform: Buffer<P>,
    hdr_color_target: TargetName,
    depth_target: Option<TargetName>,
    sampler: engine_render::Sampler<P>,
    bloom_down: Texture<P>,
) -> Result<ComputePipeline<P>, Error> {
    use shaders::post::bloom;
    match depth_target {
        Some(depth) => cx.compute_pipeline(
            pass,
            bloom::CS_BLOOM_PREFILTER_SHAFTS,
            &[],
            &[
                (bloom::POST_PARAMS, Bind::Buffer(uniform)),
                (bloom::HDR_SRC, Bind::Target(hdr_color_target)),
                (bloom::DEPTH_TEX, Bind::Target(depth)),
                (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
                (bloom::BLOOM_DST, Bind::TextureMip(bloom_down, 0)),
            ],
        ),
        None => cx.compute_pipeline(
            pass,
            bloom::CS_BLOOM_PREFILTER,
            &[],
            &[
                (bloom::POST_PARAMS, Bind::Buffer(uniform)),
                (bloom::HDR_SRC, Bind::Target(hdr_color_target)),
                (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
                (bloom::BLOOM_DST, Bind::TextureMip(bloom_down, 0)),
            ],
        ),
    }
}

fn install_downsamples<P: GpuPart>(
    cx: &mut Install<'_, P>,
    pass: Pass<P>,
    bloom_down: Texture<P>,
    sampler: engine_render::Sampler<P>,
) -> Result<Vec<ComputePipeline<P>>, Error> {
    use shaders::post::bloom;
    let mut downsamples = Vec::with_capacity(MAX_BLOOM_MIPS - 1);
    for mip in 0..(MAX_BLOOM_MIPS - 1) as u32 {
        let pipeline = cx.compute_pipeline(
            pass,
            bloom::CS_BLOOM_DOWNSAMPLE,
            &[],
            &[
                (bloom::BLOOM_SRC, Bind::TextureMip(bloom_down, mip)),
                (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
                (bloom::BLOOM_DST, Bind::TextureMip(bloom_down, mip + 1)),
            ],
        )?;
        downsamples.push(pipeline);
    }
    Ok(downsamples)
}

fn install_upsamples_first<P: GpuPart>(
    cx: &mut Install<'_, P>,
    pass: Pass<P>,
    bloom_down: Texture<P>,
    bloom_up: Texture<P>,
    sampler: engine_render::Sampler<P>,
) -> Result<Vec<ComputePipeline<P>>, Error> {
    use shaders::post::bloom;
    let mut upsamples = Vec::with_capacity(MAX_BLOOM_MIPS - 1);
    for dst_mip in 0..(MAX_BLOOM_MIPS - 1) as u32 {
        let pipeline = cx.compute_pipeline(
            pass,
            bloom::CS_BLOOM_UPSAMPLE,
            &[],
            &[
                (bloom::BLOOM_SRC, Bind::TextureMip(bloom_down, dst_mip + 1)),
                (bloom::BLOOM_ADD, Bind::TextureMip(bloom_down, dst_mip)),
                (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
                (bloom::BLOOM_DST, Bind::TextureMip(bloom_up, dst_mip)),
            ],
        )?;
        upsamples.push(pipeline);
    }
    Ok(upsamples)
}

fn install_upsamples_chain<P: GpuPart>(
    cx: &mut Install<'_, P>,
    pass: Pass<P>,
    bloom_down: Texture<P>,
    bloom_up: Texture<P>,
    sampler: engine_render::Sampler<P>,
) -> Result<Vec<ComputePipeline<P>>, Error> {
    use shaders::post::bloom;
    let mut upsamples = Vec::with_capacity(MAX_BLOOM_MIPS - 2);
    for dst_mip in 0..(MAX_BLOOM_MIPS - 2) as u32 {
        let pipeline = cx.compute_pipeline(
            pass,
            bloom::CS_BLOOM_UPSAMPLE,
            &[],
            &[
                (bloom::BLOOM_SRC, Bind::TextureMip(bloom_up, dst_mip + 1)),
                (bloom::BLOOM_ADD, Bind::TextureMip(bloom_down, dst_mip)),
                (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
                (bloom::BLOOM_DST, Bind::TextureMip(bloom_up, dst_mip)),
            ],
        )?;
        upsamples.push(pipeline);
    }
    Ok(upsamples)
}

fn install_composite<P: GpuPart>(
    cx: &mut Install<'_, P>,
    composite_pass: Pass<P>,
    uniform: Buffer<P>,
    hdr_color_target: TargetName,
    bloom_up: Texture<P>,
    sampler: engine_render::Sampler<P>,
) -> Result<Pipeline<P, Own>, Error> {
    use shaders::post::composite;
    cx.render_pipeline(
        composite_pass,
        &composite_desc(),
        &[
            (composite::POST_PARAMS, Bind::Buffer(uniform)),
            (composite::HDR_COLOR, Bind::Target(hdr_color_target)),
            (composite::BLOOM_TEX, Bind::TextureMip(bloom_up, 0)),
            (composite::POST_SAMPLER, Bind::Sampler(sampler)),
        ],
    )
}

impl<P: GpuPart> PostPipeline<P> {
    pub fn install(
        cx: &mut Install<'_, P>,
        compute_pass: Pass<P>,
        composite_pass: Pass<P>,
        hdr_color_target: TargetName,
        depth_target_opt: Option<TargetName>,
    ) -> Result<Self, Error> {
        let bloom_down = cx.texture("post_bloom_down", wgpu::TextureFormat::Rgba16Float, Lifetime::Kept);
        let bloom_up = cx.texture("post_bloom_up", wgpu::TextureFormat::Rgba16Float, Lifetime::Kept);
        let uniform = cx.buffer("post_params", Lifetime::Kept);
        let sampler = cx.sampler(LINEAR_CLAMP);
        let prefilter = install_prefilter(
            cx,
            compute_pass,
            uniform,
            hdr_color_target,
            depth_target_opt,
            sampler,
            bloom_down,
        )?;
        let downsamples = install_downsamples(cx, compute_pass, bloom_down, sampler)?;
        let upsamples_first =
            install_upsamples_first(cx, compute_pass, bloom_down, bloom_up, sampler)?;
        let upsamples_chain =
            install_upsamples_chain(cx, compute_pass, bloom_down, bloom_up, sampler)?;
        let composite = install_composite(
            cx,
            composite_pass,
            uniform,
            hdr_color_target,
            bloom_up,
            sampler,
        )?;
        Ok(Self {
            bloom_down,
            bloom_up,
            uniform,
            sampler,
            prefilter,
            downsamples,
            upsamples_first,
            upsamples_chain,
            composite,
            has_depth: depth_target_opt.is_some(),
            bloom_size: (0, 0),
            mip_count: 0,
            params: PostParams::cinematic(),
            uploaded: None,
        })
    }

    pub fn install_composite_pass(
        &self,
        cx: &mut Install<'_, P>,
        composite_pass: Pass<P>,
        hdr_color_target: TargetName,
    ) -> Result<Pipeline<P, Own>, Error> {
        install_composite(
            cx,
            composite_pass,
            self.uniform,
            hdr_color_target,
            self.bloom_up,
            self.sampler,
        )
    }

    pub fn prepare(&mut self, cx: &mut Prepare<'_, P>, params: PostParams) -> Allocation {
        let size = cx.canvas_size();
        self.prepare_sized(cx, size, params)
    }

    pub fn prepare_sized(
        &mut self,
        cx: &mut Prepare<'_, P>,
        src_size: (u32, u32),
        mut params: PostParams,
    ) -> Allocation {
        let (mip0_w, mip0_h) = bloom_mip0_size(src_size.0, src_size.1);
        let mip_count = bloom_mip_count(mip0_w, mip0_h);
        let extent = wgpu::Extent3d {
            width: mip0_w,
            height: mip0_h,
            depth_or_array_layers: 1,
        };
        let down_alloc = cx.allocate_mips(self.bloom_down, extent, mip_count);
        let up_alloc = cx.allocate_mips(self.bloom_up, extent, mip_count);
        if !self.has_depth {
            params.shaft_intensity = 0.0;
            params.flags |= POST_FLAG_DISABLE_SHAFTS;
        }
        let buf_alloc = cx.allocate_buffer(self.uniform, size_of::<PostParams>() as u64);
        if buf_alloc == Allocation::Fresh || self.uploaded != Some(params) {
            cx.write(self.uniform, 0, &[params]);
            self.uploaded = Some(params);
        }
        self.bloom_size = (mip0_w, mip0_h);
        self.mip_count = mip_count;
        self.params = params;
        if down_alloc == Allocation::Fresh || up_alloc == Allocation::Fresh {
            Allocation::Fresh
        } else {
            Allocation::Kept
        }
    }

    pub fn encode_bloom(&self, cx: &mut Encode<'_, P>) {
        let (mip0_w, mip0_h) = self.bloom_size;
        if mip0_w == 0 || mip0_h == 0 || self.mip_count < 2 || !self.bloom_pass_active() {
            return;
        }
        cx.dispatch(
            &self.prefilter,
            [
                mip0_w.div_ceil(WORKGROUP_SIZE),
                mip0_h.div_ceil(WORKGROUP_SIZE),
                1,
            ],
        );
        let steps = (self.mip_count.saturating_sub(1) as usize).min(self.downsamples.len());
        for (mip_idx, pipeline) in self.downsamples.iter().take(steps).enumerate() {
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
        if steps == 0 {
            return;
        }
        let last_dst = steps - 1;
        let (first_w, first_h) = mip_extent(mip0_w, mip0_h, last_dst as u32);
        cx.dispatch(
            &self.upsamples_first[last_dst],
            [
                first_w.div_ceil(WORKGROUP_SIZE),
                first_h.div_ceil(WORKGROUP_SIZE),
                1,
            ],
        );
        for dst_mip in (0..last_dst).rev() {
            let (dst_w, dst_h) = mip_extent(mip0_w, mip0_h, dst_mip as u32);
            cx.dispatch(
                &self.upsamples_chain[dst_mip],
                [
                    dst_w.div_ceil(WORKGROUP_SIZE),
                    dst_h.div_ceil(WORKGROUP_SIZE),
                    1,
                ],
            );
        }
    }

    pub fn encode_composite(&self, draw: &mut Draw<'_, P, Own>) -> Recorded {
        Self::encode_composite_with(draw, &self.composite)
    }

    pub fn encode_composite_with(
        draw: &mut Draw<'_, P, Own>,
        pipeline: &Pipeline<P, Own>,
    ) -> Recorded {
        draw.draw(
            pipeline,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
                key: None,
            },
        )
    }

    fn bloom_pass_active(&self) -> bool {
        let bloom_on = (self.params.flags & POST_FLAG_DISABLE_BLOOM) == 0
            && self.params.bloom_intensity > 0.0;
        let shafts_on = self.has_depth
            && (self.params.flags & POST_FLAG_DISABLE_SHAFTS) == 0
            && self.params.shaft_intensity > 0.0
            && self.params.sun_visibility > 0.0;
        bloom_on || shafts_on
    }

    #[must_use]
    pub fn bloom_texture(&self) -> Texture<P> {
        self.bloom_up
    }

    #[must_use]
    pub fn uniform(&self) -> Buffer<P> {
        self.uniform
    }

    #[must_use]
    pub fn bloom_size(&self) -> (u32, u32) {
        self.bloom_size
    }

    #[must_use]
    pub fn mip_count(&self) -> u32 {
        self.mip_count
    }

    #[must_use]
    pub fn params(&self) -> PostParams {
        self.params
    }
}
