use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, Cursor, DepthBias, DeviceLimits, Entry, Error, MaterialHandle, PassRole,
    ProvidedAssets, Views,
};
use engine_open_world_kit::ground::{GroundTiles, TerrainTiles};
use engine_open_world_kit::water::Waterline;
use engine_places_rendering::probes::ProbeCaptures;
use engine_places_rendering::shadows::{CasterMap, KeptRefresh, Refresh, ShadowMaps};
use engine_render::{
    Allocation, Attach, AttachPoint, Bind, Buffer, ComputePipeline, Count, Draw, DrawInput, Encode,
    FaceCull, Form, FormColor, FormDepth, FormFragment, FormSide, FragmentPass, FragmentUse,
    GpuPart, Install, KindForm, KindSurface, Lifetime, Own, OwnFragments, Pass, PassFragment,
    PerPass, Pipeline, Prepare, RoleFragments, Sampler, Scene, SceneDraw, Shape, Texture, Varyings,
    VertexLayout, wgpu,
};
use glam::{Mat4, Vec3};

use super::atlas::TileAtlas;
use super::patches::{PATCH_STREAM, PatchDraws, view_uniforms};
use super::surface::SurfaceTrees;
use super::tables::TerrainTables;
use super::{IDS_FORMAT, NEVER_BLENDS, SURFACE_SLOTS, TERRAIN, TerrainFrame, reached};
use crate::gpu::{HEIGHT_FORMAT, SUMMARY_FORMAT, SUMMARY_MIPS, SURFACE_ERROR_MIPS};
use crate::holes::HoledGround;
use crate::settings::TerrainShading;
use crate::state::{
    Atlas, AtlasMoves, Bakes, DrawnStats, PART_STATS, PartStats, SURFACE_TOTALS, SurfaceTotals,
};
use crate::terrain_bounce::{Bounce, BounceSources};
use crate::terrain_horizons::Horizons;
use crate::terrain_macro::MacroMaps;
use crate::terrain_maps::TerrainMaps;
use crate::terrain_rough::RoughLights;
use crate::terrain_sky::SkyMaps;
use crate::terrain_sun::{SunGpu, SunPlan};

pub(crate) struct TerrainViews<'a> {
    pub(crate) camera: Mat4,
    pub(crate) culling: Mat4,
    pub(crate) eye: Vec3,
    pub(crate) regions: Vec<(usize, KeptRefresh)>,
    pub(crate) settings: &'a TerrainShading,
    pub(crate) water: &'a Waterline,
}

pub(crate) struct Handles {
    pub(crate) heights: Texture<TerrainGpu>,
    pub(crate) tile_maps: Texture<TerrainGpu>,
    pub(crate) rivers: Texture<TerrainGpu>,
    pub(crate) errors: Texture<TerrainGpu>,
    pub(crate) water_errors: Texture<TerrainGpu>,
    pub(crate) summaries: Texture<TerrainGpu>,
    pub(crate) ids: Texture<TerrainGpu>,
    pub(crate) tables: Buffer<TerrainGpu>,
    pub(crate) palette: Buffer<TerrainGpu>,
    pub(crate) leaves: Buffer<TerrainGpu>,
    pub(crate) nodes: Buffer<TerrainGpu>,
    pub(crate) watered: Buffer<TerrainGpu>,
    pub(crate) counters: Buffer<TerrainGpu>,
    pub(crate) args: Buffer<TerrainGpu>,
    pub(crate) meshlet: Buffer<TerrainGpu>,
    pub(crate) patches: Buffer<TerrainGpu>,
    pub(crate) grid: Buffer<TerrainGpu>,
    pub(crate) normal_jobs: Buffer<TerrainGpu>,
    pub(crate) control_jobs: Buffer<TerrainGpu>,
    pub(crate) peak_jobs: Buffer<TerrainGpu>,
    pub(crate) summary_jobs: Buffer<TerrainGpu>,
    pub(crate) control_ancestors: Buffer<TerrainGpu>,
    pub(crate) normal_ancestors: Buffer<TerrainGpu>,
    pub(crate) slots: [(Buffer<TerrainGpu>, Buffer<TerrainGpu>); SURFACE_SLOTS],
    pub(crate) control_sampler: Sampler<TerrainGpu>,
    pub(crate) linear: Sampler<TerrainGpu>,
}

struct Passes {
    normals: Pass<TerrainGpu>,
    control: Pass<TerrainGpu>,
    peaks: Pass<TerrainGpu>,
    ancestors: Pass<TerrainGpu>,
    summaries: Pass<TerrainGpu>,
    clear_counters: Pass<TerrainGpu>,
    begin: Pass<TerrainGpu>,
    copy_args: Pass<TerrainGpu>,
    subdivide: Pass<TerrainGpu>,
    bounce: Pass<TerrainGpu>,
    sun: Pass<TerrainGpu>,
    ids: Pass<TerrainGpu>,
    resolve: Pass<TerrainGpu>,
}

struct Pipelines {
    normals: ComputePipeline<TerrainGpu>,
    control: ComputePipeline<TerrainGpu>,
    peaks: [ComputePipeline<TerrainGpu>; SURFACE_ERROR_MIPS as usize],
    ancestors: [ComputePipeline<TerrainGpu>; 2],
    summaries: SummaryPipelines,
    begin: [ComputePipeline<TerrainGpu>; SURFACE_SLOTS],
    decide: [ComputePipeline<TerrainGpu>; SURFACE_SLOTS],
    apply: [ComputePipeline<TerrainGpu>; SURFACE_SLOTS],
}

struct Inputs {
    args: DrawInput<TerrainGpu, Own>,
    meshlet: DrawInput<TerrainGpu, Own>,
    subdivision_args: DrawInput<TerrainGpu, Own>,
}

const CASTER_STREAMS: &[VertexLayout] = &[PATCH_STREAM];

const CASTER_BIAS: DepthBias = DepthBias {
    constant: 1,
    slope: 2.5,
};

const TERRAIN_FORM: KindForm = KindForm {
    kind: TERRAIN,
    placed: FormSide {
        vertex: crate::shaders::terrain_shading::draw::VS_TERRAIN_SURFACE,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    },
    owned: Some(FormSide {
        vertex: crate::shaders::terrain_shading::shadow::VS_TERRAIN_SHADOW,
        streams: CASTER_STREAMS,
        depth_bias: CASTER_BIAS,
    }),
    varyings: Varyings::Object(FragmentUse::Masked),
};

const SEEN_VARYINGS: &str = "TerrainOutput";

const fn seen_surface(point: &'static str, shown: Entry) -> KindSurface {
    let roles = RoleFragments {
        shown: Some(shown),
        ..RoleFragments::NONE
    };
    KindSurface {
        module: crate::shaders::terrain_shading::draw::MODULE,
        varyings: SEEN_VARYINGS,
        point,
        own: OwnFragments {
            placed: roles,
            owned: roles,
            cull: FaceCull::Back,
        },
    }
}

const SEEN_SURFACE: KindSurface = seen_surface(
    "terrain_surface",
    crate::shaders::terrain_shading::debug::FS_DEBUG_TERRAIN,
);

const SEEN_HOLES_SURFACE: KindSurface = seen_surface(
    "terrain_surface_holes",
    crate::shaders::terrain_shading::debug::FS_DEBUG_TERRAIN_HOLES,
);

const fn seen_form(surface: &'static KindSurface) -> KindForm {
    KindForm {
        kind: TERRAIN,
        placed: FormSide {
            vertex: crate::shaders::terrain_shading::draw::VS_TERRAIN,
            streams: &[],
            depth_bias: FormSide::UNBIASED,
        },
        owned: None,
        varyings: Varyings::Surfaced(surface),
    }
}

const SEEN_FORM: KindForm = seen_form(&SEEN_SURFACE);
const SEEN_HOLES_FORM: KindForm = seen_form(&SEEN_HOLES_SURFACE);

const IDS_PASS: &str = "terrain_ids";
const RESOLVE_PASS: &str = "terrain_resolve";

const fn ids_fragment(entry: Entry) -> PassFragment {
    PassFragment {
        pass: FragmentPass::Own(IDS_PASS),
        side: None,
        fragment: FormFragment::Entry {
            entry,
            cull: FaceCull::Back,
            depth: FormDepth::Written,
            color: FormColor::Written,
        },
    }
}

const IDS: PassFragment = ids_fragment(crate::shaders::terrain_shading::draw::FS_TERRAIN_IDS);
const IDS_HOLES: PassFragment =
    ids_fragment(crate::shaders::terrain_shading::draw::FS_TERRAIN_IDS_HOLES);

const RESOLVE_FORM: KindForm = KindForm {
    kind: TERRAIN,
    placed: FormSide::pulled(crate::shaders::terrain_shading::draw::VS_TERRAIN_RESOLVE),
    owned: None,
    varyings: Varyings::Own(&OwnFragments::NONE),
};

const fn resolve_fragment(entry: Entry) -> PassFragment {
    PassFragment {
        pass: FragmentPass::Own(RESOLVE_PASS),
        side: None,
        fragment: FormFragment::Entry {
            entry,
            cull: FaceCull::None,
            depth: FormDepth::Overwritten,
            color: FormColor::Written,
        },
    }
}

const RESOLVE_POST: PassFragment =
    resolve_fragment(crate::shaders::terrain_shading::draw::FS_TERRAIN_RESOLVE_POST);
const RESOLVE_TEMPORAL: PassFragment =
    resolve_fragment(crate::shaders::terrain_shading::draw::FS_TERRAIN_RESOLVE_TEMPORAL);

const NO_FORM: &str = "terrain has no form in this pass";
const TERRAIN_TILES: &str = "the terrain tiles";
const HOLED_MATERIAL: &str = "the holed ground material";
const NOT_LAID_OUT: &str = "no terrain is laid out this frame";
const NO_PATCHES: &str = "no terrain patch is laid out for this view";

enum Joined {
    Ground {
        args: DrawInput<TerrainGpu, Scene>,
        meshlet: DrawInput<TerrainGpu, Scene>,
    },
    Seen {
        args: DrawInput<TerrainGpu, Scene>,
        meshlet: DrawInput<TerrainGpu, Scene>,
    },
    Casters {
        patches: DrawInput<TerrainGpu, Scene>,
        grid: DrawInput<TerrainGpu, Scene>,
    },
    Elsewhere(&'static str),
}

type Surface = (Pipeline<TerrainGpu, Scene>, Option<MaterialHandle>);

type Surfaces<'a> = [Result<Surface, Option<&'a Error>>; 2];

pub(crate) struct TerrainGpu {
    handles: Handles,
    passes: Passes,
    pipelines: Pipelines,
    inputs: Inputs,
    form: Form<TerrainGpu>,
    seen: [Form<TerrainGpu>; 2],
    resolves: [Form<TerrainGpu>; 2],
    holed: Result<Option<MaterialHandle>, Error>,
    joined: PerPass<Joined>,
    atlas: TileAtlas,
    tables: TerrainTables,
    surface: SurfaceTrees,
    patches: PatchDraws,
    horizons: Horizons,
    maps: TerrainMaps,
    sky: SkyMaps,
    bounce: Bounce,
    sun: SunGpu,
    macro_maps: MacroMaps,
    frame: TerrainFrame,
    cursor: Cursor,
    drawn: bool,
    tiles_waiting: bool,
    deferred: bool,
    temporal: bool,
    render: (u32, u32),
}

type Lists = (
    Res<'static, TerrainShading>,
    Res<'static, TerrainTiles>,
    Res<'static, Atlas>,
    Res<'static, Bakes>,
    Res<'static, GroundTiles>,
);

type Neighbors = (
    Res<'static, Waterline>,
    Res<'static, ShadowMaps>,
    Res<'static, ProbeCaptures>,
);

type Frame = Res<'static, Views>;

impl GpuPart for TerrainGpu {
    type Reads = (
        Lists,
        Neighbors,
        Frame,
        (
            Res<'static, DeviceLimits>,
            Res<'static, AssetLibrary>,
            Res<'static, HoledGround>,
            Option<Res<'static, ProvidedAssets>>,
            Res<'static, RoughLights>,
            Res<'static, SunPlan>,
        ),
    );
    const NAME: &'static str = "terrain";

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::terrain_shading::{
            ancestors, control, draw, faces, field, horizons, normals, peaks, sky, subdivision,
            surface, tables,
        };
        let handles = Handles {
            heights: cx.texture("heights", HEIGHT_FORMAT, Lifetime::Kept),
            tile_maps: cx.texture("tile_maps", super::TILE_MAPS_FORMAT, Lifetime::Kept),
            rivers: cx.texture("rivers", super::RIVER_FORMAT, Lifetime::Kept),
            errors: cx.texture("errors", wgpu::TextureFormat::R16Float, Lifetime::Kept),
            water_errors: cx.texture(
                "water_errors",
                wgpu::TextureFormat::R16Float,
                Lifetime::Kept,
            ),
            summaries: cx.texture("tile_summaries", SUMMARY_FORMAT, Lifetime::Kept),
            ids: cx.texture("terrain_ids", IDS_FORMAT, Lifetime::Frame),
            tables: cx.buffer("tables", Lifetime::Kept),
            palette: cx.buffer("palette", Lifetime::Kept),
            leaves: cx.buffer("leaves", Lifetime::Kept),
            nodes: cx.buffer("nodes", Lifetime::Kept),
            watered: cx.buffer("watered_leaves", Lifetime::Kept),
            counters: cx.buffer("counters", Lifetime::Kept),
            args: cx.buffer("surface_args", Lifetime::Kept),
            meshlet: cx.buffer("meshlet_indices", Lifetime::Kept),
            patches: cx.buffer("patches", Lifetime::Kept),
            grid: cx.buffer("grid_indices", Lifetime::Kept),
            normal_jobs: cx.buffer("normal_jobs", Lifetime::Kept),
            control_jobs: cx.buffer("control_jobs", Lifetime::Kept),
            peak_jobs: cx.buffer("peak_jobs", Lifetime::Kept),
            summary_jobs: cx.buffer("summary_jobs", Lifetime::Kept),
            control_ancestors: cx.buffer("control_ancestors", Lifetime::Kept),
            normal_ancestors: cx.buffer("normal_ancestors", Lifetime::Kept),
            slots: std::array::from_fn(|_| {
                (
                    cx.buffer("subdivision", Lifetime::Kept),
                    cx.buffer("subdivision_tree", Lifetime::Kept),
                )
            }),
            control_sampler: cx.sampler(engine_core::Sampler::NEAREST),
            linear: cx.sampler(engine_core::Sampler::LINEAR_CLAMP),
        };
        let h = &handles;
        let maps = TerrainMaps::install(cx)?;
        let sky_maps = SkyMaps::install(cx)?;
        let macro_maps = MacroMaps::install(cx)?;
        let horizon_maps = Horizons::install(cx)?;
        cx.frame(tables::TERRAIN_TABLES, Bind::Buffer(h.tables))?;
        cx.frame(
            faces::heightfield::TERRAIN_SURFACE_LEAVES,
            Bind::Buffer(h.leaves),
        )?;
        cx.frame(
            faces::heightfield::TERRAIN_SURFACE_ARGS,
            Bind::Buffer(h.counters),
        )?;
        cx.frame(
            faces::heightfield::TERRAIN_WATERED_LEAVES,
            Bind::Buffer(h.watered),
        )?;
        cx.frame(field::TERRAIN_HEIGHTS, Bind::Texture(h.heights))?;
        cx.frame(draw::TERRAIN_TILE_MAPS, Bind::Texture(h.tile_maps))?;
        cx.frame(
            draw::TERRAIN_CONTROL_SAMPLER,
            Bind::Sampler(h.control_sampler),
        )?;
        cx.frame(draw::TERRAIN_PALETTE, Bind::Buffer(h.palette))?;
        cx.frame(draw::TERRAIN_ALBEDO, Bind::Texture(maps.albedo))?;
        cx.frame(draw::TERRAIN_MATERIAL_DATA, Bind::Texture(maps.data))?;
        cx.frame(draw::TERRAIN_MATERIAL_SAMPLER, Bind::Sampler(maps.sampler))?;
        cx.frame(sky::TERRAIN_SKY, Bind::Texture(sky_maps.texture))?;
        cx.frame(sky::TERRAIN_SKY_SAMPLER, Bind::Sampler(h.linear))?;
        cx.frame(surface::TERRAIN_ERRORS, Bind::Texture(h.errors))?;
        cx.frame(surface::WATER_ERRORS, Bind::Texture(h.water_errors))?;
        cx.frame(
            horizons::TERRAIN_HORIZONS,
            Bind::Texture(horizon_maps.texture),
        )?;
        cx.frame(horizons::TERRAIN_HORIZON_SAMPLER, Bind::Sampler(h.linear))?;
        cx.frame(faces::ground_tiles::HEIGHTS, Bind::Texture(h.heights))?;
        cx.frame(faces::ground_tiles::TILE_MAPS, Bind::Texture(h.tile_maps))?;
        cx.frame(
            faces::ground_tiles::MACRO_MAPS,
            Bind::Texture(macro_maps.texture),
        )?;
        cx.frame(faces::ground_tiles::PALETTE, Bind::Buffer(h.palette))?;
        cx.frame(
            faces::ground_tiles::GROUND_SKY_MAP,
            Bind::Texture(sky_maps.texture),
        )?;
        cx.frame(
            faces::ground_tiles::GROUND_SKY_SAMPLER,
            Bind::Sampler(h.linear),
        )?;

        let normals_pass = cx.pass("tile_normals").label("engine/terrain-normals").id();
        let control_pass = cx
            .pass("tile_control")
            .label("engine/terrain-control")
            .after(normals_pass)
            .id();
        let peaks_pass = cx.pass("tile_peaks").label("engine/terrain-peaks").id();
        let ancestors_pass = cx
            .pass("tile_ancestors")
            .label("engine/terrain-ancestors")
            .after(control_pass)
            .after(normals_pass)
            .id();
        let (summaries_pass, summary_pipelines) =
            install_summaries(cx, h, [control_pass, ancestors_pass])?;
        let clear_counters = cx.clear("clear_surface_counters", h.counters);
        let begin = cx
            .pass("subdivision_begin")
            .label("engine/terrain-subdivision-arguments")
            .after(clear_counters)
            .after(control_pass)
            .after(normals_pass)
            .after(peaks_pass)
            .after(ancestors_pass)
            .id();
        let copy_args = cx.copy(
            "subdivision_args",
            engine_render::Copied::Buffer(h.counters),
            engine_render::Copied::Buffer(h.args),
        );
        let subdivide = cx
            .pass("terrain_subdivision")
            .label("engine/terrain-subdivision")
            .span("terrain_subdivision")
            .after(copy_args)
            .after(begin)
            .id();
        let bounce_pass = cx
            .pass("bounce")
            .label("engine/terrain-bounce")
            .span("bounce")
            .id();
        let sun_pass = cx
            .pass("terrain_sun")
            .label("engine/terrain-sun")
            .span("terrain_sun")
            .after(peaks_pass)
            .after(bounce_pass)
            .id();
        let ids_pass = cx
            .pass(IDS_PASS)
            .label("engine/terrain-ids")
            .at(AttachPoint::Opaque)
            .color(h.ids, engine_render::Access::Clear)
            .id();
        let resolve_pass = cx
            .pass(RESOLVE_PASS)
            .label("engine/terrain-resolve")
            .at(AttachPoint::Opaque)
            .after(ids_pass)
            .id();

        let normals_pipeline = cx.compute_pipeline(
            normals_pass,
            normals::TILE_NORMALS,
            &[
                (normals::NORMAL_JOBS, Bind::Buffer(h.normal_jobs)),
                (normals::NORMALS, Bind::Texture(h.tile_maps)),
            ],
        )?;
        let control_pipeline = cx.compute_pipeline(
            control_pass,
            control::TILE_CONTROL,
            &[
                (control::CONTROL_JOBS, Bind::Buffer(h.control_jobs)),
                (control::CONTROL, Bind::Texture(h.tile_maps)),
                (control::CONTROL_RIVERS, Bind::Texture(h.rivers)),
            ],
        )?;
        let height_mip = |mip: u32| Bind::Slice {
            texture: h.heights,
            mips: mip..mip + 1,
            layers: 0..u32::MAX,
        };
        let mut peaks_pipelines = Vec::with_capacity(SURFACE_ERROR_MIPS as usize);
        for mip in 0..SURFACE_ERROR_MIPS {
            let entry = if mip == 0 {
                peaks::TILE_PEAKS_FIRST
            } else {
                peaks::TILE_PEAKS_DOWN
            };
            peaks_pipelines.push(cx.compute_pipeline(
                peaks_pass,
                entry,
                &[
                    (peaks::PEAK_LAYERS, Bind::Buffer(h.peak_jobs)),
                    (peaks::PEAKS_IN, height_mip(mip)),
                    (peaks::PEAKS_OUT, height_mip(mip + 1)),
                ],
            )?);
        }
        let ancestors_pipelines = [
            cx.compute_pipeline(
                ancestors_pass,
                ancestors::TILE_ANCESTORS,
                &[
                    (ancestors::ANCESTOR_JOBS, Bind::Buffer(h.control_ancestors)),
                    (ancestors::ANCESTOR_TEXELS, Bind::Texture(h.tile_maps)),
                ],
            )?,
            cx.compute_pipeline(
                ancestors_pass,
                ancestors::TILE_ANCESTORS,
                &[
                    (ancestors::ANCESTOR_JOBS, Bind::Buffer(h.normal_ancestors)),
                    (ancestors::ANCESTOR_TEXELS, Bind::Texture(h.tile_maps)),
                ],
            )?,
        ];
        let slot_binds = |slot: usize| {
            let (uniform, tree) = h.slots[slot];
            [
                (subdivision::SUBDIVISION, Bind::Buffer(uniform)),
                (subdivision::SUBDIVISION_TREE, Bind::Buffer(tree)),
                (subdivision::SUBDIVISION_NODES, Bind::Buffer(h.nodes)),
                (subdivision::SUBDIVISION_LEAVES, Bind::Buffer(h.leaves)),
                (subdivision::SUBDIVISION_ARGS, Bind::Buffer(h.counters)),
                (subdivision::SUBDIVISION_WATERED, Bind::Buffer(h.watered)),
            ]
        };
        let mut begins = Vec::with_capacity(SURFACE_SLOTS);
        let mut decides = Vec::with_capacity(SURFACE_SLOTS);
        let mut applies = Vec::with_capacity(SURFACE_SLOTS);
        for slot in 0..SURFACE_SLOTS {
            let bound = reached(cx, subdivision::SUBDIVIDE_BEGIN, slot_binds(slot));
            begins.push(cx.compute_pipeline(begin, subdivision::SUBDIVIDE_BEGIN, &bound)?);
            let bound = reached(cx, subdivision::SUBDIVIDE_DECIDE, slot_binds(slot));
            decides.push(cx.compute_pipeline(subdivide, subdivision::SUBDIVIDE_DECIDE, &bound)?);
            let bound = reached(cx, subdivision::SUBDIVIDE_APPLY, slot_binds(slot));
            applies.push(cx.compute_pipeline(subdivide, subdivision::SUBDIVIDE_APPLY, &bound)?);
        }
        let resolve_bound = [(draw::TERRAIN_IDS_TEXTURE, Bind::Texture(h.ids))];
        let resolves = [
            cx.form_owning(&RESOLVE_FORM, &[RESOLVE_POST], &resolve_bound)?,
            cx.form_owning(&RESOLVE_FORM, &[RESOLVE_TEMPORAL], &resolve_bound)?,
        ];
        let form = cx.form_for(&TERRAIN_FORM, &[PassRole::Depth], &[])?;
        let seen = [
            cx.form_owning(&SEEN_FORM, &[IDS], &[])?,
            cx.form_owning(&SEEN_HOLES_FORM, &[IDS_HOLES], &[])?,
        ];
        let joined = PerPass::of(form.passes(), |pass| match (pass.role, pass.owned) {
            (PassRole::Depth, true) => Joined::Casters {
                patches: cx.draw_input(pass.scene, h.patches),
                grid: cx.draw_input(pass.scene, h.grid),
            },
            (PassRole::Depth, false) => Joined::Ground {
                args: cx.draw_input(pass.scene, h.args),
                meshlet: cx.draw_input(pass.scene, h.meshlet),
            },
            (PassRole::Lit | PassRole::Shown, _) => Joined::Seen {
                args: cx.draw_input(pass.scene, h.args),
                meshlet: cx.draw_input(pass.scene, h.meshlet),
            },
            (PassRole::Blended, _) => Joined::Elsewhere(NEVER_BLENDS),
        });
        let inputs = Inputs {
            args: cx.draw_input(ids_pass, h.args),
            meshlet: cx.draw_input(ids_pass, h.meshlet),
            subdivision_args: cx.draw_input(subdivide, h.args),
        };
        let sources = BounceSources {
            horizons: &horizon_maps,
            sky: &sky_maps,
            colors: &macro_maps,
            palette: h.palette,
        };
        let bounce = Bounce::install(cx, &sources, bounce_pass)?;
        let sun = SunGpu::install(cx, &horizon_maps, sun_pass)?;
        let to_array = |list: Vec<ComputePipeline<TerrainGpu>>| -> [ComputePipeline<TerrainGpu>; SURFACE_SLOTS] {
            std::array::from_fn(|slot| list[slot])
        };
        Ok(Self {
            passes: Passes {
                normals: normals_pass,
                control: control_pass,
                peaks: peaks_pass,
                ancestors: ancestors_pass,
                summaries: summaries_pass,
                clear_counters,
                begin,
                copy_args,
                subdivide,
                bounce: bounce_pass,
                sun: sun_pass,
                ids: ids_pass,
                resolve: resolve_pass,
            },
            pipelines: Pipelines {
                normals: normals_pipeline,
                control: control_pipeline,
                peaks: std::array::from_fn(|mip| peaks_pipelines[mip]),
                ancestors: ancestors_pipelines,
                summaries: summary_pipelines,
                begin: to_array(begins),
                decide: to_array(decides),
                apply: to_array(applies),
            },
            inputs,
            form,
            seen,
            resolves,
            holed: Ok(None),
            joined,
            atlas: TileAtlas::default(),
            tables: TerrainTables::default(),
            surface: SurfaceTrees::default(),
            patches: PatchDraws::default(),
            horizons: horizon_maps,
            maps,
            sky: sky_maps,
            bounce,
            sun,
            macro_maps,
            frame: TerrainFrame::default(),
            cursor: Cursor::default(),
            drawn: false,
            tiles_waiting: false,
            deferred: false,
            temporal: false,
            render: (0, 0),
            handles,
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn prepare(
        &mut self,
        (lists, neighbors, frame, (limits, assets, holed, provided, lights, sun_plan)): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.holed.clone_from(&holed.held);
        let (settings, tiles, atlas, bakes, ground) = lists;
        let (water, shadows, captures) = neighbors;
        let main = frame.main();
        self.temporal = main.is_some_and(|main| main.jitter.is_some());
        let size = main.map_or(glam::UVec2::ZERO, |main| main.render);
        self.render = (size.x, size.y);
        let moves = AtlasMoves::since(&atlas, &mut self.cursor);
        let Some(main) = main.filter(|_| !tiles.terrains.is_empty() || !atlas.slots.is_empty())
        else {
            self.atlas.rest();
            self.tables = TerrainTables::default();
            self.surface.rest();
            self.horizons.rest();
            self.sun.rest();
            self.drawn = false;
            self.tiles_waiting = false;
            self.deferred = false;
            cx.release(self.handles.ids);
            cx.deliver::<PartStats>(
                PART_STATS,
                bytemuck::bytes_of(&DrawnStats::default()).to_vec(),
            );
            cx.deliver::<SurfaceTotals>(SURFACE_TOTALS, Vec::new());
            return;
        };
        let h = &self.handles;
        let eye = main.eye;
        let culling = main.view_projection * Mat4::from_translation(eye);
        let jitter = main.jitter.map_or([0.0; 2], |offset| {
            engine_render::jitter_clip(offset.to_array(), size.x, size.y)
        });
        let regions: Vec<(usize, KeptRefresh)> = shadows
            .views()
            .filter(|at| at.map == CasterMap::Sky)
            .filter_map(|at| match at.refresh {
                Refresh::Kept(kept) => Some((at.view() as usize, kept)),
                Refresh::Redraw => None,
            })
            .collect();
        let views = TerrainViews {
            camera: jittered(culling, jitter),
            culling,
            eye,
            regions,
            settings: &settings,
            water: &water,
        };
        let watered = water.is_changed();
        let p = &self.passes;
        let ran = [
            cx.ran(p.normals),
            cx.ran(p.control),
            cx.ran(p.peaks),
            cx.ran(p.ancestors),
            cx.ran(p.summaries),
        ];
        let subdivided = cx.ran(p.subdivide);
        self.atlas.upload(
            cx,
            h,
            (&tiles, &atlas, &bakes),
            (&moves, ran),
            (&water, watered),
        );
        if self.atlas.normal_jobs() > 0 {
            cx.run(self.passes.normals);
        }
        if self.atlas.control_jobs() > 0 {
            cx.run(self.passes.control);
        }
        if self.atlas.peak_jobs() > 0 {
            cx.run(self.passes.peaks);
        }
        if self.atlas.ancestor_jobs() > 0 {
            cx.run(self.passes.ancestors);
        }
        if self.atlas.summary_jobs() > 0 {
            cx.run(self.passes.summaries);
        }
        let view_list = view_uniforms(&views, &captures);
        self.tables.prepare(
            cx,
            h,
            (&tiles, &atlas, &moves, tiles.is_changed()),
            view_list.len(),
        );
        self.tables.write_header(cx, h, (atlas.now, &settings.look()));
        self.tables.write_views(cx, h, &view_list);
        let unprovided = ProvidedAssets::default();
        let provided = provided.as_deref().unwrap_or(&unprovided);
        self.maps.prepare(cx, (&tiles, &assets, provided));
        self.tables.write_palette(
            cx,
            h,
            (&tiles, &ground.palette),
            (&self.maps, &assets, &lights),
        );
        self.sky
            .prepare(cx, (&tiles, &bakes, &settings, limits.texture_layers));
        self.macro_maps
            .prepare(cx, (&tiles, &bakes, &assets, limits.texture_layers));
        let slots = crate::terrain_bounce::relit_slots(&crate::terrain_bounce::slots(
            &tiles.terrains,
            &settings,
            limits.texture_layers,
        ));
        self.horizons.prepare(
            cx,
            (&tiles, &bakes, &settings, limits.texture_layers),
            slots,
        );
        self.sun
            .prepare(cx, (&sun_plan, &settings, &tiles, &atlas), &self.horizons);
        let relit = self.bounce.prepare(
            cx,
            (
                &tiles,
                &bakes,
                &settings,
                main.detail_eye,
                (limits.texture_layers, limits.storage_room),
            ),
            (
                &self.horizons,
                &self.sky,
                &self.macro_maps,
                &self.tables.palette_bases[..],
            ),
        );
        if relit {
            cx.run(self.passes.bounce);
        }
        let held = tiles.terrains.iter().any(|entry| {
            atlas
                .slots
                .keys()
                .any(|(terrain, _)| *terrain == entry.terrain)
        });
        self.drawn = held;
        self.tiles_waiting = !held && !tiles.terrains.is_empty();
        if !held {
            self.surface.rest();
            self.frame = TerrainFrame::default();
            self.deferred = false;
            cx.release(h.ids);
            cx.deliver::<SurfaceTotals>(SURFACE_TOTALS, Vec::new());
            self.deliver_stats(cx);
            return;
        }
        let drawn: Vec<bool> = tiles
            .terrains
            .iter()
            .map(|entry| {
                atlas
                    .slots
                    .keys()
                    .any(|(terrain, _)| *terrain == entry.terrain)
            })
            .collect();
        self.surface.prepare(
            cx,
            h,
            (&tiles, &views, limits.storage_room, &drawn),
            subdivided,
        );
        self.frame = self.patches.lay_out(
            cx,
            h,
            (
                &tiles,
                &atlas,
                &self.tables,
                &self.surface,
                &self.sky,
                &self.macro_maps,
            ),
            &views,
        );
        self.tables.write_terrains(cx, h, self.patches.uniforms());
        if self.surface.any() {
            cx.run(self.passes.clear_counters);
            cx.run(self.passes.begin);
            let bytes = self.surface.counter_bytes().end;
            cx.copy(
                self.passes.copy_args,
                &[engine_render::CopyRegion::Bytes {
                    from: 0,
                    to: 0,
                    size: bytes,
                }],
            );
            cx.run(self.passes.subdivide);
            cx.read_back::<SurfaceTotals>(
                SURFACE_TOTALS,
                h.counters,
                self.surface.counter_bytes(),
                engine_render::AfterRead::Keep,
            );
        }
        self.deferred = !self.frame.terrains.is_empty();
        if self.deferred {
            let display = frame.frame().display_size;
            let size = wgpu::Extent3d {
                width: display.x,
                height: display.y,
                depth_or_array_layers: 1,
            };
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate(h.ids, size, 1);
            cx.run(self.passes.ids);
            cx.run(self.passes.resolve);
        } else {
            cx.release(h.ids);
        }
        self.deliver_stats(cx);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.passes.normals {
            return cx.dispatch(
                &self.pipelines.normals,
                Count::Groups(self.atlas.normal_groups()),
            );
        }
        if pass == self.passes.control {
            return cx.dispatch(
                &self.pipelines.control,
                Count::Groups(self.atlas.control_groups()),
            );
        }
        if pass == self.passes.peaks {
            return self.atlas.peaks(cx, &self.pipelines.peaks);
        }
        if pass == self.passes.ancestors {
            return self.atlas.ancestors(cx, &self.pipelines.ancestors);
        }
        if pass == self.passes.summaries {
            return self.atlas.summaries(cx, &self.pipelines.summaries);
        }
        if pass == self.passes.begin {
            let mut dispatched = 0;
            for slot in self.surface.slots() {
                dispatched += cx.dispatch(&self.pipelines.begin[slot], Count::Groups([1, 1, 1]));
            }
            return dispatched;
        }
        if pass == self.passes.subdivide {
            let mut dispatched = 0;
            for slot in self.surface.slots() {
                let offset = self.surface.args_offset(slot);
                dispatched += cx.dispatch(
                    &self.pipelines.decide[slot],
                    Count::Indirect(self.inputs.subdivision_args, offset),
                );
                dispatched += cx.dispatch(
                    &self.pipelines.apply[slot],
                    Count::Indirect(self.inputs.subdivision_args, offset),
                );
            }
            return dispatched;
        }
        if pass == self.passes.bounce {
            return self.bounce.encode(cx);
        }
        if pass == self.passes.sun {
            return self.sun.encode(cx);
        }
        if pass == self.passes.ids {
            let mut draw = cx.render(Attach::Declared);
            return self.draw_ids(&mut draw);
        }
        if pass == self.passes.resolve {
            let mut draw = cx.render(Attach::Declared);
            return self.draw_resolve(&mut draw);
        }
        0
    }
}

impl TerrainGpu {
    fn deliver_stats(&self, cx: &mut Prepare<'_, Self>) {
        let wanted = self.surface.slots().count() as u32 + self.surface.dropped();
        let stats = DrawnStats {
            shallow_subdivisions: self.surface.shallow(),
            shadow_patches: self.frame.shadow_patches,
            gpu_controls: self.atlas.controls,
            horizons: self.horizons.count(),
            sky_maps: self.sky.count(),
            lighting_uploads: self.horizons.uploaded + self.sky.uploaded,
            bounce_relit: self.bounce.relit,
            bounce_probes: self.bounce.relit_probes,
            bounce_levels: self.bounce.levels,
            bounce_dropped: self.bounce.dropped,
            macro_maps: self.macro_maps.count(),
            dropped_materials: self.tables.dropped_materials,
            surface_wanted: wanted,
            surface_kept: self.surface.slots().count() as u32,
        };
        cx.deliver::<PartStats>(PART_STATS, bytemuck::bytes_of(&stats).to_vec());
    }

    fn draw_ids(&self, draw: &mut Draw<'_, Self, Own>) -> u32 {
        let mut draws = 0;
        for holes in [false, true] {
            let Some(pipeline) = draw.form(&self.seen[usize::from(holes)]) else {
                continue;
            };
            for terrain in self
                .frame
                .terrains
                .iter()
                .filter(|terrain| terrain.holes == holes)
            {
                let Some(offset) = self.surface.draw_offset(terrain.terrain) else {
                    continue;
                };
                draw.draw(
                    &pipeline,
                    Shape::Indirect {
                        vertices: &[],
                        indices: Some(self.inputs.meshlet),
                        args: self.inputs.args,
                        offset,
                        material: None,
                    },
                );
                draws += 1;
            }
        }
        draws
    }

    fn draw_resolve(&self, draw: &mut Draw<'_, Self, Own>) -> u32 {
        let Some(pipeline) = draw.form(&self.resolves[usize::from(self.temporal)]) else {
            return 0;
        };
        let mut draws = 0;
        for terrain in &self.frame.terrains {
            let Some(rect) = terrain.scissor(self.render) else {
                continue;
            };
            draw.scissor(rect);
            let index = terrain.terrain as u32;
            draw.draw(
                &pipeline,
                Shape::Pulled {
                    vertices: 0..3,
                    instances: index..index + 1,
                    material: None,
                },
            );
            draws += 1;
        }
        draw.scissor([0, 0, self.render.0, self.render.1]);
        draws
    }

    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        let Some(joined) = draw.per_pass(&self.joined) else {
            return draw.skip(NO_FORM);
        };
        match joined {
            Joined::Ground { args, meshlet } => {
                let Some(pipeline) = draw.form(&self.form) else {
                    return draw.skip(NO_FORM);
                };
                let holed = match &self.holed {
                    Ok(Some(material)) => Ok((pipeline, Some(*material))),
                    Ok(None) => Err(None),
                    Err(error) => Err(Some(error)),
                };
                self.draw_surface(draw, [Ok((pipeline, None)), holed], (*args, *meshlet))
            }
            Joined::Seen { args, meshlet } => {
                if !self.drawn {
                    return self.unlaid(draw);
                }
                let [plain, holes] = &self.seen;
                let (Some(plain), Some(holes)) = (draw.form(plain), draw.form(holes)) else {
                    return draw.skip(NO_FORM);
                };
                self.draw_surface(
                    draw,
                    [Ok((plain, None)), Ok((holes, None))],
                    (*args, *meshlet),
                )
            }
            Joined::Casters { patches, grid } => {
                if !self.drawn {
                    return self.unlaid(draw);
                }
                let Some(range) = scene
                    .view
                    .and_then(|region| self.frame.map_records(region as usize))
                else {
                    return draw.skip(NO_PATCHES);
                };
                let Some(pipeline) = draw.form(&self.form) else {
                    return draw.skip(NO_FORM);
                };
                draw.draw(
                    &pipeline,
                    Shape::Buffers {
                        vertices: &[*patches],
                        indices: Some(*grid),
                        elements: 0..self.patches.index_count(),
                        instances: range,
                    },
                );
                1
            }
            Joined::Elsewhere(reason) => draw.skip(reason),
        }
    }

    fn unlaid(&self, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if self.tiles_waiting {
            draw.wait(TERRAIN_TILES)
        } else {
            draw.skip(NOT_LAID_OUT)
        }
    }

    fn draw_surface(
        &self,
        draw: &mut Draw<'_, Self, Scene>,
        surfaces: Surfaces<'_>,
        (args, meshlet): (DrawInput<Self, Scene>, DrawInput<Self, Scene>),
    ) -> u32 {
        let mut draws = 0;
        let mut unheld = None;
        for holes in [false, true] {
            let drawn = surfaces[usize::from(holes)];
            for terrain in self
                .frame
                .terrains
                .iter()
                .filter(|terrain| terrain.holes == holes)
            {
                let Some(offset) = self.surface.draw_offset(terrain.terrain) else {
                    continue;
                };
                let (pipeline, material) = match drawn {
                    Ok(drawn) => drawn,
                    Err(reason) => {
                        unheld = Some(reason);
                        continue;
                    }
                };
                draw.draw(
                    &pipeline,
                    Shape::Indirect {
                        vertices: &[],
                        indices: Some(meshlet),
                        args,
                        offset,
                        material,
                    },
                );
                draws += 1;
            }
        }
        match unheld {
            Some(None) => {
                draw.wait(HOLED_MATERIAL);
            }
            Some(Some(error)) => {
                draw.refuse(HOLED_MATERIAL, error);
            }
            None if draws == 0 => return draw.skip(NOT_LAID_OUT),
            None => {}
        }
        draws
    }
}

type SummaryPipelines = [ComputePipeline<TerrainGpu>; SUMMARY_MIPS as usize];

fn install_summaries(
    cx: &mut Install<'_, TerrainGpu>,
    h: &Handles,
    [control, ancestors]: [Pass<TerrainGpu>; 2],
) -> Result<(Pass<TerrainGpu>, SummaryPipelines), Error> {
    use crate::shaders::terrain_shading::{faces, summary};
    cx.frame(
        faces::ground_tiles::TILE_SUMMARIES,
        Bind::Texture(h.summaries),
    )?;
    let pass = cx
        .pass("tile_summaries")
        .label("engine/terrain-summaries")
        .after(control)
        .after(ancestors)
        .id();
    let level = |texture: Texture<TerrainGpu>, mip: u32| Bind::Slice {
        texture,
        mips: mip..mip + 1,
        layers: 0..u32::MAX,
    };
    let mut pipelines = Vec::with_capacity(SUMMARY_MIPS as usize);
    pipelines.push(cx.compute_pipeline(
        pass,
        summary::TILE_SUMMARY_FIRST,
        &[
            (summary::SUMMARY_LAYERS, Bind::Buffer(h.summary_jobs)),
            (summary::SUMMARY_IN, Bind::Texture(h.tile_maps)),
            (summary::SUMMARY_OUT, level(h.summaries, 0)),
            (summary::SUMMARY_HEIGHTS, level(h.heights, 0)),
        ],
    )?);
    for mip in 1..SUMMARY_MIPS {
        pipelines.push(cx.compute_pipeline(
            pass,
            summary::TILE_SUMMARY_DOWN,
            &[
                (summary::SUMMARY_LAYERS, Bind::Buffer(h.summary_jobs)),
                (summary::SUMMARY_IN, level(h.summaries, mip - 1)),
                (summary::SUMMARY_OUT, level(h.summaries, mip)),
            ],
        )?);
    }
    Ok((pass, std::array::from_fn(|mip| pipelines[mip])))
}

fn jittered(view_projection: Mat4, jitter: [f32; 2]) -> Mat4 {
    Mat4::from_cols(
        glam::Vec4::X,
        glam::Vec4::Y,
        glam::Vec4::Z,
        glam::Vec4::new(jitter[0], jitter[1], 0.0, 1.0),
    ) * view_projection
}
