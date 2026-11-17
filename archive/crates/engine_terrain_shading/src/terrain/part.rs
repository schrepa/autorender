use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, Cursor, DepthBias, Detail, DeviceLimits, Entry, Error, MaterialHandle, PassRole,
    PassViews, ProvidedAssets, Views,
};
use engine_open_world_kit::ground::{GroundTiles, TerrainTiles};
use engine_open_world_kit::water::Waterline;
use engine_places_rendering::probes::ProbeCaptures;
use engine_places_rendering::shadows::{CasterMap, KeptRefresh, Refresh, ShadowMaps};
use engine_render::{
    Allocation, Attach, AttachPoint, Bind, Buffer, ComputePipeline, Count, Draw, DrawInput, Encode,
    FaceCull, Form, FormColor, FormDepth, FormFragment, FormSide, FragmentPass, FragmentUse,
    GpuPart, Install, KindForm, KindSurface, Lifetime, Own, OwnFragments, Pass, PassFragment,
    PerPass, Pipeline, Prepare, Recorded, RoleFragments, Sampler, Scene, SceneDraw, Shape, Texture,
    Varyings, VertexLayout, wgpu,
};
use glam::{Mat4, Vec3};

use super::atlas::TileAtlas;
use super::patches::{PATCH_STREAM, PatchDraws, view_uniforms};
use super::surface::SurfaceTrees;
use super::tables::TerrainTables;
use super::{
    FrameTerrain, IDS_FORMAT, NEVER_BLENDS, SURFACE_SLOTS, TERRAIN, TerrainFrame, reached,
};
use crate::gpu::{HEIGHT_FORMAT, SUMMARY_FORMAT, SUMMARY_MIPS, SURFACE_ERROR_MIPS};
use crate::holes::HoledGround;
use crate::settings::TerrainShading;
use crate::state::{
    Atlas, AtlasMoves, Bakes, DERIVED_LAYERS, Derived, DrawnStats, GROUND_LOOKS, PART_STATS,
    PartStats, SURFACE_TOTALS, SurfaceTotals,
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
    pub(crate) temporal: bool,
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
    pub(crate) lists: Buffer<TerrainGpu>,
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
    listed: Pass<TerrainGpu>,
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
    listed: [ComputePipeline<TerrainGpu>; SURFACE_SLOTS],
}

struct Inputs {
    args: DrawInput<TerrainGpu, Own>,
    meshlet: DrawInput<TerrainGpu, Own>,
    subdivision_args: DrawInput<TerrainGpu, Own>,
    lists: DrawInput<TerrainGpu, Own>,
    listed: DrawInput<TerrainGpu, Own>,
}

#[derive(Default)]
struct Listing {
    main: bool,
    surface: bool,
    built: bool,
}

impl Listing {
    fn ready(&self) -> bool {
        self.main && self.built
    }
}

struct SubdivisionInstall {
    clear_counters: Pass<TerrainGpu>,
    begin: Pass<TerrainGpu>,
    copy_args: Pass<TerrainGpu>,
    subdivide: Pass<TerrainGpu>,
    listed: Pass<TerrainGpu>,
    pipelines: [[ComputePipeline<TerrainGpu>; SURFACE_SLOTS]; 4],
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

const fn resolve_form(vertex: Entry) -> KindForm {
    KindForm {
        kind: TERRAIN,
        placed: FormSide::pulled(vertex),
        owned: None,
        varyings: Varyings::Own(&OwnFragments::NONE),
    }
}

const RESOLVE_FORMS: [KindForm; 2] = [
    resolve_form(crate::shaders::terrain_shading::resolve::VS_TERRAIN_RESOLVE_FAR),
    resolve_form(crate::shaders::terrain_shading::resolve::VS_TERRAIN_RESOLVE),
];

const fn resolve_fragment(entry: Entry, depth: FormDepth) -> PassFragment {
    PassFragment {
        pass: FragmentPass::Own(RESOLVE_PASS),
        side: None,
        fragment: FormFragment::Entry {
            entry,
            cull: FaceCull::None,
            depth,
            color: FormColor::Written,
        },
    }
}

const RESOLVES: [PassFragment; 2] = [
    resolve_fragment(
        crate::shaders::terrain_shading::resolve::FS_TERRAIN_RESOLVE_FAR,
        FormDepth::Tested,
    ),
    resolve_fragment(
        crate::shaders::terrain_shading::resolve::FS_TERRAIN_RESOLVE,
        FormDepth::Overwritten,
    ),
];

const WHOLE_SCREEN: std::ops::Range<u32> = 0..3;
const UNSPLIT_SCREEN: std::ops::Range<u32> = 3..6;

fn resolve_forms(
    cx: &mut Install<'_, TerrainGpu>,
    ids: Texture<TerrainGpu>,
) -> Result<[Form<TerrainGpu>; 2], Error> {
    let bound = [(
        crate::shaders::terrain_shading::resolve::TERRAIN_IDS_TEXTURE,
        Bind::Texture(ids),
    )];
    Ok([
        cx.speedup_form_owning(&RESOLVE_FORMS[0], &[RESOLVES[0]], &bound)?,
        cx.form_owning(&RESOLVE_FORMS[1], &[RESOLVES[1]], &bound)?,
    ])
}

const NO_FORM: &str = "terrain has no form in this pass";
const TERRAIN_TILES: &str = "the terrain tiles";
const HOLED_MATERIAL: &str = "the holed ground material";
const NOT_LAID_OUT: &str = "no terrain is laid out this frame";
const DRAWN_WITH_IDS: &str = "terrain depth is drawn with its ids";
const NO_PATCHES: &str = "no terrain patch is laid out for this view";

enum Joined {
    Ground,
    Seen {
        args: DrawInput<TerrainGpu, Scene>,
        meshlet: DrawInput<TerrainGpu, Scene>,
        lists: DrawInput<TerrainGpu, Scene>,
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
    resolve: [Form<TerrainGpu>; 2],
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
    atlas_short: [bool; 5],
    subdivision_recorded: bool,
    fading: bool,
    drawn: bool,
    tiles_waiting: bool,
    deferred: bool,
    temporal: bool,
    render: (u32, u32),
    listing: Listing,
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

type Frame = (Res<'static, Views>, Option<Res<'static, Detail>>);

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
            ancestors, control, draw, faces, field, horizons, normals, peaks, sky, surface, tables,
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
            lists: cx.buffer("leaf_lists", Lifetime::Kept),
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
        cx.frame(surface::TERRAIN_LEAF_LISTS, Bind::Buffer(h.lists))?;
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
        let after = [control_pass, normals_pass, peaks_pass, ancestors_pass];
        let tree = install_subdivision(cx, h, after)?;
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
            .at(AttachPoint::Depth)
            .after(tree.listed)
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
        let resolve = resolve_forms(cx, h.ids)?;
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
            (PassRole::Depth, false) => Joined::Ground,
            (PassRole::Lit | PassRole::Shown, _) => Joined::Seen {
                args: cx.args_input(pass.scene, h.args),
                meshlet: cx.draw_input(pass.scene, h.meshlet),
                lists: cx.args_input(pass.scene, h.counters),
            },
            (PassRole::Blended, _) => Joined::Elsewhere(NEVER_BLENDS),
        });
        let inputs = Inputs {
            args: cx.args_input(ids_pass, h.args),
            meshlet: cx.draw_input(ids_pass, h.meshlet),
            subdivision_args: cx.draw_input(tree.subdivide, h.args),
            lists: cx.args_input(ids_pass, h.counters),
            listed: cx.draw_input(tree.listed, h.args),
        };
        let sources = BounceSources {
            horizons: &horizon_maps,
            sky: &sky_maps,
            colors: &macro_maps,
            palette: h.palette,
        };
        let bounce = Bounce::install(cx, &sources, bounce_pass)?;
        let sun = SunGpu::install(cx, &horizon_maps, sun_pass)?;
        let [begin, decide, apply, listed] = tree.pipelines;
        Ok(Self {
            passes: Passes {
                normals: normals_pass,
                control: control_pass,
                peaks: peaks_pass,
                ancestors: ancestors_pass,
                summaries: summaries_pass,
                clear_counters: tree.clear_counters,
                begin: tree.begin,
                copy_args: tree.copy_args,
                subdivide: tree.subdivide,
                listed: tree.listed,
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
                begin,
                decide,
                apply,
                listed,
            },
            inputs,
            form,
            seen,
            resolve,
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
            atlas_short: [false; 5],
            subdivision_recorded: false,
            fading: false,
            drawn: false,
            tiles_waiting: false,
            deferred: false,
            temporal: false,
            render: (0, 0),
            listing: Listing::default(),
            handles,
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn prepare(
        &mut self,
        (lists, neighbors, (frame, detail), (limits, assets, holed, provided, lights, sun_plan)): SystemParamItem<
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
            temporal: self.temporal,
        };
        let watered = water.is_changed();
        let p = &self.passes;
        let short = self.atlas_short;
        let ran = [
            cx.ran(p.normals) && !short[0],
            cx.ran(p.control) && !short[1],
            cx.ran(p.peaks) && !short[2],
            cx.ran(p.ancestors) && !(short[3] || short[0] || short[1]),
            cx.ran(p.summaries) && !short[4],
        ];
        let subdivided = cx.ran(p.subdivide) && self.subdivision_recorded;
        let peaks_missed = self.atlas.peak_jobs() > 0 && !ran[2];
        let derived_layers = self.atlas.derived_layers(ran);
        if !derived_layers.is_empty() {
            cx.deliver::<Derived>(DERIVED_LAYERS, bytemuck::cast_slice(&derived_layers).to_vec());
        }
        let uploaded = self.atlas.upload(
            cx,
            h,
            (&tiles, &atlas, &bakes),
            (&moves, ran),
            (&water, watered),
        );
        let derived = [
            self.atlas.normal_jobs(),
            self.atlas.control_jobs(),
            self.atlas.peak_jobs(),
            self.atlas.ancestor_jobs(),
            self.atlas.summary_jobs(),
        ]
        .iter()
        .any(|&jobs| jobs > 0);
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
        let laid_out = self.tables.prepare(
            cx,
            h,
            (&tiles, &atlas, &moves, tiles.is_changed()),
            view_list.len(),
        );
        let look = settings.look();
        self.tables.write_header(cx, h, (atlas.now, &look));
        self.tables.write_views(cx, h, &view_list);
        let unprovided = ProvidedAssets::default();
        let provided = provided.as_deref().unwrap_or(&unprovided);
        let mapped = self.maps.prepare(cx, (&tiles, &assets, provided));
        self.tables.write_palette(
            cx,
            h,
            (&tiles, &ground.palette, &atlas),
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
        self.sun.prepare(
            cx,
            (&sun_plan, &settings, &tiles, &atlas),
            (&self.horizons, peaks_missed),
        );
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
        let seconds = look.shore[2];
        let fading = atlas
            .arrivals
            .values()
            .chain(atlas.surfaced.values())
            .any(|&began| atlas.now - began < 2.0 * seconds);
        let moved = moves.all
            || !moves.uploads.is_empty()
            || !moves.vacated.is_empty()
            || !moves.changed.is_empty();
        let changed =
            moved || uploaded || derived || laid_out || mapped || watered || fading || self.fading;
        self.fading = fading;
        let unset = Detail::default();
        let seen = main.detail(detail.as_deref().unwrap_or(&unset));
        let mut inputs: Vec<u32> = bytemuck::cast_slice(&[
            seen.eye.x,
            seen.eye.y,
            seen.eye.z,
            seen.pixel_angle,
            seen.texel,
            seen.pixel_error,
        ])
        .to_vec();
        let macros: Vec<_> = (tiles.terrains.iter().enumerate())
            .map(|(at, entry)| (bakes.macro_map(entry.terrain), self.macro_maps.layer(at)))
            .collect();
        if let Some(looks) = self.tables.looks(self.patches.uniforms(), &macros) {
            cx.deliver::<Derived>(GROUND_LOOKS, looks.to_le_bytes().to_vec());
        }
        let (palette, rungs) = self.tables.palette_rows();
        inputs.extend_from_slice(bytemuck::cast_slice(self.patches.uniforms()));
        inputs.extend_from_slice(bytemuck::cast_slice(&[look]));
        inputs.extend_from_slice(bytemuck::cast_slice(palette));
        inputs.extend_from_slice(bytemuck::cast_slice(rungs));
        self.surface.settle(cx, h, (inputs, changed));
        self.listing.built = false;
        self.run_subdivision(cx);
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

    fn view(&mut self, facts: &engine_core::ViewFacts, cx: &mut Prepare<'_, Self>) {
        self.listing.main = cx.view_index() == 0;
        self.listing.surface = facts.target.surface();
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.passes.normals {
            let dispatched = cx.dispatch(
                &self.pipelines.normals,
                Count::Groups(self.atlas.normal_groups()),
            );
            self.atlas_short[0] = dispatched < 1;
            return dispatched;
        }
        if pass == self.passes.control {
            let dispatched = cx.dispatch(
                &self.pipelines.control,
                Count::Groups(self.atlas.control_groups()),
            );
            self.atlas_short[1] = dispatched < 1;
            return dispatched;
        }
        if pass == self.passes.peaks {
            let dispatched = self.atlas.peaks(cx, &self.pipelines.peaks);
            self.atlas_short[2] = dispatched < self.pipelines.peaks.len() as u32;
            return dispatched;
        }
        if pass == self.passes.ancestors {
            let dispatched = self.atlas.ancestors(cx, &self.pipelines.ancestors);
            self.atlas_short[3] = dispatched < self.atlas.ancestor_lists();
            return dispatched;
        }
        if pass == self.passes.summaries {
            let dispatched = self.atlas.summaries(cx, &self.pipelines.summaries);
            self.atlas_short[4] = dispatched < self.pipelines.summaries.len() as u32;
            return dispatched;
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
            self.subdivision_recorded = dispatched == 2 * self.surface.slots().count() as u32;
            return dispatched;
        }
        if pass == self.passes.listed {
            let mut dispatched = 0;
            for slot in self.surface.slots() {
                dispatched += cx.dispatch(
                    &self.pipelines.listed[slot],
                    Count::Indirect(self.inputs.listed, self.surface.listed_offset(slot)),
                );
            }
            self.listing.built =
                cx.view_index() == 0 && dispatched == self.surface.slots().count() as u32;
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
    fn run_subdivision(&self, cx: &mut Prepare<'_, Self>) {
        if !self.surface.any() {
            return;
        }
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
        cx.run(self.passes.listed);
        cx.read_back::<SurfaceTotals>(
            SURFACE_TOTALS,
            self.handles.counters,
            self.surface.counter_bytes(),
            engine_render::AfterRead::Keep,
        );
    }

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
        if !self.listing.surface {
            return draws;
        }
        let pipelines = [draw.form(&self.seen[0]), draw.form(&self.seen[1])];
        let listed = self.listing.ready();
        let args = if listed {
            self.inputs.lists
        } else {
            self.inputs.args
        };
        let held = matches!(self.holed, Ok(Some(_)));
        for holes in [false, true] {
            for terrain in self
                .frame
                .terrains
                .iter()
                .filter(|terrain| terrain.holes == holes && (held || !holes))
            {
                let split = if listed {
                    self.listed_draws(terrain, pipelines)
                } else {
                    self.split_draws(terrain, pipelines)
                };
                for (pipeline, offset, cut) in split {
                    draw.draw_timed(
                        &pipeline,
                        Shape::Indirect {
                            vertices: &[],
                            indices: Some(self.inputs.meshlet),
                            args,
                            offset,
                            material: None,
                        },
                        || drawn_label(IDS_PASS, None, cut, terrain.terrain),
                    );
                    draws += 1;
                }
            }
        }
        draws
    }

    fn listed_draws<T: Copy>(
        &self,
        terrain: &FrameTerrain,
        [plain, holed]: [Option<T>; 2],
    ) -> Vec<(T, u64, bool)> {
        let Some([whole, cut]) = self.surface.list_offsets(terrain.terrain) else {
            return Vec::new();
        };
        if !terrain.holes {
            return plain
                .map(|plain| vec![(plain, whole, false), (plain, cut, false)])
                .unwrap_or_default();
        }
        let Some(holed) = holed else {
            return Vec::new();
        };
        vec![
            (plain.unwrap_or(holed), whole, plain.is_none()),
            (holed, cut, true),
        ]
    }

    fn split_draws<T: Copy>(
        &self,
        terrain: &FrameTerrain,
        [plain, holed]: [Option<T>; 2],
    ) -> Vec<(T, u64, bool)> {
        let Some(offset) = self.surface.draw_offset(terrain.terrain) else {
            return Vec::new();
        };
        if !terrain.holes {
            return plain
                .map(|plain| (plain, offset, false))
                .into_iter()
                .collect();
        }
        let Some(holed) = holed else {
            return Vec::new();
        };
        match self.surface.split_offsets(terrain.terrain) {
            Some([whole, cut]) => vec![
                (plain.unwrap_or(holed), whole, plain.is_none()),
                (holed, cut, true),
            ],
            None => vec![(holed, offset, true)],
        }
    }

    fn draw_resolve(&self, draw: &mut Draw<'_, Self, Own>) -> u32 {
        let (Some(far), Some(near)) = (draw.form(&self.resolve[0]), draw.form(&self.resolve[1]))
        else {
            return 0;
        };
        let mut draws = 0;
        for terrain in &self.frame.terrains {
            let Some(rect) = terrain.scissor(self.render) else {
                continue;
            };
            draw.scissor(rect);
            let index = terrain.terrain as u32;
            let shape = |vertices| Shape::Pulled {
                vertices,
                instances: index..index + 1,
                material: None,
            };
            let vertices = match draw.draw(&far, shape(WHOLE_SCREEN)) {
                Recorded::Exact => WHOLE_SCREEN,
                Recorded::StandIn | Recorded::Waiting | Recorded::Refused => UNSPLIT_SCREEN,
            };
            draw.draw(&near, shape(vertices));
            draws += 2;
        }
        draw.scissor([0, 0, self.render.0, self.render.1]);
        draws
    }

    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        let Some(joined) = draw.per_pass(&self.joined) else {
            return draw.skip(NO_FORM);
        };
        match joined {
            Joined::Ground => {
                let holed = self.frame.terrains.iter().any(|terrain| {
                    terrain.holes && self.surface.draw_offset(terrain.terrain).is_some()
                });
                match &self.holed {
                    Ok(None) if holed => draw.wait(HOLED_MATERIAL),
                    Err(error) if holed => draw.refuse(HOLED_MATERIAL, error),
                    Ok(_) | Err(_) => draw.skip(DRAWN_WITH_IDS),
                }
            }
            Joined::Seen {
                args,
                meshlet,
                lists,
            } => {
                if !self.drawn {
                    return self.unlaid(draw);
                }
                let [plain, holes] = &self.seen;
                let (Some(plain), Some(holes)) = (draw.form(plain), draw.form(holes)) else {
                    return draw.skip(NO_FORM);
                };
                self.draw_surface(
                    (scene, draw),
                    [Ok((plain, None)), Ok((holes, None))],
                    [*args, *meshlet, *lists],
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
                draw.draw_timed(
                    &pipeline,
                    Shape::Buffers {
                        vertices: &[*patches],
                        indices: Some(*grid),
                        elements: 0..self.patches.index_count(),
                        instances: range,
                    },
                    || caster_label(scene),
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
        (scene, draw): (&SceneDraw, &mut Draw<'_, Self, Scene>),
        surfaces: Surfaces<'_>,
        [args, meshlet, lists]: [DrawInput<Self, Scene>; 3],
    ) -> u32 {
        let mut draws = 0;
        let mut unheld = None;
        let pipelines = [surfaces[0].ok(), surfaces[1].ok()];
        let mut sped = None;
        let listed = scene.view.is_none() && self.listing.ready();
        for holes in [false, true] {
            for terrain in self
                .frame
                .terrains
                .iter()
                .filter(|terrain| terrain.holes == holes)
            {
                if let Err(reason) = surfaces[usize::from(holes)]
                    && self.surface.draw_offset(terrain.terrain).is_some()
                {
                    unheld = Some(reason);
                    continue;
                }
                let pipelines = if holes {
                    let plain = *sped.get_or_insert_with(|| {
                        pipelines[0].and_then(|(plain, material)| {
                            draw.speedup(plain).map(|plain| (plain, material))
                        })
                    });
                    [plain, pipelines[1]]
                } else {
                    pipelines
                };
                let (args, split) = if listed {
                    (lists, self.listed_draws(terrain, pipelines))
                } else {
                    (args, self.split_draws(terrain, pipelines))
                };
                for ((pipeline, material), offset, cut) in split {
                    draw.draw_timed(
                        &pipeline,
                        Shape::Indirect {
                            vertices: &[],
                            indices: Some(meshlet),
                            args,
                            offset,
                            material,
                        },
                        || drawn_label(scene.pass.name(), scene.view, cut, terrain.terrain),
                    );
                    draws += 1;
                }
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

fn caster_label(scene: &SceneDraw) -> String {
    let pass = scene.pass.name();
    match scene.view {
        Some(view) => format!("terrain {pass} casters view {view}"),
        None => format!("terrain {pass} casters"),
    }
}

fn drawn_label(pass: &str, view: Option<u32>, holes: bool, terrain: usize) -> String {
    let cut = if holes { "holes" } else { "plain" };
    match view {
        Some(view) => format!("terrain {pass} {cut} terrain {terrain} view {view}"),
        None => format!("terrain {pass} {cut} terrain {terrain}"),
    }
}

type SummaryPipelines =[ComputePipeline<TerrainGpu>; SUMMARY_MIPS as usize];

fn install_subdivision(
    cx: &mut Install<'_, TerrainGpu>,
    h: &Handles,
    after: [Pass<TerrainGpu>; 4],
) -> Result<SubdivisionInstall, Error> {
    use crate::shaders::terrain_shading::subdivision;
    let clear_counters = cx.clear("clear_surface_counters", h.counters);
    let mut begin = cx
        .pass("subdivision_begin")
        .label("engine/terrain-subdivision-arguments")
        .after(clear_counters);
    for pass in after {
        begin = begin.after(pass);
    }
    let begin = begin.id();
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
    let listed = cx
        .pass("terrain_leaf_lists")
        .label("engine/terrain-leaf-lists")
        .at(AttachPoint::Inputs)
        .views(PassViews::Main)
        .after(subdivide)
        .id();
    let slot_binds = |slot: usize| {
        let (uniform, tree) = h.slots[slot];
        [
            (subdivision::SUBDIVISION, Bind::Buffer(uniform)),
            (subdivision::SUBDIVISION_TREE, Bind::Buffer(tree)),
            (subdivision::SUBDIVISION_NODES, Bind::Buffer(h.nodes)),
            (subdivision::SUBDIVISION_LEAVES, Bind::Buffer(h.leaves)),
            (subdivision::SUBDIVISION_ARGS, Bind::Buffer(h.counters)),
            (subdivision::SUBDIVISION_WATERED, Bind::Buffer(h.watered)),
            (subdivision::SUBDIVISION_LISTS, Bind::Buffer(h.lists)),
        ]
    };
    let entries = [
        (begin, subdivision::SUBDIVIDE_BEGIN),
        (subdivide, subdivision::SUBDIVIDE_DECIDE),
        (subdivide, subdivision::SUBDIVIDE_APPLY),
        (listed, subdivision::SUBDIVIDE_LIST_DRAWN),
    ];
    let mut pipelines = Vec::with_capacity(entries.len() * SURFACE_SLOTS);
    for (pass, entry) in entries {
        for slot in 0..SURFACE_SLOTS {
            let bound = reached(cx, entry, slot_binds(slot));
            pipelines.push(cx.compute_pipeline(pass, entry, &bound)?);
        }
    }
    Ok(SubdivisionInstall {
        clear_counters,
        begin,
        copy_args,
        subdivide,
        listed,
        pipelines: std::array::from_fn(|entry| {
            std::array::from_fn(|slot| pipelines[entry * SURFACE_SLOTS + slot])
        }),
    })
}

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
