//! Sun shadows: a clipmap of depth maps seen from the atmosphere's sun,
//! around the detail eye (the camera's, unless the freeze key holds one).
//!
//! Its levels, each with texels twice the size and a window twice the
//! reach of the one inside it, lie in light space on fixed grids of square
//! tiles; each window follows the detail eye a tile at a time, and the
//! tiles it uncovers start over (`shadow_fit`). A tile has memory, one slot
//! of a depth atlas, only while the depth buffer's pixels read it: after
//! the opaque passes a compute pass marks, for each pixel, the tile holding
//! its point at the level whose texels match what the pixel spans, seen
//! from the detail eye, and the next coarser where lookups blend it in
//! (`marks`); read back a few frames later, the marks give those tiles free
//! slots, then the slots of the tiles pixels read least recently. Each
//! frame a budget of tiles redraws, coarse levels first, those never drawn
//! before those whose still casters changed: in regions, squares of tiles
//! drawn together into a layer of their own, every caster over a region
//! drawn once, then each tile composed from its region into its slot.
//! On GPU-presented frames casters whose pose stayed the same for a few
//! frames stay in the tiles they were drawn into; one that moves or comes
//! to rest marks the drawn tiles under it dirty. CPU-prepared frames track
//! no stillness: they keep casters with virtual geometry alone, and every
//! other caster moves. Moving casters draw every frame over the drawn tiles
//! they cross, into slots of their own past the still casters', through
//! each tile's route onto its slot, and lookups take the lesser of a tile's
//! two slots. Lookups read the level a pixel's size picks and
//! blend in the next coarser (`scene.wgsl`'s `map_visibility`); where a
//! level is not drawn yet, the first coarser one that is stands in, and
//! past the last level terrain's horizons alone shadow.
//!
//! Terrain's ground casts into each region at a level of detail its texels
//! choose, unmorphed, rather than the camera's (`terrain::patches`), so a
//! kept tile never depends on where the eye stood when it was drawn; the
//! ground's own pixels look each level up from their height on that level's
//! ground (`meshes/terrain.wgsl`'s `terrain_kept_lift`), so the ground a
//! level holds never shadows itself whatever level the ground draws at;
//! tiles of ground arriving, leaving or changing mark the tiles over them
//! dirty. Placed objects cast their mean bend in the wind there. Casters
//! with virtual geometry are selected like the others; virtual geometry
//! selects their clusters for each region, within half its texels whatever
//! the camera sees, and draws them after its own selection:
//! [`ShadowPass::encode_selection`], then [`ShadowPass::encode_maps`].
//!
//! Sky occlusion's maps (`sky`) are drawn by the same pass: still mesh
//! casters seen from fixed directions of the sky, one map a frame at most,
//! selected beside the regions.
//!
//! Its parts: the textures and their lookups (`maps`), each frame's caster
//! groups (`casters`), what each frame draws besides its tiles (`regions`),
//! the selection of each map's casters (`selection`), the marks (`marks`),
//! the pipelines (`pipelines`) and the sky maps (`sky`). What the clipmap
//! holds is `shadow_fit::Clipmap`.

#![expect(
    clippy::expect_used,
    reason = "the pipelines are created before they draw"
)]

mod casters;
mod maps;
mod marks;
mod pipelines;
mod regions;
mod selection;
mod sky;

use std::cell::Cell;

use engine_core::{MaterialHandle, MeshHandle, RenderFrame, RenderSnapshot, glam::Vec3};
use engine_core::terrain_data::Pyramid;

use crate::batches::BatchPlan;
use crate::bindings::SHADOW_POSES;
use crate::cached::{Bound, GroupCache};
use crate::gpu_data::{ComposeData, DRAW_ARGS_BYTES, RouteUniform, ShadowCullUniform};
use crate::mesh_arena::{Bindings, Ids, MeshDraw};
use crate::profiling::GpuProfiler;
use crate::renderer::FramePoses;
use crate::shadow_fit::{ClipPlan, ClipSettings, ClipStats, Clipmap, LightBasis, ground_level};

use crate::gpu_layouts::constants::MOVING_MAP;
use crate::gpu_layouts::constants::SHADOW_GROUP as GROUP;
use crate::gpu_layouts::constants::SHADOW_MAPS as MAPS;
use crate::gpu_layouts::constants::SHADOW_REGIONS;
use crate::gpu_layouts::constants::SHADOW_SKINNED_GROUP as SKINNED_GROUP;
use crate::gpu_layouts::constants::SKY_MAP;
use crate::gpu_layouts::constants::SKY_REFRESH;
pub(crate) use crate::gpu_layouts::constants::VIRTUAL_SHADOW_PLANES;
use casters::CasterGroups;
pub(crate) use casters::ShadowGroup;
pub(crate) use maps::SHADOW_FORMAT;
use maps::{ShadowMaps, map_pass};
use marks::Marks;
use pipelines::ShadowPipelines;
use regions::{FrameRegions, SelectionCounts};
use selection::{Selection, map_factor, map_start};
use sky::{SkyDraw, SkyMaps};

// Uniform offset alignment between views: each region's, the moving
// tiles', the compose view and the redrawn sky map's.
const VIEW_STRIDE: u64 = 256;
const MOVING_VIEW: u64 = SHADOW_REGIONS as u64 * VIEW_STRIDE;
const COMPOSE_VIEW: u64 = MOVING_VIEW + VIEW_STRIDE;
const SKY_VIEW: u64 = COMPOSE_VIEW + VIEW_STRIDE;
const VIEWS_BYTES: u64 = SKY_VIEW + VIEW_STRIDE;

/// A region of tiles drawn this frame whose virtual casters' clusters
/// virtual geometry selects: the map its casters list as, its planes
/// around where casters can darken its receivers (unused ones hold
/// everything), the direction it looks along (away from the sun), the
/// offset of its view matrix, and the world error its clusters keep
/// whatever the camera sees.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VirtualShadowView {
    pub map: usize,
    pub planes: [[f32; 4]; VIRTUAL_SHADOW_PLANES],
    pub direction: Vec3,
    pub view_offset: u32,
    pub error: f32,
}

/// Where this frame's casters come from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Source {
    /// GPU presentation's candidates, batches and flags.
    Presented,
    /// Every caster of a CPU-prepared frame.
    Listed,
}

/// Inputs of GPU-presented frames' selection, from GPU presentation.
pub(crate) struct Presented<'a> {
    pub candidates: &'a wgpu::Buffer,
    pub batches: &'a wgpu::Buffer,
    pub flags: &'a wgpu::Buffer,
    pub count: u32,
    /// Skinned slots and their world bounds from this
    /// frame's palettes.
    pub skinned: &'a wgpu::Buffer,
    pub skin_bounds: &'a wgpu::Buffer,
    pub skinned_count: u32,
    /// Advances when any of these buffers is replaced.
    pub generation: u64,
}

/// How a caster group's mesh draws: its index count, first
/// index and base vertex, and its coarser levels of detail (first index,
/// index count and geometric error in mesh units, finest first).
pub(crate) struct CasterDraw {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub lods: Vec<(u32, u32, f32)>,
}

impl CasterDraw {
    /// The coarsest level whose error is at most `error` world units at unit
    /// scale; 0 for the mesh itself.
    fn level_within(&self, error: f32) -> u8 {
        self.lods
            .iter()
            .take_while(|&&(_, _, lod_error)| lod_error <= error)
            .count() as u8
    }

    /// Index count, first index and base vertex of level `level`.
    fn range(&self, level: u8) -> (u32, u32, i32) {
        match level.checked_sub(1) {
            Some(lod) => {
                let (first_index, index_count, _) = self.lods[lod as usize];
                (index_count, first_index, self.base_vertex)
            }
            None => (self.index_count, self.first_index, self.base_vertex),
        }
    }
}

/// What a frame's shadows select from: its snapshot, GPU presentation's
/// buffers on GPU-presented frames, the frame's pose storage and the slots
/// it holds, the mesh arena layout and each mesh's placement
/// in it.
pub(crate) struct ShadowInputs<'a> {
    pub snapshot: &'a RenderSnapshot,
    pub presented: Option<Presented<'a>>,
    pub poses: FramePoses<'a>,
    pub slot_capacity: usize,
    pub mesh_layout: u64,
    pub placement: &'a dyn Fn(MeshHandle) -> CasterDraw,
    /// Whether terrain casts this frame: shadows then run
    /// without mesh casters too.
    pub terrain: bool,
    /// The angle a display pixel spans at the view's center
    /// ([`pixel_angle`](crate::gpu_data::pixel_angle)), which sizes the
    /// wind's shaking on screen as meshes see it.
    pub pixel_angle: f32,
}

/// Draws placed objects into a region's layer with the caster pipelines
/// (by `masked + 2 * skinned`), and their impostor cards with the cards'
/// pipeline once prepared; returns the draws.
pub(crate) type ScatterDraw<'a> = dyn Fn(
        &mut wgpu::RenderPass<'_>,
        usize,
        &[wgpu::RenderPipeline; 4],
        Option<&wgpu::RenderPipeline>,
    ) -> u32
    + 'a;

/// A region of tiles a frame redraws, as terrain's ground and placed
/// objects cull their casters for it: the inward world planes around it
/// and its texel's world size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KeptRefresh {
    pub planes: [[f32; 4]; 6],
    pub texel: f32,
}

/// What depth passes draw: each mesh's uploaded draw (and whether it is
/// skinned), each masked material's group, the clusters of a region's
/// virtual casters, terrain's patches and placed objects, each returning
/// its draws.
pub(crate) struct MapSources<'a> {
    pub meshes: &'a dyn Fn(MeshHandle) -> Option<(MeshDraw<'a>, bool)>,
    pub materials: &'a dyn Fn(MaterialHandle) -> Option<&'a wgpu::BindGroup>,
    pub virtual_draw: &'a dyn Fn(&mut wgpu::RenderPass<'_>, usize) -> u32,
    /// Terrain's patches in a region, drawn first.
    pub terrain_draw: &'a dyn Fn(&mut wgpu::RenderPass<'_>, usize) -> u32,
    /// Placed objects in a region, drawn next.
    pub scatter_draw: &'a ScatterDraw<'a>,
}

/// What each region draws.
#[derive(Clone, Copy)]
enum Casters {
    /// Terrain's patches.
    Terrain,
    /// Objects placed on terrain.
    Scatter,
    /// The selected mesh casters.
    Meshes,
    /// Virtual geometry's clusters.
    Virtual,
}

/// Each region's casters in drawing order: depth alone is written, so
/// the order changes no texel, but it is one order for every region.
const CASTERS: [Casters; 4] = [
    Casters::Terrain,
    Casters::Scatter,
    Casters::Meshes,
    Casters::Virtual,
];

/// A caster group as depth passes draw it: its pipeline choice (masked,
/// skinned), mesh, joints stream, material and group index.
struct Caster<'m> {
    choice: usize,
    draw: MeshDraw<'m>,
    skin: Option<&'m wgpu::Buffer>,
    material: Option<&'m wgpu::BindGroup>,
    index: usize,
}

/// `storage` emptied, as a vector of `U`, which is `T` but for lifetimes:
/// collecting an emptied vector's items into one of the same layout keeps
/// its allocation, so frames list items borrowing their own data in storage
/// kept from frame to frame.
fn recycle<T, U>(mut storage: Vec<T>) -> Vec<U> {
    storage.clear();
    storage.into_iter().filter_map(|_| None).collect()
}

/// One frame's shadow work, from [`ShadowPass::prepare`].
pub(crate) struct ShadowFrame {
    candidates: u32,
    // Skinned slots selected by their palette bounds (GPU-presented frames).
    skinned: u32,
    // The regions redrawn, the tiles composed from them and the moving
    // tiles.
    regions: u32,
    composes: u32,
    moving: u32,
    // The sky map redrawn.
    sky: Option<SkyDraw>,
    pose_index: usize,
    tiles: ClipStats,
}

impl ShadowFrame {
    /// This frame's counts of the clipmap's tiles.
    pub(crate) fn tiles(&self) -> ClipStats {
        self.tiles
    }
}

pub(crate) struct ShadowPass {
    // Created with the first frame that has shadows: games without them
    // never compile them.
    pipelines: Option<ShadowPipelines>,
    // The material group layout of masked casters.
    material_layout: crate::pipelines::GroupLayout,
    maps: ShadowMaps,
    // Terrains whose horizon records lookups loop over, as
    // last set, which the lookup uniform carries.
    horizon_records: f32,
    clipmap: Clipmap,
    // This frame's regions, composes and moving tiles, and its regions as
    // terrain and placed objects cull for them.
    plan: ClipPlan,
    region_views: Vec<(usize, KeptRefresh)>,
    casters: CasterGroups,
    selection: Selection,
    marks: Marks,
    // Pose bindings of each pose buffer, while it exists, and the same with
    // the regions' layers, which composing binds.
    pose_groups: [GroupCache<4>; 2],
    compose_groups: [GroupCache<4>; 2],
    // Regions drawn this frame that select virtual casters' clusters.
    virtual_views: Vec<VirtualShadowView>,
    // Advances whenever the caster groups change.
    groups_generation: u64,
    // Each caster group's mesh placement, and the caster groups, mesh
    // layout and whether levels of detail were on that it was found for.
    caster_draws: Vec<CasterDraw>,
    draws_key: Option<(u64, u64, bool)>,
    // The views, cull uniform and routes last written.
    written_views: Option<Vec<f32>>,
    written_cull: Option<ShadowCullUniform>,
    written_routes: Option<Box<RouteUniform>>,
    // The tiles composed this frame, an instance each, and their buffer.
    composes: Vec<ComposeData>,
    compose_buffer: crate::buffers::Growable,
    // The storage each frame lists its mesh casters in, which borrow the
    // frame's meshes and materials, kept from frame to frame.
    caster_storage: Cell<Vec<Caster<'static>>>,
    // Sky occlusion's maps, the caster groups' generation they hold, and
    // the redrawn sky map's view matrix last written.
    sky: SkyMaps,
    sky_groups: u64,
    written_sky_view: Option<engine_core::glam::Mat4>,
}

impl ShadowPass {
    /// The bindings the main passes read; pipelines follow with the first
    /// frame that has shadows. `material_layout`: the mesh material group.
    pub(crate) fn new(
        device: &wgpu::Device,
        material_layout: &crate::pipelines::GroupLayout,
    ) -> Self {
        Self {
            pipelines: None,
            material_layout: material_layout.clone(),
            maps: ShadowMaps::new(device),
            horizon_records: 0.0,
            clipmap: Clipmap::default(),
            plan: ClipPlan::default(),
            region_views: Vec::new(),
            casters: CasterGroups::new(),
            selection: selection::Selection::new(device),
            marks: Marks::new(device),
            pose_groups: Default::default(),
            compose_groups: Default::default(),
            virtual_views: Vec::new(),
            groups_generation: 0,
            caster_draws: Vec::new(),
            draws_key: None,
            written_views: None,
            written_cull: None,
            written_routes: None,
            composes: Vec::new(),
            compose_buffer: crate::buffers::Growable::new(
                device,
                "engine/shadow-composes",
                size_of::<ComposeData>() as u64,
                wgpu::BufferUsages::VERTEX,
            ),
            caster_storage: Cell::default(),
            sky: SkyMaps::new(device),
            sky_groups: 0,
            written_sky_view: None,
        }
    }

    /// What the screen effects' occlusion pass binds of sky occlusion: its
    /// maps, their comparison sampler and its lookup uniform.
    pub(crate) fn sky_bindings(&self) -> [(&'static str, crate::cached::Bound<'_>); 3] {
        self.sky.bindings()
    }

    /// Sky maps drawn around the camera, and drawn this frame.
    pub(crate) fn sky_counts(&self) -> (u32, u32) {
        self.sky.counts()
    }

    /// Advances whenever the caster groups change.
    pub(crate) fn groups_generation(&self) -> u64 {
        self.groups_generation
    }

    /// Bindings of the main passes' first group (`bindings::CAMERA`): the
    /// lookup uniform, the atlas, the table of slots, the comparison
    /// sampler and the texel sampler of receiver planes.
    pub(crate) fn entries(&self) -> [(&'static str, crate::cached::Bound<'_>); 5] {
        self.maps.entries()
    }

    /// Meshes this frame's shadows draw; CPU-prepared frames may draw culled
    /// meshes the plan's batches lack.
    pub(crate) fn groups(&self) -> &[ShadowGroup] {
        self.casters.groups()
    }

    /// Group of plan batch `index`, for presentation's batch records.
    pub(crate) fn batch_group(&self, index: usize) -> u32 {
        self.casters.batch_group(index)
    }

    /// Regions drawn this frame whose virtual casters' clusters virtual
    /// geometry selects; empty without such casters.
    pub(crate) fn virtual_views(&self) -> &[VirtualShadowView] {
        &self.virtual_views
    }

    /// This frame's regions of tiles as terrain's ground and placed objects
    /// cull their casters for them: the map whose lists each reads, the
    /// inward world planes around it and its texel's world size.
    pub(crate) fn region_views(&self) -> &[(usize, KeptRefresh)] {
        &self.region_views
    }

    /// Marks dirty the clipmap's drawn tiles under the world box
    /// `min`..`max`: placed objects within it appeared or left.
    pub(crate) fn dirty_kept(&mut self, min: Vec3, max: Vec3) {
        self.clipmap.dirty_box(min, max);
    }

    /// Marks every drawn tile dirty: what placed objects cast changed.
    pub(crate) fn dirty_kept_all(&mut self) {
        self.clipmap.dirty_all();
    }

    /// Marks dirty the drawn tiles under the world box `min`..`max`, where
    /// a tile of level `level` of a terrain over `pyramid` arrived, left or
    /// changed: those of the levels whose ground draws at that level or
    /// finer, which may draw that tile.
    pub(crate) fn dirty_ground(
        &mut self,
        (min, max): (Vec3, Vec3),
        (pyramid, level): (&Pyramid, u32),
    ) {
        for index in 0..self.clipmap.len() {
            if level >= ground_level(pyramid, self.clipmap.texel(index)) {
                self.clipmap.dirty_level_box(index, min, max);
            }
        }
    }

    /// The selected casters' IDs (each map's region from
    /// `selection::map_start`, then each group's) and indirect arguments
    /// (five words per map and group, `map * groups + group`), the ID
    /// capacity of a region's map, and a key that changes when the buffers
    /// do.
    pub(crate) fn caster_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, u32, u64) {
        (
            &self.selection.ids,
            &self.selection.args,
            self.selection.id_capacity as u32,
            self.selection.generation(),
        )
    }

    /// The view matrices of every view, at each view's offset, once
    /// shadows ran.
    pub(crate) fn view_buffer(&self) -> Option<&wgpu::Buffer> {
        self.pipelines
            .as_ref()
            .map(|pipelines| &pipelines.view_buffer)
    }

    /// The latest caster totals read back from GPU-selected frames, a few
    /// frames old: the casters the regions drew, the moving casters' draws
    /// over the moving tiles, and those the moving map left out.
    pub(crate) fn caster_counts(&self) -> Option<[u32; 3]> {
        self.selection.caster_counts()
    }

    /// Bytes of the textures, marks and selection buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.maps.capacity_bytes() + self.selection.capacity_bytes() + self.marks.capacity_bytes()
    }

    /// Plans this frame's draw groups, one per mesh: of the plan's batches
    /// (GPU-presented frames) or of `frame`'s casters (CPU-prepared frames).
    /// Call before mesh uploads, which must include every group's mesh.
    pub(crate) fn plan_groups(&mut self, plan: &BatchPlan, frame: RenderFrame<'_>, gpu: bool) {
        if !frame.snapshot().environment.shadows.enabled {
            if !self.casters.groups().is_empty() {
                self.groups_generation += 1;
            }
            self.casters.clear();
            return;
        }
        if self.casters.plan(plan, frame, gpu) {
            self.groups_generation += 1;
            self.selection.groups_changed();
            // Removed casters leave their shadows in the tiles kept, and
            // frames of the other source keep other casters: they redraw.
            self.clipmap.dirty_all();
        }
    }

    /// Prepares this frame's shadows: places the clipmap around the detail
    /// eye, takes what earlier frames read back, plans its slots, redraws
    /// and moving tiles, and writes what its passes and lookups read; or
    /// disables lookups when shadows are off.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &ShadowInputs<'_>,
    ) -> Option<ShadowFrame> {
        let ShadowInputs {
            snapshot,
            presented,
            poses,
            ..
        } = inputs;
        self.virtual_views.clear();
        self.region_views.clear();
        let settings = snapshot.environment.shadows.sanitized();
        if !settings.enabled || (self.casters.groups().is_empty() && !inputs.terrain) {
            if !settings.enabled {
                self.maps.release();
            }
            self.sky.release(queue);
            // Shadows off: lookups see state.x 0.
            self.maps.write_uniform(queue, bytemuck::Zeroable::zeroed());
            return None;
        }
        let source = self.casters.source();
        let classify = source == Source::Presented;
        self.pipelines
            .get_or_insert_with(|| ShadowPipelines::new(device, &self.material_layout));
        // The textures, then the clipmap placed around the detail eye, which
        // the freeze key pins.
        let limit = device.limits().max_texture_dimension_2d;
        let (layout, made) = self.maps.ensure(device, &settings, limit);
        let basis = LightBasis::new(snapshot.environment.atmosphere.sanitized().sun_direction);
        let restarted = self.clipmap.place(
            basis,
            snapshot.detail_eye,
            ClipSettings {
                levels: settings.levels,
                finest_texel: settings.finest_texel,
                tile_texels: settings.tile_texels,
                slots: layout.still,
            },
        );
        if made && !restarted {
            // New textures hold nothing: every tile redraws.
            self.clipmap.invalidate();
        }
        // What earlier frames read back of this placement: the tiles pixels
        // read, those whose still casters changed (the sky maps hold them
        // too) and those moving casters crossed.
        let epoch = self.clipmap.epoch();
        let marked = self.marks.take(epoch);
        let (dirty, moving) = self.selection.take_masks(epoch);
        let still_changed = dirty.iter().any(|&word| word != 0);
        self.clipmap.merge_marks(&marked);
        self.clipmap.merge_dirty(&dirty);
        self.clipmap.merge_moving(&moving);
        let mask_words = self.clipmap.mask_words();
        self.marks.ensure(device, mask_words);
        if classify
            && self
                .selection
                .ensure_stillness(device, inputs.slot_capacity)
        {
            // Stillness starts over: kept casters' footprints are lost, so
            // every tile redraws.
            self.clipmap.dirty_all();
        }
        // Selection outputs: masks not read back from counters made anew are
        // lost, so every tile redraws.
        if self.selection.ensure_outputs(
            device,
            (self.casters.groups().len(), self.casters.total()),
            mask_words,
        ) {
            self.clipmap.dirty_all();
        }
        // Moving casters draw over moving tiles where there are mesh
        // casters.
        let moving_budget = if self.casters.groups().is_empty() {
            0
        } else {
            layout.moving
        };
        self.clipmap.plan(
            (
                settings.redraw_budget,
                crate::shadow_fit::region_tiles(settings.tile_texels),
            ),
            moving_budget,
            &mut self.plan,
        );
        let frame_regions = FrameRegions {
            basis,
            clipmap: &self.clipmap,
            plan: &self.plan,
            layout,
        };
        frame_regions.region_views(&mut self.region_views);
        // The draw arguments' template: each group's mesh placement at the
        // level of detail each map draws it at, written where either or the
        // selection's buffers changed.
        let draws_key = (
            self.groups_generation,
            inputs.mesh_layout,
            snapshot.environment.level_of_detail.enabled,
        );
        if self.draws_key != Some(draws_key) {
            self.draws_key = Some(draws_key);
            self.caster_draws = self
                .casters
                .groups()
                .iter()
                .map(|group| (inputs.placement)(group.mesh))
                .collect();
        }
        let draws = &self.caster_draws;
        let levels: Vec<u8> = frame_regions
            .errors()
            .iter()
            .flat_map(|&error| draws.iter().map(move |draw| draw.level_within(error)))
            .collect();
        self.selection.update_template(
            device,
            queue,
            &self.casters,
            levels,
            |index, level| draws[index].range(level),
            inputs.mesh_layout,
        );
        if self
            .casters
            .groups()
            .iter()
            .any(|group| group.virtual_geometry)
        {
            frame_regions.virtual_views(&mut self.virtual_views);
        }
        let pipelines = self.pipelines.as_ref().expect("created above");
        // The views: the regions' with the wind's mean bend placed objects
        // cast, the moving tiles' with the wind at this frame's time and the
        // detail eye whose distance sizes its shaking on screen, with the
        // pixels a radian spans there, and the compose view.
        let wind = snapshot.environment.wind;
        let gusting = [
            wind.velocity[0],
            wind.velocity[1],
            wind.gusts,
            snapshot.elapsed_seconds,
        ];
        let viewer = snapshot.detail_eye.extend(1.0 / inputs.pixel_angle);
        let views = frame_regions.view_matrices((wind.velocity, gusting), viewer);
        if self.written_views.as_deref() != Some(&views[..]) {
            queue.write_buffer(&pipelines.view_buffer, 0, bytemuck::cast_slice(&views));
            self.written_views = Some(views);
        }
        let routes = frame_regions.routes();
        if self
            .written_routes
            .as_deref()
            .is_none_or(|written| *written != routes)
        {
            queue.write_buffer(&pipelines.routes, 0, bytemuck::bytes_of(&routes));
            self.written_routes = Some(Box::new(routes));
        }
        frame_regions.composes(settings.tile_texels, &mut self.composes);
        if !self.composes.is_empty() {
            let bytes: &[u8] = bytemuck::cast_slice(&self.composes);
            self.compose_buffer.grow(device, bytes.len() as u64);
            queue.write_buffer(&self.compose_buffer, 0, bytes);
        }
        let (candidates, skinned) = match (presented, source) {
            (Some(presented), Source::Presented) => (presented.count, presented.skinned_count),
            (_, Source::Listed) => (self.casters.listed.len() as u32, 0),
            _ => (0, 0),
        };
        let mut cull = frame_regions.selection(
            &SelectionCounts {
                candidates,
                skinned,
                groups: self.casters.groups().len() as u32,
                capacity: self.selection.id_capacity as u32,
                classify,
            },
            mask_words as u32,
        );
        let lookup = frame_regions.lookup_uniform(
            settings.tile_texels,
            snapshot.environment.foliage.sanitized().crown_depth,
        );
        // Sky occlusion's map redrawn this frame, after still casters or the
        // caster groups changed.
        let groups_changed = self.sky_groups != self.groups_generation;
        self.sky_groups = self.groups_generation;
        let sky = self.sky.prepare(
            device,
            queue,
            &snapshot.environment.sky_occlusion.sanitized(),
            snapshot.camera_position,
            still_changed || groups_changed,
        );
        let pipelines = self.pipelines.as_ref().expect("created above");
        if let Some(draw) = sky {
            cull.sky = draw.rows;
            cull.sky_rect = draw.rect;
            cull.sky_depths = [draw.depth, 0.0, 0.0, 0.0];
            cull.mode |= SKY_REFRESH;
            if self.written_sky_view != Some(draw.clip) {
                let mut view = [0.0_f32; 20];
                view[..16].copy_from_slice(&draw.clip.to_cols_array());
                view[16..].copy_from_slice(&gusting);
                queue.write_buffer(
                    &pipelines.view_buffer,
                    SKY_VIEW,
                    bytemuck::cast_slice(&view),
                );
                self.written_sky_view = Some(draw.clip);
            }
        }
        if self
            .written_cull
            .is_none_or(|written| bytemuck::bytes_of(&written) != bytemuck::bytes_of(&cull))
        {
            queue.write_buffer(&pipelines.cull, 0, bytemuck::bytes_of(&cull));
            self.written_cull = Some(cull);
        }
        let mut lookup = lookup;
        lookup.level_state[3] = self.horizon_records;
        self.maps.write_uniform(queue, lookup);
        let (table, changed) = self.clipmap.take_table();
        self.maps.write_table(queue, table, &changed);
        // Bindings: the selection inputs, and poses of this frame.
        self.selection.bind(
            device,
            pipelines,
            source,
            presented.as_ref(),
            poses,
            skinned,
            self.maps.table(),
        );
        self.bind_poses(device, poses);
        Some(ShadowFrame {
            candidates,
            skinned,
            regions: self.plan.regions.len() as u32,
            composes: self.plan.composes.len() as u32,
            moving: self.plan.moving.len() as u32,
            sky,
            pose_index: poses.parity,
            tiles: self.clipmap.stats(),
        })
    }

    /// Binds `poses` for depth passes, with a placeholder for the regions'
    /// layers, which those may draw into, and with the layers for
    /// composing.
    fn bind_poses(&mut self, device: &wgpu::Device, poses: &FramePoses<'_>) {
        let Some(pipelines) = &self.pipelines else {
            return;
        };
        let groups = self.pose_groups.iter_mut().zip(&mut self.compose_groups);
        for (parity, (group, compose)) in groups.enumerate() {
            let Some(buffer) = poses.bound(parity) else {
                group.clear();
                compose.clear();
                continue;
            };
            let entries = |regions| {
                [
                    ("slot_poses", buffer.clone()),
                    ("slot_colors", poses.colors.bound()),
                    ("slot_palette", poses.palettes[parity].bound()),
                    ("shadow_regions", regions),
                ]
            };
            let layout = (&SHADOW_POSES, &pipelines.pose_layout);
            group.bind(
                device,
                "engine/shadow-pose-group",
                layout,
                entries(self.maps.regions_placeholder()),
            );
            match self.maps.regions_bound() {
                Some(regions) => {
                    compose.bind(device, "engine/shadow-compose-group", layout, entries(regions));
                }
                None => compose.clear(),
            }
        }
    }

    /// Records caster selection of `frame`, timed by `profiler`
    /// (`shadow_select`).
    pub(crate) fn encode_selection(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &ShadowFrame,
        profiler: &GpuProfiler,
    ) {
        let (Some(pipelines), Some(_)) = (&self.pipelines, self.maps.textures()) else {
            return;
        };
        // Terrain alone casts: no mesh casters to select.
        if self.casters.groups().is_empty() {
            return;
        }
        let epoch = self.clipmap.epoch();
        self.selection.encode(
            device,
            encoder,
            pipelines,
            (frame, self.casters.groups().len()),
            epoch,
            profiler,
        );
    }

    /// Records the marks of the tiles the frame's pixels read, `size` of
    /// them, from the depth buffer `depth` seen through the camera uniform
    /// `camera`, after the opaque passes; timed by `profiler`
    /// (`shadow_marks`).
    pub(crate) fn encode_marks(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        (camera, depth): (&wgpu::Buffer, Bound<'_>),
        size: (u32, u32),
        profiler: &GpuProfiler,
    ) {
        let (Some(pipelines), Some(_)) = (&self.pipelines, self.maps.textures()) else {
            return;
        };
        let epoch = self.clipmap.epoch();
        self.marks.encode(
            device,
            encoder,
            pipelines,
            (camera, self.maps.uniform()),
            depth,
            size,
            epoch,
            profiler,
        );
    }

    /// Creates the pipeline of placed objects' impostor cards,
    /// whose group 2 `impostor_layout` binds, once shadows have
    /// pipelines.
    pub(crate) fn prepare_impostor_casters(
        &mut self,
        device: &wgpu::Device,
        impostor_layout: &crate::pipelines::GroupLayout,
    ) {
        if let Some(pipelines) = &mut self.pipelines
            && pipelines.impostor_casters.is_none()
        {
            pipelines.impostor_casters = Some(pipelines.impostor_caster(device, impostor_layout));
        }
    }

    /// Sets how many terrains' horizon records the lookups loop over
    /// once the terrain's horizon maps are prepared, after
    /// the frame's lookup uniform: written before the frame is submitted.
    pub(crate) fn set_horizon_records(&mut self, queue: &wgpu::Queue, count: u32) {
        self.horizon_records = count as f32;
        self.maps.write_horizon_records(queue, self.horizon_records);
    }

    /// Records the depth passes of `frame` from `sources`, after
    /// [`Self::encode_selection`] and virtual geometry's selection, timed by
    /// `profiler` (`shadow_regions` over the regions' layers, `shadow_atlas`
    /// over the atlas's pass and `shadow_sky`); returns the indexed draws,
    /// one per mesh group and region or the moving map drawn. Each region's
    /// casters draw into its layer; then in the atlas each tile redrawn is
    /// composed from its region into its slot, and the moving tiles' slots
    /// are cleared and their moving casters drawn over them.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn encode_maps(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &ShadowFrame,
        sources: &MapSources<'_>,
        profiler: &GpuProfiler,
    ) -> u32 {
        let (Some(pipelines), Some(maps)) = (&self.pipelines, self.maps.textures()) else {
            return 0;
        };
        let MapSources {
            meshes,
            materials,
            virtual_draw,
            terrain_draw,
            scatter_draw,
        } = sources;
        let draws = Cell::new(0_u32);
        let groups = self.casters.groups();
        let group_count = groups.len() as u64;
        let (ids, args, id_capacity) = (
            &self.selection.ids,
            &self.selection.args,
            self.selection.id_capacity,
        );
        let pose_group = self.pose_groups[frame.pose_index].get();
        let compose_group = self.compose_groups[frame.pose_index].get();
        // Casters in pipeline, vertex buffer and material order,
        // which passes switch once per run: depth alone is written, so the
        // order changes no texel. Virtual geometry draws its casters'
        // clusters, and masked materials without bindings
        // draw nothing.
        let mut casters: Vec<Caster<'_>> = recycle(self.caster_storage.take());
        if pose_group.is_some() {
            casters.extend(
                groups
                    .iter()
                    .enumerate()
                    .filter(|(_, group)| !group.virtual_geometry)
                    .filter_map(|(index, group)| {
                        let (draw, skinned) = meshes(group.mesh)?;
                        let material = match group.material.map(materials) {
                            Some(None) => return None,
                            Some(Some(material)) => Some(material),
                            None => None,
                        };
                        let skin = draw.skin.filter(|_| skinned);
                        Some(Caster {
                            choice: usize::from(material.is_some())
                                + 2 * usize::from(skin.is_some()),
                            draw,
                            skin,
                            material,
                            index,
                        })
                    }),
            );
        }
        casters.sort_unstable_by_key(|caster| {
            (
                caster.choice,
                std::ptr::from_ref(caster.draw.vertices) as usize,
                caster
                    .material
                    .map_or(0, |group| std::ptr::from_ref(group) as usize),
                caster.index,
            )
        });
        // The mesh casters of map `map` through `set`, the casters' or the
        // moving casters' pipelines.
        let draw_meshes =
            |pass: &mut wgpu::RenderPass<'_>, map: usize, set: &[wgpu::RenderPipeline; 4]| {
                let Some(pose_group) = pose_group else {
                    return;
                };
                pass.set_pipeline(&set[0]);
                pass.set_bind_group(1, pose_group, &[]);
                let mut bound = 0;
                let mut bindings = Bindings::default();
                let start = map_start(map, id_capacity);
                for caster in &casters {
                    if caster.choice != bound {
                        pass.set_pipeline(&set[caster.choice]);
                        bound = caster.choice;
                    }
                    if let Some(material) = caster.material {
                        bindings.group(pass, 2, material);
                    }
                    if let Some(skin) = caster.skin {
                        bindings.vertex(pass, 2, skin);
                    }
                    bindings.vertex(pass, 0, caster.draw.vertices);
                    bindings.index(pass, caster.draw.indices);
                    let caster_ids = if self.selection.regions_in_args {
                        Ids::Whole(ids)
                    } else {
                        let first =
                            start + map_factor(map) as usize * groups[caster.index].start as usize;
                        Ids::Region(ids.slice(first as u64 * 4..))
                    };
                    bindings.ids(pass, 1, caster_ids);
                    pass.draw_indexed_indirect(
                        args,
                        (map as u64 * group_count + caster.index as u64) * DRAW_ARGS_BYTES,
                    );
                    draws.set(draws.get() + 1);
                }
            };
        let draw_selected = |pass: &mut wgpu::RenderPass<'_>, map: usize| {
            for casters in CASTERS {
                match casters {
                    Casters::Terrain => draws.set(draws.get() + terrain_draw(pass, map)),
                    Casters::Scatter => draws.set(
                        draws.get()
                            + scatter_draw(
                                pass,
                                map,
                                &pipelines.casters,
                                pipelines.impostor_casters.as_ref(),
                            ),
                    ),
                    Casters::Meshes => draw_meshes(pass, map, &pipelines.casters),
                    // Clusters draw in the frames binding poses, as meshes.
                    Casters::Virtual => {
                        if pose_group.is_some() {
                            draws.set(draws.get() + virtual_draw(pass, map));
                        }
                    }
                }
            }
        };
        // Profiled frames time the regions together, the atlas's pass and
        // the sky map.
        let last_region = (frame.regions as usize).saturating_sub(1);
        let writes = |span: &'static str, begin: bool, end: bool| {
            (begin || end)
                .then(|| profiler.span_in("shadow_maps", span))
                .flatten()
                .map(|span| span.render_part(begin, end))
        };
        // Each region redrawn: its layer cleared, then its casters, through
        // its view; GPU-presented and CPU-prepared frames alike.
        for (region, layer) in maps.region_layers.iter().enumerate() {
            if region >= frame.regions as usize {
                break;
            }
            let timestamps = writes("shadow_regions", region == 0, region == last_region);
            let mut pass = map_pass(encoder, layer, wgpu::LoadOp::Clear(1.0), timestamps);
            pass.set_bind_group(
                0,
                &pipelines.view_group,
                &[(region as u64 * VIEW_STRIDE) as u32],
            );
            draw_selected(&mut pass, region);
        }
        // The atlas: each tile redrawn composed from its region into its
        // slot, then the moving tiles' slots cleared and their moving
        // casters drawn over them through their routes.
        let composing = frame.composes > 0 && compose_group.is_some();
        let moving = frame.moving > 0 && pose_group.is_some();
        if composing || moving {
            let mut pass = map_pass(
                encoder,
                &maps.atlas_view,
                wgpu::LoadOp::Load,
                writes("shadow_atlas", true, true),
            );
            if let (true, Some(compose)) = (composing, compose_group) {
                pass.set_pipeline(&pipelines.compose);
                pass.set_bind_group(0, &pipelines.view_group, &[COMPOSE_VIEW as u32]);
                pass.set_bind_group(1, compose, &[]);
                pass.set_vertex_buffer(0, self.compose_buffer.slice(..));
                pass.draw(0..6, 0..frame.composes);
            }
            if let (true, Some(pose_group)) = (moving, pose_group) {
                pass.set_pipeline(&pipelines.clear_moving);
                pass.set_bind_group(0, &pipelines.view_group, &[MOVING_VIEW as u32]);
                pass.set_bind_group(1, pose_group, &[]);
                pass.draw(0..6, 0..frame.moving);
                draw_meshes(&mut pass, MOVING_MAP, &pipelines.moving);
            }
        }
        // Sky occlusion's map redrawn this frame: its still mesh casters
        // alone, neither terrain, placed objects nor virtual geometry.
        if let Some(view) = frame.sky.and_then(|draw| self.sky.layer_view(draw.layer)) {
            let mut pass = map_pass(
                encoder,
                view,
                wgpu::LoadOp::Clear(1.0),
                writes("shadow_sky", true, true),
            );
            pass.set_bind_group(0, &pipelines.view_group, &[SKY_VIEW as u32]);
            draw_meshes(&mut pass, SKY_MAP, &pipelines.casters);
        }
        self.caster_storage.set(recycle(casters));
        draws.get()
    }

    /// Requests maps of what the frame just submitted read back.
    pub(crate) fn after_submit(&mut self) {
        self.selection.after_submit();
        self.marks.after_submit();
    }
}
