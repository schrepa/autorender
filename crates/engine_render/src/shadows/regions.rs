//! Where a frame's maps look: the light basis,
//! the far region the moving and kept maps share, the near cascades in use
//! and the kept levels, with what caster selection and lookups read about
//! them.

use engine_core::{
    Shadows,
    glam::{Vec3, Vec4},
    math,
};

use super::{
    FAR_VIEW, KEPT_LEVELS, KEPT_MAP, LEVEL_REFRESH, LEVEL_VIEW, MAPS, MOVING_MAP, REFRESH,
    STILLNESS, VIEW_STRIDE, VIRTUAL_SHADOW_PLANES, VirtualShadowView,
};
use crate::gpu_data::{CASTER_PLANES, MAX_CASCADES, MAX_LEVELS, ShadowCullUniform, ShadowUniform};
use crate::shadow_fit::{
    KeptLevels, LevelRefresh, LightBasis, MapRegion, Refresh, TILE_WORDS, TILES, caster_planes,
    frustum_footprint, receivers_from, view_sphere,
};

// Lookups move this many texels along the surface normal. Receiver planes
// keep tilted surfaces off their own depths; the offset only
// covers where a lookup's texels meet another surface.
const NORMAL_OFFSET_TEXELS: f32 = 0.5;
// Receiver planes steeper than this, in texels of depth per texel (86
// degrees from the sun), are clamped.
const SLOPE_LIMIT_TEXELS: f32 = 16.0;
// Shadows fade over the last part of their distance.
const FADE_START: f32 = 0.85;

/// The casters a frame selects: candidates, skinned slots among them,
/// caster groups, the ID capacity of each map, whether any
/// group draws virtual geometry, and the kept levels.
pub(super) struct SelectionCounts<'a> {
    pub(super) candidates: u32,
    pub(super) skinned: u32,
    pub(super) groups: u32,
    pub(super) capacity: u32,
    pub(super) virtual_geometry: bool,
    /// The kept levels, whose tiles still casters that
    /// move or come to rest mark dirty.
    pub(super) levels: &'a KeptLevels,
}

/// This frame's redraw of a kept level: which, its
/// light-space rectangle and the region of its page, which its view maps
/// onto the level's whole map.
#[derive(Clone, Copy)]
pub(super) struct LevelDraw {
    pub(super) refresh: LevelRefresh,
    pub(super) rect: [f32; 4],
    pub(super) page: MapRegion,
}

pub(super) struct Regions {
    basis: LightBasis,
    // Far maps `resolution` texels a side, near cascades and kept levels
    // half as many.
    resolution: u32,
    far: MapRegion,
    // The near cascades in use, nearest first, with the view depth each
    // reaches.
    near: [(f32, MapRegion); MAX_CASCADES],
    cascades: usize,
    // The finest kept level's texel, 0 without levels, and
    // this frame's level redraw.
    level_texel: f32,
    level_draw: Option<LevelDraw>,
}

impl Regions {
    /// The regions of `settings`' maps in light `basis` around the far
    /// region `far`, seen through `view_projection` from `eye`. Each near
    /// cascade is the smallest sphere around the view within
    /// its reach, nested; a reach inside the camera's near plane covers
    /// nothing and has no cascade.
    pub(super) fn new(
        basis: LightBasis,
        far: MapRegion,
        settings: &Shadows,
        view_projection: engine_core::glam::Mat4,
        eye: Vec3,
    ) -> Self {
        let resolution = settings.resolution;
        let mut near = [(0.0, far); MAX_CASCADES];
        let mut cascades = 0;
        for reach in settings.cascade_reaches() {
            if let Some((center, radius)) = view_sphere(view_projection, eye, reach) {
                let region = MapRegion::of_sphere(&basis, center, radius, resolution / 2);
                near[cascades] = (reach, region);
                cascades += 1;
            }
        }
        Self {
            basis,
            resolution,
            far,
            near,
            cascades,
            level_texel: 0.0,
            level_draw: None,
        }
    }

    /// Adds the kept levels: the finest one's texel, and
    /// this frame's `refresh` of one of them.
    pub(super) fn with_levels(
        mut self,
        levels: &KeptLevels,
        refresh: Option<LevelRefresh>,
    ) -> Self {
        let texels = self.near_resolution();
        self.level_texel = levels.texel(0, texels);
        self.level_draw = refresh.map(|refresh| {
            let (rect, page) = levels.refresh_regions(refresh, levels.texel(refresh.level, texels));
            LevelDraw {
                refresh,
                rect,
                page,
            }
        });
        self
    }

    /// This frame's redraw of a kept level, if any.
    pub(super) fn level_draw(&self) -> Option<LevelDraw> {
        self.level_draw
    }

    /// World to clip space of this frame's kept-level redraw: its page's.
    pub(super) fn level_clip(&self) -> Option<engine_core::glam::Mat4> {
        self.level_draw.map(|draw| draw.page.clip(&self.basis))
    }

    /// The near cascades in use, with the view depth each reaches.
    pub(super) fn near(&self) -> &[(f32, MapRegion)] {
        &self.near[..self.cascades]
    }

    fn near_resolution(&self) -> u32 {
        self.resolution / 2
    }

    /// The world error each map's casters may draw with:
    /// half a texel, so rasterized depths differ from the mesh's by less
    /// than rounding, and a whole texel in the moving map, whose casters are
    /// drawn every frame: its outlines may then move by one texel, and it
    /// draws half the geometry. Kept levels draw within half
    /// the finest one's texel.
    pub(super) fn errors(&self) -> [f32; MAPS] {
        let far = self.far.texel(self.resolution);
        std::array::from_fn(|map| match map {
            MOVING_MAP => far,
            KEPT_MAP => far * 0.5,
            KEPT_LEVELS => self.level_texel * 0.5,
            cascade => self.near().get(cascade).map_or(0.0, |(_, region)| {
                region.texel(self.near_resolution()) * 0.5
            }),
        })
    }

    /// World to clip space of each map: every near cascade's (the far
    /// region's where unused), then the far maps'.
    pub(super) fn clips(
        &self,
    ) -> (
        [engine_core::glam::Mat4; MAX_CASCADES],
        engine_core::glam::Mat4,
    ) {
        let far = self.far.clip(&self.basis);
        let mut near = [far; MAX_CASCADES];
        for (clip, (_, region)) in near.iter_mut().zip(self.near()) {
            *clip = region.clip(&self.basis);
        }
        (near, far)
    }

    /// The maps' view matrices at their view offsets: each near cascade's,
    /// then the far maps' at [`FAR_VIEW`] and the redrawn kept level's page
    /// at [`LEVEL_VIEW`] (the far maps' without one).
    pub(super) fn view_matrices(&self) -> [f32; (MAX_CASCADES + 2) * VIEW_STRIDE as usize / 4] {
        let (near, far) = self.clips();
        let level = self.level_clip().unwrap_or(far);
        let mut views = [0.0_f32; (MAX_CASCADES + 2) * VIEW_STRIDE as usize / 4];
        for (index, clip) in near.iter().chain([&far, &level]).enumerate() {
            let start = index * VIEW_STRIDE as usize / 4;
            views[start..start + 16].copy_from_slice(&clip.to_cols_array());
        }
        views
    }

    /// The light-space rectangle of the kept map `refresh` redraws, with a
    /// texel of margin keeping casters at its tiles' edges; zero without
    /// one.
    pub(super) fn refresh_rect(&self, refresh: Option<Refresh>) -> [f32; 4] {
        match refresh {
            Some(Refresh::Whole) => self.far.rect,
            Some(Refresh::Tiles(region)) => {
                region.rect(self.far.rect, self.far.texel(self.resolution))
            }
            None => [0.0; 4],
        }
    }

    /// Inward world planes around the light-space rectangle `rect`, then
    /// two passing everything: what casts into it.
    pub(super) fn rect_planes(&self, rect: [f32; 4]) -> [[f32; 4]; 6] {
        let [x0, y0, x1, y1] = rect;
        let [right, up, _] = self.basis.rows;
        let plane = |normal: Vec3, distance: f32| normal.extend(distance).to_array();
        [
            plane(right, -x0),
            plane(-right, x1),
            plane(up, -y0),
            plane(-up, y1),
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// The selection pass's uniform for `counts`' casters: each map's
    /// selection region, and the kept map's `refresh`, while keeping
    /// stillness (`classify`). Maps drawn every frame whose virtual casters'
    /// clusters virtual geometry selects go to
    /// `virtual_views`.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(super) fn selection(
        &self,
        snapshot: &engine_core::RenderSnapshot,
        settings: &Shadows,
        refresh: Option<Refresh>,
        classify: bool,
        counts: SelectionCounts<'_>,
        virtual_views: &mut Vec<VirtualShadowView>,
    ) -> ShadowCullUniform {
        let (basis, far, near) = (&self.basis, &self.far, self.near());
        let eye = snapshot.camera_position;
        let refresh_rect = self.refresh_rect(refresh);
        let row = |axis: Vec3| axis.extend(0.0).to_array();
        // Casters matter only above receivers a map shadows: within the
        // view frustum up to how far the map reaches, widened by the
        // lookups' normal offset and texel gathers. Maps keep their
        // regions; selection skips casters beyond that part of them.
        let selection = |depth: f32, region: &MapRegion, texel: f32| {
            let margin = texel * (NORMAL_OFFSET_TEXELS + 2.0);
            match frustum_footprint(basis, snapshot.view_projection, eye, depth) {
                Some((low, high)) => (
                    [
                        math::max(region.rect[0], low.x - margin),
                        math::max(region.rect[1], low.y - margin),
                        math::min(region.rect[2], high.x + margin),
                        math::min(region.rect[3], high.y + margin),
                    ],
                    math::min(region.depth[1], high.z + margin),
                ),
                None => (region.rect, region.depth[1]),
            }
        };
        let mut near_rects = [[0.0; 4]; MAX_CASCADES];
        let mut near_depths = [0.0; 4];
        for (index, (reach, region)) in near.iter().enumerate() {
            (near_rects[index], near_depths[index]) =
                selection(*reach, region, region.texel(self.near_resolution()));
        }
        // The far maps shadow receivers up to the shadow distance, and
        // sun shafts sample them along view rays.
        let shafts = snapshot.environment.sun_shafts.sanitized();
        let far_reach = if shafts.enabled() {
            math::max(settings.distance, shafts.distance)
        } else {
            settings.distance
        };
        let (moving_rect, moving_depth) = selection(far_reach, far, far.texel(self.resolution));
        // Where casters can darken a map's receivers: the part of the view
        // they lie in, from `start` (radially) to `reach`, swept toward the
        // sun, widened like the regions. Receivers pick a cascade by their
        // distance, and read the next one across the last tenth of its
        // reach; any of them may read the far maps.
        let toward_sun = -basis.rows[2];
        let volume = |start: Option<f32>, reach: f32, texel: f32| {
            let margin = Vec4::new(0.0, 0.0, 0.0, texel * (NORMAL_OFFSET_TEXELS + 2.0));
            let swept = caster_planes(snapshot.view_projection, eye, reach, toward_sun);
            let first = start
                .and_then(|start| receivers_from(snapshot.view_projection, eye, start, toward_sun));
            let mut planes = [[0.0, 0.0, 0.0, 1.0]; CASTER_PLANES];
            for (slot, plane) in planes.iter_mut().zip(swept.iter().chain(first.iter())) {
                *slot = (*plane + margin).to_array();
            }
            planes
        };
        let mut planes = [[[0.0, 0.0, 0.0, 1.0]; CASTER_PLANES]; MAX_CASCADES + 1];
        for (index, (reach, region)) in near.iter().enumerate() {
            let start = index.checked_sub(1).map(|previous| near[previous].0 * 0.9);
            planes[index] = volume(start, *reach, region.texel(self.near_resolution()));
        }
        planes[MAX_CASCADES] = volume(None, far_reach, far.texel(self.resolution));
        // Maps drawn every frame select virtual casters' clusters within
        // the same regions. A cascade's receivers start
        // where the previous one hands over to it, across its last
        // tenth; the far maps' at the last cascade's reach.
        if counts.virtual_geometry {
            // The selection region, tightened by the view slice swept
            // toward the sun.
            let planes = |rect: [f32; 4], depth: f32, reach: f32| {
                let mut planes = [[0.0, 0.0, 0.0, 1.0]; VIRTUAL_SHADOW_PLANES];
                let region = region_planes(basis, rect, depth);
                let swept = caster_planes(snapshot.view_projection, eye, reach, -basis.rows[2]);
                for (slot, plane) in planes.iter_mut().zip(
                    region
                        .into_iter()
                        .chain(swept.iter().map(|plane| plane.to_array())),
                ) {
                    *slot = plane;
                }
                planes
            };
            for (index, (reach, region)) in near.iter().enumerate() {
                virtual_views.push(VirtualShadowView {
                    map: index,
                    planes: planes(near_rects[index], near_depths[index], *reach),
                    min_distance: index.checked_sub(1).map(|previous| near[previous].0 * 0.9),
                    texel: region.texel(self.near_resolution()),
                    direction: basis.rows[2],
                    view_offset: (index as u64 * VIEW_STRIDE) as u32,
                    error: None,
                });
            }
            virtual_views.push(VirtualShadowView {
                map: MOVING_MAP,
                planes: planes(moving_rect, moving_depth, far_reach),
                min_distance: near.last().map(|(reach, _)| *reach),
                texel: far.texel(self.resolution),
                direction: basis.rows[2],
                view_offset: FAR_VIEW as u32,
                error: None,
            });
            // A redrawn kept level likewise, within the error
            // of its finest level's texels.
            if let Some(draw) = self.level_draw.filter(|_| classify) {
                let mut kept = [[0.0, 0.0, 0.0, 1.0]; VIRTUAL_SHADOW_PLANES];
                for (slot, plane) in
                    kept.iter_mut()
                        .zip(region_planes(basis, draw.rect, draw.page.depth[1]))
                {
                    *slot = plane;
                }
                virtual_views.push(VirtualShadowView {
                    map: KEPT_LEVELS,
                    planes: kept,
                    min_distance: None,
                    texel: draw.page.texel(self.near_resolution()),
                    direction: basis.rows[2],
                    view_offset: LEVEL_VIEW as u32,
                    error: Some(self.errors()[KEPT_LEVELS]),
                });
            }
            // The kept map holds still casters whatever the camera sees: on
            // frames redrawing it, their clusters within the error its other
            // casters draw with (half a texel), in the region redrawn.
            if classify && refresh.is_some() {
                let mut kept = [[0.0, 0.0, 0.0, 1.0]; VIRTUAL_SHADOW_PLANES];
                for (slot, plane) in
                    kept.iter_mut()
                        .zip(region_planes(basis, refresh_rect, far.depth[1]))
                {
                    *slot = plane;
                }
                virtual_views.push(VirtualShadowView {
                    map: KEPT_MAP,
                    planes: kept,
                    min_distance: None,
                    texel: far.texel(self.resolution),
                    direction: basis.rows[2],
                    view_offset: FAR_VIEW as u32,
                    error: Some(self.errors()[KEPT_MAP]),
                });
            }
        }
        let level_draw = self.level_draw.filter(|_| classify);
        ShadowCullUniform {
            light: [row(basis.rows[0]), row(basis.rows[1]), row(basis.rows[2])],
            near: near_rects,
            far: far.rect,
            refresh: refresh_rect,
            moving: moving_rect,
            near_depths,
            depths: [0.0, far.depth[1], far.depth[1], moving_depth],
            candidates: counts.candidates,
            groups: counts.groups,
            capacity: counts.capacity,
            mode: [
                (classify, STILLNESS),
                (refresh.is_some(), REFRESH),
                (level_draw.is_some(), LEVEL_REFRESH),
            ]
            .iter()
            .filter(|(set, _)| *set)
            .fold(0, |mode, (_, bit)| mode | bit),
            tiles: [
                far.rect[0],
                far.rect[1],
                (far.rect[2] - far.rect[0]) / TILES as f32,
                (far.rect[3] - far.rect[1]) / TILES as f32,
            ],
            cascades: near.len() as u32,
            skinned: counts.skinned,
            levels: counts.levels.len() as u32,
            pad: 0,
            level_refresh: level_draw.map_or([0.0; 4], |draw| draw.rect),
            level_depths: level_draw.map_or([0.0; 4], |draw| {
                [draw.page.depth[0], draw.page.depth[1], 0.0, 0.0]
            }),
            level_tiles: std::array::from_fn(|index| counts.levels.tiles(index)),
            planes,
        }
    }

    /// The main passes' lookup uniform, which reads still slots' stillness
    /// when `classify`, from the far maps and `levels`.
    pub(super) fn lookup_uniform(
        &self,
        settings: &Shadows,
        classify: bool,
        levels: &KeptLevels,
    ) -> ShadowUniform {
        let (near_clips, far_clip) = self.clips();
        // Cascades beyond those in use are never reached.
        let mut reach = [f32::MAX; 4];
        let mut near_offsets = [0.0; 4];
        let mut near_limits = [0.0; 4];
        for (index, (distance, region)) in self.near().iter().enumerate() {
            reach[index] = *distance;
            near_offsets[index] = region.texel(self.near_resolution()) * NORMAL_OFFSET_TEXELS;
            near_limits[index] = slope_limit(region, self.near_resolution());
        }
        let far_texel = self.far.texel(self.resolution);
        let texels = self.near_resolution();
        let mut level_valid = [[0; 4]; 2 * MAX_LEVELS];
        for (index, words) in level_valid.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let valid: [u32; TILE_WORDS] = levels.valid(index);
            words[0].copy_from_slice(&valid[..4]);
            words[1].copy_from_slice(&valid[4..]);
        }
        ShadowUniform {
            near: near_clips.map(|clip| clip.to_cols_array_2d()),
            far: far_clip.to_cols_array_2d(),
            state: [
                1.0,
                if classify { 1.0 } else { 0.0 },
                self.cascades as f32,
                settings.distance * FADE_START,
            ],
            reach,
            near_offsets,
            near_limits,
            offsets: [
                settings.distance,
                far_texel * NORMAL_OFFSET_TEXELS,
                slope_limit(&self.far, self.resolution),
                0.0,
            ],
            sizes: [
                self.near_resolution() as f32,
                self.resolution as f32,
                0.0,
                0.0,
            ],
            light: self.basis.rows.map(|row| row.extend(0.0).to_array()),
            level_scales: std::array::from_fn(|index| levels.scales(index)),
            level_windows: std::array::from_fn(|index| {
                let [x, y] = levels.window(index);
                let texel = levels.texel(index, texels);
                let [near, far] = levels.band(index);
                let limit = if far > near {
                    SLOPE_LIMIT_TEXELS * texel / (far - near)
                } else {
                    0.0
                };
                [x, y, texel * NORMAL_OFFSET_TEXELS, limit]
            }),
            level_valid,
            level_state: [
                levels.len() as f32,
                texels as f32,
                (0..levels.len())
                    .filter(|&index| levels.has_pending(index))
                    .fold(0, |bits, index| bits | 1 << index) as f32,
                // The horizon records, which the shadows set.
                0.0,
            ],
        }
    }
}

/// Inward world planes of the light-space rectangle `rect` (minimum x and y,
/// maximum x and y) up to light-space depth `depth` in `basis`: where a
/// caster can darken receivers of a map's selection region.
fn region_planes(basis: &LightBasis, rect: [f32; 4], depth: f32) -> [[f32; 4]; 5] {
    let [x, y, z] = basis.rows;
    [
        x.extend(-rect[0]).to_array(),
        (-x).extend(rect[2]).to_array(),
        y.extend(-rect[1]).to_array(),
        (-y).extend(rect[3]).to_array(),
        (-z).extend(depth).to_array(),
    ]
}

/// Steepest receiver plane of `region` with `texels` texels a side, in its
/// depth (0 to 1 over the region's depths) per texel.
fn slope_limit(region: &MapRegion, texels: u32) -> f32 {
    SLOPE_LIMIT_TEXELS * region.texel(texels) / (region.depth[1] - region.depth[0])
}
