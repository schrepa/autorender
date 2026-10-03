//! Where a frame's maps look (decisions 020 and 028): the light basis, the
//! far region the moving and kept maps share, and the near cascades in use,
//! with what caster selection and lookups read about them.

use engine_core::{Shadows, glam::Vec3, math};

use super::{
    FAR_VIEW, KEPT_MAP, MAPS, MOVING_MAP, REFRESH, STILLNESS, VIEW_STRIDE, VIRTUAL_SHADOW_PLANES,
    VirtualShadowView,
};
use crate::gpu_data::{MAX_CASCADES, ShadowCullUniform, ShadowUniform};
use crate::shadow_fit::{
    LightBasis, MapRegion, Refresh, TILES, caster_planes, frustum_footprint, view_sphere,
};

// Lookups move this many texels along the surface normal. Receiver planes
// (decision 028) keep tilted surfaces off their own depths; the offset only
// covers where a lookup's texels meet another surface.
const NORMAL_OFFSET_TEXELS: f32 = 0.5;
// Receiver planes steeper than this, in texels of depth per texel (86
// degrees from the sun), are clamped.
const SLOPE_LIMIT_TEXELS: f32 = 16.0;
// Shadows fade over the last part of their distance.
const FADE_START: f32 = 0.85;

/// The casters a frame selects: candidates, skinned slots among them
/// (decision 036), caster groups, the ID capacity of each map, and whether
/// any group draws virtual geometry (decision 044).
pub(super) struct SelectionCounts {
    pub(super) candidates: u32,
    pub(super) skinned: u32,
    pub(super) groups: u32,
    pub(super) capacity: u32,
    pub(super) virtual_geometry: bool,
}

pub(super) struct Regions {
    basis: LightBasis,
    // Far maps `resolution` texels a side, near cascades half as many.
    resolution: u32,
    far: MapRegion,
    // The near cascades in use, nearest first, with the view depth each
    // reaches.
    near: [(f32, MapRegion); MAX_CASCADES],
    cascades: usize,
}

impl Regions {
    /// The regions of `settings`' maps in light `basis` around the far
    /// region `far`, seen through `view_projection` from `eye`. Each near
    /// cascade (decision 028) is the smallest sphere around the view within
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
        }
    }

    /// The near cascades in use, with the view depth each reaches.
    pub(super) fn near(&self) -> &[(f32, MapRegion)] {
        &self.near[..self.cascades]
    }

    fn near_resolution(&self) -> u32 {
        self.resolution / 2
    }

    /// The world error each map's casters may draw with (decision 047):
    /// half a texel, so rasterized depths differ from the mesh's by less
    /// than rounding, and a whole texel in the moving map, whose casters are
    /// drawn every frame: its outlines may then move by one texel, and it
    /// draws half the geometry.
    pub(super) fn errors(&self) -> [f32; MAPS] {
        let far = self.far.texel(self.resolution);
        std::array::from_fn(|map| match map {
            MOVING_MAP => far,
            KEPT_MAP => far * 0.5,
            cascade => self.near().get(cascade).map_or(0.0, |(_, region)| {
                region.texel(self.near_resolution()) * 0.5
            }),
        })
    }

    /// World to clip space of each map: every near cascade's (the far
    /// region's where unused), then the far maps'.
    fn clips(
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
    /// then the far maps' at [`FAR_VIEW`].
    pub(super) fn view_matrices(&self) -> [f32; (MAX_CASCADES + 1) * VIEW_STRIDE as usize / 4] {
        let (near, far) = self.clips();
        let mut views = [0.0_f32; (MAX_CASCADES + 1) * VIEW_STRIDE as usize / 4];
        for (index, clip) in near.iter().chain([&far]).enumerate() {
            let start = index * VIEW_STRIDE as usize / 4;
            views[start..start + 16].copy_from_slice(&clip.to_cols_array());
        }
        views
    }

    /// The selection pass's uniform for `counts`' casters: each map's
    /// selection region, and the kept map's `refresh`, while keeping
    /// stillness (`classify`). Maps drawn every frame whose virtual casters'
    /// clusters virtual geometry selects (decision 044) go to
    /// `virtual_views`.
    pub(super) fn selection(
        &self,
        snapshot: &engine_core::RenderSnapshot,
        settings: &Shadows,
        refresh: Option<Refresh>,
        classify: bool,
        counts: SelectionCounts,
        virtual_views: &mut Vec<VirtualShadowView>,
    ) -> ShadowCullUniform {
        let (basis, far, near) = (&self.basis, &self.far, self.near());
        let eye = snapshot.camera_position;
        let refresh_rect = match refresh {
            Some(Refresh::Whole) => far.rect,
            // A texel of margin keeps casters at the tiles' edges.
            Some(Refresh::Tiles(region)) => region.rect(far.rect, far.texel(self.resolution)),
            None => [0.0; 4],
        };
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
        // sun shafts (decision 024) sample them along view rays.
        let shafts = snapshot.sun_shafts.sanitized();
        let far_reach = if shafts.enabled() {
            math::max(settings.distance, shafts.distance)
        } else {
            settings.distance
        };
        let (moving_rect, moving_depth) = selection(far_reach, far, far.texel(self.resolution));
        // Maps drawn every frame select virtual casters' clusters within
        // the same regions (decision 044). A cascade's receivers start
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
                });
            }
            virtual_views.push(VirtualShadowView {
                map: MOVING_MAP,
                planes: planes(moving_rect, moving_depth, far_reach),
                min_distance: near.last().map(|(reach, _)| *reach),
                texel: far.texel(self.resolution),
                direction: basis.rows[2],
                view_offset: FAR_VIEW as u32,
            });
        }
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
            mode: [(classify, STILLNESS), (refresh.is_some(), REFRESH)]
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
            pad: [0; 2],
        }
    }

    /// The main passes' lookup uniform, which reads still slots' stillness
    /// when `classify`.
    pub(super) fn lookup_uniform(&self, settings: &Shadows, classify: bool) -> ShadowUniform {
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
