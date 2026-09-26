//! Retained results of the pure mesh-bounds/frustum test. Asset resolution and
//! snapshot ownership stay in extraction; this cache holds only weak handles.

use crate::{
    Aabb, MAX_TRANSFORM_NODES, MeshHandle,
    culling::{Classification, Frustum},
};
use glam::Mat4;

struct Entry {
    mesh: MeshHandle,
    model: [u32; 16],
    outside: bool,
}

/// A bounded prefix of valid render inputs in query order. Entity identity is
/// unnecessary: equal immutable mesh handles and matrix bits give the same
/// geometric result, even if a different entity now occupies that query row.
#[derive(Default)]
pub(crate) struct VisibilityCache {
    view: Option<([u32; 16], bool)>,
    frustum: Option<Frustum>,
    entries: Vec<Entry>,
    cursor: usize,
    tested: usize,
    interior: usize,
    reused: usize,
}

impl VisibilityCache {
    pub(crate) fn begin(&mut self, view: Mat4, enabled: bool, candidates: usize) {
        let key = (view.to_cols_array().map(f32::to_bits), enabled);
        if self.view != Some(key) {
            self.entries.clear();
            self.frustum = enabled.then(|| Frustum::new(view)).flatten();
            self.view = Some(key);
        }
        if self.frustum.is_some() {
            let capacity = candidates.min(MAX_TRANSFORM_NODES);
            if self.entries.capacity() < capacity {
                self.entries.reserve_exact(capacity - self.entries.len());
            }
        }
        self.cursor = 0;
        self.tested = 0;
        self.interior = 0;
        self.reused = 0;
    }

    /// Call only after resolving the live immutable mesh in AssetLibrary. A
    /// removed/replaced asset must still fail normal extraction validation.
    pub(crate) fn outside(&mut self, mesh: MeshHandle, model: Mat4, bounds: Aabb) -> bool {
        let Some(frustum) = &self.frustum else {
            return false;
        };
        let index = self.cursor;
        self.cursor += 1;
        let bits = model.to_cols_array().map(f32::to_bits);
        if let Some(entry) = self.entries.get(index)
            && entry.mesh == mesh
            && entry.model == bits
        {
            self.reused += 1;
            return entry.outside;
        }
        let classification = frustum.classify(model, bounds);
        let outside = classification.outside();
        self.tested += 1;
        self.interior += usize::from(classification == Classification::Interior);
        if index < MAX_TRANSFORM_NODES {
            let entry = Entry {
                mesh,
                model: bits,
                outside,
            };
            if index < self.entries.len() {
                self.entries[index] = entry;
            } else {
                self.entries.push(entry);
            }
        }
        outside
    }

    pub(crate) fn finish(&mut self) {
        self.entries.truncate(self.cursor);
        engine_profile::counter("extraction.culling_tests", self.tested as f64);
        engine_profile::counter("extraction.culling_interior", self.interior as f64);
        engine_profile::counter("extraction.culling_reused", self.reused as f64);
        engine_profile::counter(
            "memory.visibility_cache_capacity_bytes",
            (self.entries.capacity() * size_of::<Entry>()) as f64,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AssetLibrary, Engine, EngineConfig, Mesh, Mesh3d, RenderSnapshot, Transform, edit_transform,
    };
    use glam::Vec3;

    #[test]
    fn static_cache_visibility_reuses_only_matching_geometry_and_view() {
        let mut assets = AssetLibrary::default();
        let mesh = assets.insert_mesh(Mesh::cube()).unwrap();
        let other = assets.insert_mesh(Mesh::cube()).unwrap();
        let bounds = assets.mesh(mesh).unwrap().bounds();
        let inside = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5));
        let outside = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.5));
        let mut cache = VisibilityCache::default();
        cache.begin(Mat4::IDENTITY, true, 2);
        assert!(!cache.outside(mesh, inside, bounds));
        assert!(cache.outside(mesh, outside, bounds));
        cache.finish();
        for _ in 0..2 {
            cache.begin(Mat4::IDENTITY, true, 2);
            assert!(!cache.outside(mesh, inside, bounds));
            assert!(cache.outside(mesh, outside, bounds));
            cache.finish();
            assert_eq!((cache.tested, cache.reused), (0, 2));
        }
        // A reordered/moved row and a new immutable asset both invalidate.
        cache.begin(Mat4::IDENTITY, true, 2);
        assert!(cache.outside(mesh, outside, bounds));
        assert!(cache.outside(other, outside, bounds));
        cache.finish();
        assert_eq!((cache.tested, cache.reused), (2, 0));
        cache.begin(Mat4::from_translation(-Vec3::X * 10.0), true, 1);
        assert!(!cache.outside(mesh, outside, bounds));
        cache.finish();
        assert_eq!((cache.tested, cache.reused, cache.entries.len()), (1, 0, 1));
        cache.begin(Mat4::IDENTITY, false, 1);
        assert!(!cache.outside(mesh, outside, bounds));
        cache.finish();
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn interior_counter_reports_only_proven_interior_tests() {
        let mut assets = AssetLibrary::default();
        let mesh = assets.insert_mesh(Mesh::cube()).unwrap();
        let bounds = assets.mesh(mesh).unwrap().bounds();
        let centered = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5));
        let mut cache = VisibilityCache::default();
        cache.begin(Mat4::IDENTITY, true, 3);
        assert!(!cache.outside(mesh, centered * Mat4::from_scale(Vec3::splat(0.5)), bounds));
        // Touching the near and far planes is visible but not proven interior.
        assert!(!cache.outside(mesh, centered, bounds));
        assert!(cache.outside(mesh, Mat4::from_translation(Vec3::X * 10.0), bounds));
        cache.finish();
        assert_eq!((cache.tested, cache.interior, cache.reused), (3, 1, 0));
    }

    #[test]
    fn static_cache_extraction_still_validates_assets_and_replaces_caller_output() {
        let mut engine = Engine::new(EngineConfig::default());
        let material = engine
            .app
            .world_mut()
            .resource_mut::<AssetLibrary>()
            .insert_material(crate::Material::new([1.0; 4]).unwrap())
            .unwrap();
        let mesh = engine.cube_mesh;
        let entity = engine
            .app
            .world_mut()
            .spawn((Transform::default(), Mesh3d { mesh, material }))
            .id();
        let mut snapshot = RenderSnapshot::default();
        engine.extract(1.0, &mut snapshot);
        assert_eq!(snapshot.instances.len(), 1);
        let expected = snapshot.instances[0].model;
        snapshot.instances[0].model = Mat4::ZERO;
        snapshot.meshes.clear();
        engine.extract(1.0, &mut snapshot);
        assert_eq!(snapshot.instances[0].model, expected);
        assert_eq!(snapshot.meshes.len(), 1);
        assert_eq!((engine.visibility.tested, engine.visibility.reused), (0, 1));
        edit_transform(engine.app.world_mut(), entity, |transform| {
            transform.translation = Vec3::X * 10_000.0;
        })
        .unwrap();
        engine.extract(1.0, &mut snapshot);
        assert_eq!(snapshot.culled_instances, 1);
        assert!(snapshot.instances.is_empty());
        engine.set_frustum_culling(false);
        engine.extract(1.0, &mut snapshot);
        assert_eq!(snapshot.instances.len(), 1);
        assert_eq!(snapshot.culled_instances, 0);
        engine.set_frustum_culling(true);
        edit_transform(engine.app.world_mut(), entity, |transform| {
            transform.translation = Vec3::ZERO;
        })
        .unwrap();
        engine.extract(1.0, &mut snapshot);
        assert_eq!(snapshot.instances.len(), 1);
        engine
            .app
            .world_mut()
            .resource_mut::<AssetLibrary>()
            .remove_material(material);
        engine.extract(1.0, &mut snapshot);
        assert_eq!(snapshot.missing_assets, 1);
        assert!(snapshot.instances.is_empty());
        assert!(snapshot.meshes.is_empty());
        assert!(engine.visibility.entries.is_empty());
        engine.app.world_mut().despawn(entity);
        engine.extract(1.0, &mut snapshot);
        assert_eq!(snapshot.render_candidates, 0);
    }
}
