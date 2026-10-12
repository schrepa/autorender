//! Generating tiles in place: the frame makes each tile itself within its
//! work time, a generator per tile carried over to the next frame when the
//! time runs out.

use super::*;

/// The frame generating a terrain's tiles from `layers`, the layers they
/// generate from now, in slices of `budget` while `work`'s time lasts;
/// `slots`, where workers ran, stops what they left. Each tile generates
/// whole, its shape handed out first where none of it is resident.
pub(super) struct InPlace<'a, 'b> {
    layers: Arc<Layers>,
    work: &'a Work,
    budget: &'a mut u64,
    slots: Option<&'a mut JobBudget<'b>>,
    /// Whether no tile was resident as the frame's tasks began: the
    /// coarsest level, which always draws, then generates whole at once.
    first: bool,
}

impl<'a, 'b> InPlace<'a, 'b> {
    pub(super) fn new(
        layers: Arc<Layers>,
        work: &'a Work,
        budget: &'a mut u64,
        slots: Option<&'a mut JobBudget<'b>>,
    ) -> Self {
        Self {
            layers,
            work,
            budget,
            slots,
            first: false,
        }
    }
}

impl Executor for InPlace<'_, '_> {
    /// Tiles handed to workers that stopped generate here, whole: their
    /// jobs stop, and resident tiles waiting on objects standing far away
    /// leave, to come back whole.
    fn take_over(&mut self, state: &mut TerrainState) {
        for (key, (id, _, _)) in state.dispatched.drain() {
            if let Some(slots) = self.slots.as_deref_mut() {
                slots.jobs.cancel(id);
            }
            state.failed.remove(&key);
        }
        for (key, (id, _, _)) in state.far_jobs.drain() {
            if let Some(slots) = self.slots.as_deref_mut() {
                slots.jobs.cancel(id);
            }
            state.far_wanted.insert(key, 0);
        }
        let banded = state
            .banded
            .drain()
            .into_iter()
            .flat_map(|(_, banding)| banding.jobs.into_iter().map(|(id, _)| id));
        let placing = state
            .placements
            .drain()
            .into_iter()
            .flat_map(|(_, placements)| placements.jobs.into_iter().map(|(id, _)| id));
        for id in banded.chain(placing).collect::<Vec<_>>() {
            if let Some(slots) = self.slots.as_deref_mut() {
                slots.jobs.cancel(id);
            }
        }
        for key in std::mem::take(&mut state.far_wanted).keys() {
            state.store.leave(&key);
        }
    }

    /// Nothing runs between frames here: generators continue as their
    /// tasks start.
    fn collect(&mut self, _: &mut TerrainState, _: &[(TileKey, u32)]) {}

    /// Tiles generate whole here, so no task places objects standing far
    /// away alone; every tile starts, or continues, in its turn.
    fn waits(&self, _: &TerrainState, task: &TileTask) -> bool {
        task.far
    }

    fn begin(&mut self, state: &TerrainState, _: &[TileTask]) {
        self.first = state.store.is_empty();
    }

    /// Generates `task`'s tile within the frame's time, or all of it at once
    /// for the coarsest level on a terrain with no tile resident.
    fn start(&mut self, state: &mut TerrainState, task: TileTask) -> bool {
        let TileTask { key, priority, .. } = task;
        let pyramid = state.terrain.pyramid;
        let whole = self.first && u32::from(key.level) + 1 == pyramid.levels();
        if !whole && *self.budget == 0 && !self.work.refill(self.budget) {
            return false;
        }
        let rules = &state.terrain.surface.rules;
        // Tiles not resident hand out their shape first.
        let staged = !state.store.contains(&key);
        let materials = state.materials;
        let layers = &self.layers;
        let job = state.jobs.entry(key).or_insert_with(|| {
            Generator::new(layers.clone(), &pyramid, key)
                .with_surface(rules.clone())
                .with_staged(staged)
                .with_materials(materials)
        });
        let tile = if whole {
            let mut unlimited = u64::MAX;
            job.advance(&mut unlimited)
        } else {
            loop {
                if let Some(tile) = job.advance(self.budget) {
                    break Some(tile);
                }
                if !self.work.refill(self.budget) {
                    break None;
                }
            }
        };
        if let Some(tile) = tile {
            state.jobs.remove(&key);
            state.arrive(tile, priority, None);
        } else if let Some(shape) = state.jobs.get_mut(&key).and_then(Generator::take_shape) {
            state.shape(shape);
        }
        true
    }

    fn ahead(&mut self, _: &mut TerrainState, _: &[(TileKey, u32)]) {}
}
