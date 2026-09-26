//! Bounded component replacements for tracked transform mutation.

use crate::{MAX_TRANSFORM_NODES, Transform, hierarchy::same_local_bits};
use bevy_ecs::{
    prelude::*,
    system::{Deferred, SystemBuffer, SystemMeta, SystemParam},
    world::DeferredWorld,
};
use std::fmt;

/// Maximum queued replacements per system between deferred-application points.
pub const MAX_TRANSFORM_EDITS: usize = MAX_TRANSFORM_NODES;

/// A rejected edit. Already accepted deferred edits remain queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformEditError {
    CapacityExceeded,
    MissingEntity(Entity),
    MissingTransform(Entity),
}

impl fmt::Display for TransformEditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityExceeded => write!(
                formatter,
                "transform edit buffer exceeds {MAX_TRANSFORM_EDITS} replacements"
            ),
            Self::MissingEntity(entity) => write!(formatter, "entity {entity:?} does not exist"),
            Self::MissingTransform(entity) => {
                write!(formatter, "entity {entity:?} has no local Transform")
            }
        }
    }
}

impl std::error::Error for TransformEditError {}

/// Immediately edits an existing local pose without advancing the simulation.
///
/// Writes through ordinary mutable component access, preserving the added tick
/// without invoking Insert observers. A bit-identical result leaves the change
/// tick untouched. Propagate transforms before querying derived globals; reset
/// interpolation for a teleport. Use `TransformEdits` when deferred replacement
/// and its component lifecycle observers are required.
pub fn edit_transform<R>(
    world: &mut World,
    entity: Entity,
    edit: impl FnOnce(&mut Transform) -> R,
) -> Result<R, TransformEditError> {
    let source = world.get::<Transform>(entity).copied().ok_or_else(|| {
        if world.entities().contains(entity) {
            TransformEditError::MissingTransform(entity)
        } else {
            TransformEditError::MissingEntity(entity)
        }
    })?;
    let mut next = source;
    let result = edit(&mut next);
    if !same_local_bits(&source, &next) {
        *world
            .get_mut::<Transform>(entity)
            .expect("editing a source copy cannot remove its world component") = next;
    }
    Ok(result)
}

/// Queues local-pose replacements using one reusable buffer per system.
///
/// Read poses with ordinary `Query<&Transform>` and submit edited copies here.
/// Edits become visible at Bevy's deferred-application points, including the end
/// of each engine schedule. Chain dependent systems to apply earlier edits
/// before reading them. Two edits based on the same source do not compose:
/// replacements apply in submission order, with the last replacement winning.
///
/// Targets without a Transform when application begins are skipped, including
/// despawned entities; this API is for editing existing poses, not spawning them.
/// Bevy component observers run normally during the batch. Ordinary application
/// retains buffer capacity; an explicit `System::queue_deferred` transfers the
/// buffer into one command. Detailed profiling reports queued/applied/skipped
/// counts. No per-entity commands or hash tables are created.
#[derive(SystemParam)]
pub struct TransformEdits<'s> {
    buffer: Deferred<'s, EditBuffer>,
}

impl TransformEdits<'_> {
    /// Queue a replacement. This does not inspect the entity until application.
    pub fn set(&mut self, entity: Entity, transform: Transform) -> Result<(), TransformEditError> {
        self.buffer.push(entity, transform)
    }

    /// Edit a source copy and queue it only if at least one field's bits change.
    pub fn edit(
        &mut self,
        entity: Entity,
        source: &Transform,
        edit: impl FnOnce(&mut Transform),
    ) -> Result<(), TransformEditError> {
        let mut next = *source;
        edit(&mut next);
        if same_local_bits(source, &next) {
            return Ok(());
        }
        self.set(entity, next)
    }

    /// Append a batch, rejecting the whole new batch if it exceeds capacity.
    /// Previously accepted replacements are retained on rejection.
    pub fn set_batch(
        &mut self,
        edits: impl IntoIterator<Item = (Entity, Transform)>,
    ) -> Result<(), TransformEditError> {
        let start = self.buffer.values.len();
        for (entity, transform) in edits {
            if let Err(error) = self.buffer.push(entity, transform) {
                self.buffer.values.truncate(start);
                return Err(error);
            }
        }
        Ok(())
    }

    /// Accepted replacements waiting for the next deferred-application point.
    pub fn pending(&self) -> usize {
        self.buffer.values.len()
    }
}

#[derive(Default)]
struct EditBuffer {
    values: Vec<(Entity, Transform)>,
}

impl EditBuffer {
    fn push(&mut self, entity: Entity, transform: Transform) -> Result<(), TransformEditError> {
        if self.values.len() == MAX_TRANSFORM_EDITS {
            return Err(TransformEditError::CapacityExceeded);
        }
        if self.values.len() == self.values.capacity() {
            let capacity = (self.values.capacity().max(32) * 2).min(MAX_TRANSFORM_EDITS);
            self.values.reserve_exact(capacity - self.values.len());
        }
        self.values.push((entity, transform));
        Ok(())
    }

    fn apply(&mut self, world: &mut World) {
        if self.values.is_empty() {
            return;
        }
        let _profile = engine_profile::span("hierarchy", "apply_transform_edits");
        let queued = self.values.len();
        // Snapshot admission before any replacement can trigger observers. Only
        // membership is needed: avoid resolving Transform's type and column for
        // every row, and share the membership result within an archetype run.
        if let Some(component) = world.component_id::<Transform>() {
            let entities = world.entities();
            let archetypes = world.archetypes();
            let mut previous = None;
            self.values.retain(|(entity, _)| {
                let Ok(location) = entities.get_spawned(*entity) else {
                    return false;
                };
                if let Some((archetype, contains)) = previous
                    && archetype == location.archetype_id
                {
                    return contains;
                }
                let contains = archetypes[location.archetype_id].contains(component);
                previous = Some((location.archetype_id, contains));
                contains
            });
        } else {
            self.values.clear();
        }
        // Keep Bevy's batch insertion, including its entity rechecks, so hooks,
        // observer order, added ticks and archetype/bundle reuse stay unchanged.
        let admitted = self.values.len();
        let missing = world
            .try_insert_batch(self.values.drain(..))
            .err()
            .map_or(0, |error| error.entities.len());
        let applied = admitted - missing;
        // Batch insertion queues observer commands; publish their results at
        // this deferred boundary before the next ordered system can read them.
        world.flush();
        engine_profile::counter("transform_edits.queued", queued as f64);
        engine_profile::counter("transform_edits.applied", applied as f64);
        engine_profile::counter("transform_edits.skipped", (queued - applied) as f64);
        engine_profile::counter(
            "transform_edits.buffer_capacity_bytes",
            (self.values.capacity() * size_of::<(Entity, Transform)>()) as f64,
        );
    }
}

impl SystemBuffer for EditBuffer {
    fn apply(&mut self, _system_meta: &SystemMeta, world: &mut World) {
        self.apply(world);
    }

    fn queue(&mut self, _system_meta: &SystemMeta, mut world: DeferredWorld) {
        if !self.values.is_empty() {
            let mut buffer = std::mem::take(self);
            world
                .commands()
                .queue(move |world: &mut World| buffer.apply(world));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::system::SystemState;

    #[test]
    fn direct_edits_preserve_lifetime_and_skip_bit_identical_values() {
        #[derive(Resource, Default)]
        struct Insertions(usize);

        let mut world = World::new();
        let entity = world.spawn(Transform::default()).id();
        world.init_resource::<Insertions>();
        world.add_observer(|_: On<Insert, Transform>, mut count: ResMut<Insertions>| {
            count.0 += 1;
        });
        world.flush();
        let added = world.entity(entity).get_ref::<Transform>().unwrap().added();
        world.increment_change_tick();
        assert_eq!(
            edit_transform(&mut world, entity, |pose| {
                pose.translation.x = f32::from_bits(0x7fc0_0042);
                7
            }),
            Ok(7)
        );
        let changed = world
            .entity(entity)
            .get_ref::<Transform>()
            .unwrap()
            .last_changed();
        world.increment_change_tick();
        edit_transform(&mut world, entity, |_| {}).unwrap();
        let pose = world.entity(entity).get_ref::<Transform>().unwrap();
        assert_eq!(pose.added(), added);
        assert_eq!(pose.last_changed(), changed);
        assert_ne!(changed, added);
        assert_eq!(pose.translation.x.to_bits(), 0x7fc0_0042);
        assert_eq!(world.resource::<Insertions>().0, 0);
        world.entity_mut(entity).remove::<Transform>();
        assert_eq!(
            edit_transform(&mut world, entity, |_| {}),
            Err(TransformEditError::MissingTransform(entity))
        );
        world.entity_mut(entity).despawn();
        assert_eq!(
            edit_transform(&mut world, entity, |_| {}),
            Err(TransformEditError::MissingEntity(entity))
        );
    }

    #[test]
    fn chained_systems_see_deferred_edits_and_stale_targets_are_skipped() {
        fn increment(poses: Query<(Entity, &Transform)>, mut edits: TransformEdits) {
            for (entity, source) in &poses {
                edits
                    .edit(entity, source, |pose| pose.translation.x += 1.0)
                    .unwrap();
            }
        }

        let mut world = World::new();
        let entity = world.spawn(Transform::default()).id();
        let mut schedule = Schedule::default();
        schedule.add_systems((increment, increment).chain());
        schedule.run(&mut world);
        assert_eq!(world.get::<Transform>(entity).unwrap().translation.x, 2.0);

        let removed = world.spawn(Transform::default()).id();
        let despawned = world.spawn(Transform::default()).id();
        let mut state = SystemState::<TransformEdits>::new(&mut world);
        {
            let mut edits = state.get_mut(&mut world).unwrap();
            edits
                .set_batch([entity, removed, despawned].map(|id| {
                    (
                        id,
                        Transform {
                            translation: glam::Vec3::X * 3.0,
                            ..Default::default()
                        },
                    )
                }))
                .unwrap();
            assert_eq!(edits.pending(), 3);
        }
        assert_eq!(world.get::<Transform>(entity).unwrap().translation.x, 2.0);
        world.entity_mut(removed).remove::<Transform>();
        world.entity_mut(despawned).despawn();
        state.apply(&mut world);
        assert_eq!(world.get::<Transform>(entity).unwrap().translation.x, 3.0);
        assert!(world.get::<Transform>(removed).is_none());
        assert!(!world.entities().contains(despawned));
    }

    #[test]
    fn capacity_rejection_keeps_earlier_edits_and_rejects_a_whole_new_batch() {
        let mut world = World::new();
        let entity = world.spawn(Transform::default()).id();
        let mut state = SystemState::<TransformEdits>::new(&mut world);
        let mut edits = state.get_mut(&mut world).unwrap();
        edits.set(entity, Transform::default()).unwrap();
        assert_eq!(
            edits.set_batch(std::iter::repeat_n(
                (entity, Transform::default()),
                MAX_TRANSFORM_EDITS
            )),
            Err(TransformEditError::CapacityExceeded)
        );
        assert_eq!(edits.pending(), 1);
        edits
            .set_batch(std::iter::repeat_n(
                (entity, Transform::default()),
                MAX_TRANSFORM_EDITS - 1,
            ))
            .unwrap();
        assert_eq!(edits.pending(), MAX_TRANSFORM_EDITS);
        assert_eq!(
            edits.set(entity, Transform::default()),
            Err(TransformEditError::CapacityExceeded)
        );
        edits.edit(entity, &Transform::default(), |_| {}).unwrap();
        assert_eq!(edits.pending(), MAX_TRANSFORM_EDITS);
        assert!(edits.buffer.values.capacity() <= MAX_TRANSFORM_EDITS);
    }

    #[test]
    fn admission_checks_archetype_runs_before_observers_and_keeps_replacement_order() {
        #[derive(Component)]
        struct Tagged;
        #[derive(Resource, Default)]
        struct Observed(Vec<(Entity, f32)>);

        let pose = |x| Transform {
            translation: glam::Vec3::X * x,
            ..Default::default()
        };
        let mut world = World::new();
        let missing = world.spawn_empty().id();
        let mut buffer = EditBuffer::default();
        buffer.push(missing, pose(7.0)).unwrap();
        // Transform has not even been registered in this world yet.
        buffer.apply(&mut world);
        assert!(world.get::<Transform>(missing).is_none());

        let first = world.spawn(Transform::default()).id();
        let second = world.spawn(Transform::default()).id();
        let tagged = world.spawn((Transform::default(), Tagged)).id();
        let other_missing = world.spawn(Tagged).id();
        let removed = world.spawn(Transform::default()).id();
        world.entity_mut(removed).despawn();
        world.init_resource::<Observed>();
        world.add_observer(
            move |event: On<Insert, Transform>,
                  poses: Query<&Transform>,
                  mut observed: ResMut<Observed>,
                  mut commands: Commands| {
                observed
                    .0
                    .push((event.entity, poses.get(event.entity).unwrap().translation.x));
                if event.entity == first && observed.0.len() == 1 {
                    commands.entity(missing).insert(Transform {
                        translation: glam::Vec3::X * 99.0,
                        ..Default::default()
                    });
                }
            },
        );
        let added = world.entity(first).get_ref::<Transform>().unwrap().added();
        world.increment_change_tick();
        for (entity, x) in [
            (first, 1.0),
            (second, 2.0),
            (missing, 7.0),
            (other_missing, 8.0),
            (removed, 9.0),
            (tagged, 3.0),
            (first, 4.0),
        ] {
            buffer.push(entity, pose(x)).unwrap();
        }
        let capacity = buffer.values.capacity();
        buffer.apply(&mut world);
        world.flush();

        assert_eq!(
            world.resource::<Observed>().0,
            vec![
                (first, 1.0),
                (second, 2.0),
                (tagged, 3.0),
                (first, 4.0),
                (missing, 99.0),
            ]
        );
        assert!(world.get::<Transform>(other_missing).is_none());
        assert!(!world.entities().contains(removed));
        assert_eq!(
            world.entity(first).get_ref::<Transform>().unwrap().added(),
            added
        );
        assert!(buffer.values.is_empty());
        assert_eq!(buffer.values.capacity(), capacity);
    }
}
