//! The records walks read, held for every search: tiles' grounds (from the
//! tiles' records) and their portals of each gait's walk graph, asked of
//! the host as searches want them and kept for the searches after, the
//! least recently read going first past a count.

use super::*;

/// Grounds and portals kept at most once no search reads them, beyond
/// which the least recently read go.
const GROUNDS_KEPT: usize = 64;
const PORTALS_KEPT: usize = 8192;
/// Ticks a record that failed waits before it is asked for again; searches
/// reading it meanwhile find no walk.
const RETRY_AFTER: u64 = 120;
/// Tiles' records made grounds a tick at most: each is a whole tile
/// decoded.
const GROUNDS_PER_TICK: usize = 1;

/// A record as walks hold it: asked for, here (with the count of reads
/// when it was last read), or failed at a tick.
enum Kept<T> {
    Asked,
    Here(Arc<T>, u64),
    Failed(u64),
}

/// A record asked for: its key, its job and what it is for: a tile's
/// ground, made from the tile's record over its terrain's layers, or a
/// tile's portals.
struct Asked {
    record: RecordKey,
    id: JobId,
    ground: Option<Arc<WalkedTerrain>>,
}

/// What a search's wants came to: some held now, none yet, or one failed.
pub(super) enum Resolved {
    Progress,
    Waiting,
    Failed,
}

/// The records walks read, by key, and how many reads and arrivals there
/// were: the least recently read go by it, whether ticks pass or not.
#[derive(Default)]
pub(super) struct WalkRecords {
    grounds: BTreeMap<RecordKey, Kept<WalkGround>>,
    portals: BTreeMap<RecordKey, Kept<TilePortals>>,
    asked: Vec<Asked>,
    reads: u64,
}

impl WalkRecords {
    /// Grounds and portals held now.
    pub(super) fn held(&self) -> (u32, u32) {
        (here(&self.grounds), here(&self.portals))
    }

    /// Takes what `jobs` handed back of the records asked for at tick
    /// `tick`: every tile's portals, and tiles' records made grounds up to
    /// [`GROUNDS_PER_TICK`]; what failed counts in `failures`.
    pub(super) fn take(&mut self, jobs: &mut Jobs, tick: u64, failures: &mut JobFailures) {
        let mut grounds = 0;
        for asked in std::mem::take(&mut self.asked) {
            if asked.ground.is_some() && grounds >= GROUNDS_PER_TICK {
                self.asked.push(asked);
                continue;
            }
            let Some(output) = jobs.take(asked.id) else {
                self.asked.push(asked);
                continue;
            };
            match &asked.ground {
                Some(terrain) => {
                    grounds += 1;
                    let made = job_output(
                        TileJob::KIND.name(),
                        output,
                        engine_terrain::jobs::tile_output,
                    )
                    .map(|tile| WalkGround::new(&tile, &terrain.layers, &terrain.pyramid));
                    let kept = kept(made, (tick, &mut self.reads), failures);
                    self.grounds.insert(asked.record, kept);
                }
                None => {
                    let made = job_output(
                        PortalsJob::KIND.name(),
                        output,
                        engine_terrain::jobs::portals_output,
                    );
                    let kept = kept(made, (tick, &mut self.reads), failures);
                    self.portals.insert(asked.record, kept);
                }
            }
        }
        forget(&mut self.grounds, GROUNDS_KEPT, tick);
        forget(&mut self.portals, PORTALS_KEPT, tick);
    }

    /// Hands `running` the records its search wants that are here, asking
    /// `slots` for those not asked for yet, at tick `tick`.
    pub(super) fn resolve(
        &mut self,
        running: &mut Running,
        slots: &mut Option<JobBudget<'_>>,
        tick: u64,
    ) -> Resolved {
        let gait = running.search.gait();
        let priority = crate::job_priority(crate::GAMEPLAY_JOBS, 0);
        let mut progress = false;
        for &want in running.search.wants() {
            let found = match want {
                Want::Ground(key) => {
                    let (input, record) = running.terrain.tile_record(key);
                    let found = read(&mut self.grounds, record, (tick, &mut self.reads));
                    if let Found::Missing = found {
                        let ground = Some(running.terrain.clone());
                        self.ask(
                            slots,
                            (TileJob::KIND.name(), input, record),
                            priority,
                            ground,
                        );
                    }
                    found.map(|ground| running.held.grounds.insert(key, ground))
                }
                Want::Portals(key) => {
                    let (input, record) = running.terrain.portals_record(key, &gait);
                    let found = read(&mut self.portals, record, (tick, &mut self.reads));
                    if let Found::Missing = found {
                        self.ask(
                            slots,
                            (PortalsJob::KIND.name(), input, record),
                            priority,
                            None,
                        );
                    }
                    found.map(|portals| running.held.portals.insert(key, portals))
                }
            };
            match found {
                Found::Here(()) => progress = true,
                Found::Failed => return Resolved::Failed,
                Found::Asked | Found::Missing => {}
            }
        }
        if progress {
            Resolved::Progress
        } else {
            Resolved::Waiting
        }
    }

    /// Asks `slots` for the portals of the tiles `running`'s search likely
    /// reads that it has not asked for yet, those neither here nor asked
    /// for, as far as they admit.
    pub(super) fn ask_ahead(
        &mut self,
        running: &mut Running,
        slots: &mut Option<JobBudget<'_>>,
        tick: u64,
    ) {
        let gait = running.search.gait();
        let priority = crate::job_priority(crate::GAMEPLAY_JOBS, 1);
        while let Some(&key) = running.search.ahead().get(running.asked_ahead) {
            let (input, record) = running.terrain.portals_record(key, &gait);
            if let Found::Missing = read(&mut self.portals, record, (tick, &mut self.reads))
                && !self.ask(
                    slots,
                    (PortalsJob::KIND.name(), input, record),
                    priority,
                    None,
                )
            {
                return;
            }
            running.asked_ahead += 1;
        }
    }

    /// Asks `slots` for the record of `job` (its kind, input and key) at
    /// `priority`, a ground made over `ground`'s terrain where given;
    /// whether a slot admitted it.
    fn ask(
        &mut self,
        slots: &mut Option<JobBudget<'_>>,
        (kind, input, record): (&'static str, Vec<u8>, RecordKey),
        priority: u32,
        ground: Option<Arc<WalkedTerrain>>,
    ) -> bool {
        let Some(slots) = slots.as_mut() else {
            return false;
        };
        if !slots.take(priority) {
            return false;
        }
        let id = slots.jobs.submit_record(kind, input, priority, record);
        if ground.is_some() {
            self.grounds.insert(record, Kept::Asked);
        } else {
            self.portals.insert(record, Kept::Asked);
        }
        self.asked.push(Asked { record, id, ground });
        true
    }

    /// Drops every record, the jobs still making any cancelled.
    pub(super) fn clear(&mut self, jobs: Option<&mut Jobs>) {
        if let Some(jobs) = jobs {
            for asked in &self.asked {
                jobs.cancel(asked.id);
            }
        }
        *self = Self::default();
    }
}

/// What a search found of a record it wants: here (and handed over),
/// asked for, failed a while ago, or neither.
enum Found<T> {
    Here(T),
    Asked,
    Failed,
    Missing,
}

impl<T> Found<T> {
    fn map<U>(self, change: impl FnOnce(T) -> U) -> Found<()> {
        match self {
            Self::Here(value) => {
                change(value);
                Found::Here(())
            }
            Self::Asked => Found::Asked,
            Self::Failed => Found::Failed,
            Self::Missing => Found::Missing,
        }
    }
}

/// How many of `records` are here.
fn here<T>(records: &BTreeMap<RecordKey, Kept<T>>) -> u32 {
    records
        .values()
        .filter(|kept| matches!(kept, Kept::Here(..)))
        .count() as u32
}

/// Record `record` of `records` as a search reads it at tick `tick`, the
/// read counted in `reads`.
fn read<T>(
    records: &mut BTreeMap<RecordKey, Kept<T>>,
    record: RecordKey,
    (tick, reads): (u64, &mut u64),
) -> Found<Arc<T>> {
    match records.get_mut(&record) {
        Some(Kept::Here(value, read)) => {
            *reads += 1;
            *read = *reads;
            Found::Here(value.clone())
        }
        Some(Kept::Asked) => Found::Asked,
        Some(Kept::Failed(at)) if tick < *at + RETRY_AFTER => Found::Failed,
        Some(Kept::Failed(_)) | None => Found::Missing,
    }
}

/// A record made at tick `tick` as kept: here, read last as the latest of
/// `reads`, or failed, counted in `failures`.
fn kept<T>(
    made: Result<T, String>,
    (tick, reads): (u64, &mut u64),
    failures: &mut JobFailures,
) -> Kept<T> {
    match made {
        Ok(value) => {
            *reads += 1;
            Kept::Here(Arc::new(value), *reads)
        }
        Err(error) => {
            failures.record(tick, error);
            Kept::Failed(tick)
        }
    }
}

/// Keeps `most` of `records` here, the least recently read going first,
/// and forgets those that failed long enough ago at tick `tick`.
fn forget<T>(records: &mut BTreeMap<RecordKey, Kept<T>>, most: usize, tick: u64) {
    records.retain(|_, kept| !matches!(kept, Kept::Failed(at) if tick >= *at + RETRY_AFTER));
    let mut here: Vec<(u64, RecordKey)> = records
        .iter()
        .filter_map(|(&record, kept)| match kept {
            Kept::Here(_, read) => Some((*read, record)),
            Kept::Asked | Kept::Failed(_) => None,
        })
        .collect();
    if here.len() <= most {
        return;
    }
    here.sort_unstable();
    let over = here.len() - most;
    for (_, record) in here.into_iter().take(over) {
        records.remove(&record);
    }
}
