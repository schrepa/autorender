use super::*;

const GROUNDS_KEPT: usize = 64;
const PORTALS_KEPT: usize = 8192;
const RETRY_AFTER: u64 = 120;
const GROUNDS_PER_TICK: usize = 1;

enum Kept<T> {
    Asked,
    Here(Arc<T>, u64),
    Failed(u64),
}

struct Asked {
    record: RecordKey,
    id: JobId,
    ground: Option<AskedGround>,
}

struct AskedGround {
    source: GroundSource,
    surfaces: Surfaces,
    key: TileKey,
    kind: &'static str,
}

pub(super) enum Resolved {
    Progress,
    Waiting,
    Failed,
}

#[derive(Default)]
pub(super) struct WalkRecords {
    grounds: BTreeMap<RecordKey, Kept<WalkGround>>,
    portals: BTreeMap<RecordKey, Kept<TilePortals>>,
    asked: Vec<Asked>,
    reads: u64,
}

impl WalkRecords {
    pub(super) fn held(&self) -> (u32, u32) {
        (here(&self.grounds), here(&self.portals))
    }

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
                Some(ground) => {
                    grounds += 1;
                    let made = job_output(ground.kind, output, |bytes| -> Result<_, Error> {
                        let grid = ground.source.regions.grid(ground.key, bytes)?;
                        let water = surface_water(&ground.surfaces, ground.key, bytes)?;
                        Ok(WalkGround::new(&grid, water, &ground.source.pyramid))
                    });
                    let kept = kept(made, (tick, &mut self.reads), failures);
                    self.grounds.insert(asked.record, kept);
                }
                None => {
                    let made = job_output(PortalsJob::KIND, output, portals_output);
                    let kept = kept(made, (tick, &mut self.reads), failures);
                    self.portals.insert(asked.record, kept);
                }
            }
        }
        forget(&mut self.grounds, GROUNDS_KEPT, tick);
        forget(&mut self.portals, PORTALS_KEPT, tick);
    }

    pub(super) fn resolve(
        &mut self,
        running: &mut Running,
        slots: &mut Option<JobBudget<'_>>,
        tick: u64,
    ) -> Resolved {
        let gait = running.search.gait();
        let priority = job_priority(GAMEPLAY_JOBS, 0);
        let mut progress = false;
        for &want in running.search.wants() {
            let found = match want {
                Want::Ground(key) => {
                    let GroundRecord {
                        kind,
                        input,
                        key: record,
                    } = running.source.regions.record(key);
                    let found = read(&mut self.grounds, record, (tick, &mut self.reads));
                    if let Found::Missing = found {
                        let ground = Some(AskedGround {
                            source: running.source.clone(),
                            surfaces: running.surfaces.clone(),
                            key,
                            kind,
                        });
                        self.ask(slots, (kind, input, record), priority, ground);
                    }
                    found.map(|ground| running.held.grounds.insert(key, ground))
                }
                Want::Portals(key) => {
                    let (input, record) = portals_record(&running.source, key, &gait);
                    let found = read(&mut self.portals, record, (tick, &mut self.reads));
                    if let Found::Missing = found {
                        self.ask(slots, (PortalsJob::KIND, input, record), priority, None);
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

    pub(super) fn ask_ahead(
        &mut self,
        running: &mut Running,
        slots: &mut Option<JobBudget<'_>>,
        tick: u64,
    ) {
        let gait = running.search.gait();
        let priority = job_priority(GAMEPLAY_JOBS, 1);
        while let Some(&key) = running.search.ahead().get(running.asked_ahead) {
            let (input, record) = portals_record(&running.source, key, &gait);
            if let Found::Missing = read(&mut self.portals, record, (tick, &mut self.reads))
                && !self.ask(slots, (PortalsJob::KIND, input, record), priority, None)
            {
                return;
            }
            running.asked_ahead += 1;
        }
    }

    fn ask(
        &mut self,
        slots: &mut Option<JobBudget<'_>>,
        (kind, input, record): (&'static str, Vec<u8>, RecordKey),
        priority: u32,
        ground: Option<AskedGround>,
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

    pub(super) fn clear(&mut self, jobs: Option<&mut Jobs>) {
        if let Some(jobs) = jobs {
            for asked in &self.asked {
                jobs.cancel(asked.id);
            }
        }
        *self = Self::default();
    }
}

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

fn here<T>(records: &BTreeMap<RecordKey, Kept<T>>) -> u32 {
    records
        .values()
        .filter(|kept| matches!(kept, Kept::Here(..)))
        .count() as u32
}

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

fn surface_water(
    surfaces: &[Arc<dyn WaterSurfaces>],
    key: TileKey,
    ground: &[u8],
) -> Result<Option<Box<[f32]>>, Error> {
    for surface in surfaces {
        if let Some(water) = surface.decode(key, ground)? {
            return Ok(Some(water));
        }
    }
    Ok(None)
}

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
