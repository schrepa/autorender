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
    input: Vec<u8>,
    ticket: Ticket,
    ground: Option<AskedGround>,
}

struct AskedGround {
    key: TileKey,
    pyramid: Pyramid,
}

pub(super) enum Resolved {
    Progress,
    Waiting,
    Failed,
}

#[derive(Default)]
pub(super) struct WalkRecords {
    grounds: BTreeMap<Vec<u8>, Kept<WalkGround>>,
    portals: BTreeMap<Vec<u8>, Kept<TilePortals>>,
    asked: Vec<Asked>,
    reads: u64,
}

impl WalkRecords {
    pub(super) fn held(&self) -> (u32, u32) {
        (here(&self.grounds), here(&self.portals))
    }

    pub(super) fn take(
        &mut self,
        records: &Records,
        jobs: &mut Jobs,
        tick: u64,
        failures: &mut Failures,
    ) {
        let mut grounds = 0;
        for asked in std::mem::take(&mut self.asked) {
            if asked.ground.is_some() && grounds >= GROUNDS_PER_TICK {
                self.asked.push(asked);
                continue;
            }
            let Some(output) = records.get(&asked.ticket) else {
                self.asked.push(asked);
                continue;
            };
            let Asked {
                input,
                ticket,
                ground,
            } = asked;
            jobs.cancel(ticket);
            match ground {
                Some(ground) => {
                    grounds += 1;
                    let made = made(WALK_GROUND_NAME, output, |bytes| walk_ground(&ground, bytes));
                    let kept = kept(made, (tick, &mut self.reads), failures);
                    self.grounds.insert(input, kept);
                }
                None => {
                    let made = made(PORTALS_NAME, output, portals_output);
                    let kept = kept(made, (tick, &mut self.reads), failures);
                    self.portals.insert(input, kept);
                }
            }
        }
        forget(&mut self.grounds, GROUNDS_KEPT, tick);
        forget(&mut self.portals, PORTALS_KEPT, tick);
    }

    pub(super) fn resolve(&mut self, running: &mut Running, jobs: &mut Jobs, tick: u64) -> Resolved {
        let gait = running.search.gait();
        let priority = job_priority(GAMEPLAY_JOBS, 0);
        let mut progress = false;
        for &want in running.search.wants() {
            let found = match want {
                Want::Ground(key) => {
                    let input = ground_input(&running.walked, key);
                    let found = read(&mut self.grounds, &input, (tick, &mut self.reads));
                    if let Found::Missing = found {
                        let ground = Some(AskedGround {
                            key,
                            pyramid: running.walked.source.pyramid,
                        });
                        self.ask(jobs, (&WALK_GROUND, input), priority, ground);
                    }
                    found.map(|ground| running.held.grounds.insert(key, ground))
                }
                Want::Portals(key) => {
                    let input = portals_input(&running.walked, key, &gait);
                    let found = read(&mut self.portals, &input, (tick, &mut self.reads));
                    if let Found::Missing = found {
                        self.ask(jobs, (&PORTALS, input), priority, None);
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

    pub(super) fn ask_ahead(&mut self, running: &mut Running, jobs: &mut Jobs, tick: u64) {
        let gait = running.search.gait();
        let priority = job_priority(GAMEPLAY_JOBS, 1);
        while let Some(&key) = running.search.ahead().get(running.asked_ahead) {
            let input = portals_input(&running.walked, key, &gait);
            if let Found::Missing = read(&mut self.portals, &input, (tick, &mut self.reads)) {
                self.ask(jobs, (&PORTALS, input), priority, None);
            }
            running.asked_ahead += 1;
        }
    }

    fn ask(
        &mut self,
        jobs: &mut Jobs,
        (kind, input): (&'static JobKind, Vec<u8>),
        priority: u32,
        ground: Option<AskedGround>,
    ) {
        let ticket = jobs.fetch(kind, &input, priority);
        if ground.is_some() {
            self.grounds.insert(input.clone(), Kept::Asked);
        } else {
            self.portals.insert(input.clone(), Kept::Asked);
        }
        self.asked.push(Asked {
            input,
            ticket,
            ground,
        });
    }

    pub(super) fn clear(&mut self, jobs: &mut Jobs) {
        for asked in std::mem::take(&mut self.asked) {
            jobs.cancel(asked.ticket);
        }
        *self = Self::default();
    }
}

fn walk_ground(ground: &AskedGround, bytes: &[u8]) -> Result<WalkGround, Error> {
    let mut grid: GroundGrid = engine_codec::decode(bytes)?;
    if grid.key != ground.key {
        return Err(engine_codec::invalid("$.key").got([
            u16::from(grid.key.level),
            grid.key.x,
            grid.key.z,
        ]));
    }
    let water = grid.water.take();
    Ok(WalkGround::new(&grid, water, &ground.pyramid))
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

fn here<T>(records: &BTreeMap<Vec<u8>, Kept<T>>) -> u32 {
    records
        .values()
        .filter(|kept| matches!(kept, Kept::Here(..)))
        .count() as u32
}

fn read<T>(
    records: &mut BTreeMap<Vec<u8>, Kept<T>>,
    input: &[u8],
    (tick, reads): (u64, &mut u64),
) -> Found<Arc<T>> {
    match records.get_mut(input) {
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
    failures: &mut Failures,
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

fn forget<T>(records: &mut BTreeMap<Vec<u8>, Kept<T>>, most: usize, tick: u64) {
    records.retain(|_, kept| !matches!(kept, Kept::Failed(at) if tick >= *at + RETRY_AFTER));
    let mut here: Vec<(u64, &[u8])> = records
        .iter()
        .filter_map(|(input, kept)| match kept {
            Kept::Here(_, read) => Some((*read, input.as_slice())),
            Kept::Asked | Kept::Failed(_) => None,
        })
        .collect();
    if here.len() <= most {
        return;
    }
    here.sort_unstable();
    let over = here.len() - most;
    let gone: Vec<Vec<u8>> = here
        .into_iter()
        .take(over)
        .map(|(_, input)| input.to_vec())
        .collect();
    for input in gone {
        records.remove(&input);
    }
}
