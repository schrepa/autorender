use std::collections::BTreeSet;

use super::*;

#[derive(Default)]
pub(super) struct Interests {
    grounds: BTreeMap<Vec<u8>, Hold>,
    portals: BTreeMap<Vec<u8>, Hold>,
}

enum Work {
    Ground(Background<WalkGround>),
    Portals(Background<TilePortals>),
}

impl Work {
    fn ground(&self) -> bool {
        match self {
            Self::Ground(_) => true,
            Self::Portals(_) => false,
        }
    }
}

struct Flight {
    input: Vec<u8>,
    work: Work,
}

enum Outcome {
    Waiting,
    Ready,
    Failed(String),
}

fn landed<T>(
    work: &mut Background<T>,
    records: &Records,
    name: &str,
) -> Option<Result<T, String>> {
    let outcome = match work.poll(records) {
        Landed::Waiting => Outcome::Waiting,
        Landed::Ready(_) => Outcome::Ready,
        Landed::Failed(error) => Outcome::Failed(format!("{name}: {}", worker_message(error))),
    };
    match outcome {
        Outcome::Waiting => None,
        Outcome::Ready => {
            let placeholder = Background::failed(Error::new(ErrorCode::InvalidRecord, "$"));
            std::mem::replace(work, placeholder).into_ready().map(Ok)
        }
        Outcome::Failed(message) => Some(Err(message)),
    }
}

fn undecoded(error: Error) -> Error {
    Error::new(ErrorCode::InvalidRecord, "$.navigation")
        .got(format!("its output does not decode: {error}"))
}

pub(super) enum Resolved {
    Progress,
    Waiting,
    Failed,
}

#[derive(Default)]
struct Kept {
    held: BTreeMap<Vec<u8>, (u64, Hold)>,
    order: BTreeMap<u64, Vec<u8>>,
    next: u64,
}

impl Kept {
    fn keep(&mut self, input: Vec<u8>, hold: Hold, most: usize) {
        self.next += 1;
        if let Some((stamp, _)) = self.held.insert(input.clone(), (self.next, hold)) {
            self.order.remove(&stamp);
        }
        self.order.insert(self.next, input);
        while self.held.len() > most {
            let Some((_, input)) = self.order.pop_first() else {
                break;
            };
            self.held.remove(&input);
        }
    }
}

#[derive(Default)]
pub(super) struct WalkRecords {
    grounds: Requests<Vec<u8>, Arc<WalkGround>>,
    portals: Requests<Vec<u8>, Arc<TilePortals>>,
    flights: Vec<Flight>,
    kept_grounds: Kept,
    kept_portals: Kept,
}

enum Found<T> {
    Here(T),
    Asked,
    Failed,
}

impl WalkRecords {
    pub(super) fn held(&self) -> (u32, u32) {
        (self.grounds.census().ready, self.portals.census().ready)
    }

    pub(super) fn settle(
        &mut self,
        (records, now, tick): (&Records, f64, u64),
        settings: &Navigation,
        failures: &mut JobFailures,
    ) {
        let grounds: BTreeSet<Vec<u8>> = self
            .grounds
            .settle(now)
            .into_iter()
            .map(|(input, _)| input)
            .collect();
        let portals: BTreeSet<Vec<u8>> = self
            .portals
            .settle(now)
            .into_iter()
            .map(|(input, _)| input)
            .collect();
        self.flights.retain(|flight| {
            let released = if flight.work.ground() {
                &grounds
            } else {
                &portals
            };
            !released.contains(&flight.input)
        });
        let most = settings.grounds_per_tick as usize;
        let mut decoded = 0;
        let mut waiting = Vec::with_capacity(self.flights.len());
        for mut flight in std::mem::take(&mut self.flights) {
            if flight.work.ground() && decoded >= most {
                waiting.push(flight);
                continue;
            }
            match &mut flight.work {
                Work::Ground(work) => {
                    let Some(made) = landed(work, records, WALK_GROUND_NAME) else {
                        waiting.push(flight);
                        continue;
                    };
                    decoded += 1;
                    end(&mut self.grounds, &flight.input, made, (tick, failures));
                }
                Work::Portals(work) => {
                    let Some(made) = landed(work, records, PORTALS_NAME) else {
                        waiting.push(flight);
                        continue;
                    };
                    end(&mut self.portals, &flight.input, made, (tick, failures));
                }
            }
        }
        self.flights = waiting;
    }

    pub(super) fn resolve(&mut self, running: &mut Running, jobs: &mut Jobs) -> Resolved {
        let gait = running.search.gait();
        let priority = job_priority(GAMEPLAY_JOBS, 0);
        let mut progress = false;
        for &want in running.search.wants() {
            let found = match want {
                Want::Ground(key) => {
                    let input = ground_input(&running.walked, key);
                    let found = wanted(&mut self.grounds, &mut running.interests.grounds, &input);
                    if let Found::Asked = found {
                        let pyramid = running.walked.source.pyramid;
                        self.fly(jobs, (&WALK_GROUND, &input), priority, |held| {
                            Work::Ground(Background::asked(held, move |bytes| {
                                walk_ground(key, &pyramid, bytes).map_err(undecoded)
                            }))
                        });
                    }
                    found.map(|ground| running.held.grounds.insert(key, ground))
                }
                Want::Portals(key) => {
                    let input = portals_input(&running.walked, key, &gait);
                    let found = wanted(&mut self.portals, &mut running.interests.portals, &input);
                    if let Found::Asked = found {
                        self.fly(jobs, (&PORTALS, &input), priority, portals_work(key));
                    }
                    found.map(|portals| running.held.portals.insert(key, portals))
                }
            };
            match found {
                Found::Here(()) => progress = true,
                Found::Failed => return Resolved::Failed,
                Found::Asked => {}
            }
        }
        if progress {
            Resolved::Progress
        } else {
            Resolved::Waiting
        }
    }

    pub(super) fn ask_ahead(&mut self, running: &mut Running, jobs: &mut Jobs) {
        let gait = running.search.gait();
        let priority = job_priority(GAMEPLAY_JOBS, 1);
        while let Some(&key) = running.search.ahead().get(running.asked_ahead) {
            let input = portals_input(&running.walked, key, &gait);
            let found = wanted(&mut self.portals, &mut running.interests.portals, &input);
            if let Found::Asked = found
                && !self.fly(jobs, (&PORTALS, &input), priority, portals_work(key))
            {
                return;
            }
            running.asked_ahead += 1;
        }
    }

    pub(super) fn keep(&mut self, interests: Interests, settings: &Navigation) {
        let Interests { grounds, portals } = interests;
        for (input, hold) in grounds {
            if self
                .grounds
                .get(&input)
                .is_some_and(|request| request.ready().is_some())
            {
                self.kept_grounds
                    .keep(input, hold, settings.kept_grounds as usize);
            }
        }
        for (input, hold) in portals {
            if self
                .portals
                .get(&input)
                .is_some_and(|request| request.ready().is_some())
            {
                self.kept_portals
                    .keep(input, hold, settings.kept_portals as usize);
            }
        }
    }

    fn fly(
        &mut self,
        jobs: &mut Jobs,
        (kind, input): (&'static JobKind, &Vec<u8>),
        priority: u32,
        work: impl FnOnce(Interest<RecordKey>) -> Work,
    ) -> bool {
        let ground = std::ptr::eq(kind, &WALK_GROUND);
        if self
            .flights
            .iter()
            .any(|flight| flight.work.ground() == ground && flight.input == *input)
        {
            return true;
        }
        if !jobs.admit(priority) {
            return false;
        }
        let job = jobs.start(kind, input.clone(), priority);
        self.flights.push(Flight {
            input: input.clone(),
            work: work(job),
        });
        true
    }
}

fn portals_work(key: TileKey) -> impl FnOnce(Interest<RecordKey>) -> Work {
    move |held| {
        Work::Portals(Background::asked(held, move |bytes| {
            walk_portals(key, bytes).map_err(undecoded)
        }))
    }
}

fn wanted<T>(
    requests: &mut Requests<Vec<u8>, Arc<T>>,
    held: &mut BTreeMap<Vec<u8>, Hold>,
    input: &Vec<u8>,
) -> Found<Arc<T>> {
    if !held.contains_key(input) {
        held.insert(input.clone(), requests.ask(input.clone()).into());
    }
    match requests.get(input) {
        Some(Request::Ready(value)) => Found::Here(Arc::clone(value)),
        Some(Request::Asked(_)) | None => Found::Asked,
        Some(Request::Failed(_) | Request::Missing) => Found::Failed,
    }
}

fn end<T>(
    requests: &mut Requests<Vec<u8>, Arc<T>>,
    input: &Vec<u8>,
    made: Result<T, String>,
    (tick, failures): (u64, &mut JobFailures),
) {
    let Some(ticket) = requests.ticket(input) else {
        return;
    };
    match made {
        Ok(value) => {
            requests.ready(input, ticket, Arc::new(value));
        }
        Err(error) => {
            failures.record(tick, error.clone());
            requests.fail(
                input,
                ticket,
                Error::new(ErrorCode::InvalidRecord, "$.navigation").got(error),
            );
        }
    }
}

fn walk_ground(key: TileKey, pyramid: &Pyramid, bytes: &[u8]) -> Result<WalkGround, Error> {
    let mut grid: GroundGrid = engine_codec::decode(bytes)?;
    if grid.key != key {
        return Err(engine_codec::invalid("$.key").got([
            u16::from(grid.key.level),
            grid.key.x,
            grid.key.z,
        ]));
    }
    let water = grid.water.take();
    Ok(WalkGround::new(&grid, water, pyramid))
}

fn walk_portals(key: TileKey, bytes: &[u8]) -> Result<TilePortals, Error> {
    let portals = portals_output(bytes)?;
    let held = portals.key();
    if held != key {
        return Err(engine_codec::invalid("$.key").got([
            u16::from(held.level),
            held.x,
            held.z,
        ]));
    }
    Ok(portals)
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
        }
    }
}
