use std::collections::BTreeSet;

use super::*;

#[derive(Default)]
pub(super) struct Interests {
    grounds: BTreeMap<Vec<u8>, Interest<Vec<u8>>>,
    portals: BTreeMap<Vec<u8>, Interest<Vec<u8>>>,
}

#[derive(Clone, Copy)]
enum Decode {
    Ground { key: TileKey, pyramid: Pyramid },
    Portals { key: TileKey },
}

impl Decode {
    fn ground(self) -> bool {
        match self {
            Self::Ground { .. } => true,
            Self::Portals { .. } => false,
        }
    }
}

struct Flight {
    input: Vec<u8>,
    job: Interest<RecordKey>,
    decode: Decode,
}

pub(super) enum Resolved {
    Progress,
    Waiting,
    Failed,
}

#[derive(Default)]
struct Kept {
    held: BTreeMap<Vec<u8>, (u64, Interest<Vec<u8>>)>,
    order: BTreeMap<u64, Vec<u8>>,
    next: u64,
}

impl Kept {
    fn keep(&mut self, interest: Interest<Vec<u8>>, most: usize) {
        self.next += 1;
        let input = interest.key().clone();
        if let Some((stamp, _)) = self.held.insert(input.clone(), (self.next, interest)) {
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
            let released = if flight.decode.ground() {
                &grounds
            } else {
                &portals
            };
            !released.contains(&flight.input)
        });
        let most = settings.grounds_per_tick as usize;
        let mut decoded = 0;
        let mut waiting = Vec::with_capacity(self.flights.len());
        for flight in std::mem::take(&mut self.flights) {
            if flight.decode.ground() && decoded >= most {
                waiting.push(flight);
                continue;
            }
            let Some(output) = records.get(&flight.job) else {
                waiting.push(flight);
                continue;
            };
            let Flight { input, job, decode } = flight;
            drop(job);
            match decode {
                Decode::Ground { key, pyramid } => {
                    decoded += 1;
                    let made = made(WALK_GROUND_NAME, output, |bytes| {
                        walk_ground(key, &pyramid, bytes)
                    });
                    end(&mut self.grounds, &input, made, (tick, failures));
                }
                Decode::Portals { key } => {
                    let made = made(PORTALS_NAME, output, |bytes| walk_portals(key, bytes));
                    end(&mut self.portals, &input, made, (tick, failures));
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
                        let decode = Decode::Ground {
                            key,
                            pyramid: running.walked.source.pyramid,
                        };
                        self.fly(jobs, (&WALK_GROUND, &input), priority, decode);
                    }
                    found.map(|ground| running.held.grounds.insert(key, ground))
                }
                Want::Portals(key) => {
                    let input = portals_input(&running.walked, key, &gait);
                    let found = wanted(&mut self.portals, &mut running.interests.portals, &input);
                    if let Found::Asked = found {
                        self.fly(jobs, (&PORTALS, &input), priority, Decode::Portals { key });
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
                && !self.fly(jobs, (&PORTALS, &input), priority, Decode::Portals { key })
            {
                return;
            }
            running.asked_ahead += 1;
        }
    }

    pub(super) fn keep(&mut self, interests: Interests, settings: &Navigation) {
        let Interests { grounds, portals } = interests;
        for interest in grounds.into_values() {
            if self
                .grounds
                .of(&interest)
                .is_some_and(|request| request.ready().is_some())
            {
                self.kept_grounds
                    .keep(interest, settings.kept_grounds as usize);
            }
        }
        for interest in portals.into_values() {
            if self
                .portals
                .of(&interest)
                .is_some_and(|request| request.ready().is_some())
            {
                self.kept_portals
                    .keep(interest, settings.kept_portals as usize);
            }
        }
    }

    fn fly(
        &mut self,
        jobs: &mut Jobs,
        (kind, input): (&'static JobKind, &Vec<u8>),
        priority: u32,
        decode: Decode,
    ) -> bool {
        if self
            .flights
            .iter()
            .any(|flight| flight.decode.ground() == decode.ground() && flight.input == *input)
        {
            return true;
        }
        if !jobs.admit(priority) {
            return false;
        }
        let job = jobs.start(kind, input.clone(), priority);
        self.flights.push(Flight {
            input: input.clone(),
            job,
            decode,
        });
        true
    }
}

fn wanted<T>(
    requests: &mut Requests<Vec<u8>, Arc<T>>,
    held: &mut BTreeMap<Vec<u8>, Interest<Vec<u8>>>,
    input: &Vec<u8>,
) -> Found<Arc<T>> {
    if !held.contains_key(input) {
        held.insert(input.clone(), requests.ask(input.clone()));
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
