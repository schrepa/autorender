use std::collections::BTreeSet;

use super::*;

#[derive(Default)]
pub(super) struct Interests {
    portals: BTreeMap<Vec<u8>, Interest<Vec<u8>>>,
}

struct Flight {
    input: Vec<u8>,
    job: Interest<RecordKey>,
    key: TileKey,
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
struct Grounds {
    seen: Option<(u64, WalkArea)>,
    held: BTreeMap<TileKey, (u64, Arc<WalkGround>)>,
    order: BTreeMap<u64, TileKey>,
    next: u64,
    sampling: Option<Sampling>,
}

impl Grounds {
    fn refresh(&mut self, seen: (u64, WalkArea)) {
        if self.seen != Some(seen) {
            self.seen = Some(seen);
            self.held.clear();
            self.order.clear();
            self.sampling = None;
        }
    }

    fn get(&mut self, key: TileKey) -> Option<Arc<WalkGround>> {
        let (stamp, ground) = self.held.get_mut(&key)?;
        self.order.remove(stamp);
        self.next += 1;
        *stamp = self.next;
        self.order.insert(self.next, key);
        Some(Arc::clone(ground))
    }

    fn hold(&mut self, key: TileKey, ground: Arc<WalkGround>, most: usize) {
        self.next += 1;
        if let Some((stamp, _)) = self.held.insert(key, (self.next, ground)) {
            self.order.remove(&stamp);
        }
        self.order.insert(self.next, key);
        while self.held.len() > most {
            let Some((_, oldest)) = self.order.pop_first() else {
                break;
            };
            self.held.remove(&oldest);
        }
    }

    fn sample(
        &mut self,
        key: TileKey,
        (area, floors): (&WalkArea, &Floors),
        budget: &mut u64,
        most: usize,
    ) -> Option<Arc<WalkGround>> {
        if let Some(ground) = self.get(key) {
            return Some(ground);
        }
        if let Some(mut sampling) = self.sampling.take() {
            if !sampling.advance(area, floors, budget) {
                self.sampling = Some(sampling);
                return None;
            }
            let done = sampling.key;
            let ground = Arc::new(sampling.ground(area));
            self.hold(done, Arc::clone(&ground), most);
            if done == key {
                return Some(ground);
            }
        }
        if *budget == 0 {
            return None;
        }
        let mut sampling = Sampling::new(key);
        if !sampling.advance(area, floors, budget) {
            self.sampling = Some(sampling);
            return None;
        }
        let ground = Arc::new(sampling.ground(area));
        self.hold(key, Arc::clone(&ground), most);
        Some(ground)
    }
}

#[derive(Default)]
pub(super) struct WalkRecords {
    grounds: Grounds,
    kept_grounds: usize,
    portals: Requests<Vec<u8>, Arc<TilePortals>>,
    flights: Vec<Flight>,
    kept_portals: Kept,
}

enum Found<T> {
    Here(T),
    Asked,
    Failed,
}

impl WalkRecords {
    pub(super) fn held(&self) -> (u32, u32) {
        (
            self.grounds.held.len() as u32,
            self.portals.census().ready,
        )
    }

    pub(super) fn refresh(&mut self, seen: (u64, WalkArea)) {
        self.grounds.refresh(seen);
    }

    pub(super) fn settle(
        &mut self,
        (records, now, tick): (&Records, f64, u64),
        settings: &Navigation,
        failures: &mut JobFailures,
    ) {
        self.kept_grounds = settings.kept_grounds as usize;
        let portals: BTreeSet<Vec<u8>> = self
            .portals
            .settle(now)
            .into_iter()
            .map(|(input, _)| input)
            .collect();
        self.flights
            .retain(|flight| !portals.contains(&flight.input));
        let mut waiting = Vec::with_capacity(self.flights.len());
        for flight in std::mem::take(&mut self.flights) {
            let Some(output) = records.get(&flight.job) else {
                waiting.push(flight);
                continue;
            };
            let Flight { input, job, key } = flight;
            drop(job);
            let made = made(PORTALS_NAME, output, |bytes| walk_portals(key, bytes));
            end(&mut self.portals, &input, made, (tick, failures));
        }
        self.flights = waiting;
    }

    fn ground(
        &mut self,
        running: &mut Running,
        key: TileKey,
        (floors, samples): (&Floors, &mut u64),
    ) -> Option<Arc<WalkGround>> {
        if let Some(ground) = running.held.grounds.get(&key) {
            return Some(Arc::clone(ground));
        }
        let ground = self.grounds.sample(
            key,
            (&running.area, floors),
            samples,
            self.kept_grounds,
        )?;
        running.held.grounds.insert(key, Arc::clone(&ground));
        Some(ground)
    }

    fn portals(
        &mut self,
        running: &mut Running,
        key: TileKey,
        (floors, jobs, samples): (&Floors, &mut Jobs, &mut u64),
        priority: u32,
    ) -> Option<Found<Arc<TilePortals>>> {
        let gait = running.search.gait();
        let input = portals_key(floors.layout(), key, &gait);
        let found = wanted(&mut self.portals, &mut running.interests.portals, &input);
        if !matches!(found, Found::Asked) || self.flying(&input) {
            return Some(found);
        }
        let ground = self.ground(running, key, (floors, samples))?;
        if !jobs.admit(priority) {
            return None;
        }
        let job = jobs.start(
            &PORTALS,
            portals_input(&running.area, &ground, &gait),
            priority,
        );
        self.flights.push(Flight { input, job, key });
        Some(found)
    }

    pub(super) fn resolve(
        &mut self,
        running: &mut Running,
        (floors, jobs): (&Floors, &mut Jobs),
        samples: &mut u64,
    ) -> Resolved {
        let priority = job_priority(GAMEPLAY_JOBS, 0);
        let mut progress = false;
        let wants = running.search.wants().to_vec();
        for want in wants {
            let found = match want {
                Want::Ground(key) => match self.ground(running, key, (floors, &mut *samples)) {
                    Some(_) => Found::Here(()),
                    None => Found::Asked,
                },
                Want::Portals(key) => {
                    match self.portals(running, key, (floors, &mut *jobs, &mut *samples), priority)
                    {
                        Some(found) => {
                            found.map(|portals| running.held.portals.insert(key, portals))
                        }
                        None => Found::Asked,
                    }
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

    pub(super) fn ask_ahead(
        &mut self,
        running: &mut Running,
        (floors, jobs): (&Floors, &mut Jobs),
        samples: &mut u64,
    ) {
        let priority = job_priority(GAMEPLAY_JOBS, 1);
        while let Some(&key) = running.search.ahead().get(running.asked_ahead) {
            if self
                .portals(running, key, (floors, &mut *jobs, &mut *samples), priority)
                .is_none()
            {
                return;
            }
            running.asked_ahead += 1;
        }
    }

    pub(super) fn keep(&mut self, interests: Interests, settings: &Navigation) {
        let Interests { portals } = interests;
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

    fn flying(&self, input: &[u8]) -> bool {
        self.flights.iter().any(|flight| flight.input == input)
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
