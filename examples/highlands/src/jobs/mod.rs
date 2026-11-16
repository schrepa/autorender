use engine_core::{Error, ErrorCode, Job, JobCx, JobKind, Mesh};
use glam::Vec3;

mod broadleaf;
mod caves;
mod conifers;
mod deadwood;
pub(crate) mod ground;
pub(crate) mod photographs;
pub(crate) mod random;
mod rocks;
pub(crate) mod sprays;
pub(crate) mod sprigs;
pub(crate) mod trees;
mod wood;

include!(concat!(env!("OUT_DIR"), "/jobs.rs"));

pub(crate) const CAVE_POINTS: usize = 5;

pub(crate) const CAVE_RADIUS: [f32; 2] = [6.0, 4.5];

const CAVE_SEED: u32 = 79;

pub(crate) static BOULDER: JobKind = JobKind::new("highlands::boulder", &SOURCE, start_boulder);

pub(crate) static PLANT: JobKind = JobKind::new("highlands::plant", &SOURCE, start_plant);

pub(crate) static CAVE: JobKind = JobKind::new("highlands::cave", &SOURCE, start_cave);

struct Making {
    input: Vec<u8>,
    make: fn(&[u8]) -> Result<Mesh, Error>,
}

impl Job for Making {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        (self.make)(&self.input).map(|mesh| Some(mesh.to_record()))
    }
}

fn making(input: &[u8], make: fn(&[u8]) -> Result<Mesh, Error>) -> Result<Box<dyn Job>, Error> {
    Ok(Box::new(Making {
        input: input.to_vec(),
        make,
    }))
}

fn words<const N: usize>(input: &[u8], kind: &JobKind) -> Result<[u32; N], Error> {
    if input.len() != N * 4 {
        return Err(
            Error::new(ErrorCode::InvalidRecord, format!("$.jobs.{}", kind.name()))
                .range((N * 4) as f64, (N * 4) as f64)
                .got(input.len()),
        );
    }
    let mut words = [0; N];
    for (word, bytes) in words.iter_mut().zip(input.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*bytes);
    }
    Ok(words)
}

fn start_boulder(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    making(input, |input| {
        let [seed] = words(input, &BOULDER)?;
        rocks::boulder(seed)
    })
}

fn start_plant(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    making(input, trees::grow)
}

fn start_cave(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    making(input, |input| {
        let bits: [u32; CAVE_POINTS * 3] = words(input, &CAVE)?;
        let points: Vec<Vec3> = bits
            .as_chunks::<3>()
            .0
            .iter()
            .map(|point| {
                Vec3::new(
                    f32::from_bits(point[0]),
                    f32::from_bits(point[1]),
                    f32::from_bits(point[2]),
                )
            })
            .collect();
        caves::cave(&points, CAVE_RADIUS, CAVE_SEED)
    })
}
