use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use crate::{Error, ErrorCode, Interest, MISSING_RECORD, RecordKey, Records, Request, Requests, Settling};

pub trait Delivers<K: Clone> {
    fn landed(&self, held: &Interest<K>) -> Option<Result<Arc<[u8]>, Error>>;
}

impl Delivers<RecordKey> for Records {
    fn landed(&self, held: &Interest<RecordKey>) -> Option<Result<Arc<[u8]>, Error>> {
        self.get(held)
    }
}

impl<K: Clone + Eq + Hash, A> Delivers<K> for Requests<K, Arc<[u8]>, A> {
    fn landed(&self, held: &Interest<K>) -> Option<Result<Arc<[u8]>, Error>> {
        match self.of(held)? {
            Request::Asked(_) => None,
            Request::Ready(bytes) => Some(Ok(Arc::clone(bytes))),
            Request::Failed(error) => Some(Err(error.clone())),
            Request::Missing => {
                Some(Err(Error::new(ErrorCode::InvalidRecord, "$").got(MISSING_RECORD)))
            }
        }
    }
}

type Decode<T> = Box<dyn FnOnce(&[u8]) -> Result<T, Error> + Send + Sync>;

enum Stage<T, K: Clone> {
    Asked(Interest<K>, Decode<T>),
    Ready(T),
    Failed(Error),
}

pub enum Landed<'a, T> {
    Waiting,
    Ready(&'a T),
    Failed(&'a Error),
}

#[must_use]
pub struct Background<T, K: Clone = RecordKey> {
    stage: Stage<T, K>,
    report: Settling,
}

impl<T, K: Clone> Background<T, K> {
    pub fn asked(
        held: Interest<K>,
        decode: impl FnOnce(&[u8]) -> Result<T, Error> + Send + Sync + 'static,
    ) -> Self {
        let report = Settling::default();
        report.left(1);
        Self {
            stage: Stage::Asked(held, Box::new(decode)),
            report,
        }
    }

    pub fn failed(error: Error) -> Self {
        let report = Settling::default();
        report.failed(Some(error.clone()));
        Self {
            stage: Stage::Failed(error),
            report,
        }
    }

    pub fn poll(&mut self, from: &impl Delivers<K>) -> Landed<'_, T> {
        if let Stage::Asked(held, _) = &self.stage
            && let Some(landed) = from.landed(held)
        {
            let placeholder = Stage::Failed(Error::new(ErrorCode::InvalidRecord, "$"));
            if let Stage::Asked(_, decode) = std::mem::replace(&mut self.stage, placeholder) {
                self.stage = match landed.and_then(|bytes| decode(&bytes[..])) {
                    Ok(value) => Stage::Ready(value),
                    Err(error) => {
                        self.report.failed(Some(error.clone()));
                        Stage::Failed(error)
                    }
                };
                self.report.left(0);
            }
        }
        self.state()
    }

    pub fn state(&self) -> Landed<'_, T> {
        match &self.stage {
            Stage::Asked(..) => Landed::Waiting,
            Stage::Ready(value) => Landed::Ready(value),
            Stage::Failed(error) => Landed::Failed(error),
        }
    }

    pub fn waiting(&self) -> bool {
        matches!(self.stage, Stage::Asked(..))
    }

    pub fn ready(&self) -> Option<&T> {
        match &self.stage {
            Stage::Ready(value) => Some(value),
            Stage::Asked(..) | Stage::Failed(_) => None,
        }
    }

    pub fn ready_mut(&mut self) -> Option<&mut T> {
        match &mut self.stage {
            Stage::Ready(value) => Some(value),
            Stage::Asked(..) | Stage::Failed(_) => None,
        }
    }

    pub fn failure(&self) -> Option<&Error> {
        match &self.stage {
            Stage::Failed(error) => Some(error),
            Stage::Asked(..) | Stage::Ready(_) => None,
        }
    }

    pub fn asked_key(&self) -> Option<&K> {
        match &self.stage {
            Stage::Asked(held, _) => Some(held.key()),
            Stage::Ready(_) | Stage::Failed(_) => None,
        }
    }

    pub fn into_ready(self) -> Option<T> {
        match self.stage {
            Stage::Ready(value) => Some(value),
            Stage::Asked(..) | Stage::Failed(_) => None,
        }
    }
}

impl<T, K: Clone + fmt::Debug> fmt::Debug for Background<T, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.stage {
            Stage::Asked(held, _) => f.debug_tuple("Asked").field(held.key()).finish(),
            Stage::Ready(_) => f.write_str("Ready"),
            Stage::Failed(error) => f.debug_tuple("Failed").field(error).finish(),
        }
    }
}
