use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use crate::{Error, ErrorCode, Expected, Interest, Request, Requests, Settling};

pub trait Delivers<K: Clone> {
    fn landed(&self, held: &Interest<K>) -> Option<Result<Arc<[u8]>, Error>>;
}

impl<K: Clone + Eq + Hash, A> Delivers<K> for Requests<K, Arc<[u8]>, A> {
    fn landed(&self, held: &Interest<K>) -> Option<Result<Arc<[u8]>, Error>> {
        match self.of(held)? {
            Request::Asked(_) => None,
            Request::Ready(bytes) => Some(Ok(Arc::clone(bytes))),
            Request::Failed(error) => Some(Err(error.clone())),
            Request::Missing => Some(Err(
                Error::new(ErrorCode::MissingResource, "$").expected(Expected::Present)
            )),
        }
    }
}

type Decode<T> = Box<dyn FnOnce(Arc<[u8]>) -> Result<T, Error> + Send + Sync>;

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

pub enum Taken<T, K: Clone> {
    Waiting(Background<T, K>),
    Ready(T),
    Failed(Error),
}

#[must_use]
pub struct Background<T, K: Clone> {
    stage: Stage<T, K>,
    report: Settling,
}

impl<T, K: Clone> Background<T, K> {
    pub(crate) fn asked(
        held: Interest<K>,
        decode: impl FnOnce(Arc<[u8]>) -> Result<T, Error> + Send + Sync + 'static,
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
                self.stage = match landed.and_then(decode) {
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

    pub fn take(mut self, from: &impl Delivers<K>) -> Taken<T, K> {
        self.poll(from);
        let Self { stage, report } = self;
        match stage {
            Stage::Asked(held, decode) => Taken::Waiting(Self {
                stage: Stage::Asked(held, decode),
                report,
            }),
            Stage::Ready(value) => Taken::Ready(value),
            Stage::Failed(error) => Taken::Failed(error),
        }
    }

    pub fn state(&self) -> Landed<'_, T> {
        match &self.stage {
            Stage::Asked(..) => Landed::Waiting,
            Stage::Ready(value) => Landed::Ready(value),
            Stage::Failed(error) => Landed::Failed(error),
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
