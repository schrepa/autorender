use std::sync::Arc;

use bevy_ecs::resource::Resource;

use crate::{
    Background, Delivers, Error, ErrorCode, Expected, Interest, IoAct, IoOrder, IoReply,
    MAX_ID_BYTES, Requests, Ticket,
};

const PRESENT: u8 = 1;
const ABSENT: u8 = 0;

#[derive(Resource, Default)]
pub struct Kept {
    requests: Requests<u64, Arc<[u8]>>,
    place: Option<String>,
    serial: u64,
    orders: Vec<IoOrder>,
    pending: Vec<(Ticket, u64, String)>,
}

impl Kept {
    pub fn keep_in(&mut self, place: impl Into<String>) {
        self.place = Some(place.into());
    }

    pub fn place(&self) -> Option<&str> {
        self.place.as_deref()
    }

    pub fn keep(&mut self, name: &str, bytes: Vec<u8>) -> Background<(), u64> {
        self.ask(
            name,
            |place, name| IoAct::Store(place, name, bytes),
            |_| Ok(()),
        )
    }

    pub fn kept(&mut self, name: &str) -> Background<Option<Vec<u8>>, u64> {
        self.ask(name, IoAct::Load, |landed| {
            Ok(landed
                .split_first()
                .filter(|(present, _)| **present == PRESENT)
                .map(|(_, bytes)| bytes.to_vec()))
        })
    }

    pub fn in_flight(&self) -> bool {
        !self.pending.is_empty()
    }

    fn ask<T>(
        &mut self,
        name: &str,
        act: impl FnOnce(String, String) -> IoAct,
        decode: impl FnOnce(Arc<[u8]>) -> Result<T, Error> + Send + Sync + 'static,
    ) -> Background<T, u64> {
        if !named(name) {
            return Background::failed(
                Error::new(ErrorCode::InvalidName, "$.kept")
                    .expected(Expected::Matching)
                    .got(name),
            );
        }
        let Some(place) = self.place.clone() else {
            return Background::failed(unplaced());
        };
        self.serial += 1;
        let serial = self.serial;
        let held = self.requests.ask(serial);
        let Some(id) = self.requests.ticket(&serial) else {
            return Background::failed(unplaced());
        };
        self.orders.push(IoOrder {
            id,
            act: act(place, name.to_owned()),
        });
        self.pending.push((id, serial, name.to_owned()));
        Background::asked(held, decode)
    }

    pub(crate) fn orders(&mut self, now: f64, orders: &mut Vec<IoOrder>) {
        self.requests.settle(now);
        orders.append(&mut self.orders);
    }

    pub(crate) fn land(&mut self, reply: IoReply) -> Option<IoReply> {
        let Some(at) = self.pending.iter().position(|(id, ..)| *id == reply.id) else {
            return Some(reply);
        };
        let (ticket, serial, name) = self.pending.swap_remove(at);
        match reply.done {
            Ok(found) => {
                let landed = match found {
                    Some(bytes) => [&[PRESENT][..], &bytes].concat(),
                    None => vec![ABSENT],
                };
                self.requests.ready(&serial, ticket, Arc::from(landed));
            }
            Err(error) => {
                let error = Error::new(ErrorCode::SourceIo, format!("$.kept.{name}")).got(error);
                self.requests.fail(&serial, ticket, error);
            }
        }
        None
    }
}

impl Delivers<u64> for Kept {
    fn landed(&self, held: &Interest<u64>) -> Option<Result<Arc<[u8]>, Error>> {
        self.requests.landed(held)
    }
}

fn named(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_ID_BYTES
        && name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
}

fn unplaced() -> Error {
    Error::new(ErrorCode::UnsupportedFeature, "$.kept").got("this host keeps nothing")
}
