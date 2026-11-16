use std::fmt::Arguments;
use std::sync::{Mutex, PoisonError};

type Listener = fn(Arguments<'_>);

static LISTENER: Mutex<Option<Listener>> = Mutex::new(None);

pub fn listen_to_threads(listener: Option<fn(Arguments<'_>)>) {
    *LISTENER.lock().unwrap_or_else(PoisonError::into_inner) = listener;
}

pub fn announce_thread(name: Arguments<'_>) {
    let listener = *LISTENER.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(listener) = listener {
        listener(name);
    }
}
