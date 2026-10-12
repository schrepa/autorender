//! Byte ranges of files natively, read on a thread of their own and handed
//! back in the order they were asked, so frames spend no time on disk and
//! the same reads arrive in the same order every run. Pages fetch theirs
//! (`engine_render::read_range`).

use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

/// Reads asked of the reading thread, each for a `T`, and their bytes or
/// why they could not be read.
pub(crate) struct RangeReads<T> {
    asks: Sender<(String, Range<u64>)>,
    results: Receiver<Result<Vec<u8>, String>>,
    // What each read asked and not yet handed back is for, in order.
    waiting: VecDeque<T>,
}

impl<T> RangeReads<T> {
    /// Reads on a thread of their own, which ends once they are dropped and
    /// it read what was asked.
    pub(crate) fn new() -> Self {
        let (asks, asked) = channel::<(String, Range<u64>)>();
        let (send, results) = channel();
        std::thread::Builder::new()
            .name("engine-range-reads".into())
            .spawn(move || {
                // One at a time, so they finish in the order asked.
                for (path, range) in asked {
                    if send.send(read(&path, range)).is_err() {
                        break;
                    }
                }
            })
            .expect("threads start");
        Self {
            asks,
            results,
            waiting: VecDeque::new(),
        }
    }

    /// Asks for bytes `range` of the file at `path`, for `what`.
    pub(crate) fn ask(&mut self, path: &str, range: Range<u64>, what: T) {
        self.waiting.push_back(what);
        // The thread ends only once this ends.
        let _ = self.asks.send((path.to_owned(), range));
    }

    /// Hands `arrive` the reads finished since, each with what it was for
    /// and its bytes (or why the read failed), in the order asked; with
    /// `all`, every read asked so far, waiting for those still reading.
    pub(crate) fn take(&mut self, all: bool, mut arrive: impl FnMut(T, Result<Vec<u8>, String>)) {
        let stopped = || Err("the reading thread stopped".to_owned());
        while !self.waiting.is_empty() {
            let bytes = if all {
                self.results.recv().unwrap_or_else(|_| stopped())
            } else {
                match self.results.try_recv() {
                    Ok(bytes) => bytes,
                    Err(TryRecvError::Empty) => return,
                    Err(TryRecvError::Disconnected) => stopped(),
                }
            };
            if let Some(what) = self.waiting.pop_front() {
                arrive(what, bytes);
            }
        }
    }
}

/// Bytes `range` of the file at `path`, or why they cannot be read.
fn read(path: &str, range: Range<u64>) -> Result<Vec<u8>, String> {
    let failed = |error: &dyn std::fmt::Display| {
        format!(
            "cannot read bytes {}..{} of {path}: {error}",
            range.start, range.end
        )
    };
    let mut file = std::fs::File::open(path).map_err(|error| failed(&error))?;
    file.seek(SeekFrom::Start(range.start))
        .map_err(|error| failed(&error))?;
    let length = usize::try_from(range.end - range.start).map_err(|error| failed(&error))?;
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)
        .map_err(|error| failed(&error))?;
    Ok(bytes)
}
