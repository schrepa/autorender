//! Byte ranges of files natively, read on a thread of their own and handed
//! back in the order they were asked, so frames spend no time on disk and
//! the same reads arrive in the same order every run. Pages fetch theirs
//! (`engine_render::read_range`).

use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

/// Reads asked of the reading thread, each for a `T`, and their bytes.
pub(crate) struct RangeReads<T> {
    asks: Sender<(String, Range<u64>)>,
    results: Receiver<Option<Vec<u8>>>,
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
    /// and its bytes (none where the read failed), in the order asked; with
    /// `all`, every read asked so far, waiting for those still reading.
    pub(crate) fn take(&mut self, all: bool, mut arrive: impl FnMut(T, Option<Vec<u8>>)) {
        while !self.waiting.is_empty() {
            let bytes = if all {
                self.results.recv().ok().flatten()
            } else {
                match self.results.try_recv() {
                    Ok(bytes) => bytes,
                    Err(TryRecvError::Empty) => return,
                    Err(TryRecvError::Disconnected) => None,
                }
            };
            if let Some(what) = self.waiting.pop_front() {
                arrive(what, bytes);
            }
        }
    }
}

/// Bytes `range` of the file at `path`; none where it cannot be read.
fn read(path: &str, range: Range<u64>) -> Option<Vec<u8>> {
    let mut file = std::fs::File::open(path).ok()?;
    file.seek(SeekFrom::Start(range.start)).ok()?;
    let mut bytes = vec![0; usize::try_from(range.end - range.start).ok()?];
    file.read_exact(&mut bytes).ok()?;
    Some(bytes)
}
