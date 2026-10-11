//! What other programs load the machine with while a measuring tool times a
//! run of another process: their share of the processors, their share of
//! the GPU engines the run uses, and the run's GPU clock. On Windows these
//! come from the system for any GPU whose driver reports them (processor
//! times, the `GPU Engine` performance counters, the display driver's
//! engine clocks); other systems report none yet.

/// What other programs did during a window of a run.
#[derive(Clone, Copy, Debug, Default)]
pub struct Load {
    /// Other processes' share of every logical processor's time, in percent.
    pub foreign_cpu_pct: Option<f64>,
    /// The largest share other processes took of a GPU engine the run had
    /// used before the window, in percent.
    pub foreign_gpu_pct: Option<f64>,
    /// The clock of the run's GPU's 3D engine, in MHz: the first and last
    /// readings while it ran in the window (drivers may report none while
    /// the engine sleeps between frames).
    pub gpu_clock_mhz: [Option<f64>; 2],
}

/// Watches a run's process and the machine around it over one window;
/// "other processes" are all but the run's and this one.
pub struct Monitor {
    #[cfg(windows)]
    inner: Option<windows::Monitor>,
}

impl Monitor {
    /// Ready to watch process `pid`; opening what it reads now keeps that
    /// out of the window.
    pub fn new(pid: u32) -> Self {
        #[cfg(not(windows))]
        let _ = pid;
        Self {
            #[cfg(windows)]
            inner: windows::Monitor::new(pid),
        }
    }

    /// The window starts.
    pub fn start(&mut self) {
        #[cfg(windows)]
        if let Some(inner) = &mut self.inner {
            inner.start();
        }
    }

    /// The window ends: what other programs did since it started.
    pub fn end(&mut self) -> Load {
        #[cfg(windows)]
        if let Some(inner) = &mut self.inner {
            return inner.end();
        }
        Load::default()
    }
}

#[cfg(windows)]
mod windows {
    use super::Load;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};
    use windows_sys::Wdk::Graphics::Direct3D::{
        D3DKMT_ADAPTERINFO, D3DKMT_CLOSEADAPTER, D3DKMT_ENUMADAPTERS2, D3DKMT_NODE_PERFDATA,
        D3DKMT_NODEMETADATA, D3DKMT_QUERYADAPTERINFO, D3DKMTCloseAdapter, D3DKMTEnumAdapters2,
        D3DKMTQueryAdapterInfo, DXGK_ENGINE_TYPE_3D, KMTQAITYPE_NODEMETADATA,
        KMTQAITYPE_NODEPERFDATA, KMTQUERYADAPTERINFOTYPE,
    };
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, HANDLE};
    use windows_sys::Win32::System::Performance::{
        PDH_HCOUNTER, PDH_HQUERY, PDH_MORE_DATA, PDH_RAW_COUNTER_ITEM_W, PdhAddEnglishCounterW,
        PdhCloseQuery, PdhCollectQueryData, PdhGetRawCounterArrayW, PdhOpenQueryW,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetProcessTimes, GetSystemTimes, OpenProcess,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };

    /// A GPU: its driver's handle, its LUID as the counters name it, and its
    /// 3D engine's node when the driver reports that node's clock.
    struct Adapter {
        handle: u32,
        luid: String,
        node: Option<u32>,
    }

    /// The machine at one moment.
    struct Snapshot {
        at: Instant,
        /// All processors' busy and total time, in 100 ns.
        system: Option<(u64, u64)>,
        /// The run's and this process's processor time, in 100 ns.
        processes: Option<(u64, u64)>,
        /// Each `GPU Engine` counter instance's running time, in 100 ns.
        engines: HashMap<String, i64>,
    }

    /// How often the window reads the clocks.
    const CLOCK_EVERY: Duration = Duration::from_millis(2);

    /// Reads each adapter's 3D engine clock until stopped: its first and
    /// last readings, in Hz.
    struct Clocks {
        stop: Arc<AtomicBool>,
        thread: JoinHandle<Vec<[Option<u64>; 2]>>,
    }

    pub(super) struct Monitor {
        pid: u32,
        process: HANDLE,
        query: PDH_HQUERY,
        counter: PDH_HCOUNTER,
        adapters: Vec<Adapter>,
        start: Option<(Snapshot, Clocks)>,
    }

    impl Monitor {
        pub(super) fn new(pid: u32) -> Option<Self> {
            // SAFETY: opens a process by its ID for reading its times.
            #[allow(unsafe_code)]
            let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
            if process.is_null() {
                return None;
            }
            let mut monitor = Self {
                pid,
                process,
                query: std::ptr::null_mut(),
                counter: std::ptr::null_mut(),
                adapters: adapters(),
                start: None,
            };
            let path: Vec<u16> = r"\GPU Engine(*)\Running Time"
                .encode_utf16()
                .chain([0])
                .collect();
            // SAFETY: a query of this process's own, and a counter added to
            // it by its English path; both closed on drop.
            #[allow(unsafe_code)]
            unsafe {
                if PdhOpenQueryW(std::ptr::null(), 0, &mut monitor.query) == 0
                    && PdhAddEnglishCounterW(monitor.query, path.as_ptr(), 0, &mut monitor.counter)
                        != 0
                {
                    monitor.counter = std::ptr::null_mut();
                }
            }
            Some(monitor)
        }

        pub(super) fn start(&mut self) {
            let nodes: Vec<(u32, Option<u32>)> = self
                .adapters
                .iter()
                .map(|adapter| (adapter.handle, adapter.node))
                .collect();
            let stop = Arc::new(AtomicBool::new(false));
            let stopped = Arc::clone(&stop);
            let thread = std::thread::spawn(move || {
                let mut readings = vec![[None; 2]; nodes.len()];
                while !stopped.load(Ordering::Relaxed) {
                    for (reading, (handle, node)) in readings.iter_mut().zip(&nodes) {
                        if let Some(hertz) = node.and_then(|node| clock(*handle, node)) {
                            reading[0] = reading[0].or(Some(hertz));
                            reading[1] = Some(hertz);
                        }
                    }
                    std::thread::sleep(CLOCK_EVERY);
                }
                readings
            });
            self.start = Some((self.snapshot(), Clocks { stop, thread }));
        }

        pub(super) fn end(&mut self) -> Load {
            let Some((start, clocks)) = self.start.take() else {
                return Load::default();
            };
            let end = self.snapshot();
            let clocks = clocks.finish();
            let foreign_cpu_pct = match (start.system, end.system, start.processes, end.processes) {
                (Some(before), Some(after), Some(watched), Some(now)) => {
                    let total = after.1.saturating_sub(before.1);
                    let busy = after.0.saturating_sub(before.0);
                    let ours = now.0.saturating_sub(watched.0) + now.1.saturating_sub(watched.1);
                    (total > 0).then(|| busy.saturating_sub(ours) as f64 / total as f64 * 100.0)
                }
                _ => None,
            };
            // The run's GPU: the adapter of the engine it had used most.
            let run = format!("pid_{}_", self.pid);
            let used: Vec<(&str, &str, i64)> = start
                .engines
                .iter()
                .filter(|(_, time)| **time > 0)
                .filter_map(|(name, time)| {
                    let (luid, engine) = engine(name.strip_prefix(&run)?)?;
                    Some((luid, engine, *time))
                })
                .collect();
            let luid = used
                .iter()
                .max_by_key(|(_, _, time)| *time)
                .map(|(luid, _, _)| *luid);
            let window = end.at.duration_since(start.at).as_nanos() as f64 / 100.0;
            let foreign_gpu_pct = luid.filter(|_| window > 0.0).map(|luid| {
                let own = format!("pid_{}_", std::process::id());
                let mut foreign: HashMap<&str, i64> = HashMap::new();
                for (name, time) in &end.engines {
                    if name.starts_with(&run) || name.starts_with(&own) {
                        continue;
                    }
                    let Some((other, engine)) = name
                        .split_once("_luid_")
                        .and_then(|(_, rest)| self::engine(rest))
                    else {
                        continue;
                    };
                    let earlier = start.engines.get(name).copied().unwrap_or(0);
                    if other == luid {
                        *foreign.entry(engine).or_default() += (time - earlier).max(0);
                    }
                }
                let busiest = used
                    .iter()
                    .filter(|(other, _, _)| *other == luid)
                    .map(|(_, engine, _)| foreign.get(engine).copied().unwrap_or(0))
                    .max()
                    .unwrap_or(0);
                (busiest as f64 / window * 100.0).min(100.0)
            });
            let readings = self
                .adapters
                .iter()
                .position(|adapter| Some(adapter.luid.as_str()) == luid)
                .map_or([None; 2], |index| clocks[index]);
            Load {
                foreign_cpu_pct,
                foreign_gpu_pct,
                gpu_clock_mhz: readings.map(|hertz| hertz.map(|hertz| hertz as f64 / 1e6)),
            }
        }

        fn snapshot(&self) -> Snapshot {
            let (mut idle, mut kernel, mut user) = Default::default();
            // SAFETY: plain calls filling the times passed.
            #[allow(unsafe_code)]
            let system =
                (unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) } != 0).then(|| {
                    // Kernel time includes idle time.
                    let total = ticks(kernel) + ticks(user);
                    (total.saturating_sub(ticks(idle)), total)
                });
            // SAFETY: this process's pseudo-handle.
            #[allow(unsafe_code)]
            let own = unsafe { GetCurrentProcess() };
            let processes = cpu_time(self.process).zip(cpu_time(own));
            Snapshot {
                at: Instant::now(),
                system,
                processes,
                engines: self.engines(),
            }
        }

        /// Each `GPU Engine` instance's running time now.
        fn engines(&self) -> HashMap<String, i64> {
            let mut engines = HashMap::new();
            if self.counter.is_null() {
                return engines;
            }
            let (mut size, mut count) = (0u32, 0u32);
            // SAFETY: the query and counter opened above; the second call
            // fills a buffer of the size the first asked for, 8-byte
            // aligned, and its names point into that buffer.
            #[allow(unsafe_code)]
            unsafe {
                if PdhCollectQueryData(self.query) != 0
                    || PdhGetRawCounterArrayW(
                        self.counter,
                        &mut size,
                        &mut count,
                        std::ptr::null_mut(),
                    ) != PDH_MORE_DATA
                {
                    return engines;
                }
                let mut buffer = vec![0u64; (size as usize).div_ceil(8)];
                let items = buffer.as_mut_ptr().cast::<PDH_RAW_COUNTER_ITEM_W>();
                if PdhGetRawCounterArrayW(self.counter, &mut size, &mut count, items) != 0 {
                    return engines;
                }
                for index in 0..count as usize {
                    let item = &*items.add(index);
                    let mut length = 0;
                    while *item.szName.add(length) != 0 {
                        length += 1;
                    }
                    let name =
                        String::from_utf16_lossy(std::slice::from_raw_parts(item.szName, length));
                    *engines.entry(name).or_default() += item.RawValue.FirstValue;
                }
            }
            engines
        }
    }

    impl Clocks {
        /// Stops reading: each adapter's first and last readings.
        fn finish(self) -> Vec<[Option<u64>; 2]> {
            self.stop.store(true, Ordering::Relaxed);
            self.thread.join().unwrap_or_default()
        }
    }

    impl Drop for Monitor {
        fn drop(&mut self) {
            // The clocks stop reading before their adapters close.
            if let Some((_, clocks)) = self.start.take() {
                clocks.finish();
            }
            // SAFETY: what `new` opened, closed once.
            #[allow(unsafe_code)]
            unsafe {
                if !self.query.is_null() {
                    PdhCloseQuery(self.query);
                }
                CloseHandle(self.process);
                for adapter in &self.adapters {
                    D3DKMTCloseAdapter(&D3DKMT_CLOSEADAPTER {
                        hAdapter: adapter.handle,
                    });
                }
            }
        }
    }

    /// `luid_..._phys_N_eng_M_engtype_T` without its `luid_`: the LUID and
    /// the engine.
    fn engine(name: &str) -> Option<(&str, &str)> {
        let rest = name.strip_prefix("luid_").unwrap_or(name);
        rest.split_once("_phys_")
    }

    /// Every GPU, with its first 3D engine node the driver answers clock
    /// queries for (Intel's reports even the most it can run at as zero
    /// while the engine sleeps).
    fn adapters() -> Vec<Adapter> {
        let mut list = D3DKMT_ENUMADAPTERS2::default();
        // SAFETY: the first call counts the adapters, the second opens
        // that many into a list of that length.
        #[allow(unsafe_code)]
        let opened = unsafe {
            if D3DKMTEnumAdapters2(&mut list) != 0 {
                return Vec::new();
            }
            let mut infos = vec![D3DKMT_ADAPTERINFO::default(); list.NumAdapters as usize];
            list.pAdapters = infos.as_mut_ptr();
            if D3DKMTEnumAdapters2(&mut list) != 0 {
                return Vec::new();
            }
            infos.truncate(list.NumAdapters as usize);
            infos
        };
        opened
            .into_iter()
            .map(|info| {
                let luid = info.AdapterLuid;
                let node = (0..64)
                    .map_while(|node| {
                        let meta = query(
                            info.hAdapter,
                            KMTQAITYPE_NODEMETADATA,
                            D3DKMT_NODEMETADATA {
                                NodeOrdinalAndAdapterIndex: node,
                                ..Default::default()
                            },
                        )?;
                        Some((node, { meta.NodeData.EngineType }))
                    })
                    .find(|(node, kind)| {
                        *kind == DXGK_ENGINE_TYPE_3D
                            && query(
                                info.hAdapter,
                                KMTQAITYPE_NODEPERFDATA,
                                D3DKMT_NODE_PERFDATA {
                                    NodeOrdinal: *node,
                                    ..Default::default()
                                },
                            )
                            .is_some()
                    })
                    .map(|(node, _)| node);
                Adapter {
                    handle: info.hAdapter,
                    luid: format!("0x{:08X}_0x{:08X}", luid.HighPart, luid.LowPart),
                    node,
                }
            })
            .collect()
    }

    /// The clock of `adapter`'s engine `node` now, in Hz, when the driver
    /// reports one.
    fn clock(adapter: u32, node: u32) -> Option<u64> {
        let perf = query(
            adapter,
            KMTQAITYPE_NODEPERFDATA,
            D3DKMT_NODE_PERFDATA {
                NodeOrdinal: node,
                ..Default::default()
            },
        )?;
        (perf.Frequency > 0).then_some(perf.Frequency)
    }

    /// The driver's answer about `adapter` of kind `kind`, asked with
    /// `value`.
    fn query<T: Copy>(adapter: u32, kind: KMTQUERYADAPTERINFOTYPE, mut value: T) -> Option<T> {
        let mut info = D3DKMT_QUERYADAPTERINFO {
            hAdapter: adapter,
            Type: kind,
            pPrivateDriverData: (&raw mut value).cast(),
            PrivateDriverDataSize: size_of::<T>() as u32,
        };
        // SAFETY: an adapter handle opened above and a value of the size
        // and kind the query names.
        #[allow(unsafe_code)]
        let status = unsafe { D3DKMTQueryAdapterInfo(&mut info) };
        (status == 0).then_some(value)
    }

    /// A process's processor time, in 100 ns.
    fn cpu_time(process: HANDLE) -> Option<u64> {
        let (mut created, mut exited, mut kernel, mut user) = Default::default();
        // SAFETY: fills the times passed, or fails for a handle that is not
        // a process's.
        #[allow(unsafe_code)]
        let read =
            unsafe { GetProcessTimes(process, &mut created, &mut exited, &mut kernel, &mut user) };
        (read != 0).then(|| ticks(kernel) + ticks(user))
    }

    fn ticks(time: FILETIME) -> u64 {
        u64::from(time.dwHighDateTime) << 32 | u64::from(time.dwLowDateTime)
    }
}
