//! Hardware counters per pass, in process: what the GPU's own counters say
//! each pass of a frame did (its shader cores' activity, stalls and
//! occupancy, its samplers', memory traffic, depth-test rejects, pixels
//! written), where the driver offers them through Vulkan's performance
//! queries: Intel's on Windows and Linux, and Mesa's for AMD, Intel and
//! Qualcomm GPUs on Linux. NVIDIA's drivers offer none (Nsight's GPU Trace
//! counts), nor AMD's on Windows (Radeon GPU Profiler does). Opt-in, as
//! counting perturbs timings and the game then draws on Vulkan; where it
//! works it needs no privileges.
//!
//! Native renderers start their Vulkan instance through this module's
//! loader entry, which sees the calls wgpu makes: each outermost debug
//! group a command buffer records outside a render pass becomes one query
//! of the counters chosen, those one run of the work can count together,
//! from a query pool of the command buffer's own. wgpu records each pass
//! into a command buffer of its own, inside the debug group the renderer
//! names the pass's graph node by: queries are named by the groups open on
//! the recording thread, and summed into the graph node's. Pools are reset in a submission of their own
//! before the command buffers using them, as Vulkan requires, and read once
//! the GPU finished them. Unsafe code, as `pix.rs` holds: the Vulkan calls
//! and the entry points wgpu calls.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::{CStr, c_char, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use ash::vk;
use serde_json::{Value, json};

/// Query pools: for the command buffers of frames in flight, one per pass.
const POOLS: usize = 256;
/// Queries in a pool: a command buffer's passes.
const PASSES: u32 = 16;
/// Reset submissions in flight.
const RESETS: usize = 8;

/// Which counters to count.
#[derive(Clone, Debug)]
pub enum Selection {
    /// The first the driver lists that one run can count together: on
    /// Intel's drivers, their basic render set.
    First,
    /// Those named, as the driver names them, that one run can count
    /// together.
    Named(Vec<String>),
}

static SELECTION: OnceLock<Selection> = OnceLock::new();
/// The Vulkan loader, kept loaded.
static LOADER: OnceLock<ash::Entry> = OnceLock::new();
/// The instance started, with the loader's own functions.
static INSTANCE: OnceLock<ash::Instance> = OnceLock::new();
/// The device counting.
static COUNTING: OnceLock<Counting> = OnceLock::new();
/// Why passes are not counted, where they are not.
static UNAVAILABLE: Mutex<Option<String>> = Mutex::new(None);
/// Whether passes are counted now, between [`begin`] and [`end`].
static ACTIVE: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// The debug groups open on the recording thread, outermost first,
    /// across the command buffers wgpu records one encoder into; and how
    /// many outermost groups it opened.
    static GROUPS: RefCell<(Vec<String>, u64)> = const { RefCell::new((Vec::new(), 0)) };
}

/// Counts each pass's hardware counters in profiles from now on: native
/// renderers start their GPU instances on Vulkan through this module.
/// Before the first renderer starts.
pub fn enable(selection: Selection) -> Result<(), String> {
    SELECTION
        .set(selection)
        .map_err(|_| "hardware counters were enabled twice".to_owned())?;
    if engine::start_native_instances_with(start) {
        Ok(())
    } else {
        Err("GPU instances already start another way".into())
    }
}

/// Whether counters are enabled: profiles count passes where the GPU's
/// driver can.
pub fn enabled() -> bool {
    SELECTION.get().is_some()
}

/// Starts counting the passes drawn from now on, where counters are
/// enabled.
pub(crate) fn begin() {
    if let Some(counting) = COUNTING.get() {
        let mut state = counting.state();
        state.results.clear();
        state.skipped = 0;
        ACTIVE.store(true, Ordering::Release);
    }
}

/// Stops counting once the GPU finished the frames counted: each pass's
/// counters, their median over the pass's instances; `None` where counters
/// are not enabled.
pub(crate) fn end() -> Option<Value> {
    SELECTION.get()?;
    ACTIVE.store(false, Ordering::Release);
    let Some(counting) = COUNTING.get() else {
        let reason = lock(&UNAVAILABLE)
            .clone()
            .unwrap_or_else(|| "no Vulkan device started".into());
        return Some(json!({ "status": format!("unavailable: {reason}") }));
    };
    Some(counting.report())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn unavailable(reason: String) {
    lock(&UNAVAILABLE).get_or_insert(reason);
}

/// A GPU instance on Vulkan, whatever `descriptor` asks: counters exist on
/// Vulkan only. wgpu's own start where Vulkan cannot start.
fn start(descriptor: wgpu::InstanceDescriptor) -> wgpu::Instance {
    match start_vulkan(&descriptor) {
        Ok(instance) => instance,
        Err(reason) => {
            unavailable(format!("Vulkan could not start: {reason}"));
            wgpu::Instance::new(descriptor)
        }
    }
}

/// The function `function` as Vulkan returns functions.
macro_rules! void {
    ($function:expr, $kind:ty) => {
        // SAFETY: Vulkan's callers cast the pointer back to `$kind`, the
        // type of the name they asked for.
        Some(unsafe { std::mem::transmute::<$kind, unsafe extern "system" fn()>($function) })
    };
}

#[allow(unsafe_code)]
fn start_vulkan(descriptor: &wgpu::InstanceDescriptor) -> Result<wgpu::Instance, String> {
    if INSTANCE.get().is_some() {
        return Err("counters follow one GPU instance per program".into());
    }
    let loader = match LOADER.get() {
        Some(loader) => loader,
        None => {
            // SAFETY: loading the system's Vulkan loader, as wgpu's start
            // does.
            let loaded = unsafe { ash::Entry::load() }.map_err(|error| error.to_string())?;
            LOADER.get_or_init(|| loaded)
        }
    };
    // The loader's entry, but this module's functions for those wgpu's
    // command recording reaches.
    // SAFETY: `get_instance_proc_addr` is Vulkan's signature and forwards
    // every other name to the loader.
    let entry = unsafe {
        ash::Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr,
        })
    };
    // SAFETY: Vulkan calls through the loader.
    let version = unsafe { entry.try_enumerate_instance_version() }
        .map_err(|error| error.to_string())?
        .unwrap_or(vk::API_VERSION_1_0);
    if version < vk::API_VERSION_1_1 {
        return Err("Vulkan 1.1 or later is needed".into());
    }
    // Debug groups name the passes counted.
    let flags = descriptor.flags | wgpu::InstanceFlags::DEBUG;
    let extensions = wgpu::hal::vulkan::Instance::desired_extensions(&entry, version, flags)
        .map_err(|error| error.to_string())?;
    let names: Vec<*const c_char> = extensions.iter().map(|name| name.as_ptr()).collect();
    // wgpu's own instance asks for Vulkan 1.3 at most.
    let application = vk::ApplicationInfo::default()
        .application_name(c"engine")
        .engine_name(c"wgpu-hal")
        .api_version(version.min(vk::API_VERSION_1_3));
    let info = vk::InstanceCreateInfo::default()
        .application_info(&application)
        .enabled_extension_names(&names);
    // SAFETY: a valid create info whose pointers outlive the call.
    let instance =
        unsafe { entry.create_instance(&info, None) }.map_err(|error| error.to_string())?;
    // SAFETY: the instance just created, with the loader's functions.
    let raw = unsafe { ash::Instance::load(loader.static_fn(), instance.handle()) };
    if INSTANCE.set(raw).is_err() {
        return Err("counters follow one GPU instance per program".into());
    }
    // SAFETY: Vulkan calls through the loader.
    let optimus = unsafe { loader.enumerate_instance_layer_properties() }.is_ok_and(|layers| {
        layers
            .iter()
            .any(|layer| layer.layer_name_as_c_str() == Ok(c"VK_LAYER_NV_optimus"))
    });
    // SAFETY: the instance was created from `entry` with `version`,
    // `extensions` (wgpu's desired ones) and `flags`; wgpu owns it now.
    let hal = unsafe {
        wgpu::hal::vulkan::Instance::from_raw(
            entry,
            instance,
            version,
            0,
            None,
            extensions,
            flags,
            descriptor.memory_budget_thresholds,
            optimus,
            None,
        )
    }
    .map_err(|error| error.to_string())?;
    // SAFETY: a valid Vulkan instance, which wgpu owns from here.
    Ok(unsafe { wgpu::Instance::from_hal::<wgpu::hal::api::Vulkan>(hal) })
}

/// Vulkan's `vkGetInstanceProcAddr`: the loader's, but this module's
/// device creation and device function lookup.
#[allow(unsafe_code)]
unsafe extern "system" fn get_instance_proc_addr(
    instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    // SAFETY: Vulkan's callers pass a NUL-terminated name.
    match unsafe { CStr::from_ptr(name) }.to_bytes() {
        b"vkGetInstanceProcAddr" => {
            void!(get_instance_proc_addr, vk::PFN_vkGetInstanceProcAddr)
        }
        b"vkGetDeviceProcAddr" => void!(get_device_proc_addr, vk::PFN_vkGetDeviceProcAddr),
        b"vkCreateDevice" => void!(create_device, vk::PFN_vkCreateDevice),
        _ => {
            let loader = LOADER
                .get()
                .expect("the loader loads before instances start");
            // SAFETY: forwarded as called.
            unsafe { (loader.static_fn().get_instance_proc_addr)(instance, name) }
        }
    }
}

/// Vulkan's `vkGetDeviceProcAddr`: the driver's functions, but this
/// module's for those that record and submit the counted device's command
/// buffers.
#[allow(unsafe_code)]
unsafe extern "system" fn get_device_proc_addr(
    device: vk::Device,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    let counted = COUNTING
        .get()
        .is_some_and(|counting| counting.device.handle() == device);
    if counted {
        // SAFETY: Vulkan's callers pass a NUL-terminated name.
        let hooked = match unsafe { CStr::from_ptr(name) }.to_bytes() {
            b"vkBeginCommandBuffer" => void!(begin_command_buffer, vk::PFN_vkBeginCommandBuffer),
            b"vkEndCommandBuffer" => void!(end_command_buffer, vk::PFN_vkEndCommandBuffer),
            b"vkCmdBeginRenderPass" => void!(begin_render_pass, vk::PFN_vkCmdBeginRenderPass),
            b"vkCmdEndRenderPass" => void!(end_render_pass, vk::PFN_vkCmdEndRenderPass),
            b"vkCmdBeginDebugUtilsLabelEXT" => {
                void!(begin_label, vk::PFN_vkCmdBeginDebugUtilsLabelEXT)
            }
            b"vkCmdEndDebugUtilsLabelEXT" => void!(end_label, vk::PFN_vkCmdEndDebugUtilsLabelEXT),
            b"vkQueueSubmit" => void!(queue_submit, vk::PFN_vkQueueSubmit),
            b"vkDestroyDevice" => void!(destroy_device, vk::PFN_vkDestroyDevice),
            _ => None,
        };
        if hooked.is_some() {
            return hooked;
        }
    }
    let instance = INSTANCE.get().expect("instances start before devices");
    // SAFETY: forwarded as called.
    unsafe { (instance.fp_v1_0().get_device_proc_addr)(device, name) }
}

/// Vulkan's `vkCreateDevice`: the driver's, with performance queries
/// enabled where the device has them, and its counting set up.
#[allow(unsafe_code)]
unsafe extern "system" fn create_device(
    physical: vk::PhysicalDevice,
    info: *const vk::DeviceCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    device: *mut vk::Device,
) -> vk::Result {
    let instance = INSTANCE.get().expect("instances start before devices");
    let create = instance.fp_v1_0().create_device;
    // SAFETY: Vulkan's callers pass a valid create info.
    let info = unsafe { &*info };
    // SAFETY: as above, `queue_create_info_count` entries.
    let family = unsafe { slice(info.p_queue_create_infos, info.queue_create_info_count) }
        .first()
        .map_or(0, |queue| queue.queue_family_index);
    let plan = match plan(instance, physical, family) {
        Ok(plan) => plan,
        Err(reason) => {
            unavailable(reason);
            // SAFETY: forwarded as called.
            return unsafe { create(physical, info, allocator, device) };
        }
    };
    // The extension and its feature, ahead of the caller's.
    // SAFETY: as above, `enabled_extension_count` names.
    let mut names: Vec<*const c_char> = unsafe {
        slice(
            info.pp_enabled_extension_names,
            info.enabled_extension_count,
        )
    }
    .to_vec();
    // SAFETY: the names are NUL-terminated.
    let enabled = names
        .iter()
        .any(|&name| unsafe { CStr::from_ptr(name) } == ash::khr::performance_query::NAME);
    if !enabled {
        names.push(ash::khr::performance_query::NAME.as_ptr());
    }
    let mut feature = vk::PhysicalDevicePerformanceQueryFeaturesKHR {
        p_next: info.p_next.cast_mut(),
        performance_counter_query_pools: vk::TRUE,
        ..Default::default()
    };
    let mut ours = *info;
    ours.enabled_extension_count = names.len() as u32;
    ours.pp_enabled_extension_names = names.as_ptr();
    ours.p_next = (&raw mut feature).cast_const().cast();
    // SAFETY: the caller's create info with one more extension and feature,
    // which outlive the call.
    let result = unsafe { create(physical, &ours, allocator, device) };
    if result == vk::Result::SUCCESS {
        // SAFETY: the device was just created.
        match unsafe { Counting::new(instance, physical, *device, family, plan) } {
            Ok(counting) => {
                let _ = COUNTING.set(counting);
            }
            Err(reason) => unavailable(reason),
        }
    }
    result
}

/// `count` items at `items`, none where null.
///
/// # Safety
///
/// `items` points at `count` items, or is null.
#[allow(unsafe_code)]
unsafe fn slice<'a, T>(items: *const T, count: u32) -> &'a [T] {
    if items.is_null() || count == 0 {
        return &[];
    }
    // SAFETY: as the caller promises.
    unsafe { std::slice::from_raw_parts(items, count as usize) }
}

/// The counters to count on a device, before it exists.
struct Plan {
    indices: Vec<u32>,
    counters: Vec<Counter>,
    // Named or listed counters one run cannot count with those chosen.
    dropped: Vec<String>,
    // Names the driver does not list.
    unknown: Vec<String>,
    available: usize,
}

struct Counter {
    name: String,
    category: String,
    description: String,
    unit: vk::PerformanceCounterUnitKHR,
    storage: vk::PerformanceCounterStorageKHR,
}

/// The counters the selection asks for that `physical`'s queue `family`
/// can count in one run; why none, where none.
#[allow(unsafe_code)]
fn plan(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    family: u32,
) -> Result<Plan, String> {
    let selection = SELECTION.get().cloned().unwrap_or(Selection::First);
    // SAFETY: Vulkan calls on a valid physical device.
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
        .map_err(|error| error.to_string())?;
    let offered = extensions.iter().any(|extension| {
        extension.extension_name_as_c_str() == Ok(ash::khr::performance_query::NAME)
    });
    if !offered {
        return Err("this GPU's driver offers no Vulkan performance queries \
                    (VK_KHR_performance_query); Intel's drivers and Mesa's do, and Nsight \
                    (NVIDIA) or Radeon GPU Profiler (AMD) count without them"
            .into());
    }
    let mut feature = vk::PhysicalDevicePerformanceQueryFeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut feature);
    // SAFETY: as above.
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    if feature.performance_counter_query_pools == vk::FALSE {
        return Err("this GPU's driver offers performance queries without query pools".into());
    }
    let loader = LOADER
        .get()
        .expect("the loader loads before instances start");
    let queries = ash::khr::performance_query::Instance::new(loader, instance);
    // SAFETY: as above.
    let listed = unsafe {
        queries
            .enumerate_physical_device_queue_family_performance_query_counters_len(physical, family)
    }
    .map_err(|error| error.to_string())?;
    let mut counters = vec![vk::PerformanceCounterKHR::default(); listed];
    let mut descriptions = vec![vk::PerformanceCounterDescriptionKHR::default(); listed];
    // SAFETY: as above, into arrays of the length listed.
    unsafe {
        queries.enumerate_physical_device_queue_family_performance_query_counters(
            physical,
            family,
            &mut counters,
            &mut descriptions,
        )
    }
    .map_err(|error| error.to_string())?;
    let text = |name: Result<&CStr, _>| {
        name.map_or_else(
            |_| String::new(),
            |name: &CStr| name.to_string_lossy().into_owned(),
        )
    };
    let names: Vec<String> = descriptions
        .iter()
        .map(|description| text(description.name_as_c_str()))
        .collect();
    let mut unknown = Vec::new();
    // Counters of a whole command buffer cannot count a pass.
    let per_pass = |index: &u32| {
        counters[*index as usize].scope != vk::PerformanceCounterScopeKHR::COMMAND_BUFFER
    };
    let candidates: Vec<u32> = match &selection {
        Selection::First => (0..listed as u32).filter(per_pass).collect(),
        Selection::Named(wanted) => wanted
            .iter()
            .filter_map(|wanted| {
                let found = names
                    .iter()
                    .position(|name| name.eq_ignore_ascii_case(wanted));
                if found.is_none() {
                    unknown.push(wanted.clone());
                }
                found.map(|index| index as u32)
            })
            .filter(per_pass)
            .collect(),
    };
    // Those one run counts together, in order.
    let runs = |indices: &[u32]| {
        let info = vk::QueryPoolPerformanceCreateInfoKHR::default()
            .queue_family_index(family)
            .counter_indices(indices);
        // SAFETY: as above.
        unsafe {
            queries.get_physical_device_queue_family_performance_query_passes(physical, &info)
        }
    };
    let mut indices = Vec::new();
    let mut dropped = Vec::new();
    for candidate in candidates {
        indices.push(candidate);
        if runs(&indices) > 1 {
            indices.pop();
            // Listing every counter the driver has is no use; those named are.
            if matches!(selection, Selection::Named(_)) {
                dropped.push(names[candidate as usize].clone());
            }
        }
    }
    if indices.is_empty() {
        return Err("none of the counters asked for are this GPU's".into());
    }
    let chosen = indices
        .iter()
        .map(|&index| {
            let (counter, description) = (&counters[index as usize], &descriptions[index as usize]);
            Counter {
                name: names[index as usize].clone(),
                category: text(description.category_as_c_str()),
                description: text(description.description_as_c_str()),
                unit: counter.unit,
                storage: counter.storage,
            }
        })
        .collect();
    Ok(Plan {
        indices,
        counters: chosen,
        dropped,
        unknown,
        available: listed,
    })
}

/// A device counting its passes.
struct Counting {
    // The driver's functions.
    device: ash::Device,
    labels: ash::ext::debug_utils::Device,
    queries: ash::khr::performance_query::Device,
    gpu: String,
    plan: Plan,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    recording: HashMap<vk::CommandBuffer, Recording>,
    pools: Vec<Pool>,
    // Pools submitted, in submission order.
    submitted: VecDeque<usize>,
    command_pool: vk::CommandPool,
    resets: Vec<(vk::CommandBuffer, vk::Fence)>,
    next_reset: usize,
    // Each pass instance counted, and its counters.
    results: Vec<(Pass, Vec<f64>)>,
    // Passes not counted: no pool or query free.
    skipped: u64,
    destroyed: bool,
}

/// The queries of one command buffer's passes, from the first: results
/// are read from a pool's first query on, as some drivers (Intel's on
/// Windows) write a query's results at its index's offset whatever query
/// a read starts at.
struct Pool {
    raw: vk::QueryPool,
    // Its queries' passes, in order.
    passes: Vec<Pass>,
    use_: Use,
}

/// What a query counted: the debug groups open, outermost (the graph
/// node's) first, and which time the outermost was open.
#[derive(Clone)]
struct Pass {
    groups: Vec<String>,
    instance: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    Free,
    Recording,
    Submitted,
}

/// A command buffer being recorded.
#[derive(Default)]
struct Recording {
    depth: u32,
    render_pass: bool,
    pool: Option<usize>,
    open: Option<u32>,
}

impl Counting {
    /// Counting on `device`, just created from `physical` with performance
    /// queries enabled.
    ///
    /// # Safety
    ///
    /// `device` was created from `physical` with the extension and feature
    /// enabled, and is not in use yet.
    #[allow(unsafe_code)]
    unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        raw: vk::Device,
        family: u32,
        plan: Plan,
    ) -> Result<Self, String> {
        // SAFETY: a valid device of `instance`.
        let device = unsafe { ash::Device::load(instance.fp_v1_0(), raw) };
        let labels = ash::ext::debug_utils::Device::new(instance, &device);
        let queries = ash::khr::performance_query::Device::new(instance, &device);
        // SAFETY: a valid physical device.
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let gpu = properties.device_name_as_c_str().map_or_else(
            |_| String::new(),
            |name| name.to_string_lossy().into_owned(),
        );
        let lock = vk::AcquireProfilingLockInfoKHR::default().timeout(u64::MAX);
        // SAFETY: the device holds the lock until it is destroyed.
        unsafe { queries.acquire_profiling_lock(&lock) }
            .map_err(|error| format!("the driver refused its profiling lock: {error}"))?;
        let mut state = State::default();
        // SAFETY: as above.
        let made = unsafe { Self::objects(&device, family, &plan, &mut state) };
        if let Err(reason) = made {
            // SAFETY: this module's objects, unused yet, and the lock taken
            // above.
            unsafe {
                Self::destroy(&device, &mut state);
                queries.release_profiling_lock();
            }
            return Err(reason);
        }
        Ok(Self {
            device,
            labels,
            queries,
            gpu,
            plan,
            state: Mutex::new(state),
        })
    }

    /// The query pools of the plan's counters, and the command pool and
    /// buffers that reset their queries, into `state`.
    ///
    /// # Safety
    ///
    /// `device` holds the profiling lock.
    #[allow(unsafe_code)]
    unsafe fn objects(
        device: &ash::Device,
        family: u32,
        plan: &Plan,
        state: &mut State,
    ) -> Result<(), String> {
        for _ in 0..POOLS {
            let mut performance = vk::QueryPoolPerformanceCreateInfoKHR::default()
                .queue_family_index(family)
                .counter_indices(&plan.indices);
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::PERFORMANCE_QUERY_KHR)
                .query_count(PASSES)
                .push_next(&mut performance);
            // SAFETY: valid create infos; `destroy` destroys the pools.
            let raw = unsafe { device.create_query_pool(&info, None) }
                .map_err(|error| format!("the driver refused the counters' query pool: {error}"))?;
            state.pools.push(Pool {
                raw,
                passes: Vec::new(),
                use_: Use::Free,
            });
        }
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: as above.
        state.command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }
            .map_err(|error| error.to_string())?;
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(state.command_pool)
            .command_buffer_count(RESETS as u32);
        // SAFETY: as above.
        let buffers = unsafe { device.allocate_command_buffers(&allocate) }
            .map_err(|error| error.to_string())?;
        let fence = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        for buffer in buffers {
            // SAFETY: as above.
            let fence =
                unsafe { device.create_fence(&fence, None) }.map_err(|error| error.to_string())?;
            state.resets.push((buffer, fence));
        }
        Ok(())
    }

    /// Destroys this module's objects in `state`.
    ///
    /// # Safety
    ///
    /// The GPU finished with them.
    #[allow(unsafe_code)]
    unsafe fn destroy(device: &ash::Device, state: &mut State) {
        // SAFETY: as the caller promises; Vulkan ignores null handles.
        unsafe {
            for (_, fence) in state.resets.drain(..) {
                device.destroy_fence(fence, None);
            }
            device.destroy_command_pool(state.command_pool, None);
            for pool in state.pools.drain(..) {
                device.destroy_query_pool(pool.raw, None);
            }
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// The counters of the `count` first queries of `pool`, once the GPU
    /// finished them; waiting for them where `wait`. `None` while not
    /// finished, or where the driver failed to tell.
    #[allow(unsafe_code)]
    fn read(&self, pool: vk::QueryPool, count: usize, wait: bool) -> Option<Vec<Vec<f64>>> {
        let counters = self.plan.counters.len();
        let mut data = vec![vk::PerformanceCounterResultKHR::default(); counters * count];
        let stride = std::mem::size_of::<vk::PerformanceCounterResultKHR>() * counters;
        let flags = if wait {
            vk::QueryResultFlags::WAIT
        } else {
            vk::QueryResultFlags::empty()
        };
        // SAFETY: room for `count` queries' results from the pool's first,
        // `stride` bytes each.
        let result = unsafe {
            (self.device.fp_v1_0().get_query_pool_results)(
                self.device.handle(),
                pool,
                0,
                count as u32,
                stride * count,
                data.as_mut_ptr().cast::<c_void>(),
                stride as vk::DeviceSize,
                flags,
            )
        };
        (result == vk::Result::SUCCESS).then(|| {
            data.chunks(counters)
                .map(|query| {
                    query
                        .iter()
                        .zip(&self.plan.counters)
                        // SAFETY: each result is read as the storage its
                        // counter declares.
                        .map(|(value, counter)| unsafe {
                            match counter.storage {
                                vk::PerformanceCounterStorageKHR::INT32 => f64::from(value.int32),
                                vk::PerformanceCounterStorageKHR::INT64 => value.int64 as f64,
                                vk::PerformanceCounterStorageKHR::UINT32 => f64::from(value.uint32),
                                vk::PerformanceCounterStorageKHR::UINT64 => value.uint64 as f64,
                                vk::PerformanceCounterStorageKHR::FLOAT32 => {
                                    f64::from(value.float32)
                                }
                                _ => value.float64,
                            }
                        })
                        .collect()
                })
                .collect()
        })
    }

    /// Reads the submitted pools the GPU finished, oldest first; all of
    /// them where `wait`.
    fn collect(&self, state: &mut State, wait: bool) {
        while let Some(&index) = state.submitted.front() {
            let pool = &state.pools[index];
            let read = self.read(pool.raw, pool.passes.len(), wait);
            if read.is_none() && !wait {
                break;
            }
            // Read, or lost where the GPU never finished it.
            let pool = &mut state.pools[index];
            for (pass, values) in pool.passes.drain(..).zip(read.into_iter().flatten()) {
                state.results.push((pass, values));
            }
            pool.use_ = Use::Free;
            state.submitted.pop_front();
        }
    }

    /// The query of the pass that `buffer` begins, from the buffer's pool:
    /// the first free one where it has none yet.
    fn take(
        &self,
        state: &mut State,
        buffer: vk::CommandBuffer,
        pass: Pass,
    ) -> Option<(vk::QueryPool, u32)> {
        let index = match state.recording.get(&buffer)?.pool {
            Some(index) => index,
            None => {
                if !state.pools.iter().any(|pool| pool.use_ == Use::Free) {
                    self.collect(state, false);
                }
                let index = state.pools.iter().position(|pool| pool.use_ == Use::Free)?;
                state.pools[index].use_ = Use::Recording;
                state.recording.get_mut(&buffer)?.pool = Some(index);
                index
            }
        };
        let pool = &mut state.pools[index];
        let query = pool.passes.len() as u32;
        if query >= PASSES {
            return None;
        }
        pool.passes.push(pass);
        Some((pool.raw, query))
    }

    /// Resets the queries of `pools` on `queue` in a submission of its own,
    /// ahead of the command buffers using them.
    #[allow(unsafe_code)]
    fn reset(
        &self,
        state: &mut State,
        queue: vk::Queue,
        pools: &[usize],
    ) -> Result<(), vk::Result> {
        let (buffer, fence) = state.resets[state.next_reset];
        state.next_reset = (state.next_reset + 1) % RESETS;
        let device = &self.device;
        // SAFETY: the buffer and fence are this module's; the fence tells
        // when the buffer's last submission finished.
        unsafe {
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.reset_fences(&[fence])?;
            device.reset_command_buffer(buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(buffer, &begin)?;
            for &index in pools {
                let pool = &state.pools[index];
                device.cmd_reset_query_pool(buffer, pool.raw, 0, pool.passes.len() as u32);
            }
            device.end_command_buffer(buffer)?;
            let buffers = [buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&buffers);
            device.queue_submit(queue, &[submit], fence)
        }
    }

    /// Each graph node's counters, its passes' summed (and its percentages
    /// and rates weighted by time), with each pass's: medians over the
    /// node's instances (once a frame, for the renderer's nodes), the
    /// costliest first where a counter is a time.
    fn report(&self) -> Value {
        let mut state = self.state();
        self.collect(&mut state, true);
        let counters = &self.plan.counters;
        let unit = |unit| counters.iter().position(|counter| counter.unit == unit);
        let time = unit(vk::PerformanceCounterUnitKHR::NANOSECONDS);
        let weight = time.or(unit(vk::PerformanceCounterUnitKHR::CYCLES));
        let averaged: Vec<bool> = counters
            .iter()
            .map(|counter| {
                matches!(
                    counter.unit,
                    vk::PerformanceCounterUnitKHR::PERCENTAGE
                        | vk::PerformanceCounterUnitKHR::HERTZ
                        | vk::PerformanceCounterUnitKHR::BYTES_PER_SECOND
                        | vk::PerformanceCounterUnitKHR::KELVIN
                        | vk::PerformanceCounterUnitKHR::WATTS
                        | vk::PerformanceCounterUnitKHR::VOLTS
                        | vk::PerformanceCounterUnitKHR::AMPS
                )
            })
            .collect();
        // Values combined: summed, or weighted by time.
        let combine = |parts: &[&Vec<f64>]| -> Vec<f64> {
            let weights: Vec<f64> = parts
                .iter()
                .map(|values| weight.map_or(1.0, |weight| values[weight]))
                .collect();
            let total: f64 = weights.iter().sum();
            (0..counters.len())
                .map(|counter| {
                    let sum: f64 = parts.iter().map(|values| values[counter]).sum();
                    if !averaged[counter] {
                        return sum;
                    }
                    if total > 0.0 {
                        parts
                            .iter()
                            .zip(&weights)
                            .map(|(values, weight)| values[counter] * weight)
                            .sum::<f64>()
                            / total
                    } else {
                        sum / parts.len() as f64
                    }
                })
                .collect()
        };
        let medians = |instances: &[Vec<f64>]| -> Vec<f64> {
            (0..counters.len())
                .map(|counter| {
                    let values: Vec<f64> = instances.iter().map(|values| values[counter]).collect();
                    engine::host::median(&values).unwrap_or(f64::NAN)
                })
                .collect()
        };
        // Each node's instances, and each of its passes' per instance.
        let mut nodes: BTreeMap<&str, Instances> = BTreeMap::new();
        let mut parts: BTreeMap<&str, BTreeMap<String, Instances>> = BTreeMap::new();
        for (pass, values) in &state.results {
            let Some(node) = pass.groups.first() else {
                continue;
            };
            nodes
                .entry(node)
                .or_default()
                .entry(pass.instance)
                .or_default()
                .push(values);
            parts
                .entry(node)
                .or_default()
                .entry(pass.groups[1..].join(" > "))
                .or_default()
                .entry(pass.instance)
                .or_default()
                .push(values);
        }
        let named = |values: &[f64]| -> serde_json::Map<String, Value> {
            counters
                .iter()
                .zip(values)
                .map(|(counter, value)| (counter.name.clone(), json!(value)))
                .collect()
        };
        let by_time = |a: &Vec<f64>, b: &Vec<f64>| match time {
            Some(time) => b[time].total_cmp(&a[time]),
            None => std::cmp::Ordering::Equal,
        };
        let mut rows: Vec<(&str, Vec<f64>, usize)> = nodes
            .iter()
            .map(|(node, instances)| {
                let combined: Vec<Vec<f64>> =
                    instances.values().map(|parts| combine(parts)).collect();
                (*node, medians(&combined), combined.len())
            })
            .collect();
        rows.sort_by(|a, b| by_time(&a.1, &b.1));
        let passes: Vec<Value> = rows
            .iter()
            .map(|(node, values, instances)| {
                let mut inner: Vec<(&String, Vec<f64>)> = parts[node]
                    .iter()
                    .map(|(part, instances)| {
                        let combined: Vec<Vec<f64>> =
                            instances.values().map(|parts| combine(parts)).collect();
                        (part, medians(&combined))
                    })
                    .collect();
                inner.sort_by(|a, b| by_time(&a.1, &b.1));
                json!({
                    "pass": node,
                    "instances": instances,
                    "median": named(values),
                    "parts": inner.iter().take(8).map(|(part, values)| json!({
                        "part": if part.is_empty() { "(between its passes)" } else { part.as_str() },
                        "median": named(values),
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        json!({
            "status": "counted",
            "gpu": self.gpu,
            "backend": "Vulkan, which counters need; timings with counters on are perturbed",
            "available": self.plan.available,
            "counters": counters.iter().map(|counter| json!({
                "name": counter.name,
                "category": counter.category,
                "unit": format!("{:?}", counter.unit).to_lowercase(),
                "description": counter.description,
            })).collect::<Vec<_>>(),
            "not_counted": {
                "needing_another_run": self.plan.dropped,
                "unknown": self.plan.unknown,
            },
            "passes_not_counted": state.skipped,
            "passes": passes,
        })
    }
}

/// Counted values by the instance of their graph node they were counted
/// in.
type Instances<'a> = BTreeMap<u64, Vec<&'a Vec<f64>>>;

/// The device counting, which hooked functions are only of.
fn counting() -> &'static Counting {
    COUNTING
        .get()
        .expect("functions are hooked for the counting device")
}

#[allow(unsafe_code)]
unsafe extern "system" fn begin_command_buffer(
    buffer: vk::CommandBuffer,
    info: *const vk::CommandBufferBeginInfo<'_>,
) -> vk::Result {
    let counting = counting();
    // SAFETY: forwarded as called.
    let result = unsafe { (counting.device.fp_v1_0().begin_command_buffer)(buffer, info) };
    if result == vk::Result::SUCCESS {
        let mut state = counting.state();
        // A pool the buffer recorded into before, never submitted.
        if let Some(index) = state
            .recording
            .insert(buffer, Recording::default())
            .and_then(|old| old.pool)
        {
            let pool = &mut state.pools[index];
            if pool.use_ == Use::Recording {
                pool.passes.clear();
                pool.use_ = Use::Free;
            }
        }
    }
    result
}

/// The pool and query `buffer` has open, which it closes.
fn close(state: &mut State, buffer: vk::CommandBuffer) -> Option<(vk::QueryPool, u32)> {
    let recording = state.recording.get_mut(&buffer)?;
    let query = recording.open.take()?;
    Some((state.pools[recording.pool?].raw, query))
}

#[allow(unsafe_code)]
unsafe extern "system" fn end_command_buffer(buffer: vk::CommandBuffer) -> vk::Result {
    let counting = counting();
    let open = close(&mut counting.state(), buffer);
    if let Some((pool, query)) = open {
        // SAFETY: the query this buffer began.
        unsafe { counting.device.cmd_end_query(buffer, pool, query) };
    }
    // SAFETY: forwarded as called.
    unsafe { (counting.device.fp_v1_0().end_command_buffer)(buffer) }
}

#[allow(unsafe_code)]
unsafe extern "system" fn begin_render_pass(
    buffer: vk::CommandBuffer,
    info: *const vk::RenderPassBeginInfo<'_>,
    contents: vk::SubpassContents,
) {
    let counting = counting();
    if let Some(recording) = counting.state().recording.get_mut(&buffer) {
        recording.render_pass = true;
    }
    // SAFETY: forwarded as called.
    unsafe { (counting.device.fp_v1_0().cmd_begin_render_pass)(buffer, info, contents) }
}

#[allow(unsafe_code)]
unsafe extern "system" fn end_render_pass(buffer: vk::CommandBuffer) {
    let counting = counting();
    // SAFETY: forwarded as called.
    unsafe { (counting.device.fp_v1_0().cmd_end_render_pass)(buffer) };
    if let Some(recording) = counting.state().recording.get_mut(&buffer) {
        recording.render_pass = false;
    }
}

#[allow(unsafe_code)]
unsafe extern "system" fn begin_label(
    buffer: vk::CommandBuffer,
    label: *const vk::DebugUtilsLabelEXT<'_>,
) {
    let counting = counting();
    // SAFETY: forwarded as called.
    unsafe { (counting.labels.fp().cmd_begin_debug_utils_label_ext)(buffer, label) };
    // SAFETY: Vulkan's callers pass a valid label with a NUL-terminated
    // name.
    let name = unsafe { (*label).label_name_as_c_str() }
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    let pass = GROUPS.with_borrow_mut(|(groups, outermost)| {
        if groups.is_empty() {
            *outermost += 1;
        }
        groups.push(name);
        Pass {
            groups: groups.clone(),
            instance: *outermost,
        }
    });
    let mut state = counting.state();
    let Some(recording) = state.recording.get_mut(&buffer) else {
        return;
    };
    recording.depth += 1;
    if recording.depth > 1 || recording.render_pass || !ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let Some((pool, query)) = counting.take(&mut state, buffer, pass) else {
        state.skipped += 1;
        return;
    };
    if let Some(recording) = state.recording.get_mut(&buffer) {
        recording.open = Some(query);
    }
    // SAFETY: a query reset before this buffer runs, begun outside a render
    // pass, the only performance query open in the buffer and of the only
    // pool the buffer uses.
    unsafe {
        counting
            .device
            .cmd_begin_query(buffer, pool, query, vk::QueryControlFlags::empty());
    }
}

#[allow(unsafe_code)]
unsafe extern "system" fn end_label(buffer: vk::CommandBuffer) {
    let counting = counting();
    GROUPS.with_borrow_mut(|(groups, _)| groups.pop());
    let open = {
        let mut state = counting.state();
        let ended = state.recording.get_mut(&buffer).is_some_and(|recording| {
            recording.depth = recording.depth.saturating_sub(1);
            recording.depth == 0
        });
        if ended {
            close(&mut state, buffer)
        } else {
            None
        }
    };
    if let Some((pool, query)) = open {
        // SAFETY: the query this buffer began at the label's start.
        unsafe { counting.device.cmd_end_query(buffer, pool, query) };
    }
    // SAFETY: forwarded as called.
    unsafe { (counting.labels.fp().cmd_end_debug_utils_label_ext)(buffer) }
}

#[allow(unsafe_code)]
unsafe extern "system" fn queue_submit(
    queue: vk::Queue,
    count: u32,
    submits: *const vk::SubmitInfo<'_>,
    fence: vk::Fence,
) -> vk::Result {
    let counting = counting();
    let mut state = counting.state();
    let mut pools = Vec::new();
    // SAFETY: Vulkan's callers pass `count` submissions, each with its
    // command buffers.
    for submit in unsafe { slice(submits, count) } {
        // SAFETY: as above.
        for buffer in unsafe { slice(submit.p_command_buffers, submit.command_buffer_count) } {
            if let Some(index) = state
                .recording
                .get_mut(buffer)
                .and_then(|recording| recording.pool.take())
            {
                pools.push(index);
            }
        }
    }
    for &index in &pools {
        if state.pools[index].passes.is_empty() {
            state.pools[index].use_ = Use::Free;
        }
    }
    pools.retain(|&index| !state.pools[index].passes.is_empty());
    if !pools.is_empty() {
        if let Err(error) = counting.reset(&mut state, queue, &pools) {
            unavailable(format!("resetting queries failed: {error}"));
            ACTIVE.store(false, Ordering::Release);
        }
        for &index in &pools {
            state.pools[index].use_ = Use::Submitted;
            state.submitted.push_back(index);
        }
    }
    // SAFETY: forwarded as called.
    let result = unsafe { (counting.device.fp_v1_0().queue_submit)(queue, count, submits, fence) };
    counting.collect(&mut state, false);
    result
}

#[allow(unsafe_code)]
unsafe extern "system" fn destroy_device(
    device: vk::Device,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    let counting = counting();
    {
        let mut state = counting.state();
        if !state.destroyed {
            state.destroyed = true;
            // SAFETY: the device's work finished, as destroying it needs;
            // these are this module's objects and the lock it took.
            unsafe {
                Counting::destroy(&counting.device, &mut state);
                counting.queries.release_profiling_lock();
            }
        }
    }
    // SAFETY: forwarded as called.
    unsafe { (counting.device.fp_v1_0().destroy_device)(device, allocator) }
}
