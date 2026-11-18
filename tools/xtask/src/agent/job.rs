#![expect(
    unsafe_code,
    reason = "puts the driver in a job object that ends its children when the driver ends"
)]

use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
    SetInformationJobObject,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

pub fn contain() -> Result<(), String> {
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    let size = u32::try_from(size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>())
        .map_err(|error| error.to_string())?;
    // SAFETY: an unnamed job with default security; the handle is kept open
    // for the driver's whole life and closes when the driver exits, which is
    // what ends every process in the job.
    let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    if job.is_null() {
        return Err(format!(
            "cannot create a job object: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: `limits` is a valid JOBOBJECT_EXTENDED_LIMIT_INFORMATION of
    // `size` bytes that outlives the call, and `job` is the handle just made.
    let limited = unsafe {
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&raw const limits).cast(),
            size,
        )
    };
    // SAFETY: the pseudo handle of this process and the job handle above;
    // processes this driver starts afterwards join the job with it.
    let assigned = limited != 0 && unsafe { AssignProcessToJobObject(job, GetCurrentProcess()) } != 0;
    if assigned {
        Ok(())
    } else {
        Err(format!(
            "cannot put the driver in a job object: {}",
            std::io::Error::last_os_error()
        ))
    }
}
