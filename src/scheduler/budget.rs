use sysinfo::System;

/// System memory information and the computed budget for model loading.
pub struct MemoryBudget {
    /// Total physical RAM in bytes.
    pub total_ram: u64,
    /// Currently available RAM in bytes.
    pub available_ram: u64,
    /// Maximum bytes allowed for loaded models (user-configured, or full
    /// `available_ram` if omitted).
    pub budget: u64,
}

impl MemoryBudget {
    /// Detect system memory. If budget_str is provided (e.g. "8G"), use that
    /// (clamped to total RAM). `"0"` means total RAM. If omitted, use the
    /// full live availability — `available_memory_platform` already excludes
    /// wired+active pages the OS is using, so reserving an extra margin on
    /// top of that double-counts the OS overhead.
    pub fn detect(budget_str: Option<&str>) -> Self {
        let sys = System::new_all();
        let total_ram = sys.total_memory();
        let available_ram = available_memory_platform(&sys);

        let budget = match budget_str {
            Some(s) => match parse_size(s) {
                Some(0) => total_ram,
                Some(n) => std::cmp::min(n, total_ram),
                None => available_ram,
            },
            None => available_ram,
        };

        Self {
            total_ram,
            available_ram,
            budget,
        }
    }

    /// Check if a model of the given size can be loaded within budget.
    pub fn can_fit(&self, model_size: u64) -> bool {
        model_size <= self.budget
    }
}

/// On macOS, sysinfo's `available_memory()` only returns free pages. macOS
/// aggressively caches files in inactive/purgeable/speculative pages that
/// are immediately reclaimable under pressure. We query `host_statistics64`
/// to include those — matches Activity Monitor's "Memory Available".
#[cfg(target_os = "macos")]
fn available_memory_platform(_sys: &System) -> u64 {
    use std::mem::{MaybeUninit, size_of};

    const HOST_VM_INFO64: i32 = 4;
    const KERN_SUCCESS: i32 = 0;

    #[repr(C)]
    #[allow(non_camel_case_types)]
    struct vm_statistics64 {
        free_count: u32,
        active_count: u32,
        inactive_count: u32,
        wire_count: u32,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: u32,
        speculative_count: u32,
        decompressions: u64,
        compressions: u64,
        swapins: u64,
        swapouts: u64,
        compressor_page_count: u32,
        throttled_count: u32,
        external_page_count: u32,
        internal_page_count: u32,
        total_uncompressed_pages_in_compressor: u64,
    }

    unsafe extern "C" {
        fn mach_host_self() -> u32;
        fn host_statistics64(host: u32, flavor: i32, info: *mut vm_statistics64, count: *mut u32) -> i32;
    }

    const VM_STATS_COUNT: u32 = (size_of::<vm_statistics64>() / size_of::<u32>()) as u32;

    unsafe {
        let host = mach_host_self();
        let mut info = MaybeUninit::<vm_statistics64>::zeroed();
        let mut count = VM_STATS_COUNT;

        let ret = host_statistics64(host, HOST_VM_INFO64, info.as_mut_ptr(), &mut count);
        if ret != KERN_SUCCESS {
            return _sys.available_memory();
        }

        let info = info.assume_init();

        // hw.pagesize — always 16384 on Apple Silicon, 4096 on Intel Macs.
        let mut page_size: u64 = 0;
        let mut len = size_of::<u64>();
        let name = b"hw.pagesize\0";
        libc_sysctl(name.as_ptr().cast(), &mut page_size as *mut u64 as *mut _, &mut len);
        if page_size == 0 {
            page_size = 16384;
        }

        // Activity Monitor's "Available" = free + inactive + purgeable +
        // speculative. Speculative pages are file-cache prefetch that the
        // kernel reclaims first under pressure, so leaving them out
        // understates by 1–2 GB on a typical Mac.
        (info.free_count as u64
            + info.inactive_count as u64
            + info.purgeable_count as u64
            + info.speculative_count as u64)
            * page_size
    }
}

#[cfg(target_os = "macos")]
unsafe fn libc_sysctl(name: *const i8, val: *mut std::ffi::c_void, len: *mut usize) {
    unsafe extern "C" {
        fn sysctlbyname(name: *const i8, oldp: *mut std::ffi::c_void, oldlenp: *mut usize, newp: *mut std::ffi::c_void, newlen: usize) -> i32;
    }
    unsafe { sysctlbyname(name, val, len, std::ptr::null_mut(), 0); }
}

#[cfg(not(target_os = "macos"))]
fn available_memory_platform(sys: &System) -> u64 {
    sys.available_memory()
}

fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (num, mult) = if s.ends_with('G') || s.ends_with('g') {
        (s[..s.len() - 1].parse::<f64>().ok()?, 1_073_741_824.0)
    } else if s.ends_with('M') || s.ends_with('m') {
        (s[..s.len() - 1].parse::<f64>().ok()?, 1_048_576.0)
    } else {
        (s.parse::<f64>().ok()?, 1.0)
    };
    Some((num * mult) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size() {
        assert_eq!(parse_size("8G"), Some(8 * 1_073_741_824));
        assert_eq!(parse_size("512M"), Some(512 * 1_048_576));
        assert_eq!(parse_size("1024"), Some(1024));
        assert_eq!(parse_size("bad"), None);
    }
}
