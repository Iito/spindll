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

    /// Calculate the load budget for a model, accounting for scheduler overhead.
    ///
    /// Takes the minimum of the configured budget and available RAM, then subtracts
    /// any scheduler overhead. This ensures the backend has accurate memory headroom
    /// for context window sizing, and reserves space for batch scheduler threads if needed.
    pub fn load_budget_with_scheduler(&self, scheduler_overhead: u64) -> u64 {
        let working_budget = std::cmp::min(self.budget, self.available_ram);
        working_budget.saturating_sub(scheduler_overhead)
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
    fn parse_size_units() {
        assert_eq!(parse_size("8G"), Some(8 * 1_073_741_824));
        assert_eq!(parse_size("512M"), Some(512 * 1_048_576));
        assert_eq!(parse_size("1024"), Some(1024));
        assert_eq!(parse_size("bad"), None);
    }

    #[test]
    fn parse_size_zero_is_some_zero() {
        // "0" must parse to Some(0) so detect() maps it to total_ram.
        assert_eq!(parse_size("0"), Some(0));
    }

    // Item 10: --budget 0 should set budget = total_ram, not zero.
    #[test]
    fn budget_zero_means_total_ram() {
        let m = MemoryBudget::detect(Some("0"));
        assert!(m.total_ram > 0, "total_ram must be nonzero");
        assert_eq!(m.budget, m.total_ram, "--budget 0 must equal total_ram");
    }

    // Item 10: an explicit value larger than total RAM must be clamped.
    #[test]
    fn explicit_budget_clamped_to_total_ram() {
        let m = MemoryBudget::detect(Some("99999G"));
        assert_eq!(m.budget, m.total_ram, "oversized budget must clamp to total_ram");
    }

    // Item 10: explicit budget within total RAM is honoured as-is.
    #[test]
    fn explicit_budget_within_total_ram_honoured() {
        let m8 = MemoryBudget::detect(Some("8G"));
        let eight_gb = 8u64 * 1_073_741_824;
        if m8.total_ram >= eight_gb {
            assert_eq!(m8.budget, eight_gb);
        } else {
            assert_eq!(m8.budget, m8.total_ram);
        }
    }

    // Item 10: invalid string falls back to available RAM, not zero.
    #[test]
    fn invalid_budget_string_falls_back_to_available() {
        let m = MemoryBudget::detect(Some("not_a_number"));
        assert_eq!(m.budget, m.available_ram, "unparseable budget must fall back to available_ram");
    }

    // Item 10: no budget string → full available RAM (no 80% margin).
    #[test]
    fn no_budget_equals_full_available_ram() {
        let m = MemoryBudget::detect(None);
        assert_eq!(m.budget, m.available_ram, "default budget must equal available_ram (no margin)");
        assert!(m.available_ram > 0, "available_ram must be nonzero");
        assert!(m.available_ram <= m.total_ram);
    }

    // Item 12: smoke test — available_ram is a sane fraction of total on all platforms.
    #[test]
    fn available_ram_is_sane() {
        let m = MemoryBudget::detect(None);
        assert!(m.available_ram > 0, "available_ram should be nonzero");
        assert!(m.available_ram <= m.total_ram, "available must not exceed total");
    }

    // Item 12 (macOS): available_ram includes reclaimable pages so it should
    // be well above the sysinfo free-only value (typically > 100 MB even under load).
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_available_includes_reclaimable_pages() {
        let m = MemoryBudget::detect(None);
        const MIN_EXPECTED: u64 = 100 * 1_048_576; // 100 MB
        assert!(
            m.available_ram >= MIN_EXPECTED,
            "macOS available_ram ({} MB) should include inactive+purgeable+speculative pages",
            m.available_ram / 1_048_576
        );
    }

    // Default-mode clamp: when no explicit budget is set, load_budget_with_scheduler
    // should clamp to min(budget, available_ram) and subtract scheduler overhead.
    #[test]
    fn default_mode_clamps_to_available_with_scheduler_offset() {
        let m = MemoryBudget::detect(None);
        let scheduler_overhead = 100 * 1_048_576; // 100 MB

        let load_budget = m.load_budget_with_scheduler(scheduler_overhead);

        // The effective working budget is min(configured budget, available_ram).
        // Since budget == available_ram when no explicit budget is set, we should get:
        let expected = m.available_ram.saturating_sub(scheduler_overhead);
        assert_eq!(load_budget, expected, "default-mode clamp must account for scheduler overhead");
    }

    // Explicit budget clamped correctly with scheduler offset: ensure that
    // explicit budgets are clamped to available_ram before scheduler subtraction.
    #[test]
    fn explicit_budget_clamped_then_scheduler_offset_applied() {
        let m = MemoryBudget::detect(Some("8G"));
        let scheduler_overhead = 50 * 1_048_576; // 50 MB

        let load_budget = m.load_budget_with_scheduler(scheduler_overhead);

        // load_budget should be min(8GB, available_ram) - scheduler_overhead
        let expected = std::cmp::min(m.budget, m.available_ram).saturating_sub(scheduler_overhead);
        assert_eq!(load_budget, expected, "explicit budget must be clamped then scheduler-offset applied");
    }
}
