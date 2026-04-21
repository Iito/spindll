use sysinfo::System;

/// System memory information and the computed budget for model loading.
pub struct MemoryBudget {
    /// Total physical RAM in bytes.
    pub total_ram: u64,
    /// Currently available RAM in bytes.
    pub available_ram: u64,
    /// Maximum bytes allowed for loaded models (user-configured or 80% of available).
    pub budget: u64,
}

impl MemoryBudget {
    /// Detect system memory. If budget_str is provided (e.g. "8G"), use that.
    /// Otherwise, use 80% of available RAM.
    pub fn detect(budget_str: Option<&str>) -> Self {
        let sys = System::new_all();
        let total_ram = sys.total_memory();
        let available_ram = sys.available_memory();

        let budget = match budget_str {
            Some(s) => parse_size(s).unwrap_or(available_ram),
            None => available_ram * 80 / 100,
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
