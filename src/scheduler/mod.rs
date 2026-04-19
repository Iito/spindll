//! Scheduler — manages model loading/unloading and memory budget.
//!
//! Knows how much RAM/VRAM is available, refuses to load models
//! that won't fit, and handles LRU eviction.

pub mod budget;
