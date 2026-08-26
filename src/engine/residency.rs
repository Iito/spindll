// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

//! Mapping a caller-supplied model name onto the key it is resident under.
//!
//! A registry entry has one canonical key (`ollama/qwen2.5/0.5b.gguf`) and any
//! number of names that resolve to it — the Ollama `name:tag` form, a bare repo
//! prefix, the display name `spindll list` prints. [`ModelManager`] keys its
//! slots by whatever string it was handed at load time, so a surface that loads
//! under one name and looks up under another either misses the resident model
//! or loads the same weights a second time. Every entry point that takes a
//! model name from a client goes through here first.

use crate::engine::ModelManager;
use crate::model_store::ModelStore;

/// The key `model` is resident under, without loading anything.
///
/// Falls back to the name as given when nothing matches, so the caller's
/// "model '<name>' not loaded" error still names what they asked for.
pub fn resident_key(mgr: &ModelManager, store: &ModelStore, model: &str) -> String {
    if mgr.is_loaded(model) {
        return model.to_string();
    }
    match store.resolve_key(model) {
        Ok(key) if mgr.is_loaded(&key) => key,
        _ => model.to_string(),
    }
}

/// The canonical key for `model`, loading it into `mgr` if it isn't resident.
///
/// Blocking: loads mmap the weights and upload to the GPU. Call it from a
/// blocking context, never straight off an async worker.
pub fn ensure_loaded(
    mgr: &ModelManager,
    store: &ModelStore,
    model: &str,
) -> anyhow::Result<String> {
    if mgr.is_loaded(model) {
        return Ok(model.to_string());
    }
    let resolved = store.resolve(model)?;
    if mgr.is_loaded(&resolved.key) {
        return Ok(resolved.key);
    }
    mgr.load_model_with_options(
        &resolved.key,
        &resolved.path,
        crate::engine::manager::LoadOptions {
            digest: resolved.digest,
            #[cfg(feature = "vision")]
            mmproj_path: resolved.mmproj_path,
            ..Default::default()
        },
    )?;
    Ok(resolved.key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendLoadParams, BackendModel, InferenceBackend};
    use crate::engine::streaming::{GenerateParams, GenerateResult};
    use crate::model_store::registry::{ModelEntry, ModelFormat, ModelSource, Registry};

    struct FakeBackend;
    impl InferenceBackend for FakeBackend {
        fn load_model(
            &self,
            _: &std::path::Path,
            _: BackendLoadParams,
        ) -> anyhow::Result<Box<dyn BackendModel>> {
            Ok(Box::new(FakeModel))
        }
        fn name(&self) -> &str { "llamacpp" }
    }
    struct FakeModel;
    impl BackendModel for FakeModel {
        fn generate(
            &self,
            _: &str,
            _: &GenerateParams,
            _: &mut dyn FnMut(&str) -> bool,
        ) -> anyhow::Result<GenerateResult> {
            Ok(GenerateResult { prompt_tokens: 0, completion_tokens: 0, cache_hit: false })
        }
        fn apply_chat_template(
            &self,
            _: &[(String, String)],
            _: &[crate::engine::tools::ToolSpec],
            _: &crate::engine::tools::ToolChoice,
        ) -> anyhow::Result<String> { Ok(String::new()) }
        fn n_ctx(&self) -> u32 { 2048 }
        fn size_bytes(&self) -> u64 { 100 }
        fn kv_bytes_per_token(&self) -> u64 { 1 }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }

    /// Store holding one Ollama-style entry keyed `ollama/qwen2.5/0.5b.gguf`,
    /// which `qwen2.5:0.5b` resolves to.
    fn store_with_one_model(dir: &std::path::Path) -> ModelStore {
        let store = ModelStore::new(Some(dir.to_path_buf()));
        let model_dir = store.models_dir().join("ollama/qwen2.5");
        std::fs::create_dir_all(&model_dir).unwrap();
        let file = model_dir.join("0.5b.gguf");
        std::fs::write(&file, b"fake-gguf").unwrap();

        let mut reg = Registry::load(&store.registry_path()).unwrap();
        reg.add("ollama/qwen2.5/0.5b.gguf".into(), ModelEntry {
            repo: "ollama/qwen2.5".into(),
            filename: "0.5b.gguf".into(),
            path: file,
            size_bytes: 9,
            downloaded_at: 1,
            digest: String::new(),
            model_name: String::new(),
            description: String::new(),
            architecture: String::new(),
            context_length: 0,
            metadata_read: true,
            format: ModelFormat::Gguf,
            base_model: String::new(),
            source: ModelSource::OllamaImported,
            mmproj_path: None,
        });
        reg.save(&store.registry_path()).unwrap();
        store
    }

    fn manager() -> ModelManager {
        ModelManager::with_backends(vec![Box::new(FakeBackend)], 0)
    }

    #[test]
    fn ensure_loaded_registers_an_alias_under_the_canonical_key() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_with_one_model(dir.path());
        let mgr = manager();

        let key = ensure_loaded(&mgr, &store, "qwen2.5:0.5b").unwrap();

        assert_eq!(key, "ollama/qwen2.5/0.5b.gguf");
        assert!(mgr.is_loaded("ollama/qwen2.5/0.5b.gguf"));
        assert!(!mgr.is_loaded("qwen2.5:0.5b"), "the alias must not get its own slot");
    }

    #[test]
    fn ensure_loaded_by_alias_twice_loads_one_copy() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_with_one_model(dir.path());
        let mgr = manager();

        ensure_loaded(&mgr, &store, "ollama/qwen2.5/0.5b.gguf").unwrap();
        ensure_loaded(&mgr, &store, "qwen2.5:0.5b").unwrap();

        assert_eq!(mgr.loaded_models().len(), 1);
    }

    #[test]
    fn resident_key_maps_an_alias_onto_the_loaded_slot() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_with_one_model(dir.path());
        let mgr = manager();
        ensure_loaded(&mgr, &store, "qwen2.5:0.5b").unwrap();

        assert_eq!(resident_key(&mgr, &store, "qwen2.5:0.5b"), "ollama/qwen2.5/0.5b.gguf");
    }

    #[test]
    fn resident_key_never_loads_and_echoes_an_unknown_name() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_with_one_model(dir.path());
        let mgr = manager();

        // Known to the registry but not resident — still no load, and the name
        // comes back unchanged so the error downstream names what was asked for.
        assert_eq!(resident_key(&mgr, &store, "qwen2.5:0.5b"), "qwen2.5:0.5b");
        assert_eq!(resident_key(&mgr, &store, "nope:7b"), "nope:7b");
        assert!(mgr.loaded_models().is_empty());
    }
}
