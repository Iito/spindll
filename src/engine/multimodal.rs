// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

//! Internal types for multimodal (vision) message handling.
//!
//! These types represent the engine's canonical form for messages that mix
//! text and image content.  API surfaces (HTTP, gRPC, CLI) convert into
//! these before dispatching to a backend.

/// A single content fragment within a multimodal message.
#[derive(Debug, Clone)]
pub enum ContentPart {
    /// Plain text.
    Text(String),
    /// Raw image bytes with an optional MIME type (e.g. `"image/png"`).
    ImageBytes {
        data: Vec<u8>,
        media_type: Option<String>,
    },
}

/// A chat message whose content may contain interleaved text and images.
#[derive(Debug, Clone)]
pub struct MultimodalMessage {
    /// Role string — `"system"`, `"user"`, `"assistant"`, etc.
    pub role: String,
    /// One or more content parts.  A text-only message has a single
    /// `ContentPart::Text` entry.
    pub content: Vec<ContentPart>,
}

impl MultimodalMessage {
    /// Returns `true` if any part is an image.
    pub fn has_images(&self) -> bool {
        self.content.iter().any(|p| matches!(p, ContentPart::ImageBytes { .. }))
    }
}

/// Merge a system-level text block (e.g. the tool-calling preamble) into a
/// multimodal message list. Appends to the first existing `system` message,
/// otherwise inserts a new one at the front. Mirrors the text-path injection
/// used by the HTTP/gRPC chat handlers so vision requests honor tool calling too.
pub fn inject_system_text(messages: &mut Vec<MultimodalMessage>, text: &str) {
    match messages.iter_mut().find(|m| m.role == "system") {
        Some(sys) => sys.content.push(ContentPart::Text(format!("\n\n{text}"))),
        None => messages.insert(
            0,
            MultimodalMessage {
                role: "system".to_string(),
                content: vec![ContentPart::Text(text.to_string())],
            },
        ),
    }
}

/// Per-image decoded byte cap shared by the HTTP and gRPC vision entry points.
/// Bounds request-handler allocation against oversized image payloads.
pub const MAX_IMAGE_BYTES: usize = 32 * 1024 * 1024;

/// Rejects an image whose decoded byte length exceeds [`MAX_IMAGE_BYTES`].
pub fn check_image_len(len: usize) -> anyhow::Result<()> {
    if len > MAX_IMAGE_BYTES {
        anyhow::bail!("image exceeds {MAX_IMAGE_BYTES} byte limit ({len} bytes)");
    }
    Ok(())
}
