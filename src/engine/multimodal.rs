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

/// Returns `true` if any message in the slice contains image parts.
pub fn contains_images(messages: &[MultimodalMessage]) -> bool {
    messages.iter().any(|m| m.has_images())
}
