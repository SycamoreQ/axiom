use serde::{Deserialize, Serialize};

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl UsageInfo {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

impl ChatCompletionResponse {
    pub fn new(id: String, model: String, message: ChatMessage, usage: UsageInfo) -> Self {
        Self {
            id,
            object: "chat.completion".to_string(),
            created: now_unix(),
            model,
            choices: vec![ChatChoice {
                index: 0,
                message,
                finish_reason: "stop".to_string(),
            }],
            usage,
        }
    }
}

impl CompletionResponse {
    pub fn new(id: String, model: String, text: String, usage: UsageInfo) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: now_unix(),
            model,
            choices: vec![CompletionChoice {
                text,
                index: 0,
                finish_reason: "stop".to_string(),
            }],
            usage,
        }
    }
}

impl ErrorResponse {
    pub fn new(message: String, error_type: String) -> Self {
        Self {
            error: ErrorDetail {
                message,
                r#type: error_type,
                code: None,
            },
        }
    }
}

impl ModelList {
    pub fn single(id: String) -> Self {
        Self {
            object: "list".to_string(),
            data: vec![ModelInfo {
                id,
                object: "model".to_string(),
                created: now_unix(),
                owned_by: "axiom".to_string(),
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_serialization() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"hello\""));
    }

    #[test]
    fn test_chat_completion_request_deserialization() {
        let json =
            r#"{"model":"llama","messages":[{"role":"user","content":"hi"}],"temperature":0.7}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.temperature, Some(0.7));
        assert!(req.max_tokens.is_none());
    }

    #[test]
    fn test_completion_request_defaults() {
        let json = r#"{"model":"x","prompt":"y"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "x");
        assert_eq!(req.prompt, "y");
        assert!(req.max_tokens.is_none());
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
    }

    #[test]
    fn test_usage_info_total() {
        let u = UsageInfo::new(10, 5);
        assert_eq!(u.total_tokens, 15);
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
    }

    #[test]
    fn test_error_response_structure() {
        let e = ErrorResponse::new(
            "bad request".to_string(),
            "invalid_request_error".to_string(),
        );
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"error\""));
        assert!(json.contains("bad request"));
        assert!(!json.contains("\"code\"")); // None skipped
    }

    #[test]
    fn test_chat_completion_response_serialization() {
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: "hi".to_string(),
        };
        let usage = UsageInfo::new(3, 1);
        let resp = ChatCompletionResponse::new("id-1".to_string(), "llama".to_string(), msg, usage);
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"choices\""));
        assert!(json.contains("\"object\":\"chat.completion\""));
    }

    #[test]
    fn test_chunk_delta_skip_none_fields() {
        let delta = ChunkDelta {
            role: None,
            content: Some("hello".to_string()),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(!json.contains("\"role\""));
        assert!(json.contains("\"content\""));
    }

    #[test]
    fn test_model_list_serialization() {
        let list = ModelList::single("llama-3".to_string());
        let json = serde_json::to_string(&list).unwrap();
        assert!(json.contains("\"object\":\"list\""));
        assert!(json.contains("llama-3"));
    }

    #[test]
    fn test_completion_response_object_field() {
        let usage = UsageInfo::new(5, 5);
        let resp =
            CompletionResponse::new("id".to_string(), "m".to_string(), "text".to_string(), usage);
        assert_eq!(resp.object, "text_completion");
    }

    #[test]
    fn test_chat_request_stop_field() {
        let json = r#"{"model":"x","messages":[],"stop":["<|end|>"]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.stop, Some(vec!["<|end|>".to_string()]));
    }
}
