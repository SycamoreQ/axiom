use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

use crate::core::backend::Backend;
use crate::inference::engine::Engine;
use crate::server::types::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionRequest,
    CompletionResponse, ErrorResponse, UsageInfo,
};
use crate::server::{app::AppState, types::ModelList};
use std::sync::{Arc, Mutex};

pub struct ApiError {
    status: StatusCode,
    message: String,
    error_type: String,
}

impl ApiError {
    pub fn new(
        status: StatusCode,
        message: impl Into<String>,
        error_type: impl Into<String>,
    ) -> Self {
        Self {
            status,
            message: message.into(),
            error_type: error_type.into(),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, msg, "server_error")
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, msg, "invalid_request_error")
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, msg, "not_found")
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = ErrorResponse::new(self.message, self.error_type);
        (self.status, Json(body)).into_response()
    }
}

// One function per route, all pub so app.rs can reference them
pub async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({"status" : "ok" , "service" : "axiom"})),
    )
}

pub async fn list_models<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
) -> impl IntoResponse {
    Json(ModelList::single(state.model_id.clone()))
}

async fn completion<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(payload): Json<CompletionRequest>,
) -> impl IntoResponse {
    if payload.model.is_empty() {
        return ApiError::bad_request("model field is required").into_response();
    }
    if payload.prompt.is_empty() {
        return ApiError::bad_request("prompt is required").into_response();
    }
    if let Some(0) = payload.max_tokens {
        return ApiError::bad_request("max_tokens must be greater than 0").into_response();
    }

    let max_new = payload.max_tokens.unwrap_or(256);

    let generated = {
        let mut engine = match state.engine.lock() {
            Ok(e) => e,
            Err(e) => return ApiError::internal(e.to_string()).into_response(),
        };

        match engine.submit_text(
            &payload.prompt,
            max_new,
            crate::tokenizer::tokenizer::EncodeOptions::default(),
        ) {
            Ok(id) => {
                let _ = engine.run_to_completion();
                engine.decode_output(id).unwrap_or_default()
            }
            Err(e) => return ApiError::internal(e.to_string()).into_response(),
        }
    };

    let prompt_tokens = payload.prompt.split_whitespace().count();
    let completion_tokens = generated.split_whitespace().count();
    let usage = UsageInfo::new(prompt_tokens, completion_tokens);
    let resp = CompletionResponse::new(uuid(), state.model_id.clone(), generated, usage);
    (StatusCode::OK, Json(resp)).into_response()
}

async fn chat_completion<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if payload.model.is_empty() {
        return ApiError::bad_request("model field is required").into_response();
    }
    if payload.messages.is_empty() {
        return ApiError::bad_request("messages array must not be empty").into_response();
    }
    if let Some(0) = payload.max_tokens {
        return ApiError::bad_request("max_tokens must be greater than 0").into_response();
    }
    let prompt: String = payload
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let max_new = payload.max_tokens.unwrap_or(256);

    let generated = {
        let mut engine = match state.engine.lock() {
            Ok(e) => e,
            Err(_) => return ApiError::internal("engine lock poisoned").into_response(),
        };

        match engine.submit_text(
            &prompt,
            max_new,
            crate::tokenizer::tokenizer::EncodeOptions::default(),
        ) {
            Ok(id) => {
                let _ = engine.run_to_completion();
                engine.decode_output(id).unwrap_or_default()
            }
            Err(e) => return ApiError::internal(e.to_string()).into_response(),
        }
    };

    let prompt_tokens = prompt.split_whitespace().count();
    let completion_tokens = generated.split_whitespace().count();
    let usage = UsageInfo::new(prompt_tokens, completion_tokens);

    let resp = ChatCompletionResponse::new(
        uuid(),
        state.model_id.clone(),
        ChatMessage {
            role: "assistant".to_string(),
            content: generated,
        },
        usage,
    );
    (StatusCode::OK, Json(resp)).into_response()
}

fn uuid() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("cmpl-{:x}", t)
}

