use crate::core::backend::Backend;
use crate::server::app::AppState;
use crate::server::types::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionRequest,
    CompletionResponse, ErrorResponse, ModelList, UsageInfo,
};
use crate::tokenizer::tokenizer::EncodeOptions;
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::sync::Arc;

pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub error_type: String,
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

pub async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "ok", "service": "axiom"})),
    )
}

pub async fn list_models<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
) -> impl IntoResponse {
    Json(ModelList::single(state.model_id.clone()))
}

pub async fn completion<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(payload): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    // Validate
    if payload.model.is_empty() {
        return Err(ApiError::bad_request("model must not be empty"));
    }
    if payload.prompt.is_empty() {
        return Err(ApiError::bad_request("prompt must not be empty"));
    }
    if let Some(0) = payload.max_tokens {
        return Err(ApiError::bad_request("max_tokens must be greater than 0"));
    }

    let max_new = payload.max_tokens.unwrap_or(256);

    let mut engine = state
        .engine
        .lock()
        .map_err(|_| ApiError::internal("engine lock poisoned"))?;

    let id = engine
        .submit_text(&payload.prompt, max_new, EncodeOptions::default())
        .map_err(|e| ApiError::internal(e.to_string()))?;

    engine
        .run_to_completion()
        .map_err(|e| ApiError::internal(e.to_string()))?;

    let generated = engine.decode_output(id).unwrap_or_default();
    let prompt_tokens = payload.prompt.split_whitespace().count();
    let completion_tokens = generated.split_whitespace().count();

    Ok(Json(CompletionResponse::new(
        uuid(),
        state.model_id.clone(),
        generated,
        UsageInfo::new(prompt_tokens, completion_tokens),
    )))
}

pub async fn chat_completion<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    // Validate
    if payload.model.is_empty() {
        return Err(ApiError::bad_request("model must not be empty"));
    }
    if payload.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    if let Some(0) = payload.max_tokens {
        return Err(ApiError::bad_request("max_tokens must be greater than 0"));
    }

    let prompt: String = payload
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let max_new = payload.max_tokens.unwrap_or(256);

    let mut engine = state
        .engine
        .lock()
        .map_err(|_| ApiError::internal("engine lock poisoned"))?;

    let id = engine
        .submit_text(&prompt, max_new, EncodeOptions::default())
        .map_err(|e| ApiError::internal(e.to_string()))?;

    engine
        .run_to_completion()
        .map_err(|e| ApiError::internal(e.to_string()))?;

    let generated = engine.decode_output(id).unwrap_or_default();
    let prompt_tokens = prompt.split_whitespace().count();
    let completion_tokens = generated.split_whitespace().count();

    Ok(Json(ChatCompletionResponse::new(
        uuid(),
        state.model_id.clone(),
        ChatMessage {
            role: "assistant".to_string(),
            content: generated,
        },
        UsageInfo::new(prompt_tokens, completion_tokens),
    )))
}

fn uuid() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("cmpl-{:x}", t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use tower::ServiceExt;

    fn make_router() -> axum::Router {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            panic!("tokenizer required");
        }
        crate::server::app::create_router(crate::server::app::tests::make_state())
    }

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn test_api_error_internal_status() {
        let e = ApiError::internal("something broke");
        assert_eq!(e.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(e.error_type, "server_error");
    }

    #[test]
    fn test_api_error_bad_request_status() {
        let e = ApiError::bad_request("bad input");
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        assert_eq!(e.error_type, "invalid_request_error");
    }

    #[test]
    fn test_api_error_not_found_status() {
        let e = ApiError::not_found("missing");
        assert_eq!(e.status, StatusCode::NOT_FOUND);
        assert_eq!(e.error_type, "not_found");
    }

    #[test]
    fn test_api_error_new_custom() {
        let e = ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "oops", "custom_error");
        assert_eq!(e.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(e.message, "oops");
        assert_eq!(e.error_type, "custom_error");
    }

    #[tokio::test]
    async fn test_api_error_into_response_body_is_json() {
        let e = ApiError::bad_request("test error");
        let resp = e.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json.get("error").is_some());
        assert_eq!(json["error"]["message"], "test error");
    }

    // ── route tests ──

    #[tokio::test]
    async fn test_health_returns_ok() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let router = make_router();
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn test_chat_completion_empty_messages_400() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let router = make_router();
        let body = serde_json::json!({
            "model": "axiom",
            "messages": []
        });
        let resp = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("messages"));
    }

    #[tokio::test]
    async fn test_completion_empty_prompt_400() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let router = make_router();
        let body = serde_json::json!({
            "model": "axiom",
            "prompt": ""
        });
        let resp = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_completion_zero_max_tokens_400() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let router = make_router();
        let body = serde_json::json!({
            "model": "axiom",
            "prompt": "hello",
            "max_tokens": 0
        });
        let resp = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_empty_model_field_400() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let router = make_router();
        let body = serde_json::json!({
            "model": "",
            "prompt": "hello"
        });
        let resp = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
