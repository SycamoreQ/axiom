use crate::core::backend::Backend;
use crate::inference::engine::Engine;
use crate::server::types::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionRequest,
    CompletionResponse, ErrorResponse, ModelList, UsageInfo,
};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

use axum::middleware as axum_mw;

pub struct AppState<B: Backend> {
    pub engine: Arc<Mutex<Engine<B>>>,
    pub model_id: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("bind error: {0}")]
    Bind(#[from] std::io::Error),
    #[error("engine error: {0}")]
    Engine(String),
}

pub fn create_router<B: Backend + 'static>(state: AppState<B>) -> Router {
    let shared = Arc::new(state);
    Router::new()
        .route("/health", get(crate::server::routes::health))
        .route("/v1/models", get(crate::server::routes::list_models::<B>))
        .route(
            "/v1/completions",
            post(crate::server::routes::completion::<B>),
        )
        .route(
            "/v1/chat/completions",
            post(crate::server::routes::chat_completion::<B>),
        )
        .layer(axum_mw::from_fn(crate::server::middleware::log_request))
        .layer(axum_mw::from_fn(crate::server::middleware::request_id))
        .layer(crate::server::middleware::cors_layer())
        .with_state(shared)
}

pub async fn serve<B: Backend + 'static>(
    state: AppState<B>,
    addr: &str,
) -> Result<(), ServerError> {
    let router = create_router(state);
    let listener = TcpListener::bind(addr).await?;
    println!("Axiom inference server listening on http://{}", addr);
    axum::serve(listener, router)
        .await
        .map_err(|e| ServerError::Bind(std::io::Error::new(std::io::ErrorKind::Other, e)))
}

async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "ok", "service": "axiom"})),
    )
}

async fn list_models<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
) -> impl IntoResponse {
    Json(ModelList::single(state.model_id.clone()))
}

async fn completion<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(payload): Json<CompletionRequest>,
) -> impl IntoResponse {
    let max_new = payload.max_tokens.unwrap_or(256);

    let generated = {
        let mut engine = match state.engine.lock() {
            Ok(e) => e,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        serde_json::to_value(ErrorResponse::new(
                            "engine lock poisoned".into(),
                            "server_error".into(),
                        ))
                        .unwrap(),
                    ),
                )
            }
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
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        serde_json::to_value(ErrorResponse::new(
                            e.to_string(),
                            "server_error".into(),
                        ))
                        .unwrap(),
                    ),
                )
            }
        }
    };

    let prompt_tokens = payload.prompt.split_whitespace().count();
    let completion_tokens = generated.split_whitespace().count();
    let usage = UsageInfo::new(prompt_tokens, completion_tokens);
    let resp = CompletionResponse::new(uuid(), state.model_id.clone(), generated, usage);
    (StatusCode::OK, Json(serde_json::to_value(resp).unwrap()))
}

async fn chat_completion<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // Concatenate messages into a single prompt
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
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        serde_json::to_value(ErrorResponse::new(
                            "engine lock poisoned".into(),
                            "server_error".into(),
                        ))
                        .unwrap(),
                    ),
                )
            }
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
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        serde_json::to_value(ErrorResponse::new(
                            e.to_string(),
                            "server_error".into(),
                        ))
                        .unwrap(),
                    ),
                )
            }
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
    (StatusCode::OK, Json(serde_json::to_value(resp).unwrap()))
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
pub mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::util::ServiceExt;

    pub fn make_state() -> AppState<crate::core::backend::CandleBackend> {
        use crate::core::backend::CandleBackend;
        use crate::core::device::Device;
        use crate::inference::engine::Engine;
        use crate::inference::sampler::SamplerConfig;
        use crate::model::config::ModelConfig;
        use crate::model::model::LlamaModel;
        use crate::tokenizer::tokenizer::Tokenizer;

        let config = ModelConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            vocab_size: 1000,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            rope_theta: 10000.0,
            rope_scaling: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            num_shared_experts: None,
            expert_interval: None,
            prefetch_threshold: None,
            torch_dtype: "float32".to_string(),
            architectures: None,
            model_type: Some("llama".to_string()),
        };

        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            panic!("testdata/tokenizer.json required for server tests");
        }

        let model = LlamaModel::<CandleBackend>::new(&config, &Device::Cpu).unwrap();
        let tokenizer = Tokenizer::from_file("testdata/tokenizer.json").unwrap();
        let engine = Engine::new(
            model,
            tokenizer,
            SamplerConfig {
                temperature: 0.0,
                seed: Some(42),
                ..Default::default()
            },
            4,
            Device::Cpu,
        );

        AppState {
            engine: Arc::new(Mutex::new(engine)),
            model_id: "axiom-test".to_string(),
        }
    }

    #[tokio::test]
    async fn test_health_route() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let router = create_router(make_state());
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
    }

    #[tokio::test]
    async fn test_models_route() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let router = create_router(make_state());
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "list");
        assert!(json["data"].is_array());
    }

    #[tokio::test]
    async fn test_unknown_route_404() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let router = create_router(make_state());
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/unknown")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_app_state_model_id() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let state = make_state();
        assert_eq!(state.model_id, "axiom-test");
    }
}
