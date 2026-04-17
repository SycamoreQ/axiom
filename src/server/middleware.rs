use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::{alloc::System, time::SystemTime, time::UNIX_EPOCH};

use tower_http::cors::{Any, CorsLayer};

pub fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
}

pub async fn log_request(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> impl IntoResponse {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let start = std::time::Instant::now();
    let resp = next.run(req).await;
    let status = resp.status();
    let ms = start.elapsed().as_millis();
    println!("[{}] {} {} {}ms", status.as_u16(), method, path, ms);
    resp
}

pub async fn request_id(
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> impl IntoResponse {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_nanos();
    let id = format!("req-{:x}", now);
    req.headers_mut()
        .insert("x-request-id", id.parse().unwrap());
    let mut resp = next.run(req).await;
    resp.headers_mut()
        .insert("x-request-id", id.parse().unwrap());
    resp
}

