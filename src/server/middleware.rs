use axum::{extract::Request, middleware::Next, response::IntoResponse};
use tower_http::cors::{Any, CorsLayer};

pub fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
}

pub async fn log_request(req: Request, next: Next) -> impl IntoResponse {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let start = std::time::Instant::now();
    let resp = next.run(req).await;
    let status = resp.status().as_u16();
    let ms = start.elapsed().as_millis();
    println!("[axiom] {} {} {} {}ms", status, method, path, ms);
    resp
}

pub async fn request_id(mut req: Request, next: Next) -> impl IntoResponse {
    let id = format!("req-{:x}", now_nanos());
    req.headers_mut().insert(
        "x-request-id",
        id.parse()
            .unwrap_or_else(|_| "req-unknown".parse().unwrap()),
    );
    let mut resp = next.run(req).await;
    resp.headers_mut().insert(
        "x-request-id",
        id.parse()
            .unwrap_or_else(|_| "req-unknown".parse().unwrap()),
    );
    resp
}

fn now_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request, http::StatusCode, middleware, routing::get, Router};
    use tower::ServiceExt;

    async fn ok_handler() -> impl IntoResponse {
        (StatusCode::OK, "ok")
    }

    fn router_with_logging() -> Router {
        Router::new()
            .route("/test", get(ok_handler))
            .layer(middleware::from_fn(log_request))
    }

    fn router_with_request_id() -> Router {
        Router::new()
            .route("/test", get(ok_handler))
            .layer(middleware::from_fn(request_id))
    }

    fn router_with_cors() -> Router {
        Router::new()
            .route("/test", get(ok_handler))
            .layer(cors_layer())
    }

    #[test]
    fn test_cors_layer_constructs() {
        let _ = cors_layer();
    }

    #[tokio::test]
    async fn test_log_request_passes_status_through() {
        let router = router_with_logging();
        let resp = router
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_log_request_unknown_path_404() {
        let router = router_with_logging();
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/missing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_request_id_header_present_in_response() {
        let router = router_with_request_id();
        let resp = router
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert!(resp.headers().contains_key("x-request-id"));
    }

    #[tokio::test]
    async fn test_request_id_is_not_empty() {
        let router = router_with_request_id();
        let resp = router
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let id = resp
            .headers()
            .get("x-request-id")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(!id.is_empty());
        assert!(id.starts_with("req-"));
    }

    #[tokio::test]
    async fn test_request_id_does_not_change_status() {
        let router = router_with_request_id();
        let resp = router
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_cors_does_not_break_response() {
        let router = router_with_cors();
        let resp = router
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_two_request_ids_are_unique() {
        let router = router_with_request_id();
        let id1 = {
            let resp = router
                .clone()
                .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
                .await
                .unwrap();
            resp.headers()
                .get("x-request-id")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string()
        };
        let id2 = {
            let resp = router
                .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
                .await
                .unwrap();
            resp.headers()
                .get("x-request-id")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string()
        };
        assert_ne!(id1, id2);
    }
}
