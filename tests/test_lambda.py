import json

from mangum import Mangum

from fraudservice.api import create_app
from tests.conftest import make_data


def api_gateway_event(method, path, body=None):
    """Minimal API Gateway HTTP API (payload v2.0) event, as AWS sends it to Lambda."""
    return {
        "version": "2.0", "routeKey": f"{method} {path}", "rawPath": path, "rawQueryString": "",
        "headers": {"content-type": "application/json", "host": "example.execute-api.eu-west-2.amazonaws.com"},
        "requestContext": {"http": {"method": method, "path": path, "sourceIp": "1.2.3.4", "protocol": "HTTP/1.1"},
                           "stage": "$default", "requestId": "test", "accountId": "123", "apiId": "abc"},
        "body": json.dumps(body) if body is not None else None, "isBase64Encoded": False,
    }


def test_lambda_health(model_dir):
    handler = Mangum(create_app(model_dir), lifespan="off")
    r = handler(api_gateway_event("GET", "/health"), {})
    assert r["statusCode"] == 200
    assert json.loads(r["body"])["status"] == "ok"


def test_lambda_score(model_dir):
    handler = Mangum(create_app(model_dir), lifespan="off")
    txs = make_data(50, seed=9).drop(columns="Class").head(2).to_dict(orient="records")
    r = handler(api_gateway_event("POST", "/score", {"transactions": txs}), {})
    assert r["statusCode"] == 200
    assert len(json.loads(r["body"])["scores"]) == 2
