from app import create_app


def test_index_page():
    application = create_app("testing")
    response = application.test_client().get("/")

    assert response.status_code == 200
    assert b"E-Banking Phishing Detection" in response.data


def test_health_endpoint():
    application = create_app("testing")
    response = application.test_client().get("/health")

    assert response.status_code == 200
    assert response.json["status"] == "ok"
