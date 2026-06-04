"""
AppTest smoke tests for wheel_dashboard.py.

These are in a separate file to avoid @st.cache_resource pollution from
other AppTest runs in the same process.

Run with:  source venv/bin/activate && python3 -m pytest tests/test_app.py -v
"""

from streamlit.testing.v1 import AppTest


class TestAppSmoke:
    """Smoke tests that verify the app handles missing/bad config gracefully.

    NOTE: @st.cache_resource on get_db() caches across runs within the same
    process.  The empty-secret test MUST run first so get_db() caches None.
    If the secret-provided test runs first, the cached non-None db object
    leaks into the empty-secret test.
    """

    def test_missing_credentials_shows_error(self):
        """Without ANY Supabase key the app shows a visible config error.

        This must run BEFORE test_app_starts_with_dummy_secrets because
        @st.cache_resource caches get_db() across runs in the same process.
        """
        at = AppTest.from_file("wheel_dashboard.py")
        at.secrets["SUPABASE_KEY"] = ""
        at.secrets["supabase_url"] = ""
        at.run()
        assert not at.exception
        error_values = [e.value for e in at.main.get("error")]
        assert any("Supabase" in v or "configured" in v for v in error_values), (
            f"Expected a Supabase config error, got: {error_values}"
        )

    def test_app_starts_with_dummy_secrets(self):
        """Even with dummy creds the app should render without crashing."""
        at = AppTest.from_file("wheel_dashboard.py")
        at.secrets["supabase_key"] = "dummy-key"
        at.secrets["admin_password"] = "dummy-pass"
        at.run()
        assert not at.exception
        assert any(
            getattr(c, "value", "") == "☸️ Wheel Strategy Manager Pro"
            for c in at.main
        ), "Page title not found"