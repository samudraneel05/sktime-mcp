"""Ambiguous handle+dataset inputs must be rejected, not silently resolved (#556 item 6).

fit/predict/update should fail when both a handle and a dataset are given
for the same slot (y or X), instead of silently letting the dataset win.
"""

import pytest
from sktime.forecasting.naive import NaiveForecaster

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.fit_predict import fit_tool, predict_tool, update_tool


@pytest.fixture
def fitted_handle():
    executor = get_executor()
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
    fit_tool(estimator_handle=handle, y_dataset="airline", fh=[1])
    yield handle
    executor._handle_manager.release_handle(handle)


@pytest.fixture
def data_handle():
    import pandas as pd

    executor = get_executor()
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    executor._data_handles["test_ambig"] = {"y": pd.Series(range(24), index=idx, dtype=float)}
    yield "test_ambig"
    executor._data_handles.pop("test_ambig", None)


class TestAmbiguousYSlot:
    def test_fit_rejects_y_handle_and_y_dataset(self, data_handle):
        executor = get_executor()
        handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
        try:
            result = fit_tool(
                estimator_handle=handle,
                y_handle=data_handle,
                y_dataset="airline",
            )
            assert not result["success"], "fit should reject both y_handle and y_dataset"
            assert "ambiguous" in result["error"].lower()
        finally:
            executor._handle_manager.release_handle(handle)

    def test_predict_rejects_y_handle_and_y_dataset(self, fitted_handle, data_handle):
        result = predict_tool(
            estimator_handle=fitted_handle,
            y_handle=data_handle,
            y_dataset="airline",
        )
        assert not result["success"], "predict should reject both y_handle and y_dataset"
        assert "ambiguous" in result["error"].lower()

    def test_update_rejects_y_handle_and_y_dataset(self, fitted_handle, data_handle):
        result = update_tool(
            estimator_handle=fitted_handle,
            y_handle=data_handle,
            y_dataset="airline",
        )
        assert not result["success"], "update should reject both y_handle and y_dataset"
        assert "ambiguous" in result["error"].lower()


class TestAmbiguousXSlot:
    def test_fit_rejects_exog_handle_and_dataset(self, data_handle):
        executor = get_executor()
        handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})
        try:
            result = fit_tool(
                estimator_handle=handle,
                X_handle=data_handle,
                X_dataset="airline",
            )
            assert not result["success"], "fit should reject both X_handle and X_dataset"
            assert "ambiguous" in result["error"].lower()
        finally:
            executor._handle_manager.release_handle(handle)
