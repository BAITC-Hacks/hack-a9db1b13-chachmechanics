"""Numerical forecasting components for TwinTurbo.ai.

The importable package remains ``windoracle`` by the shared packaging
contract; TwinTurbo.ai is the product name and source-directory name.
"""

from .baseline import (
    ConstantBaselinePredictor,
    MeanBaselinePredictor,
    TurbineMean,
    fit_constant_baseline,
    fit_mean_baseline,
    fit_turbine_means,
)
from .bias import BiasEstimator, Residual, apply_bias, update_bias
from .ensemble import (
    EnsembleExample,
    EnsemblePredictor,
    LeadWeight,
    WeightSelection,
    build_ensemble_predictor,
    select_lead_weights,
)
from .intervals import apply_intervals, fit_intervals
from .ml import MLPredictor, RidgePredictor, fit_ml_predictor, fit_ridge_predictor
from .power_curve import (
    BinnedPowerCurve,
    PowerCurve,
    PowerCurvePredictor,
    TurbinePowerCurve,
    fit_forecast_power_curve_predictor,
    fit_forecast_power_curves,
    fit_power_curve,
    fit_power_curves,
)

__all__ = [
    "BinnedPowerCurve",
    "BiasEstimator",
    "ConstantBaselinePredictor",
    "EnsembleExample",
    "EnsemblePredictor",
    "LeadWeight",
    "MLPredictor",
    "MeanBaselinePredictor",
    "PowerCurve",
    "PowerCurvePredictor",
    "Residual",
    "RidgePredictor",
    "TurbineMean",
    "TurbinePowerCurve",
    "WeightSelection",
    "apply_bias",
    "apply_intervals",
    "build_ensemble_predictor",
    "fit_constant_baseline",
    "fit_forecast_power_curve_predictor",
    "fit_forecast_power_curves",
    "fit_intervals",
    "fit_mean_baseline",
    "fit_ml_predictor",
    "fit_power_curve",
    "fit_power_curves",
    "fit_ridge_predictor",
    "fit_turbine_means",
    "select_lead_weights",
    "update_bias",
]
