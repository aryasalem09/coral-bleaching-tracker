import type { ModelInfoResponse, ModelMetricsResponse, RiskInfoResponse } from "../../lib/api";

type LayerExplainerProps = {
    riskInfo: RiskInfoResponse | null;
    modelInfo: ModelInfoResponse | null;
    modelMetrics: ModelMetricsResponse | null;
};

function metricFromModel(
    modelMetrics: ModelMetricsResponse | null,
    modelName: string,
    split: "validation" | "test",
    metric: string
): string {
    const value = modelMetrics?.candidate_results?.[modelName]?.[split]?.[metric];
    return typeof value === "number" ? value.toFixed(3) : "n/a";
}

export default function LayerExplainer({ riskInfo, modelInfo, modelMetrics }: LayerExplainerProps) {
    const selectedModelName = modelInfo?.model_name ?? "hist_gradient_boosting";

    return (
        <section className="explainer-card">
            <div className="section-heading">
                <p className="eyebrow">How This Works</p>
                <h3>Each view answers a different question.</h3>
            </div>

            <details className="explanation-block" open>
                <summary>Observed Bleaching</summary>
                <p>
                    This view shows what survey teams reported for that site and date after duplicate rows were cleaned
                    up and combined.
                </p>
                <p>Survey dates are limited and irregular. They are not a weekly heat record.</p>
            </details>

            <details className="explanation-block" open>
                <summary>Heat Stress / NOAA History</summary>
                <p>
                    {riskInfo?.definition ??
                        "This view uses NOAA heat data to show how much temperature stress built up before the selected survey date."}
                </p>
                <p>
                    It is not a confirmed bleaching result. It only shows whether heat conditions were strong enough to
                    raise concern.
                </p>
            </details>

            <details className="explanation-block" open>
                <summary>Bleaching Forecast</summary>
                <p>
                    The model estimates the chance that bleaching will be observed at that site in the next 4 weeks
                    using site details plus recent NOAA heat history. It does not predict percent bleaching.
                </p>
                <p>
                    When you pick a survey date, the forecast issue date is the nearest earlier Monday used to build
                    the 12-week NOAA history window.
                </p>
                <p>This is a model estimate, not a confirmed observation. If the model is unavailable, the app says so.</p>
                <p>
                    Production target: <strong>{modelInfo?.target_definition ?? "observed_bleaching_event_in_next_4_weeks"}</strong>.
                    Forecast unit: <strong>{modelInfo?.prediction_unit ?? "site-anchor-date"}</strong>.
                </p>
                <p>{modelInfo?.input_feature_window ?? "The model uses the nearest earlier Monday plus recent weekly heat data."}</p>
                <p>{modelInfo?.probability_meaning ?? "Chance of observed bleaching in the next 4 weeks."}</p>
                <p>
                    Test AUROC: <strong>{metricFromModel(modelMetrics, selectedModelName, "test", "auroc")}</strong>.
                    Test PR-AUC: <strong>{metricFromModel(modelMetrics, selectedModelName, "test", "pr_auc")}</strong>.
                </p>
                <p>These scores come from later years held out from training. Some sites still appear across time splits, which matches the forecasting setup.</p>
            </details>

            <div className="limitations-note">
                <strong>Why yes/no instead of percent?</strong>
                <span>
                    Percent bleaching is useful, but different data sources measure it in different ways. A yes/no
                    bleaching event is more reliable for training.
                </span>
            </div>
        </section>
    );
}
