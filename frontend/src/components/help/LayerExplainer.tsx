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
                <h3>Three layers, three different questions.</h3>
            </div>

            <details className="explanation-block" open>
                <summary>Observed Bleaching</summary>
                <p>
                    Recorded site-month outcomes from the BCO-DMO global bleaching database. This layer shows what was
                    actually reported at that site and date, after duplicate rows were standardized and aggregated.
                </p>
                <p>Observed survey dates are sparse and irregular. They should not be interpreted as a weekly environmental time series.</p>
            </details>

            <details className="explanation-block" open>
                <summary>Environmental / NOAA Weekly History</summary>
                <p>
                    {riskInfo?.definition ??
                        "A transparent heat-stress score based on hotspot-like thermal stress and accumulated heat stress."}
                </p>
                <p>
                    It is not a confirmed bleaching label. It tells us whether temperature stress conditions are
                    compatible with bleaching pressure, using weekly NOAA Monday context when that history can be reconstructed.
                </p>
            </details>

            <details className="explanation-block" open>
                <summary>Model Prediction</summary>
                <p>
                    A supervised model trained to estimate a binary bleaching event at the site-month level using site
                    factors plus weekly NOAA Monday heat-stress history. It estimates same-month event probability, not
                    percent bleaching, not a guaranteed outcome, and not a long-range forecast.
                </p>
                <p>If the model bundle is unavailable, the UI shows a model-unavailable state instead of implying the probability is below threshold.</p>
                <p>
                    Production target: <strong>{modelInfo?.target_definition ?? "binary_bleaching_event"}</strong>.
                    Prediction unit: <strong>{modelInfo?.prediction_unit ?? "site-month"}</strong>.
                </p>
                <p>{modelInfo?.input_feature_window ?? "The model uses the nearest prior Monday and lagged weekly heat-stress context."}</p>
                <p>
                    Test AUROC: <strong>{metricFromModel(modelMetrics, selectedModelName, "test", "auroc")}</strong>.
                    Test PR-AUC: <strong>{metricFromModel(modelMetrics, selectedModelName, "test", "pr_auc")}</strong>.
                </p>
                <p>Those metrics are time-held-out, but not fully site-independent.</p>
            </details>

            <div className="limitations-note">
                <strong>Why binary?</strong>
                <span>
                    Percent bleaching is valuable, but cross-source measurement rules differ enough that binary event
                    labeling is the most defensible production target in this repo.
                </span>
            </div>
        </section>
    );
}
