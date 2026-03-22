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
            </details>

            <details className="explanation-block" open>
                <summary>Environmental Stress Outlook</summary>
                <p>
                    {riskInfo?.definition ??
                        "A transparent heat-stress score based on hotspot-like thermal stress and accumulated heat stress."}
                </p>
                <p>
                    It is not a confirmed bleaching label. It tells us whether temperature stress conditions are
                    compatible with bleaching pressure.
                </p>
            </details>

            <details className="explanation-block" open>
                <summary>Model Prediction</summary>
                <p>
                    A supervised model trained to estimate a binary bleaching event at the site-month level using site
                    factors plus thermal stress features. It estimates same-month event probability, not percent
                    bleaching, not a guaranteed outcome, and not a long-range forecast.
                </p>
                <p>
                    Production target: <strong>{modelInfo?.target_definition ?? "binary_bleaching_event"}</strong>.
                    Prediction unit: <strong>{modelInfo?.prediction_unit ?? "site-month"}</strong>.
                </p>
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
