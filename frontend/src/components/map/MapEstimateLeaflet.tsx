import {
    memo,
    startTransition,
    useCallback,
    useDeferredValue,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import type { ReactNode } from "react";
import {
    CircleMarker,
    MapContainer,
    Pane,
    TileLayer,
    Tooltip,
    ZoomControl,
    useMap,
    useMapEvents,
} from "react-leaflet";
import LayerExplainer from "../help/LayerExplainer";
import { useCapabilityTier } from "../../hooks/useCapabilityTier";
import {
    ApiError,
    getModelInfo,
    getModelMetrics,
    getNoaaAvailability,
    getRiskInfo,
    getRiskScore,
    getSiteDetail,
    getSiteObservations,
    getSites,
    type NoaaAvailabilityResponse,
    predictBleaching,
    type ModelInfoResponse,
    type ModelMetricsResponse,
    type ObservationRecord,
    type PredictionResponse,
    type RiskInfoResponse,
    type RiskScoreResponse,
    type SiteMeta,
    type SitePoint,
    type SummaryResponse,
} from "../../lib/api";
import { pickNewestUsableObservationDate, sortDatesDescending } from "../../lib/dateUtils";
import type { ServerStatus } from "../../types/server";

type MapEstimateLeafletProps = {
    ensureBackendReady: () => Promise<void>;
    serverStatus: ServerStatus;
    onServerReachable: () => void;
    onServerDown: () => void;
    summary: SummaryResponse | null;
};

type LayerKey = "observed" | "risk" | "prediction";

type MapViewport = {
    south: number;
    west: number;
    north: number;
    east: number;
    zoom: number;
};

const TILE_URL = "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png";
const LABEL_URL = "https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png";

const LAYER_META: Record<
    LayerKey,
    {
        title: string;
        subtitle: string;
        accent: string;
    }
> = {
    observed: {
        title: "Observed Bleaching",
        subtitle: "Survey-backed site outcomes",
        accent: "#c34b32",
    },
    risk: {
        title: "Environmental Stress Outlook",
        subtitle: "Transparent thermal-stress score",
        accent: "#b9971c",
    },
    prediction: {
        title: "Model Prediction",
        subtitle: "Supervised site-month event probability",
        accent: "#2563eb",
    },
};

function viewportLimitForTier(tier: "low" | "medium" | "high", zoom: number): number {
    const base = tier === "low" ? 450 : tier === "medium" ? 900 : 1500;
    if (zoom <= 3) return Math.round(base * 0.7);
    if (zoom <= 5) return base;
    return Math.round(base * 1.15);
}

function markerRadiusForTier(tier: "low" | "medium" | "high", zoom: number): number {
    const base = tier === "low" ? 3.5 : tier === "medium" ? 4.5 : 5.5;
    if (zoom <= 3) return base;
    if (zoom <= 5) return base + 0.5;
    return base + 1;
}

function formatPercent(value: number | null | undefined): string {
    if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
    return `${value.toFixed(1)}%`;
}

function formatProbability(value: number | null | undefined): string {
    if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
    return `${(value * 100).toFixed(1)}%`;
}

function numericFeatureValue(features: Record<string, unknown> | undefined, key: string): number | null {
    const value = features?.[key];
    return typeof value === "number" && !Number.isNaN(value) ? value : null;
}

function riskTone(category: string | null | undefined): string {
    if (!category) return "tone-neutral";
    if (category === "severe") return "tone-severe";
    if (category === "high") return "tone-high";
    if (category === "moderate") return "tone-moderate";
    return "tone-low";
}

function riskModeLabel(mode: string | null | undefined): string {
    if (mode === "noaa_weekly_monday") return "Weekly NOAA context";
    if (mode === "historical_environmental") return "Historical aligned context";
    if (mode === "historical_observed") return "Historical observed context";
    return "Context status unknown";
}

function severityTone(category: string | null | undefined): string {
    if (!category) return "tone-neutral";
    if (category === "severe") return "tone-severe";
    if (category === "moderate") return "tone-high";
    if (category === "mild") return "tone-moderate";
    return "tone-low";
}

function isAbortError(error: unknown): boolean {
    return (
        (error instanceof DOMException && error.name === "AbortError") ||
        (error instanceof Error && error.name === "AbortError")
    );
}

function errorMessage(error: unknown): string {
    if (isAbortError(error)) return "Request cancelled.";
    if (!(error instanceof Error)) return "Unable to load site data.";
    return error.message;
}

function ViewportBridge({ onViewportChange }: { onViewportChange: (viewport: MapViewport) => void }) {
    const map = useMap();

    const reportViewport = useCallback(() => {
        const bounds = map.getBounds();
        onViewportChange({
            south: bounds.getSouth(),
            west: bounds.getWest(),
            north: bounds.getNorth(),
            east: bounds.getEast(),
            zoom: map.getZoom(),
        });
    }, [map, onViewportChange]);

    useMapEvents({
        moveend: reportViewport,
        zoomend: reportViewport,
    });

    useEffect(() => {
        reportViewport();
    }, [reportViewport]);

    return null;
}

const SiteMarkerLayer = memo(function SiteMarkerLayer({
    sites,
    selectedSiteId,
    radius,
    accent,
    onSelect,
}: {
    sites: SitePoint[];
    selectedSiteId: string | null;
    radius: number;
    accent: string;
    onSelect: (site: SitePoint) => void;
}) {
    return (
        <>
            {sites.map((site) => {
                const isSelected = site.site_id === selectedSiteId;
                return (
                    <CircleMarker
                        key={site.site_id}
                        center={[site.latitude, site.longitude]}
                        pane="site-points"
                        radius={isSelected ? radius + 1.8 : radius}
                        pathOptions={{
                            color: isSelected ? "#0f172a" : accent,
                            weight: isSelected ? 2 : 1.2,
                            fillColor: isSelected ? accent : "#f8fafc",
                            fillOpacity: isSelected ? 0.94 : 0.78,
                        }}
                        eventHandlers={{ click: () => onSelect(site) }}
                    >
                        {isSelected ? (
                            <Tooltip direction="top" offset={[0, -6]} opacity={1}>
                                {site.display_name}
                            </Tooltip>
                        ) : null}
                    </CircleMarker>
                );
            })}
        </>
    );
});

function MetricCard({ label, value, tone = "tone-neutral" }: { label: string; value: ReactNode; tone?: string }) {
    return (
        <div className={`metric-card ${tone}`}>
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
    );
}

export default function MapEstimateLeaflet({
    ensureBackendReady,
    serverStatus,
    onServerReachable,
    onServerDown,
    summary,
}: MapEstimateLeafletProps) {
    const { tier, inferredTier, overrideTier, setOverride } = useCapabilityTier();
    const [activeLayer, setActiveLayer] = useState<LayerKey>("observed");
    const [sites, setSites] = useState<SitePoint[]>([]);
    const deferredSites = useDeferredValue(sites);
    const [siteMeta, setSiteMeta] = useState<SiteMeta | null>(null);
    const [observations, setObservations] = useState<ObservationRecord[]>([]);
    const [selectedDate, setSelectedDate] = useState("");
    const [riskResult, setRiskResult] = useState<RiskScoreResponse | null>(null);
    const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
    const [riskInfo, setRiskInfo] = useState<RiskInfoResponse | null>(null);
    const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
    const [modelMetrics, setModelMetrics] = useState<ModelMetricsResponse | null>(null);
    const [noaaAvailability, setNoaaAvailability] = useState<NoaaAvailabilityResponse | null>(null);
    const [loadingSites, setLoadingSites] = useState(true);
    const [loadingSite, setLoadingSite] = useState(false);
    const [loadingAnalysis, setLoadingAnalysis] = useState(false);
    const [siteCount, setSiteCount] = useState({ total: 0, returned: 0 });
    const [error, setError] = useState("");
    const [analysisNotice, setAnalysisNotice] = useState("");
    const [mapZoom, setMapZoom] = useState(3);

    const viewportRequestRef = useRef(0);
    const siteRequestRef = useRef(0);
    const analysisRequestRef = useRef(0);
    const sitesAbortRef = useRef<AbortController | null>(null);
    const siteAbortRef = useRef<AbortController | null>(null);
    const analysisAbortRef = useRef<AbortController | null>(null);

    const sortedObservationDates = useMemo(
        () => sortDatesDescending(observations.map((record) => record.date)),
        [observations]
    );
    const selectedObservation = useMemo(
        () => observations.find((record) => record.date === selectedDate) ?? observations[0] ?? null,
        [observations, selectedDate]
    );
    const selectedDateIndex = useMemo(
        () => Math.max(0, sortedObservationDates.findIndex((date) => date === selectedDate)),
        [selectedDate, sortedObservationDates]
    );
    const layerAccent = LAYER_META[activeLayer].accent;
    const markerRadius = markerRadiusForTier(tier, mapZoom);
    const selectedModelName = modelInfo?.model_name ?? modelMetrics?.selected_model ?? "hist_gradient_boosting";

    useEffect(() => {
        if (serverStatus === "down") return;
        if (riskInfo && modelInfo && modelMetrics && noaaAvailability) return;

        const controller = new AbortController();
        void Promise.all([
            getRiskInfo({ signal: controller.signal }),
            getModelInfo({ signal: controller.signal }),
            getModelMetrics({ signal: controller.signal }),
            getNoaaAvailability({ signal: controller.signal }),
        ])
            .then(([nextRiskInfo, nextModelInfo, nextModelMetrics, nextNoaaAvailability]) => {
                setRiskInfo(nextRiskInfo);
                setModelInfo(nextModelInfo);
                setModelMetrics(nextModelMetrics);
                setNoaaAvailability(nextNoaaAvailability);
            })
            .catch(() => {
                // auxiliary metadata is helpful but non-blocking
            });

        return () => controller.abort();
    }, [modelInfo, modelMetrics, noaaAvailability, riskInfo, serverStatus]);

    const fetchViewportSites = useCallback(
        async (viewport: MapViewport) => {
            viewportRequestRef.current += 1;
            const requestId = viewportRequestRef.current;
            sitesAbortRef.current?.abort();
            const controller = new AbortController();
            sitesAbortRef.current = controller;
            setLoadingSites(true);

            try {
                const response = await getSites(
                    viewport.south,
                    viewport.west,
                    viewport.north,
                    viewport.east,
                    viewportLimitForTier(tier, viewport.zoom),
                    { signal: controller.signal }
                );
                if (viewportRequestRef.current !== requestId) return;
                startTransition(() => {
                    setMapZoom(viewport.zoom);
                    setSites(response.points);
                    setSiteCount({ total: response.total, returned: response.returned });
                    setError("");
                });
                onServerReachable();
            } catch (error: unknown) {
                if (viewportRequestRef.current !== requestId || isAbortError(error)) return;
                if (error instanceof ApiError && error.status >= 500) onServerDown();
                setError(errorMessage(error));
            } finally {
                if (viewportRequestRef.current === requestId) {
                    setLoadingSites(false);
                }
            }
        },
        [onServerDown, onServerReachable, tier]
    );

    const handleSiteSelect = useCallback(
        async (site: SitePoint) => {
            siteRequestRef.current += 1;
            const requestId = siteRequestRef.current;
            siteAbortRef.current?.abort();
            const controller = new AbortController();
            siteAbortRef.current = controller;
            setLoadingSite(true);
            setError("");

            try {
                await ensureBackendReady();
                const [detail, observationResponse] = await Promise.all([
                    getSiteDetail(site.site_id, { signal: controller.signal }),
                    getSiteObservations(site.site_id, { signal: controller.signal }),
                ]);
                if (siteRequestRef.current !== requestId) return;

                // Start on the newest reef date that has either analysis-ready QA
                // status or at least an observed bleaching value. The active layer
                // can still realign to an older backend-validated date if needed.
                const nextDate =
                    pickNewestUsableObservationDate(observationResponse.records) ??
                    detail.recommended_observed_date ??
                    observationResponse.recommended_date ??
                    "";

                startTransition(() => {
                    setSiteMeta(detail.site);
                    setObservations(observationResponse.records);
                    setSelectedDate(nextDate);
                    setRiskResult(null);
                    setPrediction(null);
                    setAnalysisNotice("");
                    setError("");
                });
                onServerReachable();
            } catch (error: unknown) {
                if (siteRequestRef.current !== requestId || isAbortError(error)) return;
                if (error instanceof ApiError && error.status >= 500) onServerDown();
                setError(errorMessage(error));
            } finally {
                if (siteRequestRef.current === requestId) {
                    setLoadingSite(false);
                }
            }
        },
        [ensureBackendReady, onServerDown, onServerReachable]
    );

    useEffect(() => {
        if (!siteMeta || !selectedDate) return;
        if (activeLayer === "observed") {
            setLoadingAnalysis(false);
            setAnalysisNotice("");
            return;
        }

        analysisRequestRef.current += 1;
        const requestId = analysisRequestRef.current;
        analysisAbortRef.current?.abort();
        const controller = new AbortController();
        analysisAbortRef.current = controller;
        setLoadingAnalysis(true);
        setAnalysisNotice("");
        setError("");

        if (activeLayer === "risk") {
            setRiskResult(null);
            void getRiskScore(
                {
                    site_id: siteMeta.site_id,
                    date: selectedDate,
                    prefer_live: true,
                },
                { signal: controller.signal }
            )
                .then((nextRisk) => {
                    if (analysisRequestRef.current !== requestId) return;
                    setRiskResult(nextRisk);
                    setError("");
                    if (nextRisk.used_date !== selectedDate && sortedObservationDates.includes(nextRisk.used_date)) {
                        setAnalysisNotice(
                            `Moved from ${selectedDate} to ${nextRisk.used_date} because the risk layer needed the newest valid earlier context.`
                        );
                        setSelectedDate(nextRisk.used_date);
                        onServerReachable();
                        return;
                    }
                    setAnalysisNotice("");
                    onServerReachable();
                })
                .catch((error: unknown) => {
                    if (analysisRequestRef.current !== requestId || isAbortError(error)) return;
                    if (error instanceof ApiError && error.status === 404) {
                        setRiskResult(null);
                        setAnalysisNotice("No valid environmental risk context exists at or before the selected reef date.");
                        return;
                    }
                    if (error instanceof ApiError && error.status >= 500) onServerDown();
                    setError(errorMessage(error));
                })
                .finally(() => {
                    if (analysisRequestRef.current === requestId) {
                        setLoadingAnalysis(false);
                    }
                });

            return () => controller.abort();
        }

        setPrediction(null);
        void predictBleaching(
            {
                site_id: siteMeta.site_id,
                date: selectedDate,
                prefer_live: true,
            },
            { signal: controller.signal }
        )
            .then((predictionPayload) => {
                if (analysisRequestRef.current !== requestId) return;
                if (predictionPayload.available === false) {
                    setPrediction(null);
                    setAnalysisNotice(predictionPayload.message ?? "Model output is unavailable because the model artifact is missing.");
                    return;
                }

                setPrediction(predictionPayload);
                setError("");
                if (predictionPayload.used_date && predictionPayload.used_date !== selectedDate && sortedObservationDates.includes(predictionPayload.used_date)) {
                    setAnalysisNotice(
                        `Moved from ${selectedDate} to ${predictionPayload.used_date} because the prediction layer needed the newest valid earlier context.`
                    );
                    setSelectedDate(predictionPayload.used_date);
                    onServerReachable();
                    return;
                }
                setAnalysisNotice("");
                onServerReachable();
            })
            .catch((error: unknown) => {
                if (analysisRequestRef.current !== requestId || isAbortError(error)) return;
                if (error instanceof ApiError && error.status === 404) {
                    setPrediction(null);
                    setAnalysisNotice("No model-eligible historical context passed QA at or before the selected reef date.");
                    return;
                }
                if (error instanceof ApiError && error.status >= 500) onServerDown();
                setError(errorMessage(error));
            })
            .finally(() => {
                if (analysisRequestRef.current === requestId) {
                    setLoadingAnalysis(false);
                }
            });

        return () => controller.abort();
    }, [activeLayer, onServerDown, onServerReachable, selectedDate, siteMeta, sortedObservationDates]);

    const moveTimeline = useCallback(
        (delta: number) => {
            if (sortedObservationDates.length === 0) return;
            const nextIndex = Math.min(
                Math.max(selectedDateIndex + delta, 0),
                Math.max(0, sortedObservationDates.length - 1)
            );
            setSelectedDate(sortedObservationDates[nextIndex]);
        },
        [selectedDateIndex, sortedObservationDates]
    );

    return (
        <main className={`experience-grid capability-${tier}`}>
            <section className="map-panel glass-panel">
                <div className="panel-header">
                    <div>
                        <p className="eyebrow">Map-first explorer</p>
                        <h2>Observed reef sites with on-demand risk and model layers.</h2>
                        <p className="muted-copy">
                            The initial load stays light by fetching only visible sites. Click a point to load its
                            observed timeline and then inspect risk or model output for the selected date.
                        </p>
                    </div>

                    <div className="status-stack">
                        <div className="status-chip">
                            <span>Capability</span>
                            <strong>{tier}</strong>
                        </div>
                        <div className="status-chip">
                            <span>Viewport sites</span>
                            <strong>{siteCount.returned}</strong>
                        </div>
                        <div className="status-chip">
                            <span>Backend</span>
                            <strong>{serverStatus === "ready" ? "ready" : serverStatus}</strong>
                        </div>
                    </div>
                </div>

                <div className="toolbar-row">
                    <div className="layer-toggle" role="tablist" aria-label="Data layers">
                        {(Object.keys(LAYER_META) as LayerKey[]).map((layer) => (
                            <button
                                key={layer}
                                type="button"
                                className={layer === activeLayer ? "layer-button layer-button--active" : "layer-button"}
                                onClick={() => setActiveLayer(layer)}
                                style={{ ["--layer-accent" as string]: LAYER_META[layer].accent }}
                            >
                                <span>{LAYER_META[layer].title}</span>
                                <small>{LAYER_META[layer].subtitle}</small>
                            </button>
                        ))}
                    </div>

                    <label className="capability-select">
                        <span>Device mode</span>
                        <select
                            value={overrideTier ?? "auto"}
                            onChange={(event) => {
                                const value = event.target.value;
                                if (value === "auto") {
                                    setOverride(null);
                                    return;
                                }
                                setOverride(value as "low" | "medium" | "high");
                            }}
                        >
                            <option value="auto">{`Auto (${inferredTier})`}</option>
                            <option value="low">Low</option>
                            <option value="medium">Medium</option>
                            <option value="high">High</option>
                        </select>
                    </label>
                </div>

                <div className="map-frame">
                    <MapContainer center={[8, 155]} zoom={3} zoomControl={false} className="leaflet-map" preferCanvas worldCopyJump>
                        <ZoomControl position="topright" />
                        <Pane name="site-points" style={{ zIndex: 430 }} />
                        <Pane name="labels" style={{ zIndex: 500, pointerEvents: "none" }} />

                        <TileLayer attribution="&copy; OpenStreetMap contributors &copy; CARTO" url={TILE_URL} />
                        {tier !== "low" ? (
                            <TileLayer attribution="&copy; OpenStreetMap contributors &copy; CARTO" url={LABEL_URL} pane="labels" />
                        ) : null}
                        <ViewportBridge onViewportChange={fetchViewportSites} />
                        <SiteMarkerLayer
                            sites={deferredSites}
                            selectedSiteId={siteMeta?.site_id ?? null}
                            radius={markerRadius}
                            accent={layerAccent}
                            onSelect={handleSiteSelect}
                        />
                    </MapContainer>

                    {loadingSites ? <div className="map-badge">Loading visible sites…</div> : null}

                    <div className="map-legend">
                        <div className="legend-row">
                            <span className="legend-dot" style={{ background: layerAccent }} />
                            Click to load the site timeline and analysis panel.
                        </div>
                        <div className="legend-row legend-row--muted">
                            Initial fetch returns viewport-limited site summaries only.
                        </div>
                    </div>
                </div>
            </section>

            <aside className="insight-panel glass-panel">
                <div className="section-heading">
                    <p className="eyebrow">Selected Site</p>
                    <h3>{siteMeta?.display_name ?? "Choose a reef site"}</h3>
                    <p className="muted-copy">
                        {siteMeta
                            ? `${siteMeta.ecoregion_name ?? "Unknown ecoregion"} · ${siteMeta.country_name ?? "Unknown country"}`
                            : "Pick a site on the map to load observed records, risk context, and the supervised model output."}
                    </p>
                </div>

                {error ? <div className="alert-banner">{error}</div> : null}
                {analysisNotice && !error ? <div className="alert-banner alert-banner--soft">{analysisNotice}</div> : null}

                {siteMeta ? (
                    <>
                        {observations.length === 0 ? (
                            <div className="empty-state">
                                <strong>No valid observed timeline is available for this site.</strong>
                                <span>The site metadata loaded, but there were no usable observed records to place on the timeline.</span>
                            </div>
                        ) : null}

                        <div className="metric-grid">
                            <MetricCard label="Observed records" value={siteMeta.observed_record_count} />
                            <MetricCard label="Positive observations" value={siteMeta.observed_positive_count} />
                            <MetricCard
                                label="Data quality"
                                value={siteMeta.mean_label_quality_score.toFixed(2)}
                                tone="tone-low"
                            />
                            <MetricCard label="Selected date" value={selectedDate || "n/a"} />
                        </div>

                        <section className="timeline-card">
                            <div className="section-heading section-heading--compact">
                                <h4>Observed timeline</h4>
                                <p>Newest valid observed date is auto-selected when a site is clicked.</p>
                            </div>

                            <div className="timeline-controls">
                                <button type="button" onClick={() => moveTimeline(1)} disabled={selectedDateIndex >= sortedObservationDates.length - 1}>
                                    Older
                                </button>
                                <input
                                    type="range"
                                    min={0}
                                    max={Math.max(sortedObservationDates.length - 1, 0)}
                                    value={selectedDateIndex}
                                    onChange={(event) => {
                                        const nextIndex = Number(event.target.value);
                                        const nextDate = sortedObservationDates[nextIndex];
                                        if (nextDate) setSelectedDate(nextDate);
                                    }}
                                    disabled={sortedObservationDates.length === 0}
                                />
                                <button type="button" onClick={() => moveTimeline(-1)} disabled={selectedDateIndex <= 0}>
                                    Newer
                                </button>
                            </div>

                            <div className="observation-list">
                                {observations.slice(0, 10).map((record) => (
                                    <button
                                        key={record.date}
                                        type="button"
                                        className={record.date === selectedDate ? "observation-row observation-row--active" : "observation-row"}
                                        onClick={() => setSelectedDate(record.date)}
                                    >
                                        <div>
                                            <strong>{record.date}</strong>
                                            <span>{record.provenance_sources.join(", ") || "Unknown source"}</span>
                                        </div>
                                        <div className={`pill ${severityTone(record.observed_severity_category)}`}>
                                            {record.observed_severity_category ?? "unrated"}
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </section>

                        {activeLayer === "observed" ? (
                            <section className="layer-card">
                                <div className="section-heading section-heading--compact">
                                    <h4>Observed Bleaching</h4>
                                    <p>Recorded outcome for the selected site-date.</p>
                                </div>

                                <div className="hero-stat">
                                    <strong>{formatPercent(selectedObservation?.observed_percent_bleaching)}</strong>
                                    <span>Observed percent bleaching</span>
                                </div>

                                <div className="metric-grid">
                                    <MetricCard
                                        label="Severity"
                                        value={selectedObservation?.observed_severity_category ?? "n/a"}
                                        tone={severityTone(selectedObservation?.observed_severity_category)}
                                    />
                                    <MetricCard
                                        label="Conflict history"
                                        value={selectedObservation?.has_conflict_history ? "Averaged duplicates" : "Clean"}
                                    />
                                    <MetricCard
                                        label="Label origin"
                                        value={
                                            selectedObservation?.is_direct_observation
                                                ? "Direct observation"
                                                : selectedObservation?.is_derived_label
                                                  ? "Comment-derived"
                                                  : "Unknown"
                                        }
                                    />
                                    <MetricCard label="Samples combined" value={selectedObservation?.sample_row_count ?? 0} />
                                    <MetricCard
                                        label="Sources"
                                        value={selectedObservation?.provenance_sources.join(", ") || "n/a"}
                                    />
                                </div>
                            </section>
                        ) : null}

                        {activeLayer === "risk" ? (
                            <section className="layer-card">
                                <div className="section-heading section-heading--compact">
                                    <h4>Environmental Stress Outlook</h4>
                                    <p>Nearest weekly NOAA Monday stress snapshot, separate from observed bleaching and the supervised model.</p>
                                </div>

                                <div className={`hero-stat hero-stat--compact ${riskTone(riskResult?.category)}`}>
                                    <strong>{riskResult?.category ?? "loading"}</strong>
                                    <span>{riskModeLabel(riskResult?.mode)}</span>
                                </div>

                                <div className="metric-grid">
                                    <MetricCard label="Hotspot-like stress" value={riskResult ? riskResult.hotspot.toFixed(2) : "n/a"} />
                                    <MetricCard label="Accumulated heat" value={riskResult ? riskResult.dhw.toFixed(2) : "n/a"} />
                                    <MetricCard label="Used date" value={riskResult?.used_date ?? "n/a"} />
                                    <MetricCard label="Warnings" value={riskResult?.warnings.length ? riskResult.warnings.join(" · ") : "None"} />
                                </div>

                                <p className="explanation-copy">{riskResult?.explanation ?? "Loading environmental stress context…"}</p>
                            </section>
                        ) : null}

                        {activeLayer === "prediction" ? (
                            <section className="layer-card">
                                <div className="section-heading section-heading--compact">
                                    <h4>Model Prediction</h4>
                                    <p>Supervised estimate of a binary bleaching event using weekly Monday NOAA history at the site-month level.</p>
                                </div>

                                <div
                                    className="probability-ring"
                                    style={{ ["--probability" as string]: `${Math.round((prediction?.probability ?? 0) * 100)}%` }}
                                >
                                    <strong>{formatProbability(prediction?.probability)}</strong>
                                    <span>{prediction?.predicted_event ? "Above model threshold" : "Below model threshold"}</span>
                                </div>

                                <div className="metric-grid">
                                    <MetricCard label="Model version" value={prediction?.model_version ?? "n/a"} />
                                    <MetricCard label="Used date" value={prediction?.used_date ?? "n/a"} />
                                    <MetricCard label="Prediction unit" value={prediction?.prediction_unit ?? "n/a"} />
                                    <MetricCard
                                        label="Weekly history"
                                        value={
                                            numericFeatureValue(prediction?.features_used, "weekly_history_weeks_available")?.toFixed(0) ??
                                            "n/a"
                                        }
                                    />
                                    <MetricCard
                                        label="Decision threshold"
                                        value={prediction?.threshold?.toFixed(2) ?? modelInfo?.decision_threshold?.toFixed(2) ?? "n/a"}
                                    />
                                    <MetricCard
                                        label="Test PR-AUC"
                                        value={modelMetrics?.candidate_results?.[selectedModelName]?.test?.pr_auc?.toFixed(3) ?? "n/a"}
                                    />
                                </div>

                                <p className="explanation-copy">
                                    {prediction?.data_quality_warning ? `${prediction.data_quality_warning} ` : ""}
                                    {typeof numericFeatureValue(prediction?.features_used, "weekly_missing_fraction_12w") === "number"
                                        ? `Weekly-history missing fraction: ${formatPercent(
                                              numericFeatureValue(prediction?.features_used, "weekly_missing_fraction_12w")! * 100
                                          )}. `
                                        : ""}
                                    {prediction?.coverage_warning ??
                                        "This model was selected over a climatology baseline, but it still weakens on held-out future years."}
                                </p>
                            </section>
                        ) : null}
                    </>
                ) : (
                    <div className="empty-state">
                        <strong>No site selected yet.</strong>
                        <span>
                            Pan or zoom the map if needed, then click a visible point. The timeline will snap to that
                            site’s newest valid observed date.
                        </span>
                    </div>
                )}

                {loadingSite || loadingAnalysis ? <div className="loading-banner">Loading site detail…</div> : null}

                <LayerExplainer riskInfo={riskInfo} modelInfo={modelInfo} modelMetrics={modelMetrics} />

                <div className="footnote-grid">
                    <div className="footnote-card">
                        <strong>Local NOAA coverage</strong>
                        <span>
                            {(summary?.latest_live_noaa_date ?? noaaAvailability?.paired_last_date)
                                ? `${summary?.live_noaa_first_date ?? noaaAvailability?.paired_first_date ?? "unknown"} to ${
                                      summary?.latest_live_noaa_date ?? noaaAvailability?.paired_last_date
                                  } (${summary?.live_noaa_schedule ?? "weekly_mondays"})`
                                : "No local weekly NOAA Monday files cached in this session."}
                        </span>
                    </div>
                    <div className="footnote-card">
                        <strong>Adaptive rendering</strong>
                        <span>
                            {tier === "low"
                                ? "Lower point density and reduced motion."
                                : tier === "medium"
                                  ? "Balanced density and interaction polish."
                                  : "Richer density and visual treatment."}
                        </span>
                    </div>
                </div>
            </aside>
        </main>
    );
}
