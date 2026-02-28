import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import {
    ApiError,
    apiAvailableDatesFor,
    apiEstimate,
    apiEstimateFromFeatures,
    apiReefPoints,
    apiSensitivity,
} from "../../lib/api";
import { findNearestDateIndex } from "../../lib/dateUtils";
import type {
    AvailableDatesForResponse,
    EstimateResponse,
    SensitivityResponse,
} from "../../lib/api";
import type { ServerStatus } from "../../types/server";

type MapEstimateLeafletProps = {
    ensureBackendReady: () => Promise<void>;
    serverStatus: ServerStatus;
    onServerReachable: () => void;
    onServerDown: () => void;
    onOpenTutorial: () => void;
    warmupElapsedSeconds: number;
};

type ReefPoint = {
    lat: number;
    lon: number;
    reefKey: string;
};

type HistoryPoint = {
    date: string;
    risk_prob: number;
    dhw: number;
    hotspot: number;
};

type RiskBand = {
    label: "Low" | "Elevated" | "High" | "Severe";
    toneClassName: string;
    color: string;
    softColor: string;
};

type DashboardTab = "overview" | "timeline" | "scenario";
type TimelineSpeed = 1 | 2 | 4;

type MapViewport = {
    south: number;
    west: number;
    north: number;
    east: number;
    zoom: number;
};

const REEF_KEY_DECIMALS = 4;
const HISTORY_SAMPLE_POINTS = 12;
const MAX_ESTIMATE_RETRIES = 2;
const RETRY_BASE_DELAY_MS = 800;
const TILE_URL = "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png";
const LABEL_URL = "https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png";

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => {
        window.setTimeout(resolve, ms);
    });
}

function isAbortError(error: unknown): boolean {
    return (
        (error instanceof DOMException && error.name === "AbortError") ||
        (error instanceof Error && error.name === "AbortError")
    );
}

function isRetriableError(error: unknown): boolean {
    if (isAbortError(error)) return false;
    if (error instanceof TypeError) return true;
    if (error instanceof ApiError && error.status >= 500) return true;
    if (!(error instanceof Error)) return false;

    const message = error.message.toLowerCase();
    return (
        message.includes("failed to fetch") ||
        message.includes("network") ||
        message.includes("timeout") ||
        message.includes("gateway") ||
        /api 5\d{2}/.test(message)
    );
}

function isServiceDownError(error: unknown): boolean {
    if (isAbortError(error)) return false;
    if (error instanceof ApiError && error.status >= 500) return true;
    if (!(error instanceof Error)) return false;

    const message = error.message.toLowerCase();
    return (
        message.includes("slow to start") ||
        message.includes("failed to fetch") ||
        message.includes("network") ||
        /api 5\d{2}/.test(message)
    );
}

function toFriendlyError(error: unknown): string {
    if (isAbortError(error)) return "Request cancelled.";
    if (!(error instanceof Error)) return "Unable to load reef analysis right now.";

    const message = error.message.toLowerCase();
    if (message.includes("invalid date")) return "Choose a valid analysis date.";
    if (message.includes("slow to start")) return "Backend is waking up. Try again in a moment.";
    if (message.includes("failed to fetch") || message.includes("network")) {
        return "Could not reach the analysis service.";
    }
    if (/api 5\d{2}/.test(message)) return "The analysis service is temporarily unavailable.";
    return "Unable to load reef analysis right now.";
}

function roundCoord(value: number, decimals: number): number {
    const factor = 10 ** decimals;
    return Math.round(value * factor) / factor;
}

function buildReefKey(lat: number, lon: number): string {
    const roundedLat = roundCoord(lat, REEF_KEY_DECIMALS).toFixed(REEF_KEY_DECIMALS);
    const roundedLon = roundCoord(lon, REEF_KEY_DECIMALS).toFixed(REEF_KEY_DECIMALS);
    return `${roundedLat}|${roundedLon}`;
}

function todayIsoDate(): string {
    return new Date().toISOString().slice(0, 10);
}

function isValidDateString(dateValue: string): boolean {
    return /^\d{4}-\d{2}-\d{2}$/.test(dateValue);
}

function normalizeDateList(dates: string[]): string[] {
    const uniqueDates = Array.from(new Set(dates.filter(isValidDateString)));
    uniqueDates.sort((left, right) => left.localeCompare(right));
    return uniqueDates;
}

function buildEstimateKey(reefKey: string, isoDate: string): string {
    return `${reefKey}|${isoDate}`;
}

function getRiskBand(riskProb: number): RiskBand {
    if (riskProb >= 0.75) {
        return { label: "Severe", toneClassName: "tone-severe", color: "#d85d41", softColor: "#ffe2d6" };
    }
    if (riskProb >= 0.5) {
        return { label: "High", toneClassName: "tone-high", color: "#f2a541", softColor: "#ffefcf" };
    }
    if (riskProb >= 0.25) {
        return { label: "Elevated", toneClassName: "tone-elevated", color: "#d0b434", softColor: "#fff6cc" };
    }
    return { label: "Low", toneClassName: "tone-low", color: "#0e9f8c", softColor: "#dff8f3" };
}

function clampPercent(value: number): number {
    return Math.max(0, Math.min(100, value * 100));
}

function formatPercent(value: number): string {
    return `${clampPercent(value).toFixed(1)}%`;
}

function sampleDates(dates: string[], maxPoints: number, selectedDate?: string): string[] {
    if (dates.length <= maxPoints) return dates;

    const selected = new Set<string>();
    for (let index = 0; index < maxPoints; index += 1) {
        const ratio = index / (maxPoints - 1);
        selected.add(dates[Math.round(ratio * (dates.length - 1))]);
    }
    if (selectedDate && dates.includes(selectedDate)) {
        selected.add(selectedDate);
    }
    return dates.filter((date) => selected.has(date));
}

function formatCompactNumber(value: number): string {
    return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function severityBars(history: HistoryPoint[]) {
    const counts = { Low: 0, Elevated: 0, High: 0, Severe: 0 };
    for (const entry of history) {
        counts[getRiskBand(entry.risk_prob).label] += 1;
    }
    return counts;
}

async function estimateWithRetry(
    payload: { lat: number; lon: number; date: string },
    signal?: AbortSignal
): Promise<EstimateResponse> {
    let attempt = 0;

    while (true) {
        try {
            return await apiEstimate(payload, { signal });
        } catch (error: unknown) {
            if (!isRetriableError(error) || attempt >= MAX_ESTIMATE_RETRIES) {
                throw error;
            }
            attempt += 1;
            await sleep(RETRY_BASE_DELAY_MS * attempt);
        }
    }
}

function viewportLimitForZoom(zoom: number): number {
    if (zoom <= 3) return 500;
    if (zoom <= 4) return 800;
    if (zoom <= 5) return 1200;
    return 1800;
}

function markerRadiusForZoom(zoom: number): number {
    if (zoom <= 3) return 4;
    if (zoom <= 5) return 5;
    if (zoom <= 7) return 6;
    return 7;
}

function DashboardSection({
    title,
    subtitle,
    children,
}: {
    title: string;
    subtitle?: string;
    children: ReactNode;
}) {
    return (
        <section className="dashboard-section">
            <div className="dashboard-section__header">
                <h3>{title}</h3>
                {subtitle ? <p>{subtitle}</p> : null}
            </div>
            {children}
        </section>
    );
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

function HistoryChart({
    data,
    selectedDate,
    onSelectDate,
}: {
    data: HistoryPoint[];
    selectedDate: string;
    onSelectDate: (date: string) => void;
}) {
    const [hoveredDate, setHoveredDate] = useState<string | null>(null);

    const displayEntry = useMemo(() => {
        const preferredDate = hoveredDate ?? selectedDate;
        return data.find((entry) => entry.date === preferredDate) ?? data[data.length - 1] ?? null;
    }, [data, hoveredDate, selectedDate]);

    const svg = useMemo(() => {
        if (data.length === 0) return null;

        const width = 320;
        const height = 160;
        const padX = 20;
        const padY = 16;
        const innerWidth = width - padX * 2;
        const innerHeight = height - padY * 2;

        const coordinates = data.map((entry, index) => ({
            x: padX + (index / Math.max(1, data.length - 1)) * innerWidth,
            y: height - padY - entry.risk_prob * innerHeight,
            entry,
        }));

        const linePath = coordinates
            .map(({ x, y }, index) => `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`)
            .join(" ");
        const areaPath = `${linePath} L ${(padX + innerWidth).toFixed(1)} ${(height - padY).toFixed(1)} L ${padX.toFixed(1)} ${(height - padY).toFixed(1)} Z`;

        return { width, height, padX, padY, innerWidth, innerHeight, coordinates, linePath, areaPath };
    }, [data]);

    if (!svg) {
        return <div className="chart-empty">Timeline loads after a reef is selected.</div>;
    }

    return (
        <div className="history-chart">
            <div className="history-chart__summary">
                <div>
                    <span>Hovered date</span>
                    <strong>{displayEntry?.date ?? "No data"}</strong>
                </div>
                <div>
                    <span>Risk</span>
                    <strong>{displayEntry ? formatPercent(displayEntry.risk_prob) : "No data"}</strong>
                </div>
            </div>

            <svg viewBox={`0 0 ${svg.width} ${svg.height}`} className="chart-surface" role="img" aria-label="Risk history chart">
                <defs>
                    <linearGradient id="riskArea" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#0e9f8c" stopOpacity="0.38" />
                        <stop offset="100%" stopColor="#0e9f8c" stopOpacity="0.04" />
                    </linearGradient>
                </defs>

                {[0, 0.25, 0.5, 0.75, 1].map((value) => {
                    const y = svg.height - svg.padY - value * svg.innerHeight;
                    return (
                        <g key={value}>
                            <line x1={svg.padX} y1={y} x2={svg.padX + svg.innerWidth} y2={y} className="chart-grid" />
                            <text x={4} y={y + 4} className="chart-axis-label">
                                {Math.round(value * 100)}%
                            </text>
                        </g>
                    );
                })}

                <path d={svg.areaPath} fill="url(#riskArea)" />
                <path d={svg.linePath} className="chart-line" />

                {svg.coordinates.map(({ x, y, entry }) => {
                    const isSelected = entry.date === selectedDate;
                    return (
                        <circle
                            key={entry.date}
                            cx={x}
                            cy={y}
                            r={isSelected ? 5 : 4}
                            className={isSelected ? "chart-point chart-point--selected" : "chart-point"}
                            onMouseEnter={() => setHoveredDate(entry.date)}
                            onMouseLeave={() => setHoveredDate(null)}
                            onClick={() => onSelectDate(entry.date)}
                        />
                    );
                })}
            </svg>

            <div className="history-chart__labels">
                <span>{data[0]?.date}</span>
                <span>{data[data.length - 1]?.date}</span>
            </div>
        </div>
    );
}

export default function MapEstimateLeaflet({
    ensureBackendReady,
    serverStatus,
    onServerReachable,
    onServerDown,
    onOpenTutorial,
    warmupElapsedSeconds,
}: MapEstimateLeafletProps) {
    const [mapZoom, setMapZoom] = useState(3);
    const [visibleReefs, setVisibleReefs] = useState<ReefPoint[]>([]);
    const [visibleReefMeta, setVisibleReefMeta] = useState({ total: 0, returned: 0 });
    const [loadingReefs, setLoadingReefs] = useState(true);

    const [selectedReef, setSelectedReef] = useState<ReefPoint | null>(null);
    const [availableDates, setAvailableDates] = useState<string[]>([]);
    const [selectedDateIndex, setSelectedDateIndex] = useState(0);
    const [dateStr, setDateStr] = useState(todayIsoDate());

    const [loadingCoverage, setLoadingCoverage] = useState(false);
    const [loadingEstimate, setLoadingEstimate] = useState(false);
    const [loadingHistory, setLoadingHistory] = useState(false);
    const [loadingScenario, setLoadingScenario] = useState(false);

    const [result, setResult] = useState<EstimateResponse | null>(null);
    const [history, setHistory] = useState<HistoryPoint[]>([]);
    const [scenarioRisk, setScenarioRisk] = useState<number | null>(null);
    const [sensitivity, setSensitivity] = useState<SensitivityResponse | null>(null);
    const [scenarioDhwDelta, setScenarioDhwDelta] = useState(0);
    const [scenarioHotspotDelta, setScenarioHotspotDelta] = useState(0);
    const [activeTab, setActiveTab] = useState<DashboardTab>("overview");
    const [error, setError] = useState("");

    const [isTimelinePlaying, setTimelinePlaying] = useState(false);
    const [timelineSpeed, setTimelineSpeed] = useState<TimelineSpeed>(1);

    const viewportRequestRef = useRef(0);
    const coverageRequestRef = useRef(0);
    const estimateRequestRef = useRef(0);
    const historyRequestRef = useRef(0);
    const scenarioRequestRef = useRef(0);

    const availableDatesCacheRef = useRef<Map<string, AvailableDatesForResponse>>(new Map());
    const estimateCacheRef = useRef<Map<string, EstimateResponse>>(new Map());
    const historyCacheRef = useRef<Map<string, HistoryPoint[]>>(new Map());

    const selectedDate = useMemo(() => {
        if (availableDates.length === 0) return dateStr;
        return availableDates[Math.min(Math.max(selectedDateIndex, 0), availableDates.length - 1)] ?? dateStr;
    }, [availableDates, dateStr, selectedDateIndex]);

    const selectedBand = useMemo(() => getRiskBand(result?.risk_prob ?? 0), [result]);
    const selectedHistoryCounts = useMemo(() => severityBars(history), [history]);
    const selectedMarkerRadius = useMemo(() => markerRadiusForZoom(mapZoom), [mapZoom]);
    const historyCoverageText = useMemo(() => {
        if (availableDates.length === 0) return "No reef date coverage loaded yet.";
        return `${availableDates.length.toLocaleString()} valid days from ${availableDates[0]} to ${availableDates[availableDates.length - 1]}.`;
    }, [availableDates]);

    const scenarioDhw = result ? Math.max(0, Number(result.dhw) + scenarioDhwDelta) : 0;
    const scenarioHotspot = result ? Number(result.hotspot) + scenarioHotspotDelta : 0;

    const fetchVisibleReefs = useCallback(async (viewport: MapViewport) => {
        const requestId = viewportRequestRef.current + 1;
        viewportRequestRef.current = requestId;
        setLoadingReefs(true);
        setMapZoom(viewport.zoom);

        try {
            const response = await apiReefPoints(
                viewport.south,
                viewport.west,
                viewport.north,
                viewport.east,
                viewportLimitForZoom(viewport.zoom)
            );

            if (viewportRequestRef.current !== requestId) return;

            setVisibleReefs(
                response.points.map((point) => ({
                    lat: point.lat,
                    lon: point.lon,
                    reefKey: buildReefKey(point.lat, point.lon),
                }))
            );
            setVisibleReefMeta({ total: response.total, returned: response.returned });
        } catch (error: unknown) {
            if (viewportRequestRef.current !== requestId || isAbortError(error)) return;
            setError(toFriendlyError(error));
        } finally {
            if (viewportRequestRef.current === requestId) {
                setLoadingReefs(false);
            }
        }
    }, []);

    const loadAvailableDatesForReef = useCallback(async (reef: ReefPoint) => {
        const cached = availableDatesCacheRef.current.get(reef.reefKey);
        if (cached) return cached;

        const response = await apiAvailableDatesFor(reef.lat, reef.lon);
        const normalizedResponse: AvailableDatesForResponse = {
            reef_key: response.reef_key ?? reef.reefKey,
            lat: response.lat,
            lon: response.lon,
            count: response.count,
            min_date: response.min_date,
            max_date: response.max_date,
            dates: normalizeDateList(response.dates),
        };

        availableDatesCacheRef.current.set(reef.reefKey, normalizedResponse);
        if (normalizedResponse.reef_key) {
            availableDatesCacheRef.current.set(normalizedResponse.reef_key, normalizedResponse);
        }
        return normalizedResponse;
    }, []);

    const handleReefSelect = useCallback(
        async (reef: ReefPoint) => {
            const requestId = coverageRequestRef.current + 1;
            coverageRequestRef.current = requestId;

            setSelectedReef(reef);
            setAvailableDates([]);
            setSelectedDateIndex(0);
            setLoadingCoverage(true);
            setLoadingEstimate(false);
            setLoadingHistory(false);
            setResult(null);
            setHistory([]);
            setScenarioRisk(null);
            setSensitivity(null);
            setScenarioDhwDelta(0);
            setScenarioHotspotDelta(0);
            setError("");
            setActiveTab("overview");

            try {
                const coverage = await loadAvailableDatesForReef(reef);
                if (coverageRequestRef.current !== requestId) return;

                if (coverage.dates.length === 0) {
                    setError("This reef does not have valid date coverage.");
                    return;
                }

                const nextReef: ReefPoint = {
                    lat: coverage.lat,
                    lon: coverage.lon,
                    reefKey: coverage.reef_key ?? reef.reefKey,
                };
                const preferredDate = isValidDateString(dateStr) ? dateStr : coverage.dates[coverage.dates.length - 1];
                const nearestIndex = Math.max(0, findNearestDateIndex(coverage.dates, preferredDate));

                setSelectedReef(nextReef);
                setAvailableDates(coverage.dates);
                setSelectedDateIndex(nearestIndex);
                setDateStr(coverage.dates[nearestIndex]);
            } catch (error: unknown) {
                if (coverageRequestRef.current !== requestId) return;
                if (isServiceDownError(error)) {
                    onServerDown();
                }
                setError(toFriendlyError(error));
            } finally {
                if (coverageRequestRef.current === requestId) {
                    setLoadingCoverage(false);
                }
            }
        },
        [dateStr, loadAvailableDatesForReef, onServerDown]
    );

    useEffect(() => {
        if (!selectedReef || !selectedDate || availableDates.length === 0 || loadingCoverage) return;

        const requestId = estimateRequestRef.current + 1;
        estimateRequestRef.current = requestId;
        const cacheKey = buildEstimateKey(selectedReef.reefKey, selectedDate);
        const cached = estimateCacheRef.current.get(cacheKey);
        if (cached) {
            setResult(cached);
            setScenarioRisk(cached.risk_prob);
            setError("");
            onServerReachable();
            return;
        }

        const controller = new AbortController();
        setLoadingEstimate(true);

        void (async () => {
            try {
                await ensureBackendReady();
                const response = await estimateWithRetry(
                    { lat: selectedReef.lat, lon: selectedReef.lon, date: selectedDate },
                    controller.signal
                );

                if (estimateRequestRef.current !== requestId) return;

                estimateCacheRef.current.set(cacheKey, response);
                setResult(response);
                setScenarioRisk(response.risk_prob);
                setError("");
                onServerReachable();
            } catch (error: unknown) {
                if (estimateRequestRef.current !== requestId || isAbortError(error)) return;
                if (isServiceDownError(error)) {
                    onServerDown();
                }
                setError(toFriendlyError(error));
            } finally {
                if (estimateRequestRef.current === requestId) {
                    setLoadingEstimate(false);
                }
            }
        })();

        return () => {
            controller.abort();
        };
    }, [
        availableDates.length,
        ensureBackendReady,
        loadingCoverage,
        onServerDown,
        onServerReachable,
        selectedDate,
        selectedReef,
    ]);

    useEffect(() => {
        if (!selectedReef || availableDates.length === 0) return;

        const cached = historyCacheRef.current.get(selectedReef.reefKey);
        if (cached) {
            setHistory(cached);
            return;
        }

        const requestId = historyRequestRef.current + 1;
        historyRequestRef.current = requestId;
        const controller = new AbortController();
        const sampledDates = sampleDates(availableDates, HISTORY_SAMPLE_POINTS, selectedDate);

        setLoadingHistory(true);

        void (async () => {
            try {
                await ensureBackendReady();
                const entries = await Promise.all(
                    sampledDates.map(async (date) => {
                        const cacheKey = buildEstimateKey(selectedReef.reefKey, date);
                        const cachedEstimate = estimateCacheRef.current.get(cacheKey);
                        const estimate =
                            cachedEstimate ??
                            (await estimateWithRetry(
                                { lat: selectedReef.lat, lon: selectedReef.lon, date },
                                controller.signal
                            ));

                        estimateCacheRef.current.set(cacheKey, estimate);
                        return { date, risk_prob: estimate.risk_prob, dhw: estimate.dhw, hotspot: estimate.hotspot };
                    })
                );

                if (historyRequestRef.current !== requestId) return;

                historyCacheRef.current.set(selectedReef.reefKey, entries);
                setHistory(entries);
                onServerReachable();
            } catch (error: unknown) {
                if (historyRequestRef.current !== requestId || isAbortError(error)) return;
                if (isServiceDownError(error)) {
                    onServerDown();
                }
            } finally {
                if (historyRequestRef.current === requestId) {
                    setLoadingHistory(false);
                }
            }
        })();

        return () => {
            controller.abort();
        };
    }, [availableDates, ensureBackendReady, onServerDown, onServerReachable, selectedDate, selectedReef]);

    useEffect(() => {
        if (!result) {
            setScenarioRisk(null);
            setSensitivity(null);
            return;
        }

        const requestId = scenarioRequestRef.current + 1;
        scenarioRequestRef.current = requestId;
        const controller = new AbortController();

        setSensitivity(null);

        void apiSensitivity(
            {
                lat: result.used_lat,
                lon: result.used_lon,
                dhw: result.dhw,
                hotspot: result.hotspot,
            },
            { signal: controller.signal }
        )
            .then((response) => {
                if (scenarioRequestRef.current !== requestId) return;
                setSensitivity(response);
                onServerReachable();
            })
            .catch((error: unknown) => {
                if (scenarioRequestRef.current !== requestId || isAbortError(error)) return;
                if (isServiceDownError(error)) {
                    onServerDown();
                }
            });

        return () => {
            controller.abort();
        };
    }, [onServerDown, onServerReachable, result]);

    useEffect(() => {
        if (!result) {
            setScenarioRisk(null);
            return;
        }

        if (scenarioDhwDelta === 0 && scenarioHotspotDelta === 0) {
            setScenarioRisk(result.risk_prob);
            return;
        }

        const requestId = scenarioRequestRef.current + 1;
        scenarioRequestRef.current = requestId;
        const controller = new AbortController();
        const timeoutId = window.setTimeout(() => {
            setLoadingScenario(true);
            void apiEstimateFromFeatures(
                {
                    lat: result.used_lat,
                    lon: result.used_lon,
                    dhw: scenarioDhw,
                    hotspot: scenarioHotspot,
                },
                { signal: controller.signal }
            )
                .then((response) => {
                    if (scenarioRequestRef.current !== requestId) return;
                    setScenarioRisk(response.risk_prob);
                    onServerReachable();
                })
                .catch((error: unknown) => {
                    if (scenarioRequestRef.current !== requestId || isAbortError(error)) return;
                    if (isServiceDownError(error)) {
                        onServerDown();
                    }
                })
                .finally(() => {
                    if (scenarioRequestRef.current === requestId) {
                        setLoadingScenario(false);
                    }
                });
        }, 180);

        return () => {
            controller.abort();
            window.clearTimeout(timeoutId);
        };
    }, [
        onServerDown,
        onServerReachable,
        result,
        scenarioDhw,
        scenarioDhwDelta,
        scenarioHotspot,
        scenarioHotspotDelta,
    ]);

    useEffect(() => {
        if (!isTimelinePlaying || availableDates.length < 2) return;

        const intervalMs = timelineSpeed === 4 ? 180 : timelineSpeed === 2 ? 360 : 640;
        const intervalId = window.setInterval(() => {
            setSelectedDateIndex((current) => {
                if (current >= availableDates.length - 1) {
                    setTimelinePlaying(false);
                    return current;
                }
                const next = current + 1;
                setDateStr(availableDates[next]);
                return next;
            });
        }, intervalMs);

        return () => {
            window.clearInterval(intervalId);
        };
    }, [availableDates, isTimelinePlaying, timelineSpeed]);

    const selectDate = useCallback(
        (date: string) => {
            if (!availableDates.includes(date)) return;
            const nextIndex = availableDates.indexOf(date);
            if (nextIndex < 0) return;
            setTimelinePlaying(false);
            setSelectedDateIndex(nextIndex);
            setDateStr(date);
        },
        [availableDates]
    );

    const shiftDate = useCallback(
        (delta: number) => {
            if (availableDates.length === 0) return;
            const nextIndex = Math.min(Math.max(selectedDateIndex + delta, 0), availableDates.length - 1);
            setTimelinePlaying(false);
            setSelectedDateIndex(nextIndex);
            setDateStr(availableDates[nextIndex]);
        },
        [availableDates, selectedDateIndex]
    );

    const mapPrompt = selectedReef
        ? `Reef node selected at ${selectedReef.lat.toFixed(3)}, ${selectedReef.lon.toFixed(3)}.`
        : "Pan and zoom to load highlighted reef circles, then click one to open its dashboard.";

    const scenarioDeltaText =
        result && scenarioRisk !== null
            ? `${(clampPercent(scenarioRisk) - clampPercent(result.risk_prob)).toFixed(1)} pts`
            : "0.0 pts";

    return (
        <main className="map-experience">
            <section className="map-stage glass-panel">
                <div className="map-stage__hud">
                    <div className="map-callout">
                        <p className="map-callout__eyebrow">Live reef explorer</p>
                        <h2>Click a reef circle to open a focused reef dashboard.</h2>
                        <p>{mapPrompt}</p>
                    </div>

                    <div className="map-chip-row">
                        <div className="info-chip">
                            <span>Visible nodes</span>
                            <strong>{formatCompactNumber(visibleReefMeta.returned)}</strong>
                        </div>
                        <div className="info-chip">
                            <span>In viewport</span>
                            <strong>{formatCompactNumber(visibleReefMeta.total)}</strong>
                        </div>
                        <div className="info-chip">
                            <span>Status</span>
                            <strong>
                                {serverStatus === "ready"
                                    ? "Ready"
                                    : serverStatus === "warming"
                                      ? `Waking ${warmupElapsedSeconds}s`
                                      : "Offline"}
                            </strong>
                        </div>
                        <button type="button" className="info-chip info-chip--button" onClick={onOpenTutorial}>
                            Tutorial
                        </button>
                    </div>
                </div>

                <div className="map-frame">
                    <MapContainer
                        center={[12, 10]}
                        zoom={3}
                        zoomControl={false}
                        className="leaflet-map"
                        preferCanvas
                        worldCopyJump
                    >
                        <ZoomControl position="topright" />
                        <Pane name="reef-points" style={{ zIndex: 430 }} />
                        <Pane name="labels" style={{ zIndex: 500, pointerEvents: "none" }} />

                        <TileLayer attribution="&copy; OpenStreetMap contributors &copy; CARTO" url={TILE_URL} />
                        <TileLayer attribution="&copy; OpenStreetMap contributors &copy; CARTO" url={LABEL_URL} pane="labels" />

                        <ViewportBridge onViewportChange={fetchVisibleReefs} />

                        {visibleReefs.map((reef) => {
                            const isSelected = reef.reefKey === selectedReef?.reefKey;
                            return (
                                <CircleMarker
                                    key={reef.reefKey}
                                    center={[reef.lat, reef.lon]}
                                    pane="reef-points"
                                    radius={isSelected ? selectedMarkerRadius + 3 : selectedMarkerRadius}
                                    pathOptions={{
                                        color: isSelected ? "#0f172a" : "#0e9f8c",
                                        weight: isSelected ? 2.6 : 1.4,
                                        fillColor: isSelected ? "#f2a541" : "#f4fffb",
                                        fillOpacity: isSelected ? 1 : 0.88,
                                    }}
                                    eventHandlers={{
                                        click: () => {
                                            void handleReefSelect(reef);
                                        },
                                    }}
                                >
                                    {isSelected ? (
                                        <Tooltip permanent direction="top" offset={[0, -8]} opacity={1}>
                                            Selected reef
                                        </Tooltip>
                                    ) : null}
                                </CircleMarker>
                            );
                        })}
                    </MapContainer>

                    {loadingReefs ? (
                        <div className="map-overlay-badge" role="status" aria-live="polite">
                            Loading reef circles...
                        </div>
                    ) : null}

                    <div className="map-legend">
                        <div className="map-legend__row">
                            <span className="legend-dot legend-dot--reef" />
                            Clickable reef circle
                        </div>
                        <div className="map-legend__row">
                            <span className="legend-dot legend-dot--selected" />
                            Selected reef
                        </div>
                    </div>
                </div>
            </section>

            <aside className={`reef-dashboard glass-panel ${selectedReef ? "reef-dashboard--open" : ""}`}>
                <div className="reef-dashboard__header">
                    <div>
                        <p className="reef-dashboard__eyebrow">Reef dashboard</p>
                        <h2>
                            {selectedReef
                                ? `Reef ${selectedReef.lat.toFixed(3)}, ${selectedReef.lon.toFixed(3)}`
                                : "Choose a reef"}
                        </h2>
                        <p>
                            {selectedReef
                                ? historyCoverageText
                                : "The map is the primary interface. Select a highlighted reef circle to unlock graphs, timeline controls, and scenario testing."}
                        </p>
                    </div>

                    {result ? <span className={`severity-pill ${selectedBand.toneClassName}`}>{selectedBand.label}</span> : null}
                </div>

                {error ? <div className="inline-error">{error}</div> : null}

                {!selectedReef ? (
                    <div className="dashboard-empty">
                        <div className="dashboard-empty__ring" />
                        <p>
                            Start with a reef circle on the map. The dashboard then shifts into detailed mode for that
                            reef only.
                        </p>
                    </div>
                ) : (
                    <>
                        <div className="dashboard-tabs" role="tablist" aria-label="Reef dashboard tabs">
                            {(["overview", "timeline", "scenario"] as DashboardTab[]).map((tab) => (
                                <button
                                    key={tab}
                                    type="button"
                                    className={tab === activeTab ? "dashboard-tab dashboard-tab--active" : "dashboard-tab"}
                                    onClick={() => setActiveTab(tab)}
                                >
                                    {tab}
                                </button>
                            ))}
                        </div>

                        {activeTab === "overview" ? (
                            <div className="dashboard-scroll">
                                <DashboardSection title="Current risk" subtitle="Model output for the selected reef and date.">
                                    <div className="hero-metric">
                                        <div
                                            className={`hero-metric__ring ${selectedBand.toneClassName}`}
                                            style={{ ["--risk-value" as string]: `${clampPercent(result?.risk_prob ?? 0)}%` }}
                                        >
                                            <strong>{result ? formatPercent(result.risk_prob) : "--"}</strong>
                                            <span>Bleaching risk</span>
                                        </div>

                                        <div className="hero-metric__meta">
                                            <label>
                                                Analysis date
                                                <input
                                                    type="date"
                                                    value={selectedDate}
                                                    onChange={(event) => selectDate(event.target.value)}
                                                    disabled={loadingCoverage || availableDates.length === 0}
                                                />
                                            </label>

                                            <div className="metric-stack">
                                                <div className="metric-stack__card">
                                                    <span>DHW</span>
                                                    <strong>{result ? Number(result.dhw).toFixed(2) : "--"}</strong>
                                                </div>
                                                <div className="metric-stack__card">
                                                    <span>HotSpot</span>
                                                    <strong>{result ? Number(result.hotspot).toFixed(2) : "--"}</strong>
                                                </div>
                                                <div className="metric-stack__card">
                                                    <span>Snap distance</span>
                                                    <strong>{result ? `${Number(result.snap_km).toFixed(2)} km` : "--"}</strong>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </DashboardSection>

                                <DashboardSection title="Alert mix" subtitle="How sampled reef history distributes across severity levels.">
                                    <div className="distribution-bars">
                                        {(Object.entries(selectedHistoryCounts) as [RiskBand["label"], number][]).map(([label, count]) => {
                                            const band = getRiskBand(
                                                label === "Severe" ? 0.8 : label === "High" ? 0.55 : label === "Elevated" ? 0.3 : 0.1
                                            );
                                            const width = history.length > 0 ? (count / history.length) * 100 : 0;
                                            return (
                                                <div key={label} className="distribution-row">
                                                    <span>{label}</span>
                                                    <div className="distribution-row__track">
                                                        <div
                                                            className={`distribution-row__fill ${band.toneClassName}`}
                                                            style={{ width: `${width}%` }}
                                                        />
                                                    </div>
                                                    <strong>{count}</strong>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </DashboardSection>

                                <DashboardSection title="Key metrics" subtitle="Current thermal-stress signals feeding the model.">
                                    <div className="metric-grid">
                                        <div className="metric-card">
                                            <span>Used coordinates</span>
                                            <strong>
                                                {result?.used_lat.toFixed(3)}, {result?.used_lon.toFixed(3)}
                                            </strong>
                                        </div>
                                        <div className="metric-card">
                                            <span>Requested coordinates</span>
                                            <strong>
                                                {result?.input_lat.toFixed(3)}, {result?.input_lon.toFixed(3)}
                                            </strong>
                                        </div>
                                        <div className="metric-card">
                                            <span>Risk flag</span>
                                            <strong>{result?.risk_flag === 1 ? "High alert" : "Watch"}</strong>
                                        </div>
                                        <div className="metric-card">
                                            <span>Coverage</span>
                                            <strong>{availableDates.length.toLocaleString()} dates</strong>
                                        </div>
                                    </div>
                                </DashboardSection>
                            </div>
                        ) : null}

                        {activeTab === "timeline" ? (
                            <div className="dashboard-scroll">
                                <DashboardSection title="Risk timeline" subtitle="Sampled history for this reef. Click a point to jump the dashboard date.">
                                    {loadingHistory && history.length === 0 ? (
                                        <div className="chart-empty">Loading reef history...</div>
                                    ) : (
                                        <HistoryChart data={history} selectedDate={selectedDate} onSelectDate={selectDate} />
                                    )}
                                </DashboardSection>

                                <DashboardSection title="Timeline controls" subtitle="Scrub through valid dates or autoplay the reef history.">
                                    <div className="timeline-panel">
                                        <div className="timeline-panel__date">{selectedDate}</div>
                                        <input
                                            type="range"
                                            min={0}
                                            max={Math.max(0, availableDates.length - 1)}
                                            step={1}
                                            value={Math.min(selectedDateIndex, Math.max(0, availableDates.length - 1))}
                                            onChange={(event) => {
                                                const nextIndex = Number(event.target.value);
                                                if (!Number.isFinite(nextIndex) || availableDates.length === 0) return;
                                                setTimelinePlaying(false);
                                                setSelectedDateIndex(nextIndex);
                                                setDateStr(availableDates[nextIndex]);
                                            }}
                                            disabled={availableDates.length === 0 || loadingCoverage}
                                        />

                                        <div className="timeline-panel__actions">
                                            <button type="button" className="ghost-button" onClick={() => shiftDate(-1)} disabled={selectedDateIndex <= 0}>
                                                Prev
                                            </button>
                                            <button
                                                type="button"
                                                className="ghost-button ghost-button--accent"
                                                onClick={() => setTimelinePlaying((playing) => !playing)}
                                                disabled={availableDates.length < 2}
                                            >
                                                {isTimelinePlaying ? "Pause" : "Play"}
                                            </button>
                                            <button
                                                type="button"
                                                className="ghost-button"
                                                onClick={() => shiftDate(1)}
                                                disabled={selectedDateIndex >= availableDates.length - 1}
                                            >
                                                Next
                                            </button>
                                        </div>

                                        <div className="segmented-control">
                                            {[1, 2, 4].map((speed) => (
                                                <button
                                                    key={speed}
                                                    type="button"
                                                    className={timelineSpeed === speed ? "segmented-control__button segmented-control__button--active" : "segmented-control__button"}
                                                    onClick={() => setTimelineSpeed(speed as TimelineSpeed)}
                                                >
                                                    {speed}x
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                </DashboardSection>
                            </div>
                        ) : null}

                        {activeTab === "scenario" ? (
                            <div className="dashboard-scroll">
                                <DashboardSection title="Scenario lab" subtitle="Shift the thermal stress inputs and watch the model react.">
                                    <div className="scenario-grid">
                                        <div className="scenario-card">
                                            <span>Current risk</span>
                                            <strong>{result ? formatPercent(result.risk_prob) : "--"}</strong>
                                        </div>
                                        <div className="scenario-card">
                                            <span>Adjusted risk</span>
                                            <strong>{scenarioRisk !== null ? formatPercent(scenarioRisk) : "--"}</strong>
                                        </div>
                                        <div className="scenario-card">
                                            <span>Delta</span>
                                            <strong>{scenarioDeltaText}</strong>
                                        </div>
                                    </div>

                                    <label className="slider-control">
                                        <div className="slider-control__header">
                                            <span>DHW adjustment</span>
                                            <strong>
                                                {scenarioDhwDelta >= 0 ? "+" : ""}
                                                {scenarioDhwDelta.toFixed(1)}
                                            </strong>
                                        </div>
                                        <input
                                            type="range"
                                            min={-3}
                                            max={6}
                                            step={0.1}
                                            value={scenarioDhwDelta}
                                            onChange={(event) => setScenarioDhwDelta(Number(event.target.value))}
                                        />
                                    </label>

                                    <label className="slider-control">
                                        <div className="slider-control__header">
                                            <span>HotSpot adjustment</span>
                                            <strong>
                                                {scenarioHotspotDelta >= 0 ? "+" : ""}
                                                {scenarioHotspotDelta.toFixed(1)}
                                            </strong>
                                        </div>
                                        <input
                                            type="range"
                                            min={-2}
                                            max={4}
                                            step={0.1}
                                            value={scenarioHotspotDelta}
                                            onChange={(event) => setScenarioHotspotDelta(Number(event.target.value))}
                                        />
                                    </label>

                                    <div className="stress-bars">
                                        <div className="stress-bar">
                                            <div className="stress-bar__label">
                                                <span>DHW</span>
                                                <strong>{scenarioDhw.toFixed(2)}</strong>
                                            </div>
                                            <div className="stress-bar__track">
                                                <div className="stress-bar__fill" style={{ width: `${Math.min(100, (scenarioDhw / 16) * 100)}%` }} />
                                            </div>
                                        </div>
                                        <div className="stress-bar">
                                            <div className="stress-bar__label">
                                                <span>HotSpot</span>
                                                <strong>{scenarioHotspot.toFixed(2)}</strong>
                                            </div>
                                            <div className="stress-bar__track">
                                                <div className="stress-bar__fill stress-bar__fill--warm" style={{ width: `${Math.min(100, (Math.max(0, scenarioHotspot) / 6) * 100)}%` }} />
                                            </div>
                                        </div>
                                    </div>
                                </DashboardSection>

                                <DashboardSection title="Model sensitivity" subtitle="Estimated lift from raising one driver while holding the other constant.">
                                    {sensitivity ? (
                                        <div className="scenario-grid">
                                            <div className="scenario-card">
                                                <span>+{sensitivity.dhw_step} DHW</span>
                                                <strong>{`${(sensitivity.delta_dhw * 100).toFixed(1)} pts`}</strong>
                                            </div>
                                            <div className="scenario-card">
                                                <span>+{sensitivity.hotspot_step} HotSpot</span>
                                                <strong>{`${(sensitivity.delta_hotspot * 100).toFixed(1)} pts`}</strong>
                                            </div>
                                            <div className="scenario-card">
                                                <span>Lab status</span>
                                                <strong>{loadingScenario ? "Updating" : "Ready"}</strong>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="chart-empty">Loading sensitivity preview...</div>
                                    )}
                                </DashboardSection>
                            </div>
                        ) : null}
                    </>
                )}

                {loadingCoverage || loadingEstimate ? (
                    <div className="dashboard-loading" role="status" aria-live="polite">
                        {loadingCoverage ? "Loading reef coverage..." : "Running reef analysis..."}
                    </div>
                ) : null}
            </aside>
        </main>
    );
}
