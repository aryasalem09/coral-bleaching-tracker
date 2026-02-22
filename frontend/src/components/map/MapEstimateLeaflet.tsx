import { type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import iconUrl from "leaflet/dist/images/marker-icon.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";
import {
    Circle,
    CircleMarker,
    MapContainer,
    Marker,
    Polyline,
    TileLayer,
    Tooltip,
    useMapEvents,
} from "react-leaflet";
import { apiAvailableDatesFor, apiEstimate } from "../../lib/api";
import { findNearestDateIndex } from "../../lib/dateUtils";
import type { AvailableDatesForResponse, EstimateRequest, EstimateResponse } from "../../lib/api";
import type { ServerStatus } from "../../types/server";

if (typeof window !== "undefined") {
    L.Icon.Default.mergeOptions({
        iconRetinaUrl,
        iconUrl,
        shadowUrl,
    });
}

type MapEstimateLeafletProps = {
    ensureBackendReady: () => Promise<void>;
    serverStatus: ServerStatus;
    onServerReachable: () => void;
    onServerDown: () => void;
};

type ClickHandlerProps = {
    onClick: (lat: number, lon: number) => void;
    disabled: boolean;
};

type ReefPoint = {
    lat: number;
    lon: number;
};

type TimelineSpeed = 0.5 | 1 | 2;

type RiskBand = {
    label: "Low" | "Elevated" | "High" | "Severe";
    badgeClassName: string;
    markerBadgeClassName: string;
    markerStroke: string;
    markerFill: string;
};

const MAX_ESTIMATE_RETRIES = 2;
const RETRY_BASE_DELAY_MS = 800;
const ESTIMATE_DEBOUNCE_MS = 280;
const DATES_CACHE_DECIMALS = 3;
const REEF_KEY_DECIMALS = 4;
const NO_DATA_SNAP_NOTE = "No data for requested date — snapped to nearest available date.";

function isValidDateString(dateValue: string): boolean {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(dateValue)) return false;

    const [yearText, monthText, dayText] = dateValue.split("-");
    const year = Number(yearText);
    const month = Number(monthText);
    const day = Number(dayText);

    if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) return false;

    const parsedDate = new Date(Date.UTC(year, month - 1, day));
    return (
        parsedDate.getUTCFullYear() === year &&
        parsedDate.getUTCMonth() === month - 1 &&
        parsedDate.getUTCDate() === day
    );
}

function roundCoord(value: number, decimals: number): number {
    const factor = 10 ** decimals;
    return Math.round(value * factor) / factor;
}

function buildCoordKey(lat: number, lon: number, decimals: number): string {
    return `${roundCoord(lat, decimals).toFixed(decimals)}|${roundCoord(lon, decimals).toFixed(decimals)}`;
}

function buildEstimateKey(reefKey: string, isoDate: string): string {
    return `${reefKey}|${isoDate}`;
}

function buildSelectionDateKey(lat: number, lon: number, isoDate: string): string {
    const selectedKey = buildCoordKey(lat, lon, DATES_CACHE_DECIMALS);
    return `${selectedKey}|${isoDate}`;
}

function normalizeDateList(dates: string[]): string[] {
    const validDates = dates.filter((value) => isValidDateString(value));
    const uniqueDates = Array.from(new Set(validDates));
    uniqueDates.sort((a, b) => a.localeCompare(b));
    return uniqueDates;
}

function timelineIntervalMs(speed: TimelineSpeed): number {
    if (speed === 0.5) return 1000;
    if (speed === 2) return 350;
    return 600;
}

function parseTimelineSpeed(rawValue: string): TimelineSpeed {
    if (rawValue === "0.5") return 0.5;
    if (rawValue === "2") return 2;
    return 1;
}

function ClickHandler({ onClick, disabled }: ClickHandlerProps) {
    useMapEvents({
        click(event) {
            if (disabled) return;
            onClick(event.latlng.lat, event.latlng.lng);
        },
    });
    return null;
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => {
        window.setTimeout(resolve, ms);
    });
}

function isRetriableError(error: unknown): boolean {
    if (error instanceof TypeError) return true;
    if (!(error instanceof Error)) return false;

    const message = error.message.toLowerCase();
    return (
        message.includes("failed to fetch") ||
        message.includes("network") ||
        /api 5\d{2}/.test(message) ||
        message.includes("timeout") ||
        message.includes("gateway")
    );
}

function isServiceDownError(error: unknown): boolean {
    if (!(error instanceof Error)) return false;
    const message = error.message.toLowerCase();
    return (
        message.includes("slow to start") ||
        message.includes("failed to fetch") ||
        /api 5\d{2}/.test(message) ||
        message.includes("network")
    );
}

function isNoDataForDateError(error: unknown): boolean {
    if (!(error instanceof Error)) return false;
    const message = error.message.toLowerCase();
    return (
        message.includes("no valid reef point found nearby for this date") ||
        message.includes("location/date") ||
        message.includes("noaa file missing")
    );
}

function toFriendlyError(error: unknown): string {
    if (!(error instanceof Error)) {
        return "Unable to complete analysis right now. Please try again.";
    }

    const message = error.message.toLowerCase();

    if (isNoDataForDateError(error)) {
        return "No usable reef data was available for that date at this location.";
    }

    if (message.includes("invalid date format")) {
        return "Please choose a valid date in YYYY-MM-DD format.";
    }

    if (message.includes("slow to start")) {
        return "Server is slow to start. Please try again shortly.";
    }

    if (message.includes("failed to fetch") || message.includes("network")) {
        return "We could not reach the analysis server. Please try again in a moment.";
    }

    if (/api 5\d{2}/.test(message)) {
        return "The analysis service is temporarily unavailable. Please retry shortly.";
    }

    return "Unable to complete analysis right now. Please try again.";
}

async function estimateWithRetry(payload: EstimateRequest): Promise<EstimateResponse> {
    let attempt = 0;

    while (true) {
        try {
            return await apiEstimate(payload);
        } catch (error: unknown) {
            if (!isRetriableError(error) || attempt >= MAX_ESTIMATE_RETRIES) {
                throw error;
            }

            attempt += 1;
            await sleep(RETRY_BASE_DELAY_MS * attempt);
        }
    }
}

function getRiskBand(riskProb: number): RiskBand {
    if (riskProb >= 0.75) {
        return {
            label: "Severe",
            badgeClassName: "risk-badge--critical",
            markerBadgeClassName: "reef-status-tooltip--severe",
            markerStroke: "#ff6b6b",
            markerFill: "#ff9f9f",
        };
    }
    if (riskProb >= 0.5) {
        return {
            label: "High",
            badgeClassName: "risk-badge--high",
            markerBadgeClassName: "reef-status-tooltip--high",
            markerStroke: "#ff9f66",
            markerFill: "#ffd0b6",
        };
    }
    if (riskProb >= 0.25) {
        return {
            label: "Elevated",
            badgeClassName: "risk-badge--moderate",
            markerBadgeClassName: "reef-status-tooltip--elevated",
            markerStroke: "#f7d26a",
            markerFill: "#ffedb4",
        };
    }
    return {
        label: "Low",
        badgeClassName: "risk-badge--low",
        markerBadgeClassName: "reef-status-tooltip--low",
        markerStroke: "#1fa6a6",
        markerFill: "#8ce8ce",
    };
}

function todayIsoDate(): string {
    return new Date().toISOString().slice(0, 10);
}

export default function MapEstimateLeaflet({
    ensureBackendReady,
    serverStatus,
    onServerReachable,
    onServerDown,
}: MapEstimateLeafletProps) {
    const [dateStr, setDateStr] = useState(todayIsoDate);
    const [selectedReef, setSelectedReef] = useState<ReefPoint | null>(null);
    const [availableDates, setAvailableDates] = useState<string[]>([]);
    const [selectedDateIndex, setSelectedDateIndex] = useState(0);
    const [loadingDates, setLoadingDates] = useState(false);
    const [loading, setLoading] = useState(false);
    const [err, setErr] = useState("");
    const [dateSnapNote, setDateSnapNote] = useState("");
    const [res, setRes] = useState<EstimateResponse | null>(null);
    const [animatedRiskProb, setAnimatedRiskProb] = useState(0);
    const [isTimelinePlaying, setTimelinePlaying] = useState(false);
    const [timelineSpeed, setTimelineSpeed] = useState<TimelineSpeed>(1);
    const [timelineLoopEnabled, setTimelineLoopEnabled] = useState(false);
    const [timelineShouldResume, setTimelineShouldResume] = useState(false);

    const animationFrameRef = useRef<number | null>(null);
    const mountedRef = useRef(true);
    const requestInFlightRef = useRef(false);
    const debounceTimerRef = useRef<number | null>(null);
    const reefSelectionRequestIdRef = useRef(0);

    const availableDatesCacheRef = useRef<Map<string, AvailableDatesForResponse>>(new Map());
    const estimateCacheRef = useRef<Map<string, EstimateResponse>>(new Map());
    const selectedDateToReefKeyRef = useRef<Map<string, string>>(new Map());
    const selectedToLatestReefKeyRef = useRef<Map<string, string>>(new Map());

    const clearDebounceTimer = useCallback(() => {
        if (debounceTimerRef.current !== null) {
            window.clearTimeout(debounceTimerRef.current);
            debounceTimerRef.current = null;
        }
    }, []);

    useEffect(() => {
        mountedRef.current = true;

        return () => {
            mountedRef.current = false;
            requestInFlightRef.current = false;
            clearDebounceTimer();

            if (animationFrameRef.current !== null) {
                window.cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
        };
    }, [clearDebounceTimer]);

    useEffect(() => {
        if (animationFrameRef.current !== null) {
            window.cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }

        const targetValue = res ? Math.max(0, Math.min(1, res.risk_prob)) : 0;
        const startedAt = performance.now();
        const durationMs = 900;

        const animate = (timestamp: number) => {
            const progress = Math.min(1, (timestamp - startedAt) / durationMs);
            const eased = 1 - (1 - progress) ** 3;
            if (!mountedRef.current) return;
            setAnimatedRiskProb(targetValue * eased);

            if (progress < 1) {
                animationFrameRef.current = window.requestAnimationFrame(animate);
            } else {
                animationFrameRef.current = null;
            }
        };

        animationFrameRef.current = window.requestAnimationFrame(animate);

        return () => {
            if (animationFrameRef.current !== null) {
                window.cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
        };
    }, [res]);

    useEffect(() => {
        if (availableDates.length === 0) return;

        const boundedIndex = Math.min(Math.max(selectedDateIndex, 0), availableDates.length - 1);
        if (boundedIndex !== selectedDateIndex) {
            setSelectedDateIndex(boundedIndex);
            return;
        }

        const nextDate = availableDates[boundedIndex];
        if (nextDate && nextDate !== dateStr) {
            setDateStr(nextDate);
        }
    }, [availableDates, dateStr, selectedDateIndex]);

    useEffect(() => {
        if (serverStatus === "warming" && isTimelinePlaying) {
            setTimelineShouldResume(true);
            setTimelinePlaying(false);
        }

        if (serverStatus === "down") {
            setTimelineShouldResume(false);
            setTimelinePlaying(false);
        }
    }, [isTimelinePlaying, serverStatus]);

    useEffect(() => {
        if (!timelineShouldResume) return;
        if (loading || loadingDates) return;
        if (serverStatus === "warming" || serverStatus === "down") return;

        if (!selectedReef || availableDates.length === 0) {
            setTimelineShouldResume(false);
            return;
        }

        setTimelineShouldResume(false);
        setTimelinePlaying(true);
    }, [availableDates.length, loading, loadingDates, selectedReef, serverStatus, timelineShouldResume]);

    useEffect(() => {
        if (!isTimelinePlaying) return;
        if (availableDates.length < 2) return;
        if (loading || loadingDates) return;
        if (serverStatus === "warming" || serverStatus === "down") return;

        const intervalId = window.setInterval(() => {
            setDateSnapNote("");
            setSelectedDateIndex((currentIndex) => {
                const lastIndex = availableDates.length - 1;
                if (currentIndex >= lastIndex) {
                    if (timelineLoopEnabled) return 0;
                    setTimelineShouldResume(false);
                    setTimelinePlaying(false);
                    return currentIndex;
                }
                return currentIndex + 1;
            });
        }, timelineIntervalMs(timelineSpeed));

        return () => {
            window.clearInterval(intervalId);
        };
    }, [availableDates, isTimelinePlaying, loading, loadingDates, serverStatus, timelineLoopEnabled, timelineSpeed]);

    const selectedDate = useMemo(() => {
        if (availableDates.length === 0) return dateStr;
        const boundedIndex = Math.min(Math.max(selectedDateIndex, 0), availableDates.length - 1);
        return availableDates[boundedIndex] ?? dateStr;
    }, [availableDates, dateStr, selectedDateIndex]);

    const inputPos = useMemo(() => {
        if (!selectedReef) return null;
        return [selectedReef.lat, selectedReef.lon] as [number, number];
    }, [selectedReef]);

    const usedPos = useMemo(() => {
        if (!res) return null;
        return [res.used_lat, res.used_lon] as [number, number];
    }, [res]);

    const circleRadiusM = useMemo(() => {
        if (!res?.snapped) return 0;
        const km = Number(res.snap_km);
        if (!Number.isFinite(km) || km <= 0) return 0;
        return km * 1000;
    }, [res]);

    const line = useMemo(() => {
        if (!res?.snapped || !inputPos || !usedPos) return null;
        return [inputPos, usedPos] as [number, number][];
    }, [inputPos, res, usedPos]);

    const riskBand = useMemo(() => getRiskBand(res?.risk_prob ?? 0), [res]);
    const riskWidth = useMemo(() => Math.max(0, Math.min(100, animatedRiskProb * 100)), [animatedRiskProb]);
    const riskText = useMemo(() => `${riskWidth.toFixed(1)}%`, [riskWidth]);

    const coverageText = useMemo(() => {
        if (availableDates.length === 0) return "";
        const firstDate = availableDates[0];
        const lastDate = availableDates[availableDates.length - 1];
        return `This reef has data for ${availableDates.length} days (from ${firstDate} to ${lastDate}).`;
    }, [availableDates]);

    const sliderMax = Math.max(0, availableDates.length - 1);
    const timelineDisabled = availableDates.length === 0 || loadingDates || serverStatus === "down";
    const mapDisabled = loading || loadingDates;

    const getCachedEstimate = useCallback((selectedLat: number, selectedLon: number, isoDate: string) => {
        const mappedDateKey = buildSelectionDateKey(selectedLat, selectedLon, isoDate);
        const mappedReefKey = selectedDateToReefKeyRef.current.get(mappedDateKey);
        if (mappedReefKey) {
            return estimateCacheRef.current.get(buildEstimateKey(mappedReefKey, isoDate)) ?? null;
        }

        const selectedKey = buildCoordKey(selectedLat, selectedLon, DATES_CACHE_DECIMALS);
        const latestReefKey = selectedToLatestReefKeyRef.current.get(selectedKey);
        if (latestReefKey) {
            return estimateCacheRef.current.get(buildEstimateKey(latestReefKey, isoDate)) ?? null;
        }

        return null;
    }, []);

    const cacheEstimate = useCallback((selectedLat: number, selectedLon: number, estimate: EstimateResponse) => {
        const reefKey = buildCoordKey(estimate.used_lat, estimate.used_lon, REEF_KEY_DECIMALS);
        estimateCacheRef.current.set(buildEstimateKey(reefKey, estimate.date), estimate);

        const selectedKey = buildCoordKey(selectedLat, selectedLon, DATES_CACHE_DECIMALS);
        selectedToLatestReefKeyRef.current.set(selectedKey, reefKey);
        selectedDateToReefKeyRef.current.set(buildSelectionDateKey(selectedLat, selectedLon, estimate.date), reefKey);
    }, []);

    const loadAvailableDatesForReef = useCallback(async (lat: number, lon: number) => {
        const cacheKey = buildCoordKey(lat, lon, DATES_CACHE_DECIMALS);
        const cached = availableDatesCacheRef.current.get(cacheKey);
        if (cached) return cached;

        const response = await apiAvailableDatesFor(lat, lon);
        const normalizedDates = normalizeDateList(response.dates);
        const normalizedResponse: AvailableDatesForResponse = {
            lat: response.lat,
            lon: response.lon,
            count: normalizedDates.length,
            dates: normalizedDates,
        };
        availableDatesCacheRef.current.set(cacheKey, normalizedResponse);
        return normalizedResponse;
    }, []);

    const fetchEstimate = useCallback(
        async (lat: number, lon: number, isoDate: string): Promise<EstimateResponse> => {
            await ensureBackendReady();
            return estimateWithRetry({ lat, lon, date: isoDate });
        },
        [ensureBackendReady]
    );

    const runEstimateForDate = useCallback(
        async (lat: number, lon: number, isoDate: string, allowNoDataRetry: boolean) => {
            if (requestInFlightRef.current) return;
            if (!isValidDateString(isoDate)) {
                setErr("Please select a valid date before running analysis.");
                return;
            }

            requestInFlightRef.current = true;
            if (isTimelinePlaying) {
                setTimelineShouldResume(true);
                setTimelinePlaying(false);
            }

            if (mountedRef.current) {
                setErr("");
                setLoading(true);
            }

            let finalData: EstimateResponse | null = null;
            let finalError: unknown = null;

            try {
                try {
                    finalData = await fetchEstimate(lat, lon, isoDate);
                } catch (error: unknown) {
                    finalError = error;
                }

                if (!finalData && allowNoDataRetry && finalError && isNoDataForDateError(finalError)) {
                    const retryIndex = findNearestDateIndex(availableDates, isoDate);
                    if (retryIndex >= 0) {
                        const retryDate = availableDates[retryIndex];
                        if (retryDate && retryDate !== isoDate && isValidDateString(retryDate)) {
                            if (mountedRef.current) {
                                setDateSnapNote(NO_DATA_SNAP_NOTE);
                                setSelectedDateIndex(retryIndex);
                                setDateStr(retryDate);
                            }

                            try {
                                finalData = await fetchEstimate(lat, lon, retryDate);
                                finalError = null;
                            } catch (retryError: unknown) {
                                finalError = retryError;
                            }
                        }
                    }
                }

                if (finalData) {
                    if (!mountedRef.current) return;
                    cacheEstimate(lat, lon, finalData);
                    setRes(finalData);
                    setErr("");
                    if (finalData.date !== isoDate) {
                        const snappedIndex = availableDates.indexOf(finalData.date);
                        if (snappedIndex >= 0) {
                            setSelectedDateIndex(snappedIndex);
                            setDateStr(finalData.date);
                        }
                    } else if (dateSnapNote) {
                        setDateSnapNote("");
                    }
                    onServerReachable();
                    return;
                }

                if (finalError) {
                    setTimelineShouldResume(false);
                    if (mountedRef.current && isServiceDownError(finalError)) {
                        onServerDown();
                    }
                    if (mountedRef.current) {
                        setErr(toFriendlyError(finalError));
                    }
                }
            } finally {
                requestInFlightRef.current = false;
                if (mountedRef.current) {
                    setLoading(false);
                }
            }
        },
        [
            availableDates,
            cacheEstimate,
            dateSnapNote,
            fetchEstimate,
            isTimelinePlaying,
            onServerDown,
            onServerReachable,
        ]
    );

    useEffect(() => {
        clearDebounceTimer();

        if (!selectedReef || availableDates.length === 0 || loadingDates) return;
        if (loading || requestInFlightRef.current) return;
        if (!isValidDateString(selectedDate)) {
            setErr("Please select a valid date before running analysis.");
            return;
        }

        const cached = getCachedEstimate(selectedReef.lat, selectedReef.lon, selectedDate);
        if (cached) {
            setErr("");
            setRes(cached);
            onServerReachable();
            return;
        }

        debounceTimerRef.current = window.setTimeout(() => {
            void runEstimateForDate(selectedReef.lat, selectedReef.lon, selectedDate, true);
        }, ESTIMATE_DEBOUNCE_MS);

        return clearDebounceTimer;
    }, [
        availableDates.length,
        clearDebounceTimer,
        getCachedEstimate,
        loading,
        loadingDates,
        onServerReachable,
        runEstimateForDate,
        selectedDate,
        selectedReef,
    ]);

    const handleMapClick = useCallback(
        async (lat: number, lon: number) => {
            const requestId = reefSelectionRequestIdRef.current + 1;
            reefSelectionRequestIdRef.current = requestId;

            clearDebounceTimer();
            setTimelineShouldResume(false);
            setTimelinePlaying(false);
            setDateSnapNote("");
            setErr("");
            setRes(null);
            setSelectedReef({ lat, lon });
            setAvailableDates([]);
            setSelectedDateIndex(0);
            setLoadingDates(true);

            try {
                const coverage = await loadAvailableDatesForReef(lat, lon);
                if (!mountedRef.current || reefSelectionRequestIdRef.current !== requestId) return;

                if (coverage.dates.length === 0) {
                    setErr("This reef has no valid date coverage. Please click a nearby reef location.");
                    return;
                }

                const baseDate = isValidDateString(dateStr) ? dateStr : coverage.dates[0];
                const exactIndex = coverage.dates.indexOf(baseDate);
                const nextIndex = exactIndex >= 0 ? exactIndex : findNearestDateIndex(coverage.dates, baseDate);
                const safeIndex = nextIndex >= 0 ? nextIndex : 0;
                const snappedDate = coverage.dates[safeIndex] ?? coverage.dates[0];

                setAvailableDates(coverage.dates);
                setSelectedDateIndex(safeIndex);
                setDateStr(snappedDate);
            } catch (error: unknown) {
                if (!mountedRef.current || reefSelectionRequestIdRef.current !== requestId) return;
                if (isServiceDownError(error)) {
                    onServerDown();
                }
                setErr(toFriendlyError(error));
            } finally {
                if (mountedRef.current && reefSelectionRequestIdRef.current === requestId) {
                    setLoadingDates(false);
                }
            }
        },
        [clearDebounceTimer, dateStr, loadAvailableDatesForReef, onServerDown]
    );

    const handleDateInputChange = useCallback(
        (event: ChangeEvent<HTMLInputElement>) => {
            const nextDate = event.target.value;
            setDateSnapNote("");

            if (availableDates.length === 0) {
                setDateStr(nextDate);
                return;
            }

            if (!isValidDateString(nextDate)) return;

            setTimelineShouldResume(false);
            setTimelinePlaying(false);
            const snappedIndex = findNearestDateIndex(availableDates, nextDate);
            if (snappedIndex >= 0) {
                setSelectedDateIndex(snappedIndex);
                setDateStr(availableDates[snappedIndex]);
            }
        },
        [availableDates]
    );

    const handleSliderChange = useCallback(
        (event: ChangeEvent<HTMLInputElement>) => {
            if (availableDates.length === 0) return;
            const index = Number(event.target.value);
            if (!Number.isFinite(index)) return;

            const boundedIndex = Math.min(Math.max(Math.round(index), 0), availableDates.length - 1);
            setTimelineShouldResume(false);
            setTimelinePlaying(false);
            setDateSnapNote("");
            setSelectedDateIndex(boundedIndex);
            setDateStr(availableDates[boundedIndex]);
        },
        [availableDates]
    );

    const shiftDateIndex = useCallback(
        (delta: number) => {
            if (availableDates.length === 0) return;
            setTimelineShouldResume(false);
            setTimelinePlaying(false);
            setDateSnapNote("");
            setSelectedDateIndex((current) => {
                const nextIndex = Math.min(Math.max(current + delta, 0), availableDates.length - 1);
                if (nextIndex !== current) {
                    setDateStr(availableDates[nextIndex]);
                }
                return nextIndex;
            });
        },
        [availableDates]
    );

    const toggleTimelinePlay = useCallback(() => {
        if (timelineDisabled) return;
        setTimelineShouldResume(false);
        setTimelinePlaying((currentlyPlaying) => !currentlyPlaying);
    }, [timelineDisabled]);

    const mapHint = useMemo(() => {
        if (loadingDates) {
            return "Loading reef-specific date coverage...";
        }

        if (serverStatus === "warming") {
            return "Server warmup in progress. Coverage is ready now; timeline playback resumes when the backend is ready.";
        }

        if (selectedReef) {
            return "Reef selected. Use Time Scrubber or Timeline Mode to animate available dates.";
        }

        return "Click any reef location on the map to load date coverage and run bleaching risk estimates.";
    }, [loadingDates, selectedReef, serverStatus]);

    return (
        <section className="estimate-grid">
            <div className="map-panel glass-panel">
                <div className="map-toolbar">
                    <label className="date-control cursor-target">
                        <span className="date-control__label">Analysis Date</span>
                        <input
                            className="date-control__input"
                            type="date"
                            value={dateStr}
                            onChange={handleDateInputChange}
                            disabled={mapDisabled}
                        />
                    </label>

                    <p className="map-toolbar__hint">{mapHint}</p>
                </div>

                {selectedReef && loadingDates ? <p className="coverage-line">Loading coverage for selected reef...</p> : null}
                {selectedReef && !loadingDates && availableDates.length > 0 ? (
                    <p className="coverage-line">{coverageText}</p>
                ) : null}

                <div className={`map-frame ${loading ? "map-frame--loading" : ""}`}>
                    <MapContainer center={[7, -38]} zoom={3} className="leaflet-map">
                        <TileLayer
                            attribution='&copy; OpenStreetMap contributors'
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        />

                        <ClickHandler onClick={handleMapClick} disabled={mapDisabled} />

                        {inputPos ? <Marker position={inputPos} /> : null}
                        {usedPos && res ? (
                            <CircleMarker
                                center={usedPos}
                                radius={10}
                                pathOptions={{
                                    color: riskBand.markerStroke,
                                    weight: 2.6,
                                    fillColor: riskBand.markerFill,
                                    fillOpacity: 0.95,
                                }}
                            >
                                <Tooltip
                                    permanent
                                    direction="bottom"
                                    offset={[0, 12]}
                                    className={`reef-status-tooltip ${riskBand.markerBadgeClassName}`}
                                    opacity={1}
                                >
                                    {riskBand.label}
                                </Tooltip>
                            </CircleMarker>
                        ) : null}

                        {res?.snapped && inputPos ? (
                            <Circle
                                center={inputPos}
                                radius={circleRadiusM}
                                pathOptions={{
                                    color: "#8FD9FF",
                                    weight: 2,
                                    fillColor: "#8FD9FF",
                                    fillOpacity: 0.14,
                                    dashArray: "6 8",
                                }}
                            />
                        ) : null}

                        {line ? <Polyline positions={line} pathOptions={{ color: "#1FA6A6", weight: 2.6 }} /> : null}
                    </MapContainer>

                    {loading ? (
                        <div className="map-loading-overlay" role="status" aria-live="polite">
                            <div className="map-loading-card">
                                <span className="spinner" aria-hidden="true" />
                                <span>Running ocean stress analysis...</span>
                            </div>
                        </div>
                    ) : null}
                </div>

                <p className="map-note">
                    If you click off a reef, the model snaps to the nearest valid reef grid cell and visualizes the
                    snap distance.
                </p>
            </div>

            <aside className={`results-panel glass-panel ${res ? "results-panel--active" : ""}`}>
                <div className="results-heading">
                    <h2>Estimate Results</h2>
                    <p>NOAA thermal stress indicators with model probability output</p>
                </div>

                {selectedReef && availableDates.length > 0 ? (
                    <section className="time-scrubber">
                        <div className="time-scrubber__header">
                            <h3>Time Scrubber</h3>
                            {isTimelinePlaying ? <span className="timeline-playing">Playing...</span> : null}
                        </div>

                        <div className="time-scrubber__date">{selectedDate}</div>

                        <input
                            className="time-scrubber__slider"
                            type="range"
                            min={0}
                            max={sliderMax}
                            step={1}
                            value={Math.min(selectedDateIndex, sliderMax)}
                            onChange={handleSliderChange}
                            disabled={mapDisabled}
                            aria-label="Reef date scrubber"
                        />

                        <div className="time-scrubber__buttons">
                            <button
                                className="timeline-button"
                                type="button"
                                onClick={() => shiftDateIndex(-1)}
                                disabled={mapDisabled || selectedDateIndex <= 0}
                            >
                                Prev
                            </button>
                            <button
                                className="timeline-button"
                                type="button"
                                onClick={() => shiftDateIndex(1)}
                                disabled={mapDisabled || selectedDateIndex >= sliderMax}
                            >
                                Next
                            </button>
                        </div>

                        <p className="time-scrubber__hint">drag to scrub through valid reef dates</p>

                        <div className="timeline-controls">
                            <span className="timeline-controls__title">Timeline Mode</span>
                            <button
                                className="timeline-button timeline-button--play"
                                type="button"
                                onClick={toggleTimelinePlay}
                                disabled={timelineDisabled || (loading && !isTimelinePlaying)}
                            >
                                {isTimelinePlaying ? "Pause" : "Play"}
                            </button>

                            <label className="timeline-controls__speed">
                                <span>Speed</span>
                                <select
                                    value={String(timelineSpeed)}
                                    onChange={(event) => setTimelineSpeed(parseTimelineSpeed(event.target.value))}
                                    disabled={timelineDisabled}
                                >
                                    <option value="0.5">0.5x</option>
                                    <option value="1">1x</option>
                                    <option value="2">2x</option>
                                </select>
                            </label>

                            <label className="timeline-controls__loop">
                                <input
                                    type="checkbox"
                                    checked={timelineLoopEnabled}
                                    onChange={(event) => setTimelineLoopEnabled(event.target.checked)}
                                    disabled={timelineDisabled}
                                />
                                Loop
                            </label>
                        </div>

                        {dateSnapNote ? <p className="timeline-note">{dateSnapNote}</p> : null}
                    </section>
                ) : null}

                {err ? (
                    <div className="error-alert" role="alert">
                        <span className="error-alert__icon" aria-hidden="true">
                            !
                        </span>
                        <div>
                            <div className="error-alert__title">Analysis unavailable</div>
                            <p>{err}</p>
                        </div>
                    </div>
                ) : null}

                {!res && !err && !selectedReef ? (
                    <div className="results-empty">
                        Select a location on the reef map to generate a bleaching risk estimate.
                    </div>
                ) : null}

                {res ? (
                    <div className="results-content">
                        <section className="risk-summary">
                            <span className={`risk-badge ${riskBand.badgeClassName}`}>{riskBand.label} Risk</span>
                            <div className="risk-summary__percent">{riskText}</div>
                            <div className="risk-meter" aria-hidden="true">
                                <div className="risk-meter__fill" style={{ width: `${riskWidth}%` }} />
                            </div>
                        </section>

                        <section className="metric-grid">
                            <div className="metric-card">
                                <span className="metric-card__label">Risk Flag</span>
                                <strong>{res.risk_flag === 1 ? "High" : "Low"}</strong>
                            </div>
                            <div className="metric-card">
                                <span className="metric-card__label">DHW</span>
                                <strong>{Number(res.dhw).toFixed(2)}</strong>
                            </div>
                            <div className="metric-card">
                                <span className="metric-card__label">HotSpot</span>
                                <strong>{Number(res.hotspot).toFixed(2)}</strong>
                            </div>
                        </section>

                        <section className="snap-section">
                            <h3>Snap Distance</h3>
                            {res.snapped ? (
                                <p>
                                    Snapped <strong>{Number(res.snap_km).toFixed(2)} km</strong> to nearest reef grid
                                    cell.
                                </p>
                            ) : (
                                <p>No snapping required.</p>
                            )}
                        </section>

                        <section className="coords-section">
                            <div>
                                <span>Input</span>
                                <strong>
                                    {res.input_lat.toFixed(4)}, {res.input_lon.toFixed(4)}
                                </strong>
                            </div>
                            <div>
                                <span>Used</span>
                                <strong>
                                    {res.used_lat.toFixed(4)}, {res.used_lon.toFixed(4)}
                                </strong>
                            </div>
                            <div>
                                <span>Date</span>
                                <strong>{res.date}</strong>
                            </div>
                        </section>
                    </div>
                ) : null}
            </aside>
        </section>
    );
}

