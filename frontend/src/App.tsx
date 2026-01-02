import { useEffect, useMemo, useRef, useState } from "react";
import {
    MapContainer,
    TileLayer,
    CircleMarker,
    Popup,
    Tooltip,
    useMapEvents,
} from "react-leaflet";
import type { LatLngExpression, LeafletMouseEvent, PathOptions } from "leaflet";
import "./App.css";

// local
const ESTIMATE_URL =
    import.meta.env.VITE_ESTIMATE_URL ?? "http://127.0.0.1:8000/estimate";
const NEAREST_REEF_URL =
    import.meta.env.VITE_NEAREST_REEF_URL ?? "http://127.0.0.1:8000/nearest-reef";
const AVAILABLE_DATES_FOR_URL =
    import.meta.env.VITE_AVAILABLE_DATES_FOR_URL ??
    "http://127.0.0.1:8000/available-dates-for";
const ESTIMATE_FROM_FEATURES_URL =
    import.meta.env.VITE_ESTIMATE_FROM_FEATURES_URL ??
    "http://127.0.0.1:8000/estimate-from-features";
const SENSITIVITY_URL =
    import.meta.env.VITE_SENSITIVITY_URL ?? "http://127.0.0.1:8000/sensitivity";

const LS_HIDE_INTRO = "cbr_hide_intro";
const LS_DATES_CACHE = "cbr_dates_cache_v1";
const LS_LAST_SEL = "cbr_last_sel_v1";

type EstimateResponse = {
    lat: number;
    lon: number;
    date: string;
    dhw: number;
    hotspot: number;
    risk_prob: number;
    risk_flag: number;
};

type SavedEstimate = EstimateResponse & { id: string; kind: "pin" | "sample" };

type NearestReef = {
    lat: number;
    lon: number;
    distance_km: number;
};

type AvailableDatesForResponse = {
    lat: number;
    lon: number;
    count: number;
    dates: string[];
};

type FeatureEstimateResponse = {
    risk_prob: number;
    risk_flag: number;
};

type SensitivityResponse = {
    base: number;
    dhw_step: number;
    hotspot_step: number;
    delta_dhw: number;
    delta_hotspot: number;
};

function MapClickHandler({ onSelect }: { onSelect: (lat: number, lon: number) => void }) {
    useMapEvents({
        click(e: LeafletMouseEvent) {
            onSelect(e.latlng.lat, e.latlng.lng);
        },
    });
    return null;
}

function riskLabel(p: number): string {
    if (p >= 0.8) return "🟥 Very High";
    if (p >= 0.5) return "🟧 High";
    if (p >= 0.2) return "🟨 Moderate";
    return "🟩 Low";
}

function riskColor(p: number): string {
    if (p >= 0.8) return "#ef4444";
    if (p >= 0.5) return "#f97316";
    if (p >= 0.2) return "#eab308";
    return "#22c55e";
}

function fmt(n: number): string {
    if (!Number.isFinite(n)) return "—";
    return n.toFixed(2);
}

function fmt3(n: number): string {
    if (!Number.isFinite(n)) return "—";
    return n.toFixed(3);
}

function clamp(x: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, x));
}

function radiusFromDhw(dhw: number, kind: "pin" | "sample"): number {
    const d = Number.isFinite(dhw) ? dhw : 0;
    const r = 4 + 1.4 * Math.sqrt(Math.max(0, d));
    const base = clamp(r, 4, 14);
    return kind === "sample" ? base * 0.75 : base;
}

function signFmt(x: number): string {
    if (!Number.isFinite(x)) return "—";
    const s = x >= 0 ? "+" : "";
    return `${s}${x.toFixed(3)}`;
}

function keyFor(lat: number, lon: number): string {
    return `${lat.toFixed(3)},${lon.toFixed(3)}`;
}

function safeLoadDatesCache(): Map<string, string[]> {
    try {
        const raw = localStorage.getItem(LS_DATES_CACHE);
        if (!raw) return new Map();
        const arr = JSON.parse(raw) as Array<[string, string[]]>;
        if (!Array.isArray(arr)) return new Map();
        const m = new Map<string, string[]>();
        for (const pair of arr) {
            if (!Array.isArray(pair) || pair.length !== 2) continue;
            const k = pair[0];
            const v = pair[1];
            if (typeof k !== "string" || !Array.isArray(v)) continue;
            m.set(k, v.filter((x) => typeof x === "string"));
        }
        return m;
    } catch {
        return new Map();
    }
}

function persistDatesCache(m: Map<string, string[]>) {
    try {
        const arr = Array.from(m.entries());
        localStorage.setItem(LS_DATES_CACHE, JSON.stringify(arr));
    } catch {
        // ignore (storage full / blocked)
    }
}

function safeLoadLastSel():
    | { lat: number; lon: number; date: string }
    | null {
    try {
        const raw = localStorage.getItem(LS_LAST_SEL);
        if (!raw) return null;
        const o = JSON.parse(raw) as any;
        const la = Number(o?.lat);
        const lo = Number(o?.lon);
        const d = typeof o?.date === "string" ? o.date : "";
        if (!Number.isFinite(la) || !Number.isFinite(lo)) return null;
        return { lat: la, lon: lo, date: d };
    } catch {
        return null;
    }
}

function persistLastSel(lat: number, lon: number, date: string) {
    try {
        localStorage.setItem(LS_LAST_SEL, JSON.stringify({ lat, lon, date }));
    } catch {
        // ignore
    }
}

async function fetchEstimate(lat: number, lon: number, date: string) {
    const res = await fetch(ESTIMATE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lon, date }),
    });

    if (!res.ok) {
        let detail = "Estimate failed.";
        try {
            const err = await res.json();
            if (err?.detail) detail = String(err.detail);
        } catch {}
        throw new Error(detail);
    }

    return (await res.json()) as EstimateResponse;
}

async function fetchNearestReef(lat: number, lon: number) {
    const url = `${NEAREST_REEF_URL}?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("nearest reef lookup failed.");
    return (await res.json()) as NearestReef;
}

async function fetchAvailableDatesFor(lat: number, lon: number) {
    const url = `${AVAILABLE_DATES_FOR_URL}?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("No valid dates for this reef :(");
    return (await res.json()) as AvailableDatesForResponse;
}

async function fetchEstimateFromFeatures(lat: number, lon: number, dhw: number, hotspot: number) {
    const res = await fetch(ESTIMATE_FROM_FEATURES_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lon, dhw, hotspot }),
    });

    if (!res.ok) {
        let detail = "Model sandbox call failed.";
        try {
            const err = await res.json();
            if (err?.detail) detail = String(err.detail);
        } catch {}
        throw new Error(detail);
    }

    return (await res.json()) as FeatureEstimateResponse;
}

async function fetchSensitivity(lat: number, lon: number, dhw: number, hotspot: number) {
    const res = await fetch(SENSITIVITY_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lon, dhw, hotspot }),
    });

    if (!res.ok) {
        let detail = "Sensitivity call failed.";
        try {
            const err = await res.json();
            if (err?.detail) detail = String(err.detail);
        } catch {}
        throw new Error(detail);
    }

    return (await res.json()) as SensitivityResponse;
}

function IntroOverlay({
                          open,
                          onClose,
                      }: {
    open: boolean;
    onClose: (dontShowAgain: boolean) => void;
}) {
    const [dontShowAgain, setDontShowAgain] = useState<boolean>(false);

    if (!open) return null;

    return (
        <div
            style={{
                position: "fixed",
                inset: 0,
                background: "rgba(0,0,0,0.75)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                zIndex: 9999,
                padding: 18,
            }}
        >
            <div
                style={{
                    width: "min(880px, 100%)",
                    background: "#0b1220",
                    border: "1px solid rgba(255,255,255,0.10)",
                    borderRadius: 16,
                    padding: 18,
                    color: "white",
                    fontFamily: "system-ui",
                }}
            >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
                    <div>
                        <div style={{ fontSize: 22, fontWeight: 950 }}>what you’re looking at</div>
                        <div style={{ opacity: 0.85, marginTop: 6, lineHeight: 1.45 }}>
                            this app estimates coral bleaching risk using a neural network. i used the noaa database for this proejct.
                        </div>
                    </div>

                    <button
                        onClick={() => onClose(dontShowAgain)}
                        style={{
                            padding: "0.6rem 0.8rem",
                            borderRadius: 12,
                            border: "1px solid rgba(255,255,255,0.18)",
                            background: "rgba(255,255,255,0.06)",
                            color: "white",
                            fontWeight: 900,
                            cursor: "pointer",
                            whiteSpace: "nowrap",
                        }}
                    >
                        enter →
                    </button>
                </div>

                <div
                    style={{
                        marginTop: 14,
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr",
                        gap: 12,
                    }}
                >
                    <div style={{ background: "rgba(255,255,255,0.05)", borderRadius: 14, padding: 12 }}>
                        <div style={{ fontWeight: 950, marginBottom: 6 }}>core flow</div>
                        <div style={{ opacity: 0.9, lineHeight: 1.5, fontSize: "0.95rem" }}>
                            <div>1) click on the map → snaps to nearest reef</div>
                            <div>2) pick a date that’s valid at that reef</div>
                            <div>3) noaa provides: dhw + hotspot</div>
                            <div>4) neural net outputs: risk_prob (0→1)</div>
                        </div>
                    </div>

                    <div style={{ background: "rgba(255,255,255,0.05)", borderRadius: 14, padding: 12 }}>
                        <div style={{ fontWeight: 950, marginBottom: 6 }}>map + visuals</div>
                        <div style={{ opacity: 0.9, lineHeight: 1.5, fontSize: "0.95rem" }}>
                            <div><b>color</b> = risk_prob</div>
                            <div><b>circle size</b> = dhw (more heat stress → bigger)</div>
                            <div><b>samples</b> = extra points around the reef (mini heatmap)</div>
                        </div>
                    </div>

                    <div style={{ background: "rgba(255,255,255,0.05)", borderRadius: 14, padding: 12 }}>
                        <div style={{ fontWeight: 950, marginBottom: 6 }}>what dhw + hotspot mean</div>
                        <div style={{ opacity: 0.9, lineHeight: 1.5, fontSize: "0.95rem" }}>
                            <div><b>dhw</b> (degree heating weeks): accumulated heat stress over recent weeks</div>
                            <div><b>hotspot</b>: current temperature anomaly vs a baseline</div>
                        </div>
                    </div>

                    <div style={{ background: "rgba(255,255,255,0.05)", borderRadius: 14, padding: 12 }}>
                        <div style={{ fontWeight: 950, marginBottom: 6 }}>model sandbox (ml flex)</div>
                        <div style={{ opacity: 0.9, lineHeight: 1.5, fontSize: "0.95rem" }}>
                            <div>tweak dhw/hotspot sliders to probe the model directly</div>
                            <div>use the threshold to see where the model “flips”</div>
                        </div>
                    </div>
                </div>

                <div style={{ marginTop: 12, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                    <label style={{ display: "flex", alignItems: "center", gap: 10, opacity: 0.9 }}>
                        <input
                            type="checkbox"
                            checked={dontShowAgain}
                            onChange={(e) => setDontShowAgain(e.target.checked)}
                        />
                        don’t show this again
                    </label>

                    <div style={{ opacity: 0.75, fontSize: "0.9rem" }}>
                        you can reopen this later with the “?” button.
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function App() {
    const [lat, setLat] = useState<number | null>(null);
    const [lon, setLon] = useState<number | null>(null);

    const [availableDates, setAvailableDates] = useState<string[]>([]);
    const [datesLoading, setDatesLoading] = useState<boolean>(false);
    const [date, setDate] = useState<string>("");

    const [saved, setSaved] = useState<SavedEstimate[]>([]);
    const [errorMsg, setErrorMsg] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);

    const [snapLoading, setSnapLoading] = useState<boolean>(false);
    const [nearestInfo, setNearestInfo] = useState<NearestReef | null>(null);

    const [autoSample, setAutoSample] = useState<boolean>(false);
    const [gridSize, setGridSize] = useState<number>(3);
    const [gridStepDeg, setGridStepDeg] = useState<number>(1.0);

    const [datesCache, setDatesCache] = useState<Map<string, string[]>>(new Map());

    const [introOpen, setIntroOpen] = useState<boolean>(false);

    const [sandboxOpen, setSandboxOpen] = useState<boolean>(false);
    const [sbDhw, setSbDhw] = useState<number>(0);
    const [sbHotspot, setSbHotspot] = useState<number>(0);
    const [sbProb, setSbProb] = useState<number | null>(null);
    const [sbLoading, setSbLoading] = useState<boolean>(false);

    const [sbSens, setSbSens] = useState<SensitivityResponse | null>(null);
    const [sbSensLoading, setSbSensLoading] = useState<boolean>(false);

    const [riskThresh, setRiskThresh] = useState<number>(0.6);
    const [showSamples, setShowSamples] = useState<boolean>(true);

    const [clickGhost, setClickGhost] = useState<{ lat: number; lon: number } | null>(null);

    const sbTimerRef = useRef<number | null>(null);

    const latestPin = useMemo(
        () => saved.find((x) => x.kind === "pin") ?? null,
        [saved]
    );

    const samples = useMemo(
        () => saved.filter((x) => x.kind === "sample"),
        [saved]
    );

    const sampleAlertPct = useMemo(() => {
        if (!samples.length) return null;
        const hits = samples.filter((s) => s.risk_prob >= riskThresh).length;
        return (hits / samples.length) * 100;
    }, [samples, riskThresh]);

    useEffect(() => {
        const hide = localStorage.getItem(LS_HIDE_INTRO) === "1";
        setIntroOpen(!hide);

        const dc = safeLoadDatesCache();
        setDatesCache(dc);

        const last = safeLoadLastSel();
        if (last) {
            setLat(last.lat);
            setLon(last.lon);
            setDate(last.date ?? "");
        }
    }, []);

    useEffect(() => {
        // keep sandbox synced to the latest pin so the sliders start at a real baseline
        if (!latestPin) return;
        setSbDhw(latestPin.dhw);
        setSbHotspot(latestPin.hotspot);
        setSbProb(latestPin.risk_prob);
        setSbSens(null);
    }, [latestPin]);

    useEffect(() => {
        //  last selection
        if (lat === null || lon === null) return;
        persistLastSel(lat, lon, date);
    }, [lat, lon, date]);

    useEffect(() => {
        // cache dates
        persistDatesCache(datesCache);
    }, [datesCache]);

    useEffect(() => {
        // if we restored a reef from localStorage, load its valid dates automatically
        const boot = async () => {
            if (lat === null || lon === null) return;
            if (availableDates.length) return;

            setDatesLoading(true);
            try {
                const k = keyFor(lat, lon);
                const hit = datesCache.get(k);
                if (hit) {
                    setAvailableDates(hit);
                    if (!date && hit.length) setDate(hit[hit.length - 1]);
                    return;
                }

                const d = await fetchAvailableDatesFor(lat, lon);
                const ds = Array.isArray(d.dates) ? d.dates : [];
                setAvailableDates(ds);
                if (!date && ds.length) setDate(ds[ds.length - 1]);

                setDatesCache((prev) => {
                    const next = new Map(prev);
                    next.set(k, ds);
                    return next;
                });
            } catch (e: any) {
                setAvailableDates([]);
                setErrorMsg(e?.message ?? "couldn't load dates.");
            } finally {
                setDatesLoading(false);
            }
        };

        void boot();

    }, [lat, lon]);

    const mapCenter: LatLngExpression = [0, 0];

    const markerStyles = useMemo(() => {
        const m = new Map<string, PathOptions>();
        for (const p of saved) {
            const c = riskColor(p.risk_prob);
            const isLatest = latestPin ? p.id === latestPin.id : false;

            m.set(p.id, {
                color: c,
                fillColor: c,
                fillOpacity: p.kind === "sample" ? 0.45 : 0.75,
                weight: isLatest ? 3 : 1, // make the newest pin pop
                opacity: 1,
            });
        }
        return m;
    }, [saved, latestPin]);

    const runSamplingGrid = async (centerLat: number, centerLon: number, d: string) => {
        const n = clamp(gridSize, 3, 9);
        const step = clamp(gridStepDeg, 0.25, 4);

        const half = Math.floor(n / 2);
        const pts: Array<{ la: number; lo: number }> = [];
        for (let i = -half; i <= half; i++) {
            for (let j = -half; j <= half; j++) {
                if (i === 0 && j === 0) continue;
                pts.push({ la: centerLat + i * step, lo: centerLon + j * step });
            }
        }

        setSaved((prev) => prev.filter((x) => x.kind === "pin"));

        const newSamples: SavedEstimate[] = [];
        for (const p of pts) {
            try {
                const data = await fetchEstimate(p.la, p.lo, d);
                newSamples.push({
                    ...data,
                    id: `${Date.now()}-${Math.random().toString(16).slice(2)}-${p.la.toFixed(3)}-${p.lo.toFixed(3)}`,
                    kind: "sample",
                });
            } catch {
                // skip if fail
            }
        }

        setSaved((prev) => {
            const pins = prev.filter((x) => x.kind === "pin");
            return [...pins, ...newSamples].slice(0, 200);
        });
    };

    const handleEstimatePin = async () => {
        if (lat === null || lon === null || !date) {
            setErrorMsg("Select a reef and a valid date for that reef.");
            return;
        }

        setErrorMsg("");
        setLoading(true);
        try {
            const data = await fetchEstimate(lat, lon, date);
            const item: SavedEstimate = {
                ...data,
                id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
                kind: "pin",
            };

            setSaved((prev) => {
                const withoutSamples = prev.filter((x) => x.kind === "pin");
                return [item, ...withoutSamples].slice(0, 25);
            });

            if (autoSample) {
                await runSamplingGrid(data.lat, data.lon, date);
            }
        } catch (e: any) {
            setErrorMsg(e?.message ?? "Estimate failed.");
        } finally {
            setLoading(false);
        }
    };

    const reefSelected = lat !== null && lon !== null;

    const clearAll = () => {
        setSaved([]);
        setNearestInfo(null);
        setLat(null);
        setLon(null);
        setAvailableDates([]);
        setDate("");
        setSbSens(null);
        setSbProb(null);
        setClickGhost(null);

        try {
            localStorage.removeItem(LS_LAST_SEL);
        } catch {
            // ignore
        }
    };

    const scheduleSandboxRun = (nextDhw: number, nextHotspot: number) => {
        if (sbTimerRef.current) window.clearTimeout(sbTimerRef.current);
        if (lat === null || lon === null) return;

        sbTimerRef.current = window.setTimeout(async () => {
            setSbLoading(true);
            setSbSensLoading(true);
            setErrorMsg("");
            try {
                const out = await fetchEstimateFromFeatures(lat, lon, nextDhw, nextHotspot);
                setSbProb(out.risk_prob);

                const sens = await fetchSensitivity(lat, lon, nextDhw, nextHotspot);
                setSbSens(sens);
            } catch (e: any) {
                setSbProb(null);
                setSbSens(null);
                setErrorMsg(e?.message ?? "model sandbox call failed.");
            } finally {
                setSbLoading(false);
                setSbSensLoading(false);
            }
        }, 250);
    };

    const sandboxBadge = useMemo(() => {
        if (sbProb === null) return null;
        const alert = sbProb >= riskThresh;
        return {
            text: alert ? "ALERT" : "OK",
            bg: alert ? "rgba(239, 68, 68, 0.18)" : "rgba(34, 197, 94, 0.16)",
            border: alert ? "rgba(239, 68, 68, 0.35)" : "rgba(34, 197, 94, 0.32)",
            color: alert ? "#fecaca" : "#bbf7d0",
        };
    }, [sbProb, riskThresh]);

    const visiblePoints = useMemo(() => {
        if (showSamples) return saved;
        return saved.filter((x) => x.kind === "pin");
    }, [saved, showSamples]);

    return (
        <div style={{ display: "flex", height: "100vh", width: "100vw" }}>
            <IntroOverlay
                open={introOpen}
                onClose={(dontShowAgain) => {
                    if (dontShowAgain) localStorage.setItem(LS_HIDE_INTRO, "1");
                    setIntroOpen(false);
                }}
            />

            <div
                style={{
                    width: 380,
                    padding: "1rem",
                    background: "#111827",
                    color: "white",
                    fontFamily: "system-ui",
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.8rem",
                    boxSizing: "border-box",
                    overflow: "auto",
                    position: "relative",
                }}
            >
                <button
                    onClick={() => setIntroOpen(true)}
                    style={{
                        position: "absolute",
                        top: 12,
                        right: 12,
                        width: 36,
                        height: 36,
                        borderRadius: 999,
                        border: "1px solid rgba(255,255,255,0.18)",
                        background: "rgba(255,255,255,0.06)",
                        color: "white",
                        fontWeight: 950,
                        cursor: "pointer",
                    }}
                    aria-label="help"
                    title="help"
                >
                    ?
                </button>

                <div>
                    <h2 style={{ margin: 0 }}>Estimate Bleaching Risk </h2>
                    <div style={{ opacity: 0.85, marginTop: 6, fontSize: "0.92rem" }}>
                        historical risk estimate using NOAA DHW + hotspot for the selected date.
                    </div>
                </div>

                {snapLoading && <div style={{ opacity: 0.9 }}>Snapping to nearest reef…</div>}
                {nearestInfo && (
                    <div style={{ fontSize: "0.9rem", opacity: 0.9 }}>
                        <b>Snapped distance:</b> {nearestInfo.distance_km.toFixed(2)} km
                    </div>
                )}

                <div style={{ display: "flex", gap: 10 }}>
                    <button
                        onClick={handleEstimatePin}
                        disabled={loading || datesLoading || !reefSelected || !date}
                        style={{
                            flex: 1,
                            padding: "0.7rem",
                            borderRadius: 10,
                            border: "none",
                            background: loading ? "#059669" : "#10b981",
                            color: "white",
                            fontWeight: 900,
                            cursor: loading ? "not-allowed" : "pointer",
                            opacity: loading ? 0.85 : 1,
                        }}
                    >
                        {loading ? "Estimating..." : "Estimate Risk"}
                    </button>

                    <button
                        onClick={clearAll}
                        disabled={!saved.length && !reefSelected}
                        style={{
                            padding: "0.7rem",
                            borderRadius: 10,
                            border: "1px solid rgba(255,255,255,0.18)",
                            background: "transparent",
                            color: "white",
                            fontWeight: 800,
                            cursor: saved.length || reefSelected ? "pointer" : "not-allowed",
                            opacity: saved.length || reefSelected ? 1 : 0.55,
                        }}
                    >
                        Clear
                    </button>
                </div>

                {errorMsg && <div style={{ color: "#f87171", fontWeight: 900 }}>{errorMsg}</div>}

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    <div>
                        <div style={{ fontWeight: 800, marginBottom: 6 }}>Latitude</div>
                        <input
                            type="number"
                            step="0.01"
                            value={lat ?? ""}
                            onChange={(e) => {
                                setNearestInfo(null);
                                setLat(e.target.value === "" ? null : Number(e.target.value));
                                setAvailableDates([]);
                                setDate("");
                            }}
                            style={{ width: "100%", padding: "0.55rem", borderRadius: 8, border: "none" }}
                        />
                    </div>

                    <div>
                        <div style={{ fontWeight: 800, marginBottom: 6 }}>Longitude</div>
                        <input
                            type="number"
                            step="0.01"
                            value={lon ?? ""}
                            onChange={(e) => {
                                setNearestInfo(null);
                                setLon(e.target.value === "" ? null : Number(e.target.value));
                                setAvailableDates([]);
                                setDate("");
                            }}
                            style={{ width: "100%", padding: "0.55rem", borderRadius: 8, border: "none" }}
                        />
                    </div>
                </div>

                <div>
                    <div style={{ fontWeight: 800, marginBottom: 6 }}>Date (valid at this reef)</div>
                    <select
                        value={date}
                        onChange={(e) => setDate(e.target.value)}
                        disabled={datesLoading || !reefSelected || !availableDates.length}
                        style={{ width: "100%", padding: "0.55rem", borderRadius: 8, border: "none" }}
                    >
                        {datesLoading ? (
                            <option value="">Loading dates…</option>
                        ) : !reefSelected ? (
                            <option value="">Click a reef first</option>
                        ) : !availableDates.length ? (
                            <option value="">No valid dates for this reef</option>
                        ) : (
                            availableDates
                                .slice()
                                .reverse()
                                .map((d) => (
                                    <option key={d} value={d}>
                                        {d}
                                    </option>
                                ))
                        )}
                    </select>
                </div>

                <div
                    style={{
                        background: "#0b1220",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: 12,
                        padding: 12,
                        display: "flex",
                        flexDirection: "column",
                        gap: 10,
                    }}
                >
                    <div style={{ fontWeight: 900 }}>Mini Heatmap Sampling</div>

                    <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <input type="checkbox" checked={autoSample} onChange={(e) => setAutoSample(e.target.checked)} />
                        Auto-sample a grid after Estimate
                    </label>

                    <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <input type="checkbox" checked={showSamples} onChange={(e) => setShowSamples(e.target.checked)} />
                        Show samples on map
                    </label>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                        <div>
                            <div style={{ fontWeight: 700, marginBottom: 6 }}>Grid size</div>
                            <select
                                value={gridSize}
                                onChange={(e) => setGridSize(Number(e.target.value))}
                                style={{ width: "100%", padding: "0.55rem", borderRadius: 8, border: "none" }}
                            >
                                <option value={3}>3 × 3</option>
                                <option value={5}>5 × 5</option>
                                <option value={7}>7 × 7</option>
                            </select>
                        </div>

                        <div>
                            <div style={{ fontWeight: 700, marginBottom: 6 }}>Spacing (°)</div>
                            <input
                                type="number"
                                step="0.25"
                                value={gridStepDeg}
                                onChange={(e) => setGridStepDeg(Number(e.target.value))}
                                style={{ width: "100%", padding: "0.55rem", borderRadius: 8, border: "none" }}
                            />
                        </div>
                    </div>

                    <button
                        onClick={async () => {
                            if (lat === null || lon === null || !date) {
                                setErrorMsg("Pick a reef and a valid date first.");
                                return;
                            }
                            setErrorMsg("");
                            setLoading(true);
                            try {
                                await runSamplingGrid(lat, lon, date);
                            } finally {
                                setLoading(false);
                            }
                        }}
                        disabled={loading || !reefSelected || !date}
                        style={{
                            padding: "0.65rem",
                            borderRadius: 10,
                            border: "1px solid rgba(255,255,255,0.18)",
                            background: "rgba(255,255,255,0.06)",
                            color: "white",
                            fontWeight: 900,
                            cursor: loading ? "not-allowed" : "pointer",
                        }}
                    >
                        {loading ? "Sampling..." : "Sample Grid Now"}
                    </button>

                    <div style={{ fontSize: "0.86rem", opacity: 0.85 }}>
                        samples are smaller circles; hover for a quick read.
                    </div>
                </div>

                <div
                    style={{
                        background: "#1f2937",
                        padding: "1rem",
                        borderRadius: 12,
                        fontSize: "0.92rem",
                        lineHeight: 1.45,
                        border: "1px solid rgba(255,255,255,0.08)",
                    }}
                >
                    <div style={{ fontWeight: 950, marginBottom: 8 }}>
                        {latestPin ? "Latest Estimate" : "No estimate yet"}
                    </div>

                    {latestPin ? (
                        <>
                            <div><b>Date:</b> {latestPin.date}</div>
                            <div><b>Lat:</b> {latestPin.lat.toFixed(2)}</div>
                            <div><b>Lon:</b> {latestPin.lon.toFixed(2)}</div>
                            <div style={{ marginTop: 8 }}><b>DHW:</b> {fmt(latestPin.dhw)}</div>
                            <div><b>Hotspot:</b> {fmt(latestPin.hotspot)}</div>
                            <div style={{ marginTop: 8 }}>
                                <b>Risk:</b> {latestPin.risk_prob.toFixed(2)} ({riskLabel(latestPin.risk_prob)})
                            </div>
                        </>
                    ) : (
                        <div style={{ opacity: 0.85 }}>
                            Click a reef, pick a valid date, then estimate.
                        </div>
                    )}
                </div>

                <div
                    style={{
                        background: "#0b1220",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: 12,
                        padding: 12,
                    }}
                >
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                        <div>
                            <div style={{ fontWeight: 950 }}>Model Sandbox</div>
                            <div style={{ opacity: 0.85, fontSize: "0.86rem", marginTop: 4 }}>
                                tweak inputs and probe the decision boundary
                            </div>
                        </div>

                        <button
                            onClick={() => setSandboxOpen((v) => !v)}
                            disabled={!latestPin || lat === null || lon === null}
                            style={{
                                padding: "0.55rem 0.7rem",
                                borderRadius: 10,
                                border: "1px solid rgba(255,255,255,0.18)",
                                background: "rgba(255,255,255,0.06)",
                                color: "white",
                                fontWeight: 900,
                                cursor: latestPin && lat !== null && lon !== null ? "pointer" : "not-allowed",
                                opacity: latestPin && lat !== null && lon !== null ? 1 : 0.6,
                                whiteSpace: "nowrap",
                            }}
                        >
                            {sandboxOpen ? "Hide" : "Open"}
                        </button>
                    </div>

                    {!latestPin ? (
                        <div style={{ marginTop: 10, opacity: 0.85, fontSize: "0.9rem" }}>
                            run one estimate first so we have a baseline.
                        </div>
                    ) : sandboxOpen ? (
                        <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 12 }}>
                            <div style={{ fontSize: "0.9rem", opacity: 0.9 }}>
                                <b>x</b> = [{fmt3(latestPin.lat)}, {fmt3(latestPin.lon)}, {fmt3(sbDhw)}, {fmt3(sbHotspot)}]
                            </div>

                            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                                    <div style={{ fontWeight: 800 }}>DHW</div>
                                    <input
                                        type="number"
                                        step="0.1"
                                        value={sbDhw}
                                        onChange={(e) => {
                                            const v = Number(e.target.value);
                                            setSbDhw(v);
                                            scheduleSandboxRun(v, sbHotspot);
                                        }}
                                        style={{ width: 110, padding: "0.45rem", borderRadius: 8, border: "none" }}
                                    />
                                </div>

                                <input
                                    type="range"
                                    min={0}
                                    max={Math.max(16, Math.ceil(latestPin.dhw + 8))}
                                    step={0.1}
                                    value={sbDhw}
                                    onChange={(e) => {
                                        const v = Number(e.target.value);
                                        setSbDhw(v);
                                        scheduleSandboxRun(v, sbHotspot);
                                    }}
                                />
                            </div>

                            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                                    <div style={{ fontWeight: 800 }}>Hotspot</div>
                                    <input
                                        type="number"
                                        step="0.1"
                                        value={sbHotspot}
                                        onChange={(e) => {
                                            const v = Number(e.target.value);
                                            setSbHotspot(v);
                                            scheduleSandboxRun(sbDhw, v);
                                        }}
                                        style={{ width: 110, padding: "0.45rem", borderRadius: 8, border: "none" }}
                                    />
                                </div>

                                <input
                                    type="range"
                                    min={Math.floor(latestPin.hotspot - 3)}
                                    max={Math.ceil(latestPin.hotspot + 3)}
                                    step={0.1}
                                    value={sbHotspot}
                                    onChange={(e) => {
                                        const v = Number(e.target.value);
                                        setSbHotspot(v);
                                        scheduleSandboxRun(sbDhw, v);
                                    }}
                                />
                            </div>

                            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                                    <div style={{ fontWeight: 900 }}>Risk threshold</div>
                                    <div style={{ fontWeight: 900, opacity: 0.9 }}>{riskThresh.toFixed(2)}</div>
                                </div>

                                <input
                                    type="range"
                                    min={0.05}
                                    max={0.95}
                                    step={0.01}
                                    value={riskThresh}
                                    onChange={(e) => setRiskThresh(Number(e.target.value))}
                                />

                                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                                    <div style={{ opacity: 0.85, fontSize: "0.88rem" }}>
                                        flips at: risk ≥ {riskThresh.toFixed(2)}
                                    </div>
                                    <div style={{ opacity: 0.85, fontSize: "0.88rem" }}>
                                        {sampleAlertPct === null ? "no samples" : `${sampleAlertPct.toFixed(0)}% of samples above`}
                                    </div>
                                </div>
                            </div>

                            <div
                                style={{
                                    background: "rgba(255,255,255,0.06)",
                                    border: "1px solid rgba(255,255,255,0.10)",
                                    borderRadius: 12,
                                    padding: 12,
                                    fontSize: "0.92rem",
                                }}
                            >
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                                    <div style={{ fontWeight: 950 }}>Model output</div>
                                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                        {sandboxBadge && (
                                            <span
                                                style={{
                                                    padding: "0.2rem 0.55rem",
                                                    borderRadius: 999,
                                                    background: sandboxBadge.bg,
                                                    border: `1px solid ${sandboxBadge.border}`,
                                                    color: sandboxBadge.color,
                                                    fontWeight: 950,
                                                    fontSize: "0.82rem",
                                                    letterSpacing: 0.4,
                                                }}
                                            >
                                                {sandboxBadge.text}
                                            </span>
                                        )}
                                        <div style={{ opacity: 0.85 }}>
                                            {sbLoading ? "running..." : "live"}
                                        </div>
                                    </div>
                                </div>

                                {sbProb === null ? (
                                    <div style={{ marginTop: 8, opacity: 0.85 }}>no output yet</div>
                                ) : (
                                    <div style={{ marginTop: 8, lineHeight: 1.45 }}>
                                        <div>
                                            <b>risk:</b> {sbProb.toFixed(3)} ({riskLabel(sbProb)})
                                        </div>
                                        <div style={{ opacity: 0.9 }}>
                                            <b>delta vs baseline:</b> {signFmt(sbProb - latestPin.risk_prob)}
                                        </div>
                                        <div style={{ opacity: 0.9 }}>
                                            <b>threshold delta:</b> {signFmt(sbProb - riskThresh)}
                                        </div>

                                        <div style={{ marginTop: 10, opacity: 0.92 }}>
                                            <b>sensitivity</b>{" "}
                                            <span style={{ opacity: 0.75 }}>
                                                {sbSensLoading ? "(calc...)" : ""}
                                            </span>
                                        </div>

                                        {sbSens ? (
                                            <div style={{ marginTop: 6, opacity: 0.92 }}>
                                                <div>
                                                    +{sbSens.dhw_step} dhw → {signFmt(sbSens.delta_dhw)} risk
                                                </div>
                                                <div>
                                                    +{sbSens.hotspot_step} hotspot → {signFmt(sbSens.delta_hotspot)} risk
                                                </div>
                                            </div>
                                        ) : (
                                            <div style={{ marginTop: 6, opacity: 0.8 }}>
                                                move a slider to compute it
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    ) : null}
                </div>
            </div>

            <div style={{ flex: 1 }}>
                <div className="mapWrap">
                    <MapContainer
                        center={mapCenter}
                        zoom={2}
                        style={{ height: "100%", width: "100%" }}
                        worldCopyJump={true}
                    >
                        <TileLayer
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                            attribution="&copy; OpenStreetMap contributors"
                        />

                        <MapClickHandler
                            onSelect={async (la, lo) => {
                                setErrorMsg("");
                                setSnapLoading(true);
                                setDatesLoading(true);
                                setClickGhost({ lat: la, lon: lo });

                                try {
                                    const reef = await fetchNearestReef(la, lo);
                                    setLat(reef.lat);
                                    setLon(reef.lon);
                                    setNearestInfo(reef);

                                    const k = keyFor(reef.lat, reef.lon);
                                    const hit = datesCache.get(k);

                                    if (hit) {
                                        setAvailableDates(hit);
                                        setDate(hit.length ? hit[hit.length - 1] : "");
                                    } else {
                                        const d = await fetchAvailableDatesFor(reef.lat, reef.lon);
                                        const ds = Array.isArray(d.dates) ? d.dates : [];
                                        setAvailableDates(ds);
                                        setDate(ds.length ? ds[ds.length - 1] : "");

                                        setDatesCache((prev) => {
                                            const next = new Map(prev);
                                            next.set(k, ds);
                                            return next;
                                        });
                                    }

                                    persistLastSel(reef.lat, reef.lon, date);
                                } catch (e: any) {
                                    setAvailableDates([]);
                                    setDate("");
                                    setNearestInfo(null);
                                    setErrorMsg(e?.message ?? "couldn't load reef or dates.");
                                } finally {
                                    setSnapLoading(false);
                                    setDatesLoading(false);
                                    setClickGhost(null);
                                }
                            }}
                        />

                        {clickGhost && snapLoading && (
                            <CircleMarker
                                center={[clickGhost.lat, clickGhost.lon] as LatLngExpression}
                                radius={7}
                                pathOptions={{
                                    color: "#60a5fa",
                                    fillColor: "#60a5fa",
                                    fillOpacity: 0.18,
                                    weight: 2,
                                }}
                            >
                                <Tooltip direction="top" offset={[0, -6]} opacity={0.95}>
                                    <span style={{ fontFamily: "system-ui" }}>your click</span>
                                </Tooltip>
                            </CircleMarker>
                        )}

                        {visiblePoints.map((p) => {
                            const st =
                                markerStyles.get(p.id) ??
                                ({
                                    color: "cyan",
                                    fillColor: "cyan",
                                    fillOpacity: 0.75,
                                    weight: 1,
                                } as PathOptions);

                            const isLatest = latestPin ? p.id === latestPin.id : false;

                            return (
                                <CircleMarker
                                    key={p.id}
                                    center={[p.lat, p.lon] as LatLngExpression}
                                    radius={radiusFromDhw(p.dhw, p.kind)}
                                    pathOptions={st}
                                >
                                    <Tooltip direction="top" offset={[0, -6]} opacity={0.95}>
                                        <span style={{ fontFamily: "system-ui" }}>
                                            {p.kind === "pin" ? "pin" : "sample"} • {p.risk_prob.toFixed(2)} • {riskLabel(p.risk_prob)}
                                            {isLatest ? " • latest" : ""}
                                        </span>
                                    </Tooltip>

                                    <Popup>
                                        <div style={{ fontSize: "0.9rem" }}>
                                            <b>{p.kind === "pin" ? "Estimate" : "Sample"}</b>
                                            <br />
                                            Date: {p.date}
                                            <br />
                                            Risk: {p.risk_prob.toFixed(2)} ({riskLabel(p.risk_prob)})
                                            <br />
                                            DHW: {fmt(p.dhw)} | Hotspot: {fmt(p.hotspot)}
                                            <br />
                                            ({p.lat.toFixed(2)}, {p.lon.toFixed(2)})
                                            {p.kind === "sample" && (
                                                <>
                                                    <br />
                                                    <span style={{ opacity: 0.85 }}>
                                                        threshold: {p.risk_prob >= riskThresh ? "above" : "below"} ({riskThresh.toFixed(2)})
                                                    </span>
                                                </>
                                            )}
                                        </div>
                                    </Popup>
                                </CircleMarker>
                            );
                        })}
                    </MapContainer>

                    <div className="legend">
                        <div className="legendTitle">Legend</div>

                        <div className="legendRow">
                            <div className="legendDot" style={{ background: riskColor(0.1) }} />
                            <div>Low (0.00–0.19)</div>
                        </div>
                        <div className="legendRow">
                            <div className="legendDot" style={{ background: riskColor(0.3) }} />
                            <div>Moderate (0.20–0.49)</div>
                        </div>
                        <div className="legendRow">
                            <div className="legendDot" style={{ background: riskColor(0.6) }} />
                            <div>High (0.50–0.79)</div>
                        </div>
                        <div className="legendRow">
                            <div className="legendDot" style={{ background: riskColor(0.9) }} />
                            <div>Very High (0.80–1.00)</div>
                        </div>

                        <div style={{ marginTop: 10, opacity: 0.9 }}>
                            <div><b>Color</b> = risk</div>
                            <div><b>Size</b> = DHW</div>
                            <div style={{ marginTop: 6 }}>
                                <b>Threshold</b> = {riskThresh.toFixed(2)}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
