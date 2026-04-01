import { useCallback, useEffect, useRef, useState } from "react";
import MapEstimateLeaflet from "./components/map/MapEstimateLeaflet";
import WarmupBanner from "./components/ui/WarmupBanner";
import { getModelStatus, getSummary, warmBackend, type ModelStatusResponse, type SummaryResponse } from "./lib/api";
import type { ServerStatus } from "./types/server";

const GITHUB_URL = "https://github.com/aryasalem09/coral-bleaching-tracker";

function statusLabel(serverStatus: ServerStatus): string {
    if (serverStatus === "ready") return "Backend ready";
    if (serverStatus === "warming") return "Warming backend";
    if (serverStatus === "down") return "Backend unavailable";
    return "Checking backend";
}

function modelStatusLabel(modelStatus: ModelStatusResponse | null): string {
    if (!modelStatus) return "Checking model bundle";
    if (modelStatus.status === "ready") return "Model bundle ready";
    if (modelStatus.status === "invalid") return "Model bundle unavailable";
    return "Model bundle missing";
}

export default function App() {
    const [serverStatus, setServerStatus] = useState<ServerStatus>("unknown");
    const [summary, setSummary] = useState<SummaryResponse | null>(null);
    const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);
    const [warmElapsedSeconds, setWarmElapsedSeconds] = useState(0);
    const mountedRef = useRef(true);
    const serverStatusRef = useRef<ServerStatus>("unknown");
    const warmupPromiseRef = useRef<Promise<void> | null>(null);

    useEffect(() => {
        serverStatusRef.current = serverStatus;
    }, [serverStatus]);

    const refreshSummary = useCallback(async () => {
        try {
            const [nextSummary, nextModelStatus] = await Promise.all([getSummary(), getModelStatus()]);
            if (mountedRef.current) {
                setSummary(nextSummary);
                setModelStatus(nextModelStatus);
            }
        } catch {
            // summary is helpful, but not critical for first paint
        }
    }, []);

    const ensureBackendReady = useCallback(async () => {
        if (serverStatusRef.current === "ready") return;
        if (warmupPromiseRef.current) return warmupPromiseRef.current;

        setServerStatus("warming");
        warmupPromiseRef.current = warmBackend({
            onTick: ({ elapsedMs }) => {
                if (!mountedRef.current) return;
                setWarmElapsedSeconds(Math.floor(elapsedMs / 1000));
            },
        })
            .then(async () => {
                if (!mountedRef.current) return;
                setServerStatus("ready");
                setWarmElapsedSeconds(0);
                await refreshSummary();
            })
            .finally(() => {
                warmupPromiseRef.current = null;
            });

        return warmupPromiseRef.current;
    }, [refreshSummary]);

    useEffect(() => {
        mountedRef.current = true;
        const timer = window.setTimeout(() => {
            void ensureBackendReady().catch(() => {
                if (mountedRef.current) setServerStatus("down");
            });
        }, 0);
        return () => {
            mountedRef.current = false;
            window.clearTimeout(timer);
        };
    }, [ensureBackendReady]);

    return (
        <div className={`app-shell app-shell--${serverStatus}`}>
            <div className="ambient-orb ambient-orb--left" />
            <div className="ambient-orb ambient-orb--right" />
            <WarmupBanner visible={serverStatus === "warming"} elapsedSeconds={warmElapsedSeconds} />

            <header className="app-topbar">
                <div className="brand-block">
                    <p className="eyebrow">Coral Bleaching Tracker</p>
                    <h1>Observed outcomes, environmental stress, and supervised prediction in one honest workflow.</h1>
                    <p className="muted-copy">
                        This refactor separates recorded bleaching, transparent heat-stress outlooks, and a true
                        supervised site-month model driven by weekly NOAA heat-stress history so the app does not blur
                        observation, risk, and prediction.
                    </p>
                </div>

                <div className="topbar-actions">
                    <div className={`status-pill status-pill--${serverStatus}`}>{statusLabel(serverStatus)}</div>
                    <div className="status-pill status-pill--ghost" title={modelStatus?.message ?? summary?.model_status_message}>
                        {modelStatusLabel(modelStatus)}
                    </div>
                    <a className="link-button" href={GITHUB_URL} target="_blank" rel="noreferrer">
                        GitHub
                    </a>
                </div>
            </header>

            <MapEstimateLeaflet
                ensureBackendReady={ensureBackendReady}
                serverStatus={serverStatus}
                onServerReachable={() => setServerStatus("ready")}
                onServerDown={() => setServerStatus("down")}
                summary={summary}
                modelStatus={modelStatus}
            />
        </div>
    );
}
