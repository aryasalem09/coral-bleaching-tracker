import { useCallback, useEffect, useRef, useState } from "react";
import MapEstimateLeaflet from "./components/map/MapEstimateLeaflet";
import WarmupBanner from "./components/ui/WarmupBanner";
import OnboardingOverlay from "./components/ui/OnboardingOverlay";
import ShinyText from "./components/ui/ShinyText";
import GradientText from "./components/ui/GradientText";
import BlurText from "./components/ui/BlurText";
import ClickSpark from "./components/ui/ClickSpark";
import { getModelStatus, getSummary, warmBackend, type ModelStatusResponse, type SummaryResponse } from "./lib/api";
import type { ServerStatus } from "./types/server";

const GITHUB_URL = "https://github.com/aryasalem09/coral-bleaching-tracker";

function statusLabel(serverStatus: ServerStatus): string {
    if (serverStatus === "ready") return "Online";
    if (serverStatus === "warming") return "Warming";
    if (serverStatus === "down") return "Offline";
    return "Checking";
}

function modelStatusLabel(modelStatus: ModelStatusResponse | null): string {
    if (!modelStatus) return "Checking";
    if (modelStatus.status === "ready") return "Forecast ready";
    if (modelStatus.status === "invalid") return "Unavailable";
    return "Missing";
}

function StatusDot({ status }: { status: ServerStatus }) {
    const color =
        status === "ready" ? "#00d4aa" : status === "warming" ? "#f0a500" : status === "down" ? "#ff6b6b" : "#4e6678";
    return (
        <span
            className="status-dot"
            style={{
                background: color,
                boxShadow: status === "ready" ? `0 0 8px ${color}` : status === "warming" ? `0 0 8px ${color}` : "none",
            }}
        />
    );
}

export default function App() {
    const [serverStatus, setServerStatus] = useState<ServerStatus>("unknown");
    const [summary, setSummary] = useState<SummaryResponse | null>(null);
    const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);
    const [warmElapsedSeconds, setWarmElapsedSeconds] = useState(0);
    const [onboardingDone, setOnboardingDone] = useState(false);
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
            // non-critical
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
        <>
            <OnboardingOverlay onComplete={() => setOnboardingDone(true)} />

            <div className={`app-shell ${onboardingDone ? "app-shell--visible" : "app-shell--hidden"}`}>
                {/* Ambient glow layers */}
                <div className="ambient-glow ambient-glow--teal" />
                <div className="ambient-glow ambient-glow--coral" />
                <div className="ambient-glow ambient-glow--blue" />
                <div className="grain-overlay" />

                <WarmupBanner visible={serverStatus === "warming"} elapsedSeconds={warmElapsedSeconds} />

                {/* Floating top navbar */}
                <header className="floating-nav">
                    <div className="floating-nav__brand">
                        <ShinyText className="floating-nav__title" shimmerWidth={100} speed={4}>
                            Coral Bleaching Tracker
                        </ShinyText>
                    </div>

                    <div className="floating-nav__controls">
                        <div className="nav-status-pill">
                            <StatusDot status={serverStatus} />
                            <span>{statusLabel(serverStatus)}</span>
                        </div>
                        <div className="nav-status-pill">
                            <span>{modelStatusLabel(modelStatus)}</span>
                        </div>
                        <ClickSpark sparkColor="#00d4aa" sparkCount={6}>
                            <a className="nav-link" href={GITHUB_URL} target="_blank" rel="noreferrer">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" opacity="0.7">
                                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                                </svg>
                                GitHub
                            </a>
                        </ClickSpark>
                    </div>
                </header>

                {/* Hero strip below nav */}
                <div className="hero-strip">
                    <h1 className="hero-strip__title">
                        <GradientText as="span" from="#00d4aa" via="#3b82f6" to="#a78bfa" animate speed={6}>
                            Survey records, heat stress, and a 4-week forecast
                        </GradientText>
                    </h1>
                    <p className="hero-strip__subtitle">
                        <BlurText
                            text="Compare past bleaching records, NOAA heat stress, and the estimated chance of bleaching in the next 4 weeks."
                            delay={40}
                            direction="bottom"
                        />
                    </p>
                </div>

                {/* Full-bleed map experience */}
                <MapEstimateLeaflet
                    ensureBackendReady={ensureBackendReady}
                    serverStatus={serverStatus}
                    onServerReachable={() => setServerStatus("ready")}
                    onServerDown={() => setServerStatus("down")}
                    summary={summary}
                    modelStatus={modelStatus}
                />
            </div>
        </>
    );
}
