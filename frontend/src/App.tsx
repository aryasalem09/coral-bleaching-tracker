import { useCallback, useEffect, useRef, useState } from "react";
import MapEstimateLeaflet from "./components/map/MapEstimateLeaflet";
import TargetCursor from "./components/ui/TargetCursor";
import TutorialModal from "./components/ui/TutorialModal";
import { warmBackend } from "./lib/api";
import type { ServerStatus } from "./types/server";

const WARMUP_TIMEOUT_MS = 45_000;
const WARMUP_INTERVAL_MS = 1200;
const SLOW_START_MESSAGE = "Server is slow to start. Please try again shortly.";
const GITHUB_URL = "https://github.com/aryasalem09/coral-bleaching-tracker";
const TUTORIAL_STORAGE_KEY = "cbt:tutorial-dismissed";

let globalWarmupPromise: Promise<void> | null = null;

function statusLabel(serverStatus: ServerStatus): string {
    if (serverStatus === "ready") return "Live data ready";
    if (serverStatus === "warming") return "Warming backend";
    if (serverStatus === "down") return "Backend unavailable";
    return "Checking service";
}

export default function App() {
    const [serverStatus, setServerStatus] = useState<ServerStatus>("unknown");
    const [isTutorialOpen, setTutorialOpen] = useState(() => {
        try {
            return window.localStorage.getItem(TUTORIAL_STORAGE_KEY) !== "1";
        } catch {
            return true;
        }
    });
    const [isWarmupVisible, setWarmupVisible] = useState(false);
    const [warmupElapsedSeconds, setWarmupElapsedSeconds] = useState(0);

    const warmupPromiseRef = useRef<Promise<void> | null>(null);
    const warmupStartedAtRef = useRef<number | null>(null);
    const warmupTimerRef = useRef<number | null>(null);
    const mountedRef = useRef(true);

    const clearWarmupTimer = useCallback(() => {
        if (warmupTimerRef.current !== null) {
            window.clearInterval(warmupTimerRef.current);
            warmupTimerRef.current = null;
        }
        warmupStartedAtRef.current = null;
    }, []);

    const beginWarmupTimer = useCallback(() => {
        clearWarmupTimer();
        warmupStartedAtRef.current = Date.now();
        if (mountedRef.current) {
            setWarmupElapsedSeconds(0);
        }
        warmupTimerRef.current = window.setInterval(() => {
            if (!mountedRef.current || !warmupStartedAtRef.current) return;
            setWarmupElapsedSeconds(Math.floor((Date.now() - warmupStartedAtRef.current) / 1000));
        }, 250);
    }, [clearWarmupTimer]);

    const captureWarmupFinalSeconds = useCallback(() => {
        if (!warmupStartedAtRef.current || !mountedRef.current) return;
        setWarmupElapsedSeconds(Math.floor((Date.now() - warmupStartedAtRef.current) / 1000));
    }, []);

    const startWarmup = useCallback((): Promise<void> => {
        if (warmupPromiseRef.current) {
            return warmupPromiseRef.current;
        }

        if (mountedRef.current) {
            setServerStatus("warming");
            setWarmupVisible(true);
        }
        beginWarmupTimer();

        if (!globalWarmupPromise) {
            globalWarmupPromise = warmBackend({
                maxMs: WARMUP_TIMEOUT_MS,
                intervalMs: WARMUP_INTERVAL_MS,
            }).finally(() => {
                globalWarmupPromise = null;
            });
        }

        const warmupTask = globalWarmupPromise
            .then(() => {
                captureWarmupFinalSeconds();
                clearWarmupTimer();
                if (mountedRef.current) {
                    setServerStatus("ready");
                    setWarmupVisible(false);
                }
            })
            .catch((error: unknown) => {
                captureWarmupFinalSeconds();
                clearWarmupTimer();
                if (mountedRef.current) {
                    setServerStatus("down");
                    setWarmupVisible(false);
                }
                throw error;
            })
            .finally(() => {
                warmupPromiseRef.current = null;
            });

        warmupPromiseRef.current = warmupTask;
        return warmupTask;
    }, [beginWarmupTimer, captureWarmupFinalSeconds, clearWarmupTimer]);

    useEffect(() => {
        mountedRef.current = true;
        const timeoutId = window.setTimeout(() => {
            void startWarmup();
        }, 0);

        return () => {
            window.clearTimeout(timeoutId);
            mountedRef.current = false;
            clearWarmupTimer();
        };
    }, [clearWarmupTimer, startWarmup]);

    const closeTutorial = useCallback(() => {
        setTutorialOpen(false);
        try {
            window.localStorage.setItem(TUTORIAL_STORAGE_KEY, "1");
        } catch {
            // ignore storage failures
        }
    }, []);

    const ensureBackendReady = useCallback(async () => {
        if (serverStatus === "ready") return;

        try {
            await startWarmup();
        } catch (error: unknown) {
            if (error instanceof Error && error.message.toLowerCase().includes("slow to start")) {
                throw error;
            }

            throw new Error(SLOW_START_MESSAGE, {
                cause: error instanceof Error ? error : undefined,
            });
        }
    }, [serverStatus, startWarmup]);

    const markServerReady = useCallback(() => {
        if (mountedRef.current) {
            setServerStatus("ready");
        }
    }, []);

    const markServerDown = useCallback(() => {
        if (mountedRef.current) {
            setServerStatus("down");
        }
    }, []);

    return (
        <div className="app-shell">
            <TargetCursor hideDefaultCursor />
            <TutorialModal open={isTutorialOpen} onClose={closeTutorial} />

            <header className="app-topbar">
                <div className="app-topbar__brand">
                    <p className="app-kicker">Coral Bleaching Tracker</p>
                    <h1>Explore reefs through a map-first risk dashboard.</h1>
                </div>

                <div className="app-topbar__actions">
                    <span className={`status-pill status-pill--${serverStatus}`}>{statusLabel(serverStatus)}</span>
                    {isWarmupVisible ? (
                        <span className="status-pill status-pill--warmup">Wake-up {warmupElapsedSeconds}s</span>
                    ) : null}
                    <button type="button" className="ghost-button" onClick={() => setTutorialOpen(true)}>
                        Tutorial
                    </button>
                    <a className="ghost-button" href={GITHUB_URL} target="_blank" rel="noreferrer">
                        GitHub
                    </a>
                </div>
            </header>

            <MapEstimateLeaflet
                ensureBackendReady={ensureBackendReady}
                serverStatus={serverStatus}
                onServerReachable={markServerReady}
                onServerDown={markServerDown}
                onOpenTutorial={() => setTutorialOpen(true)}
                warmupElapsedSeconds={warmupElapsedSeconds}
            />
        </div>
    );
}
