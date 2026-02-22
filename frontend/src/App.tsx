import { useCallback, useEffect, useRef, useState } from "react";
import MapEstimateLeaflet from "./components/map/MapEstimateLeaflet";
import HelpModal from "./components/ui/HelpModal";
import HeroIntro from "./components/ui/HeroIntro";
import Navbar from "./components/ui/Navbar";
import TargetCursor from "./components/ui/TargetCursor";
import TutorialModal from "./components/ui/TutorialModal";
import WarmupBanner from "./components/ui/WarmupBanner";
import { warmBackend } from "./lib/api";
import type { ServerStatus } from "./types/server";

const WARMUP_TIMEOUT_MS = 45_000;
const WARMUP_INTERVAL_MS = 1200;
const SLOW_START_MESSAGE = "Server is slow to start. Please try again shortly.";
const GITHUB_URL = "https://github.com/aryasalem09/coral-bleaching-tracker";
let globalWarmupPromise: Promise<void> | null = null;

export default function App() {
    const [serverStatus, setServerStatus] = useState<ServerStatus>("unknown");
    const [isWarmupBannerVisible, setWarmupBannerVisible] = useState(false);
    const [warmupElapsedSeconds, setWarmupElapsedSeconds] = useState(0);
    const [isHelpOpen, setHelpOpen] = useState(false);
    const [isTutorialOpen, setTutorialOpen] = useState(false);

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
            if (!warmupStartedAtRef.current || !mountedRef.current) return;
            const elapsed = Math.floor((Date.now() - warmupStartedAtRef.current) / 1000);
            setWarmupElapsedSeconds(elapsed);
        }, 250);
    }, [clearWarmupTimer]);

    const captureWarmupFinalSeconds = useCallback(() => {
        if (!warmupStartedAtRef.current) return;
        const elapsed = Math.floor((Date.now() - warmupStartedAtRef.current) / 1000);
        if (mountedRef.current) {
            setWarmupElapsedSeconds(elapsed);
        }
    }, []);

    const startWarmup = useCallback((): Promise<void> => {
        if (warmupPromiseRef.current) {
            return warmupPromiseRef.current;
        }

        if (mountedRef.current) {
            setServerStatus("warming");
            setWarmupBannerVisible(true);
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
                    setWarmupBannerVisible(false);
                }
            })
            .catch((error: unknown) => {
                captureWarmupFinalSeconds();
                clearWarmupTimer();
                if (mountedRef.current) {
                    setServerStatus("down");
                    setWarmupBannerVisible(false);
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
        void startWarmup();
    }, [startWarmup]);

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            clearWarmupTimer();
        };
    }, [clearWarmupTimer]);

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

    const openAboutMetrics = useCallback(() => {
        const aboutMetrics = document.getElementById("about-metrics");
        if (!aboutMetrics) return;

        window.dispatchEvent(new CustomEvent("cbt:open-about-metrics"));

        const navHeight = document.querySelector<HTMLElement>(".top-nav")?.offsetHeight ?? 84;
        const sectionTop = window.scrollY + aboutMetrics.getBoundingClientRect().top;
        const targetTop = Math.max(0, sectionTop - navHeight - 20);

        window.scrollTo({
            top: targetTop,
            behavior: "smooth",
        });
    }, []);

    const wakeServer = useCallback(() => {
        void startWarmup();
    }, [startWarmup]);

    return (
        <div className="app-shell">
            <TargetCursor spinDuration={2} hideDefaultCursor parallaxOn hoverDuration={0.2} />
            <Navbar
                serverStatus={serverStatus}
                onHelpClick={() => setHelpOpen(true)}
                onTutorialClick={() => setTutorialOpen(true)}
                onAboutMetricsClick={openAboutMetrics}
                onWakeServerClick={wakeServer}
                githubUrl={GITHUB_URL}
            />
            <WarmupBanner visible={isWarmupBannerVisible} elapsedSeconds={warmupElapsedSeconds} />
            <HelpModal open={isHelpOpen} onClose={() => setHelpOpen(false)} />
            <TutorialModal open={isTutorialOpen} onClose={() => setTutorialOpen(false)} />

            <main className="dashboard-main">
                <HeroIntro />
                <MapEstimateLeaflet
                    ensureBackendReady={ensureBackendReady}
                    serverStatus={serverStatus}
                    onServerReachable={markServerReady}
                    onServerDown={markServerDown}
                />
            </main>
        </div>
    );
}
