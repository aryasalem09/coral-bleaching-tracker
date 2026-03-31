import { useEffect, useMemo, useState } from "react";

export type CapabilityTier = "low" | "medium" | "high";

const STORAGE_KEY = "cbt:capability-override";

function inferTier(): CapabilityTier {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const deviceMemory = Number((navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4);
    const hardwareConcurrency = Number(navigator.hardwareConcurrency ?? 4);
    const narrowViewport = window.innerWidth < 900;

    if (reducedMotion || deviceMemory <= 4 || hardwareConcurrency <= 4) return "low";
    if (!narrowViewport && deviceMemory >= 8 && hardwareConcurrency >= 8) return "high";
    return "medium";
}

export function useCapabilityTier() {
    const [override, setOverrideState] = useState<CapabilityTier | null>(() => {
        const stored = window.localStorage.getItem(STORAGE_KEY);
        if (stored === "low" || stored === "medium" || stored === "high") return stored;
        return null;
    });
    const [inferred, setInferred] = useState<CapabilityTier>(() => inferTier());

    useEffect(() => {
        const onResize = () => setInferred(inferTier());
        const media = window.matchMedia("(prefers-reduced-motion: reduce)");
        const onMotionChange = () => setInferred(inferTier());

        window.addEventListener("resize", onResize);
        media.addEventListener("change", onMotionChange);
        return () => {
            window.removeEventListener("resize", onResize);
            media.removeEventListener("change", onMotionChange);
        };
    }, []);

    const tier = override ?? inferred;

    const setOverride = (next: CapabilityTier | null) => {
        setOverrideState(next);
        if (next) {
            window.localStorage.setItem(STORAGE_KEY, next);
        } else {
            window.localStorage.removeItem(STORAGE_KEY);
        }
    };

    return useMemo(
        () => ({
            tier,
            inferredTier: inferred,
            overrideTier: override,
            setOverride,
        }),
        [inferred, override, tier]
    );
}
