export type EstimateRequest = {
    lat: number;
    lon: number;
    date: string;
};

export type EstimateResponse = {
    input_lat: number;
    input_lon: number;
    used_lat: number;
    used_lon: number;
    snapped: boolean;
    snap_km: number;
    date: string;
    dhw: number;
    hotspot: number;
    risk_prob: number;
    risk_flag: number;
};

export type NearestReefResponse = {
    lat: number;
    lon: number;
    distance_km: number;
};

export type AvailableDatesForResponse = {
    lat: number;
    lon: number;
    count: number;
    dates: string[];
};

const rawBase = (import.meta.env.VITE_API_BASE_URL ?? "").trim();
const normalized = rawBase.replace(/\/+$/, "");

let resolvedBase = normalized;
if (!resolvedBase) {
    if (import.meta.env.DEV && typeof window !== "undefined" && window.location?.origin) {
        resolvedBase = window.location.origin;
    } else {
        throw new Error("Missing VITE_API_BASE_URL. Set it to your backend URL.");
    }
}

export const API_BASE = resolvedBase;

console.info("[api] base:", API_BASE);

export function buildUrl(path: string): string {
    const safePath = path.startsWith("/") ? path : `/${path}`;
    return `${API_BASE}${safePath}`;
}

function buildUrlWithParams(
    path: string,
    params?: Record<string, string | number | boolean | undefined | null>
): string {
    if (!params) return buildUrl(path);
    const search = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
        if (value === undefined || value === null) continue;
        search.set(key, String(value));
    }
    const base = buildUrl(path);
    const query = search.toString();
    return query ? `${base}?${query}` : base;
}

async function parseJson<T>(res: Response): Promise<T> {
    const text = await res.text();
    try {
        return JSON.parse(text) as T;
    } catch {
        throw new Error(`API ${res.status} ${res.statusText}: invalid JSON response`);
    }
}

async function buildError(res: Response): Promise<Error> {
    const text = await res.text();
    const snippet = text.length > 200 ? `${text.slice(0, 200)}...` : text;
    const body = snippet.trim() ? `: ${snippet}` : "";
    return new Error(`API ${res.status} ${res.statusText}${body}`);
}

export async function apiGet<T>(
    path: string,
    params?: Record<string, string | number | boolean | undefined | null>
): Promise<T> {
    const url = buildUrlWithParams(path, params);
    const res = await fetch(url, { method: "GET" });
    if (!res.ok) throw await buildError(res);
    return parseJson<T>(res);
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
    const url = buildUrl(path);
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw await buildError(res);
    return parseJson<T>(res);
}

export function apiEstimate(payload: EstimateRequest): Promise<EstimateResponse> {
    return apiPost<EstimateResponse>("/estimate", payload);
}

export function apiNearestReef(lat: number, lon: number): Promise<NearestReefResponse> {
    return apiGet<NearestReefResponse>("/nearest-reef", { lat, lon });
}

export function apiAvailableDatesFor(lat: number, lon: number): Promise<AvailableDatesForResponse> {
    return apiGet<AvailableDatesForResponse>("/available-dates-for", { lat, lon });
}

export type HealthResponse = { ok?: boolean };

export async function apiHealth(): Promise<HealthResponse> {
    return apiGet<HealthResponse>("/health");
}

function sleep(ms: number) {
    return new Promise((r) => setTimeout(r, ms));
}

// tries /health until it responds or we give up
export async function warmBackend(opts?: {
    maxMs?: number;
    intervalMs?: number;
    onTick?: (info: { attempts: number; elapsedMs: number }) => void;
}): Promise<void> {
    const maxMs = opts?.maxMs ?? 45_000;
    const intervalMs = opts?.intervalMs ?? 1200;

    const start = Date.now();
    let attempts = 0;

    while (Date.now() - start < maxMs) {
        attempts++;
        opts?.onTick?.({ attempts, elapsedMs: Date.now() - start });

        try {
            await apiHealth();
            return;
        } catch {
            // ignore and retry
        }

        await sleep(intervalMs);
    }

    throw new Error("backend is taking too long to wake up. try again in a moment.");
}
