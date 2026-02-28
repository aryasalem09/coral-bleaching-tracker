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
    mhw_category?: number | null;
    mhw_label?: string | null;
};

export type NearestReefResponse = {
    lat: number;
    lon: number;
    distance_km: number;
};

export type AvailableDatesResponse = {
    count: number;
    min_date: string | null;
    max_date: string | null;
    source?: string;
    dates: string[];
};

export type AvailableDatesForResponse = {
    reef_key?: string;
    lat: number;
    lon: number;
    count: number;
    min_date?: string | null;
    max_date?: string | null;
    dates: string[];
};

export type HealthResponse = {
    ok?: boolean;
    available_dates?: number;
    min_date?: string | null;
    max_date?: string | null;
    dates_source?: string;
};

export type ReefPointResponse = {
    lat: number;
    lon: number;
};

export type ReefPointsResponse = {
    total: number;
    returned: number;
    points: ReefPointResponse[];
};

export type ReefPointsFallbackResponse = {
    points: ReefPointResponse[];
};

export type EstimateFromFeaturesRequest = {
    lat: number;
    lon: number;
    dhw: number;
    hotspot: number;
};

export type EstimateFromFeaturesResponse = {
    risk_prob: number;
    risk_flag: number;
};

export type SensitivityRequest = {
    lat: number;
    lon: number;
    dhw: number;
    hotspot: number;
    dhw_step?: number;
    hotspot_step?: number;
};

export type SensitivityResponse = {
    base: number;
    dhw_step: number;
    hotspot_step: number;
    delta_dhw: number;
    delta_hotspot: number;
};

export class ApiError extends Error {
    status: number;
    payload: unknown;

    constructor(status: number, message: string, payload: unknown) {
        super(message);
        this.name = "ApiError";
        this.status = status;
        this.payload = payload;
    }
}

export type RequestOptions = {
    signal?: AbortSignal;
};

const rawBase = String(import.meta.env.VITE_API_BASE_URL ?? "").trim();
if (!rawBase) {
    throw new Error("Missing VITE_API_BASE_URL. Set it to your backend URL.");
}

const normalizedBase = rawBase.replace(/\/+$/, "");
if (!/^https?:\/\//i.test(normalizedBase)) {
    throw new Error("Invalid VITE_API_BASE_URL. It must start with http:// or https://");
}

export const API_BASE = normalizedBase;

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

    const query = search.toString();
    const base = buildUrl(path);
    return query ? `${base}?${query}` : base;
}

async function readResponsePayload(res: Response): Promise<{ text: string; payload: unknown }> {
    const text = await res.text();
    if (!text.trim()) {
        return { text, payload: null };
    }

    try {
        return { text, payload: JSON.parse(text) as unknown };
    } catch {
        return { text, payload: text };
    }
}

function extractErrorMessage(res: Response, text: string, payload: unknown): string {
    const statusPrefix = `API ${res.status} ${res.statusText}`;

    if (payload && typeof payload === "object") {
        const maybeDetail = (payload as { detail?: unknown }).detail;
        if (typeof maybeDetail === "string" && maybeDetail.trim()) {
            return `${statusPrefix}: ${maybeDetail.trim()}`;
        }

        const maybeMessage = (payload as { message?: unknown }).message;
        if (typeof maybeMessage === "string" && maybeMessage.trim()) {
            return `${statusPrefix}: ${maybeMessage.trim()}`;
        }
    }

    if (typeof payload === "string" && payload.trim()) {
        const clipped = payload.trim().slice(0, 280);
        return `${statusPrefix}: ${clipped}`;
    }

    if (text.trim()) {
        const clipped = text.trim().slice(0, 280);
        return `${statusPrefix}: ${clipped}`;
    }

    return statusPrefix;
}

async function parseJson<T>(res: Response): Promise<T> {
    const { text, payload } = await readResponsePayload(res);

    if (!res.ok) {
        throw new ApiError(res.status, extractErrorMessage(res, text, payload), payload);
    }

    if (payload === null || typeof payload === "string") {
        throw new ApiError(
            res.status,
            `API ${res.status} ${res.statusText}: expected JSON object response`,
            payload
        );
    }

    return payload as T;
}

export async function apiGet<T>(
    path: string,
    params?: Record<string, string | number | boolean | undefined | null>,
    options?: RequestOptions
): Promise<T> {
    const res = await fetch(buildUrlWithParams(path, params), {
        method: "GET",
        signal: options?.signal,
    });
    return parseJson<T>(res);
}

export async function apiPost<T>(path: string, body: unknown, options?: RequestOptions): Promise<T> {
    const res = await fetch(buildUrl(path), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: options?.signal,
    });
    return parseJson<T>(res);
}

export function estimate(payload: EstimateRequest, options?: RequestOptions): Promise<EstimateResponse> {
    return apiPost<EstimateResponse>("/estimate", payload, options);
}

export function nearestReef(lat: number, lon: number, options?: RequestOptions): Promise<NearestReefResponse> {
    return apiGet<NearestReefResponse>("/nearest-reef", { lat, lon }, options);
}

export function getAvailableDates(options?: RequestOptions): Promise<AvailableDatesResponse> {
    return apiGet<AvailableDatesResponse>("/available-dates", undefined, options);
}

export function getAvailableDatesFor(
    lat: number,
    lon: number,
    options?: RequestOptions
): Promise<AvailableDatesForResponse> {
    return apiGet<AvailableDatesForResponse>("/available-dates-for", { lat, lon }, options);
}

export function health(options?: RequestOptions): Promise<HealthResponse> {
    return apiGet<HealthResponse>("/health", undefined, options);
}

export function getReefPoints(
    south: number,
    west: number,
    north: number,
    east: number,
    limit = 1800,
    options?: RequestOptions
): Promise<ReefPointsResponse> {
    return apiGet<ReefPointsResponse>("/reef-points", { south, west, north, east, limit }, options);
}

export async function getFallbackReefPoints(options?: RequestOptions): Promise<ReefPointsFallbackResponse> {
    const res = await fetch("/reef-points-fallback.json", {
        method: "GET",
        signal: options?.signal,
    });
    return parseJson<ReefPointsFallbackResponse>(res);
}

export function estimateFromFeatures(
    payload: EstimateFromFeaturesRequest,
    options?: RequestOptions
): Promise<EstimateFromFeaturesResponse> {
    return apiPost<EstimateFromFeaturesResponse>("/estimate-from-features", payload, options);
}

export function sensitivity(
    payload: SensitivityRequest,
    options?: RequestOptions
): Promise<SensitivityResponse> {
    return apiPost<SensitivityResponse>("/sensitivity", payload, options);
}

// backward-compatible exports
export const apiEstimate = estimate;
export const apiNearestReef = nearestReef;
export const apiAvailableDatesFor = getAvailableDatesFor;
export const apiHealth = health;
export const apiEstimateFromFeatures = estimateFromFeatures;
export const apiSensitivity = sensitivity;
export const apiReefPoints = getReefPoints;
export const apiFallbackReefPoints = getFallbackReefPoints;

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

// tries /health until it responds or we give up
export async function warmBackend(opts?: {
    maxMs?: number;
    intervalMs?: number;
    onTick?: (info: { attempts: number; elapsedMs: number }) => void;
}): Promise<void> {
    const maxMs = opts?.maxMs ?? 45_000;
    const intervalMs = opts?.intervalMs ?? 1200;

    const startedAt = Date.now();
    let attempts = 0;

    while (Date.now() - startedAt < maxMs) {
        attempts += 1;
        opts?.onTick?.({ attempts, elapsedMs: Date.now() - startedAt });

        try {
            await health();
            return;
        } catch {
            // ignore and retry
        }

        await sleep(intervalMs);
    }

    throw new Error("backend is taking too long to wake up. try again in a moment.");
}
