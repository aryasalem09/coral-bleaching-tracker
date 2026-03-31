export type RequestOptions = {
    signal?: AbortSignal;
};

export type SummaryResponse = {
    service: string;
    version: string;
    started_at: string;
    model_ready: boolean;
    model_status: "ready" | "missing" | "invalid";
    model_status_message: string;
    live_noaa_dates_available: number;
    latest_live_noaa_date: string | null;
    live_date_source: string;
};

export type SitePoint = {
    site_id: string;
    display_name: string;
    latitude: number;
    longitude: number;
    latest_observed_date: string | null;
    observed_record_count: number;
    observed_positive_count: number;
    mean_label_quality_score: number;
};

export type SitesResponse = {
    total: number;
    returned: number;
    points: SitePoint[];
};

export type SiteMeta = {
    site_id: string;
    display_name: string;
    latitude: number;
    longitude: number;
    ocean_name: string | null;
    realm_name: string | null;
    ecoregion_name: string | null;
    country_name: string | null;
    state_name: string | null;
    city_town_name: string | null;
    site_name: string | null;
    distance_to_shore_km: number | null;
    exposure: string | null;
    turbidity: number | null;
    cyclone_frequency: number | null;
    depth_mean_m: number | null;
    observed_record_count: number;
    observed_positive_count: number;
    latest_observed_date: string | null;
    first_observed_date: string | null;
    provenance_source_count: number;
    provenance_sources: string[];
    mean_label_quality_score: number;
};

export type SiteDetailResponse = {
    site: SiteMeta;
    recommended_observed_date: string | null;
    observed_dates: string[];
};

export type ObservationRecord = {
    date: string;
    observed_percent_bleaching: number | null;
    observed_severity_category: string | null;
    target_bleaching_event: number | null;
    label_quality_score: number;
    is_direct_observation: boolean;
    is_derived_label: boolean;
    has_conflict_history: boolean;
    sample_row_count: number;
    source_count: number;
    recommended_for_modeling: boolean;
    provenance_sources: string[];
};

export type ObservationsResponse = {
    site_id: string;
    display_name: string;
    recommended_date: string | null;
    records: ObservationRecord[];
};

export type RiskInfoResponse = {
    layer_name: string;
    definition: string;
    fallback_behavior: string;
    thresholds: Array<{
        category: string;
        min_score: number;
        min_hotspot: number;
        min_dhw: number;
        color: string;
    }>;
};

export type RiskScoreResponse = {
    site_id: string;
    display_name: string;
    requested_date: string | null;
    used_date: string;
    mode: string;
    hotspot: number;
    dhw: number;
    category: string;
    score: number;
    color: string;
    explanation: string;
    used_latitude: number;
    used_longitude: number;
    snap_km: number;
    warnings: string[];
};

export type PredictionResponse = {
    available: boolean;
    site_id?: string;
    display_name?: string;
    requested_date?: string | null;
    used_date?: string;
    mode?: string;
    predicted_event?: boolean;
    probability?: number;
    threshold?: number;
    model_version?: string;
    target_definition?: string;
    prediction_unit?: string;
    input_feature_window?: string;
    data_quality_warning?: string | null;
    coverage_warning?: string | null;
    message?: string;
    features_used?: Record<string, unknown>;
};

export type ModelInfoResponse = {
    available: boolean;
    model_name?: string;
    model_version?: string;
    target_definition?: string;
    prediction_unit?: string;
    feature_columns?: string[];
    validation_metric_used_for_selection?: string;
    decision_threshold?: number;
};

export type ModelMetricsResponse = {
    available: boolean;
    selected_model?: string;
    candidate_results?: Record<
        string,
        {
            validation: Record<string, number>;
            test: Record<string, number>;
        }
    >;
    split_overlap_summary?: Record<string, number>;
    selected_model_additional_evaluation?: Record<string, Record<string, number> | null>;
};

export type SiteAnalysisRequest = {
    site_id?: string;
    lat?: number;
    lon?: number;
    date?: string;
    prefer_live?: boolean;
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

const rawBase = String(
    import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.DEV ? "http://127.0.0.1:8000" : window.location.origin)
).trim();

const normalizedBase = rawBase.replace(/\/+$/, "");
if (!/^https?:\/\//i.test(normalizedBase)) {
    throw new Error("Invalid VITE_API_BASE_URL. It must start with http:// or https://");
}

export const API_BASE = normalizedBase;

function buildUrl(path: string): string {
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

function extractErrorMessage(res: Response, payload: unknown): string {
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
        return `${statusPrefix}: ${payload.trim().slice(0, 280)}`;
    }

    return statusPrefix;
}

async function parseJson<T>(res: Response): Promise<T> {
    const responsePayload = await readResponsePayload(res);
    if (!res.ok) {
        throw new ApiError(
            res.status,
            extractErrorMessage(res, responsePayload.payload),
            responsePayload.payload
        );
    }
    return responsePayload.payload as T;
}

async function apiGet<T>(
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

async function apiPost<T>(path: string, body: unknown, options?: RequestOptions): Promise<T> {
    const res = await fetch(buildUrl(path), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: options?.signal,
    });
    return parseJson<T>(res);
}

export function health(options?: RequestOptions): Promise<{ ok: boolean; service: string; started_at: string; version: string }> {
    return apiGet("/health", undefined, options);
}

export function getSummary(options?: RequestOptions): Promise<SummaryResponse> {
    return apiGet("/api/summary", undefined, options);
}

export function getSites(
    south: number,
    west: number,
    north: number,
    east: number,
    limit: number,
    options?: RequestOptions
): Promise<SitesResponse> {
    return apiGet("/api/sites", { south, west, north, east, limit }, options);
}

export function getSiteDetail(siteId: string, options?: RequestOptions): Promise<SiteDetailResponse> {
    return apiGet(`/api/site/${siteId}`, undefined, options);
}

export function getSiteObservations(siteId: string, options?: RequestOptions): Promise<ObservationsResponse> {
    return apiGet(`/api/site/${siteId}/observations`, undefined, options);
}

export function getRiskInfo(options?: RequestOptions): Promise<RiskInfoResponse> {
    return apiGet("/api/risk/info", undefined, options);
}

export function getRiskScore(payload: SiteAnalysisRequest, options?: RequestOptions): Promise<RiskScoreResponse> {
    return apiPost("/api/risk/score", payload, options);
}

export function getModelInfo(options?: RequestOptions): Promise<ModelInfoResponse> {
    return apiGet("/api/model/info", undefined, options);
}

export function getModelMetrics(options?: RequestOptions): Promise<ModelMetricsResponse> {
    return apiGet("/api/model/metrics", undefined, options);
}

export function predictBleaching(payload: SiteAnalysisRequest, options?: RequestOptions): Promise<PredictionResponse> {
    return apiPost("/api/predict", payload, options);
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function warmBackend(opts?: {
    maxMs?: number;
    intervalMs?: number;
    onTick?: (info: { attempts: number; elapsedMs: number }) => void;
}): Promise<void> {
    const maxMs = opts?.maxMs ?? 30_000;
    const intervalMs = opts?.intervalMs ?? 1_000;
    const startedAt = Date.now();
    let attempts = 0;

    while (Date.now() - startedAt < maxMs) {
        attempts += 1;
        opts?.onTick?.({ attempts, elapsedMs: Date.now() - startedAt });
        try {
            await health();
            return;
        } catch {
            // keep polling until maxMs
        }
        await sleep(intervalMs);
    }

    throw new Error("Backend is taking too long to wake up. Try again shortly.");
}
