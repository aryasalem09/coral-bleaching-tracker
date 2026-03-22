function toUtcMidnightMs(isoDate: string): number {
    return Date.parse(`${isoDate}T00:00:00Z`);
}

export type DateCandidate = {
    date: string;
    observed_percent_bleaching?: number | null;
    recommended_for_modeling?: boolean;
};

export function sortDatesDescending(dates: string[]): string[] {
    return [...dates].filter(Boolean).sort((left, right) => right.localeCompare(left));
}

export function findNearestDateIndex(dates: string[], targetISO: string): number {
    if (dates.length === 0) return -1;
    if (dates.length === 1) return 0;

    const targetMs = toUtcMidnightMs(targetISO);
    if (!Number.isFinite(targetMs)) return 0;

    let bestIndex = 0;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (let index = 0; index < dates.length; index += 1) {
        const currentMs = toUtcMidnightMs(dates[index]);
        if (!Number.isFinite(currentMs)) continue;
        const distance = Math.abs(targetMs - currentMs);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = index;
        }
    }

    return bestIndex;
}

export function pickRecommendedDate(dates: string[], preferredDate?: string | null): string | null {
    if (dates.length === 0) return null;
    if (!preferredDate) return dates[0];
    const index = findNearestDateIndex(dates, preferredDate);
    return dates[Math.max(0, index)] ?? dates[0];
}

export function pickNewestUsableObservationDate(records: DateCandidate[]): string | null {
    const sorted = [...records].filter((record) => Boolean(record.date)).sort((left, right) => right.date.localeCompare(left.date));
    const analysisReady = sorted.find((record) => record.recommended_for_modeling);
    if (analysisReady) return analysisReady.date;

    const observedOnly = sorted.find((record) => typeof record.observed_percent_bleaching === "number");
    return observedOnly?.date ?? null;
}
