function toUtcMidnightMs(isoDate: string): number {
    return Date.parse(`${isoDate}T00:00:00Z`);
}

// Assumes `dates` is sorted ascending (ISO yyyy-mm-dd).
export function findNearestDateIndex(dates: string[], targetIso: string): number {
    if (dates.length === 0) return -1;
    if (dates.length === 1) return 0;

    const targetMs = toUtcMidnightMs(targetIso);
    if (!Number.isFinite(targetMs)) return 0;

    let low = 0;
    let high = dates.length - 1;

    while (low <= high) {
        const mid = (low + high) >> 1;
        const midMs = toUtcMidnightMs(dates[mid]);

        if (!Number.isFinite(midMs)) {
            return Math.max(0, Math.min(dates.length - 1, mid));
        }

        if (midMs === targetMs) return mid;
        if (midMs < targetMs) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    if (low <= 0) return 0;
    if (low >= dates.length) return dates.length - 1;

    const prevIndex = low - 1;
    const prevMs = toUtcMidnightMs(dates[prevIndex]);
    const nextMs = toUtcMidnightMs(dates[low]);

    if (!Number.isFinite(prevMs)) return low;
    if (!Number.isFinite(nextMs)) return prevIndex;

    return Math.abs(targetMs - prevMs) <= Math.abs(nextMs - targetMs) ? prevIndex : low;
}
