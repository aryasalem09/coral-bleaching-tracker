import { useEffect, useState } from "react";

const VISITOR_ID_KEY = "cbt_visitor_id";
const COUNTED_KEY = "cbt_counted";
const UNIQUE_DEVICES_KEY = "cbt_unique_devices";

function createVisitorId(): string {
    if (typeof window !== "undefined" && window.crypto?.randomUUID) {
        return window.crypto.randomUUID();
    }

    const randomPart = Math.random().toString(16).slice(2, 10);
    const timePart = Date.now().toString(16);
    return `${timePart}-${randomPart}`;
}

export default function useUniqueDeviceCount(): number {
    const [uniqueDeviceCount, setUniqueDeviceCount] = useState(0);

    useEffect(() => {
        if (typeof window === "undefined") return;

        try {
            const storage = window.localStorage;

            let visitorId = storage.getItem(VISITOR_ID_KEY);
            if (!visitorId) {
                visitorId = createVisitorId();
                storage.setItem(VISITOR_ID_KEY, visitorId);
            }

            let storedCount = Number(storage.getItem(UNIQUE_DEVICES_KEY) ?? "0");
            if (!Number.isFinite(storedCount) || storedCount < 0) {
                storedCount = 0;
            }

            const isAlreadyCounted = storage.getItem(COUNTED_KEY) === "1";
            if (!isAlreadyCounted) {
                storedCount += 1;
                storage.setItem(UNIQUE_DEVICES_KEY, String(storedCount));
                storage.setItem(COUNTED_KEY, "1");
            }

            if (storedCount < 1) {
                storedCount = 1;
                storage.setItem(UNIQUE_DEVICES_KEY, "1");
            }

            setUniqueDeviceCount(storedCount);
        } catch {
            setUniqueDeviceCount(1);
        }
    }, []);

    return uniqueDeviceCount;
}
