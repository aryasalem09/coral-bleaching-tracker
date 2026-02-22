import type { ServerStatus } from "../../types/server";

type StatusIndicatorProps = {
    status: ServerStatus;
};

const STATUS_LABELS: Record<ServerStatus, string> = {
    unknown: "Status unknown",
    warming: "Server warming",
    ready: "Server ready",
    down: "Server down",
};

export default function StatusIndicator({ status }: StatusIndicatorProps) {
    return (
        <div className={`status-indicator status-indicator--${status}`} role="status" aria-live="polite">
            <span className="status-indicator__dot" aria-hidden="true" />
            <span className="status-indicator__label">{STATUS_LABELS[status]}</span>
        </div>
    );
}
