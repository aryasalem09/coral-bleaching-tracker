type WarmupBannerProps = {
    visible: boolean;
    elapsedSeconds: number;
};

export default function WarmupBanner({ visible, elapsedSeconds }: WarmupBannerProps) {
    return (
        <div
            className={`warmup-banner ${visible ? "warmup-banner--visible" : "warmup-banner--hidden"}`}
            role="status"
            aria-live="polite"
        >
            <span className="spinner" aria-hidden="true" />
            <span>Waking up analysis server... ({elapsedSeconds}s)</span>
        </div>
    );
}
