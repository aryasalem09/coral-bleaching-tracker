import { useEffect, useState } from "react";

type TutorialModalProps = {
    open: boolean;
    onClose: () => void;
};

const STEPS = [
    {
        title: "Explore the map first",
        body: "The map is the main interface. Pan, zoom, and look for the highlighted reef circles before opening any detail.",
    },
    {
        title: "Click reef circles only",
        body: "Each circle is a real reef point. Clicking the circle avoids the off-reef snapping behavior and opens a reef-specific dashboard.",
    },
    {
        title: "Use the dashboard",
        body: "After selecting a reef, switch between overview, timeline, and scenario tabs to inspect risk, scrub dates, and test stress changes.",
    },
    {
        title: "Replay the walkthrough anytime",
        body: "Use the Tutorial button in the top bar whenever you want the guided walkthrough again.",
    },
];

export default function TutorialModal({ open, onClose }: TutorialModalProps) {
    const [stepIndex, setStepIndex] = useState(0);

    useEffect(() => {
        if (!open) return;

        const previousOverflow = document.body.style.overflow;
        document.body.style.overflow = "hidden";

        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") onClose();
            if (event.key === "ArrowRight") setStepIndex((current) => Math.min(current + 1, STEPS.length - 1));
            if (event.key === "ArrowLeft") setStepIndex((current) => Math.max(current - 1, 0));
        };

        window.addEventListener("keydown", onKeyDown);
        return () => {
            document.body.style.overflow = previousOverflow;
            window.removeEventListener("keydown", onKeyDown);
            setStepIndex(0);
        };
    }, [onClose, open]);

    if (!open) return null;

    const step = STEPS[stepIndex];
    const isLastStep = stepIndex === STEPS.length - 1;

    return (
        <div className="tutorial-overlay" onClick={onClose}>
            <div className="tutorial-dialog glass-panel" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
                <div className="tutorial-dialog__header">
                    <div>
                        <p className="tutorial-dialog__eyebrow">Quick tour</p>
                        <h2>{step.title}</h2>
                    </div>
                    <button type="button" className="ghost-button" onClick={onClose}>
                        Close
                    </button>
                </div>

                <p className="tutorial-dialog__body">{step.body}</p>

                <div className="tutorial-progress" aria-label="Tutorial progress">
                    {STEPS.map((item, index) => (
                        <button
                            key={item.title}
                            type="button"
                            className={index === stepIndex ? "tutorial-progress__dot tutorial-progress__dot--active" : "tutorial-progress__dot"}
                            onClick={() => setStepIndex(index)}
                            aria-label={`Go to step ${index + 1}`}
                        />
                    ))}
                </div>

                <div className="tutorial-cards">
                    <article className="tutorial-card tutorial-card--active">
                        <span>{`0${stepIndex + 1}`}</span>
                        <strong>{step.title}</strong>
                    </article>
                    <article className="tutorial-card">
                        <span>Map</span>
                        <strong>Choose circles, not empty ocean.</strong>
                    </article>
                    <article className="tutorial-card">
                        <span>Dashboard</span>
                        <strong>Overview, timeline, scenario.</strong>
                    </article>
                </div>

                <div className="tutorial-dialog__footer">
                    <button type="button" className="ghost-button" onClick={() => setStepIndex((current) => Math.max(current - 1, 0))} disabled={stepIndex === 0}>
                        Back
                    </button>
                    <button
                        type="button"
                        className="ghost-button ghost-button--accent"
                        onClick={() => {
                            if (isLastStep) {
                                onClose();
                                return;
                            }
                            setStepIndex((current) => Math.min(current + 1, STEPS.length - 1));
                        }}
                    >
                        {isLastStep ? "Start exploring" : "Next"}
                    </button>
                </div>
            </div>
        </div>
    );
}
