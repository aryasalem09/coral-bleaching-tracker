import { useEffect } from "react";
import Stepper, { Step } from "./Stepper";

type TutorialModalProps = {
    open: boolean;
    onClose: () => void;
};

const BODY_LOCK_COUNT_KEY = "cbtLockCount";
const BODY_PREV_OVERFLOW_KEY = "cbtPrevOverflow";

function lockBodyScroll() {
    const body = document.body;
    const currentCount = Number(body.dataset[BODY_LOCK_COUNT_KEY] ?? "0");
    if (!Number.isFinite(currentCount) || currentCount <= 0) {
        body.dataset[BODY_PREV_OVERFLOW_KEY] = body.style.overflow;
        body.style.overflow = "hidden";
        body.dataset[BODY_LOCK_COUNT_KEY] = "1";
        return;
    }
    body.dataset[BODY_LOCK_COUNT_KEY] = String(currentCount + 1);
}

function unlockBodyScroll() {
    const body = document.body;
    const currentCount = Number(body.dataset[BODY_LOCK_COUNT_KEY] ?? "0");
    if (!Number.isFinite(currentCount) || currentCount <= 1) {
        body.style.overflow = body.dataset[BODY_PREV_OVERFLOW_KEY] ?? "";
        delete body.dataset[BODY_LOCK_COUNT_KEY];
        delete body.dataset[BODY_PREV_OVERFLOW_KEY];
        return;
    }
    body.dataset[BODY_LOCK_COUNT_KEY] = String(currentCount - 1);
}

export default function TutorialModal({ open, onClose }: TutorialModalProps) {
    useEffect(() => {
        if (!open) return;

        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                onClose();
            }
        };

        lockBodyScroll();
        window.addEventListener("keydown", onKeyDown);

        return () => {
            unlockBodyScroll();
            window.removeEventListener("keydown", onKeyDown);
        };
    }, [open, onClose]);

    if (!open) return null;

    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div
                className="tutorial-modal glass-panel"
                role="dialog"
                aria-modal="true"
                aria-labelledby="tutorial-modal-title"
                onClick={(event) => event.stopPropagation()}
            >
                <div className="tutorial-modal__header">
                    <h2 id="tutorial-modal-title">Quick Tutorial</h2>
                    <button type="button" className="help-modal__close cursor-target" onClick={onClose}>
                        Close
                    </button>
                </div>

                <Stepper
                    nextButtonText="Next"
                    backButtonText="Back"
                    stepCircleContainerClassName="tutorial-stepper-card"
                    contentClassName="tutorial-stepper-content"
                    footerClassName="tutorial-stepper-footer"
                >
                    <Step>
                        <article className="tutorial-step">
                            <h3>1. Welcome</h3>
                            <p>
                                This dashboard estimates coral bleaching risk from NOAA Degree Heating Weeks and
                                HotSpot thermal stress metrics.
                            </p>
                        </article>
                    </Step>

                    <Step>
                        <article className="tutorial-step">
                            <h3>2. Pick A Date + Click Reef</h3>
                            <p>Select a historical date, then click any reef area on the map to run an estimate.</p>
                            <div className="tutorial-step__placeholder" aria-hidden="true">
                                <span className="tutorial-step__placeholder-dot" />
                                <span>Map interaction preview</span>
                            </div>
                        </article>
                    </Step>

                    <Step>
                        <article className="tutorial-step">
                            <h3>3. Understand Results</h3>
                            <p>
                                Review the risk probability badge, plus DHW and HotSpot values, then inspect the risk
                                bar for quick severity context.
                            </p>
                        </article>
                    </Step>

                    <Step>
                        <article className="tutorial-step">
                            <h3>4. Tips</h3>
                            <p>
                                If a click is off-reef, the system snaps to the nearest valid reef cell. First request
                                may take longer while the Render backend wakes.
                            </p>
                        </article>
                    </Step>
                </Stepper>
            </div>
        </div>
    );
}
