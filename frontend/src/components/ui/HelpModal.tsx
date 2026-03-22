import { useEffect } from "react";

type HelpModalProps = {
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

export default function HelpModal({ open, onClose }: HelpModalProps) {
    useEffect(() => {
        if (!open) return;

        function onKeyDown(event: KeyboardEvent) {
            if (event.key === "Escape") {
                onClose();
            }
        }

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
                className="help-modal glass-panel"
                role="dialog"
                aria-modal="true"
                aria-labelledby="help-modal-title"
                onClick={(event) => event.stopPropagation()}
            >
                <div className="help-modal__header">
                    <h2 id="help-modal-title">How This Works</h2>
                    <button
                        type="button"
                        className="help-modal__close cursor-target"
                        onClick={onClose}
                        aria-label="Close help"
                    >
                        Close
                    </button>
                </div>

                <div className="help-modal__sections">
                    <section>
                        <h3>1. What is DHW?</h3>
                        <p>Degree Heating Weeks measure accumulated thermal stress.</p>
                    </section>

                    <section>
                        <h3>2. What is HotSpot?</h3>
                        <p>Difference between SST and bleaching threshold.</p>
                    </section>

                    <section>
                        <h3>3. Model event probability</h3>
                        <p>A supervised estimate between 0 and 1 for a binary bleaching event, not a confirmed outcome.</p>
                    </section>

                    <section>
                        <h3>4. Snap Distance</h3>
                        <p>When NOAA live files are used, this reports how far the selected site was from the nearest valid ocean grid cell.</p>
                    </section>
                </div>
            </div>
        </div>
    );
}
