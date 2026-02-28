import { useEffect, useMemo, useRef } from "react";
import "./TargetCursor.css";

export type TargetCursorProps = {
    targetSelector?: string;
    hideDefaultCursor?: boolean;
};

type HoverFrame = {
    x: number;
    y: number;
    width: number;
    height: number;
};

const FALLBACK_SELECTOR = [
    ".cursor-target",
    "button",
    "a[href]",
    "input",
    "select",
    "textarea",
    "summary",
    "[role='button']",
    ".leaflet-interactive",
    ".leaflet-control-zoom a",
].join(", ");

function supportsCustomCursor(): boolean {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(hover: hover) and (pointer: fine)").matches;
}

function readHoverFrame(element: Element): HoverFrame {
    const rect = element.getBoundingClientRect();
    const padding = 10;

    return {
        x: rect.left - padding,
        y: rect.top - padding,
        width: rect.width + padding * 2,
        height: rect.height + padding * 2,
    };
}

export default function TargetCursor({
    targetSelector = FALLBACK_SELECTOR,
    hideDefaultCursor = true,
}: TargetCursorProps) {
    const rootRef = useRef<HTMLDivElement>(null);
    const dotRef = useRef<HTMLDivElement>(null);
    const frameRef = useRef<HTMLDivElement>(null);
    const rafRef = useRef<number | null>(null);
    const resizeObserverRef = useRef<ResizeObserver | null>(null);

    const enabled = useMemo(() => supportsCustomCursor(), []);

    useEffect(() => {
        if (!enabled || !rootRef.current || !dotRef.current || !frameRef.current) return;

        const root = rootRef.current;
        const dot = dotRef.current;
        const frame = frameRef.current;
        const body = document.body;
        const originalCursor = body.style.cursor;

        const pointer = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
        const rendered = { x: pointer.x, y: pointer.y };
        const pointerVisible = { current: false };
        const pointerPressed = { current: false };
        const hoveredElement = { current: null as Element | null };
        const hoveredFrame = { current: null as HoverFrame | null };

        const updateHoverFrame = () => {
            if (!hoveredElement.current) {
                hoveredFrame.current = null;
                root.dataset.hover = "false";
                return;
            }

            hoveredFrame.current = readHoverFrame(hoveredElement.current);
            root.dataset.hover = "true";
        };

        const setHoveredElement = (element: Element | null) => {
            if (hoveredElement.current === element) {
                updateHoverFrame();
                return;
            }

            resizeObserverRef.current?.disconnect();
            hoveredElement.current = element;

            if (element instanceof HTMLElement) {
                resizeObserverRef.current?.observe(element);
            }

            updateHoverFrame();
        };

        const resolveTarget = (eventTarget: EventTarget | null): Element | null => {
            if (!(eventTarget instanceof Element)) return null;
            return eventTarget.closest(targetSelector);
        };

        const syncFromPoint = () => {
            const nextElement = document.elementFromPoint(pointer.x, pointer.y);
            setHoveredElement(nextElement?.closest(targetSelector) ?? null);
        };

        const tick = () => {
            rendered.x += (pointer.x - rendered.x) * 0.24;
            rendered.y += (pointer.y - rendered.y) * 0.24;

            root.style.transform = `translate3d(${rendered.x}px, ${rendered.y}px, 0)`;
            dot.style.transform = pointerPressed.current ? "translate(-50%, -50%) scale(0.74)" : "translate(-50%, -50%) scale(1)";

            const hover = hoveredFrame.current;
            if (hover) {
                const frameX = hover.x - rendered.x;
                const frameY = hover.y - rendered.y;
                frame.style.transform = `translate3d(${frameX}px, ${frameY}px, 0)`;
                frame.style.width = `${hover.width}px`;
                frame.style.height = `${hover.height}px`;
            } else {
                frame.style.transform = "translate3d(-22px, -22px, 0)";
                frame.style.width = "44px";
                frame.style.height = "44px";
            }

            rafRef.current = window.requestAnimationFrame(tick);
        };

        if (hideDefaultCursor) {
            body.style.cursor = "none";
            body.classList.add("target-cursor-active");
        }

        resizeObserverRef.current = new ResizeObserver(() => {
            updateHoverFrame();
        });

        root.dataset.visible = "false";
        root.dataset.hover = "false";
        rafRef.current = window.requestAnimationFrame(tick);

        const onPointerMove = (event: PointerEvent) => {
            pointer.x = event.clientX;
            pointer.y = event.clientY;

            if (!pointerVisible.current) {
                pointerVisible.current = true;
                root.dataset.visible = "true";
            }

            setHoveredElement(resolveTarget(event.target));
        };

        const onPointerDown = () => {
            pointerPressed.current = true;
        };

        const onPointerUp = () => {
            pointerPressed.current = false;
        };

        const onPointerLeaveWindow = (event: PointerEvent) => {
            if (event.relatedTarget !== null) return;
            root.dataset.visible = "false";
            pointerVisible.current = false;
            setHoveredElement(null);
        };

        const onFocusIn = (event: FocusEvent) => {
            const target = resolveTarget(event.target);
            if (!target) return;

            const rect = target.getBoundingClientRect();
            pointer.x = rect.left + rect.width / 2;
            pointer.y = rect.top + rect.height / 2;
            root.dataset.visible = "true";
            pointerVisible.current = true;
            setHoveredElement(target);
        };

        const onFocusOut = () => {
            window.setTimeout(syncFromPoint, 0);
        };

        const onScrollOrResize = () => {
            if (hoveredElement.current && !hoveredElement.current.isConnected) {
                setHoveredElement(null);
                return;
            }
            updateHoverFrame();
            if (!hoveredElement.current && pointerVisible.current) {
                syncFromPoint();
            }
        };

        window.addEventListener("pointermove", onPointerMove, { passive: true });
        window.addEventListener("pointerdown", onPointerDown, { passive: true });
        window.addEventListener("pointerup", onPointerUp, { passive: true });
        window.addEventListener("pointerout", onPointerLeaveWindow);
        window.addEventListener("focusin", onFocusIn);
        window.addEventListener("focusout", onFocusOut);
        window.addEventListener("scroll", onScrollOrResize, true);
        window.addEventListener("resize", onScrollOrResize);

        return () => {
            if (rafRef.current !== null) {
                window.cancelAnimationFrame(rafRef.current);
            }
            resizeObserverRef.current?.disconnect();
            body.style.cursor = originalCursor;
            body.classList.remove("target-cursor-active");
            window.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerdown", onPointerDown);
            window.removeEventListener("pointerup", onPointerUp);
            window.removeEventListener("pointerout", onPointerLeaveWindow);
            window.removeEventListener("focusin", onFocusIn);
            window.removeEventListener("focusout", onFocusOut);
            window.removeEventListener("scroll", onScrollOrResize, true);
            window.removeEventListener("resize", onScrollOrResize);
        };
    }, [enabled, hideDefaultCursor, targetSelector]);

    if (!enabled) return null;

    return (
        <div ref={rootRef} className="target-cursor" aria-hidden="true">
            <div ref={dotRef} className="target-cursor__dot" />
            <div ref={frameRef} className="target-cursor__frame" />
        </div>
    );
}
