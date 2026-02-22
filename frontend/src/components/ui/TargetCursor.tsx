import { gsap } from "gsap";
import { useCallback, useEffect, useMemo, useRef } from "react";
import "./TargetCursor.css";

export type TargetCursorProps = {
    targetSelector?: string;
    spinDuration?: number;
    hideDefaultCursor?: boolean;
    hoverDuration?: number;
    parallaxOn?: boolean;
};

type CornerPosition = { x: number; y: number };

function normalizeRotation(rawRotation: number): number {
    if (!Number.isFinite(rawRotation)) return 0;
    const normalized = rawRotation % 360;
    return normalized < 0 ? normalized + 360 : normalized;
}

export default function TargetCursor({
    targetSelector = ".cursor-target",
    spinDuration = 2,
    hideDefaultCursor = true,
    hoverDuration = 0.2,
    parallaxOn = true,
}: TargetCursorProps) {
    const cursorRef = useRef<HTMLDivElement>(null);
    const dotRef = useRef<HTMLDivElement>(null);
    const cornersRef = useRef<NodeListOf<HTMLDivElement> | null>(null);
    const spinTimelineRef = useRef<gsap.core.Timeline | null>(null);
    const targetCornerPositionsRef = useRef<CornerPosition[] | null>(null);
    const tickerFnRef = useRef<(() => void) | null>(null);
    const activeStrengthRef = useRef({ current: 0 });
    const tickerAttachedRef = useRef(false);

    const isMobile = useMemo(() => {
        if (typeof window === "undefined") return false;
        const hasTouchScreen = "ontouchstart" in window || navigator.maxTouchPoints > 0;
        const isSmallScreen = window.innerWidth <= 768;
        const operaAgent = (window as Window & { opera?: string }).opera ?? "";
        const userAgent = navigator.userAgent || navigator.vendor || operaAgent;
        const mobileRegex = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i;
        return (hasTouchScreen && isSmallScreen) || mobileRegex.test(userAgent.toLowerCase());
    }, []);

    const constants = useMemo(() => ({ borderWidth: 3, cornerSize: 12 }), []);

    const moveCursor = useCallback((x: number, y: number) => {
        if (!cursorRef.current) return;
        gsap.to(cursorRef.current, { x, y, duration: 0.1, ease: "power3.out" });
    }, []);

    const resolveClosestTarget = useCallback(
        (eventTarget: EventTarget | null): Element | null => {
            if (!eventTarget) return null;
            if (eventTarget instanceof Element) {
                return eventTarget.closest(targetSelector);
            }
            if (eventTarget instanceof Node) {
                return eventTarget.parentElement?.closest(targetSelector) ?? null;
            }
            return null;
        },
        [targetSelector]
    );

    useEffect(() => {
        if (isMobile || !cursorRef.current) return;

        const body = document.body;
        const originalCursor = body.style.cursor;
        if (hideDefaultCursor) {
            body.style.cursor = "none";
            body.classList.add("target-cursor-active");
        }

        const cursor = cursorRef.current;
        cornersRef.current = cursor.querySelectorAll<HTMLDivElement>(".target-cursor-corner");

        gsap.set(cursor, {
            xPercent: -50,
            yPercent: -50,
            x: window.innerWidth / 2,
            y: window.innerHeight / 2,
        });

        const createSpinTimeline = () => {
            if (spinTimelineRef.current) {
                spinTimelineRef.current.kill();
            }
            spinTimelineRef.current = gsap
                .timeline({ repeat: -1 })
                .to(cursor, { rotation: "+=360", duration: spinDuration, ease: "none" });
        };

        createSpinTimeline();

        const attachTicker = () => {
            if (tickerAttachedRef.current || !tickerFnRef.current) return;
            gsap.ticker.add(tickerFnRef.current);
            tickerAttachedRef.current = true;
        };

        const detachTicker = () => {
            if (!tickerAttachedRef.current || !tickerFnRef.current) return;
            gsap.ticker.remove(tickerFnRef.current);
            tickerAttachedRef.current = false;
        };

        const tickerFn = () => {
            if (!targetCornerPositionsRef.current || !cursorRef.current || !cornersRef.current) {
                return;
            }

            const strength = activeStrengthRef.current.current;
            if (strength === 0) return;

            const cursorX = gsap.getProperty(cursorRef.current, "x") as number;
            const cursorY = gsap.getProperty(cursorRef.current, "y") as number;
            const corners = Array.from(cornersRef.current);

            corners.forEach((corner, index) => {
                const currentX = gsap.getProperty(corner, "x") as number;
                const currentY = gsap.getProperty(corner, "y") as number;
                const targetX = targetCornerPositionsRef.current![index].x - cursorX;
                const targetY = targetCornerPositionsRef.current![index].y - cursorY;
                const finalX = currentX + (targetX - currentX) * strength;
                const finalY = currentY + (targetY - currentY) * strength;
                const duration = strength >= 0.99 ? (parallaxOn ? 0.2 : 0) : 0.05;

                gsap.to(corner, {
                    x: finalX,
                    y: finalY,
                    duration,
                    ease: duration === 0 ? "none" : "power1.out",
                    overwrite: "auto",
                });
            });
        };

        tickerFnRef.current = tickerFn;

        let activeTarget: Element | null = null;
        let currentLeaveHandler: (() => void) | null = null;
        let resumeTimeoutId: number | null = null;

        const releaseIfTargetDetached = () => {
            if (activeTarget && !activeTarget.isConnected) {
                currentLeaveHandler?.();
            }
        };

        const cleanupTarget = (target: Element) => {
            if (currentLeaveHandler) {
                target.removeEventListener("mouseleave", currentLeaveHandler);
            }
            currentLeaveHandler = null;
        };

        const onPointerMove = (event: PointerEvent) => {
            releaseIfTargetDetached();
            moveCursor(event.clientX, event.clientY);
        };

        const onScroll = () => {
            releaseIfTargetDetached();
            if (!activeTarget || !cursorRef.current) return;

            const mouseX = gsap.getProperty(cursorRef.current, "x") as number;
            const mouseY = gsap.getProperty(cursorRef.current, "y") as number;
            const underPointer = document.elementFromPoint(mouseX, mouseY);
            const isStillOverTarget =
                underPointer &&
                (underPointer === activeTarget || underPointer.closest(targetSelector) === activeTarget);

            if (!isStillOverTarget && currentLeaveHandler) {
                currentLeaveHandler();
            }
        };

        const onWindowBlur = () => {
            currentLeaveHandler?.();
        };

        const onWindowMouseOut = (event: MouseEvent) => {
            if (event.relatedTarget !== null) return;
            currentLeaveHandler?.();
        };

        const onMouseDown = () => {
            if (!dotRef.current || !cursorRef.current) return;
            gsap.to(dotRef.current, { scale: 0.7, duration: 0.3 });
            gsap.to(cursorRef.current, { scale: 0.9, duration: 0.2 });
        };

        const onMouseUp = () => {
            if (!dotRef.current || !cursorRef.current) return;
            gsap.to(dotRef.current, { scale: 1, duration: 0.3 });
            gsap.to(cursorRef.current, { scale: 1, duration: 0.2 });
        };

        const onPointerOver = (event: PointerEvent) => {
            const matches = resolveClosestTarget(event.target);
            if (!matches || !cursorRef.current || !cornersRef.current) return;
            if (activeTarget === matches) return;

            if (activeTarget && currentLeaveHandler) {
                currentLeaveHandler();
            }

            if (resumeTimeoutId !== null) {
                window.clearTimeout(resumeTimeoutId);
                resumeTimeoutId = null;
            }

            activeTarget = matches;
            const corners = Array.from(cornersRef.current);
            corners.forEach((corner) => gsap.killTweensOf(corner));
            gsap.killTweensOf(cursorRef.current, "rotation");
            spinTimelineRef.current?.pause();
            gsap.set(cursorRef.current, { rotation: 0 });

            const rect = matches.getBoundingClientRect();
            const { borderWidth, cornerSize } = constants;
            const cursorX = gsap.getProperty(cursorRef.current, "x") as number;
            const cursorY = gsap.getProperty(cursorRef.current, "y") as number;

            targetCornerPositionsRef.current = [
                { x: rect.left - borderWidth, y: rect.top - borderWidth },
                { x: rect.right + borderWidth - cornerSize, y: rect.top - borderWidth },
                { x: rect.right + borderWidth - cornerSize, y: rect.bottom + borderWidth - cornerSize },
                { x: rect.left - borderWidth, y: rect.bottom + borderWidth - cornerSize },
            ];

            attachTicker();

            gsap.to(activeStrengthRef.current, { current: 1, duration: hoverDuration, ease: "power2.out" });

            corners.forEach((corner, index) => {
                gsap.to(corner, {
                    x: targetCornerPositionsRef.current![index].x - cursorX,
                    y: targetCornerPositionsRef.current![index].y - cursorY,
                    duration: 0.2,
                    ease: "power2.out",
                });
            });

            let didLeave = false;
            const leaveHandler = () => {
                if (didLeave) return;
                didLeave = true;

                detachTicker();

                targetCornerPositionsRef.current = null;
                gsap.set(activeStrengthRef.current, { current: 0, overwrite: true });
                activeTarget = null;

                if (cornersRef.current) {
                    const currentCorners = Array.from(cornersRef.current);
                    gsap.killTweensOf(currentCorners);
                    const positions = [
                        { x: -constants.cornerSize * 1.5, y: -constants.cornerSize * 1.5 },
                        { x: constants.cornerSize * 0.5, y: -constants.cornerSize * 1.5 },
                        { x: constants.cornerSize * 0.5, y: constants.cornerSize * 0.5 },
                        { x: -constants.cornerSize * 1.5, y: constants.cornerSize * 0.5 },
                    ];

                    const timeline = gsap.timeline();
                    currentCorners.forEach((corner, index) => {
                        timeline.to(
                            corner,
                            {
                                x: positions[index].x,
                                y: positions[index].y,
                                duration: 0.3,
                                ease: "power3.out",
                            },
                            0
                        );
                    });
                }

                resumeTimeoutId = window.setTimeout(() => {
                    if (!activeTarget && cursorRef.current && spinTimelineRef.current) {
                        const currentRotation = gsap.getProperty(cursorRef.current, "rotation") as number;
                        const normalizedRotation = normalizeRotation(currentRotation);
                        spinTimelineRef.current.kill();
                        spinTimelineRef.current = gsap
                            .timeline({ repeat: -1 })
                            .to(cursorRef.current, { rotation: "+=360", duration: spinDuration, ease: "none" });

                        const remainingRatio = Math.max(0, Math.min(1, 1 - normalizedRotation / 360));
                        const resumeDuration = Math.max(0.0001, spinDuration * remainingRatio);

                        gsap.to(cursorRef.current, {
                            rotation: normalizedRotation + 360,
                            duration: resumeDuration,
                            ease: "none",
                            onComplete: () => {
                                spinTimelineRef.current?.restart();
                            },
                        });
                    }
                    resumeTimeoutId = null;
                }, 50);

                cleanupTarget(matches);
            };

            currentLeaveHandler = leaveHandler;
            matches.addEventListener("mouseleave", leaveHandler);
        };

        window.addEventListener("pointermove", onPointerMove);
        window.addEventListener("pointerover", onPointerOver);
        window.addEventListener("scroll", onScroll, { passive: true });
        window.addEventListener("mousedown", onMouseDown);
        window.addEventListener("mouseup", onMouseUp);
        window.addEventListener("blur", onWindowBlur);
        window.addEventListener("mouseout", onWindowMouseOut);

        return () => {
            detachTicker();
            window.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerover", onPointerOver);
            window.removeEventListener("scroll", onScroll);
            window.removeEventListener("mousedown", onMouseDown);
            window.removeEventListener("mouseup", onMouseUp);
            window.removeEventListener("blur", onWindowBlur);
            window.removeEventListener("mouseout", onWindowMouseOut);
            if (resumeTimeoutId !== null) {
                window.clearTimeout(resumeTimeoutId);
            }
            if (activeTarget) {
                cleanupTarget(activeTarget);
            }
            spinTimelineRef.current?.kill();
            targetCornerPositionsRef.current = null;
            activeStrengthRef.current.current = 0;
            body.style.cursor = originalCursor;
            body.classList.remove("target-cursor-active");
        };
    }, [
        constants,
        hideDefaultCursor,
        hoverDuration,
        isMobile,
        moveCursor,
        resolveClosestTarget,
        parallaxOn,
        spinDuration,
        targetSelector,
    ]);

    if (isMobile) return null;

    return (
        <div ref={cursorRef} className="target-cursor-wrapper" aria-hidden="true">
            <div ref={dotRef} className="target-cursor-dot" />
            <div className="target-cursor-corner corner-tl" />
            <div className="target-cursor-corner corner-tr" />
            <div className="target-cursor-corner corner-br" />
            <div className="target-cursor-corner corner-bl" />
        </div>
    );
}
