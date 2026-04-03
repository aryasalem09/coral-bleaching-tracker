import { useRef, useCallback, type ReactNode, type MouseEvent } from "react";

type ClickSparkProps = {
    children: ReactNode;
    sparkColor?: string;
    sparkCount?: number;
    sparkRadius?: number;
    duration?: number;
};

export default function ClickSpark({
    children,
    sparkColor = "#00d4aa",
    sparkCount = 8,
    sparkRadius = 24,
    duration = 400,
}: ClickSparkProps) {
    const containerRef = useRef<HTMLDivElement>(null);

    const spark = useCallback(
        (e: MouseEvent) => {
            const container = containerRef.current;
            if (!container) return;

            const rect = container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            for (let i = 0; i < sparkCount; i++) {
                const angle = (i / sparkCount) * Math.PI * 2;
                const el = document.createElement("div");
                Object.assign(el.style, {
                    position: "absolute",
                    left: `${x}px`,
                    top: `${y}px`,
                    width: "3px",
                    height: "3px",
                    borderRadius: "50%",
                    background: sparkColor,
                    boxShadow: `0 0 6px ${sparkColor}`,
                    pointerEvents: "none",
                    zIndex: "9999",
                    transition: `all ${duration}ms cubic-bezier(0.25, 0.46, 0.45, 0.94)`,
                    opacity: "1",
                });
                container.appendChild(el);

                requestAnimationFrame(() => {
                    el.style.transform = `translate(${Math.cos(angle) * sparkRadius}px, ${Math.sin(angle) * sparkRadius}px)`;
                    el.style.opacity = "0";
                });

                setTimeout(() => el.remove(), duration);
            }
        },
        [sparkColor, sparkCount, sparkRadius, duration]
    );

    return (
        <div ref={containerRef} style={{ position: "relative", display: "contents" }} onClick={spark}>
            {children}
        </div>
    );
}
