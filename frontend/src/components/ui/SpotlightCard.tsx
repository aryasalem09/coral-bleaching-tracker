import { useRef, type ReactNode, type CSSProperties } from "react";

type SpotlightCardProps = {
    children: ReactNode;
    className?: string;
    spotlightColor?: string;
    style?: CSSProperties;
};

export default function SpotlightCard({
    children,
    className = "",
    spotlightColor = "rgba(0, 212, 170, 0.07)",
    style,
}: SpotlightCardProps) {
    const cardRef = useRef<HTMLDivElement>(null);
    const overlayRef = useRef<HTMLDivElement>(null);

    const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!cardRef.current || !overlayRef.current) return;
        const rect = cardRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        overlayRef.current.style.background = `radial-gradient(320px circle at ${x}px ${y}px, ${spotlightColor}, transparent 65%)`;
        overlayRef.current.style.opacity = "1";
    };

    const handleMouseLeave = () => {
        if (!overlayRef.current) return;
        overlayRef.current.style.opacity = "0";
    };

    return (
        <div
            ref={cardRef}
            className={className}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            style={{ position: "relative", overflow: "hidden", ...style }}
        >
            <div
                ref={overlayRef}
                style={{
                    position: "absolute",
                    inset: 0,
                    pointerEvents: "none",
                    opacity: 0,
                    transition: "opacity 300ms ease",
                    zIndex: 1,
                    borderRadius: "inherit",
                }}
            />
            <div style={{ position: "relative", zIndex: 2 }}>{children}</div>
        </div>
    );
}
