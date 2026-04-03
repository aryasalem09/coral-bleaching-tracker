import type { CSSProperties, ReactNode } from "react";

type GradientTextProps = {
    children: ReactNode;
    className?: string;
    from?: string;
    via?: string;
    to?: string;
    animate?: boolean;
    speed?: number;
    as?: "span" | "p" | "h1" | "h2" | "h3" | "h4";
    style?: CSSProperties;
};

export default function GradientText({
    children,
    className = "",
    from = "#00d4aa",
    via,
    to = "#3b82f6",
    animate = false,
    speed = 4,
    as: Tag = "span",
    style,
}: GradientTextProps) {
    const gradient = via
        ? `linear-gradient(135deg, ${from}, ${via}, ${to})`
        : `linear-gradient(135deg, ${from}, ${to})`;

    return (
        <Tag
            className={`gradient-text ${animate ? "gradient-text--animated" : ""} ${className}`}
            style={{
                ...style,
                backgroundImage: gradient,
                backgroundSize: animate ? "200% auto" : "100% auto",
                WebkitBackgroundClip: "text",
                backgroundClip: "text",
                WebkitTextFillColor: "transparent",
                animation: animate ? `gradient-shift ${speed}s ease infinite` : undefined,
            } as CSSProperties}
        >
            {children}
        </Tag>
    );
}
