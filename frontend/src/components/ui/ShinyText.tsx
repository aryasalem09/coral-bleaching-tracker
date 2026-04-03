import type { CSSProperties, ReactNode } from "react";
import "./ShinyText.css";

type ShinyTextProps = {
    children: ReactNode;
    className?: string;
    shimmerWidth?: number;
    speed?: number;
    as?: "span" | "p" | "div" | "h1" | "h2" | "h3" | "h4";
    style?: CSSProperties;
};

export default function ShinyText({
    children,
    className = "",
    shimmerWidth = 120,
    speed = 3,
    as: Tag = "span",
    style,
}: ShinyTextProps) {
    return (
        <Tag
            className={`shiny-text ${className}`}
            style={{
                ...style,
                "--shimmer-width": `${shimmerWidth}px`,
                "--shimmer-speed": `${speed}s`,
            } as CSSProperties}
        >
            {children}
        </Tag>
    );
}
