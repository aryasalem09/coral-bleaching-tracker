import { useEffect, useRef } from "react";

type AuroraProps = {
    colorStops?: string[];
    amplitude?: number;
    blend?: number;
    speed?: number;
    className?: string;
};

export default function Aurora({
    colorStops = ["#00d4aa", "#3b82f6", "#ff6b6b", "#f0a500"],
    amplitude = 1.0,
    blend = 0.5,
    speed = 0.002,
    className = "",
}: AuroraProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animRef = useRef<number>(0);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        let time = 0;

        const resize = () => {
            const dpr = Math.min(window.devicePixelRatio, 2);
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
        };

        resize();
        window.addEventListener("resize", resize);

        const draw = () => {
            const w = canvas.getBoundingClientRect().width;
            const h = canvas.getBoundingClientRect().height;

            ctx.clearRect(0, 0, w, h);

            for (let i = 0; i < colorStops.length; i++) {
                const phase = (i / colorStops.length) * Math.PI * 2;
                const x = w * (0.3 + 0.4 * Math.sin(time + phase));
                const y = h * (0.3 + 0.4 * Math.cos(time * 0.7 + phase * 1.3));
                const radius = Math.max(w, h) * (0.3 + 0.15 * amplitude * Math.sin(time * 1.2 + i));

                const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
                gradient.addColorStop(0, colorStops[i] + "18");
                gradient.addColorStop(0.5, colorStops[i] + "08");
                gradient.addColorStop(1, "transparent");

                ctx.globalCompositeOperation = "lighter";
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, w, h);
            }

            time += speed;
            animRef.current = requestAnimationFrame(draw);
        };

        animRef.current = requestAnimationFrame(draw);

        return () => {
            cancelAnimationFrame(animRef.current);
            window.removeEventListener("resize", resize);
        };
    }, [colorStops, amplitude, speed, blend]);

    return (
        <canvas
            ref={canvasRef}
            className={className}
            style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
                zIndex: 0,
            }}
        />
    );
}
