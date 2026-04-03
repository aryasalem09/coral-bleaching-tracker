import { useMemo, useRef } from "react";
import { motion, useInView } from "motion/react";

type BlurTextProps = {
    text: string;
    className?: string;
    delay?: number;
    direction?: "top" | "bottom" | "left" | "right";
    threshold?: number;
    animateByWord?: boolean;
};

export default function BlurText({
    text,
    className = "",
    delay = 80,
    direction = "bottom",
    threshold = 0.1,
    animateByWord = true,
}: BlurTextProps) {
    const ref = useRef<HTMLSpanElement>(null);
    const isInView = useInView(ref, { once: true, amount: threshold });

    const directionOffset = {
        top: { y: -12 },
        bottom: { y: 12 },
        left: { x: -12 },
        right: { x: 12 },
    }[direction];

    const tokens = useMemo(() => {
        if (animateByWord) {
            return text.split(" ").map((word, i) => ({
                text: word,
                key: `${i}-${word}`,
            }));
        }
        return [...text].map((char, i) => ({
            text: char === " " ? "\u00A0" : char,
            key: `${i}-${char}`,
        }));
    }, [text, animateByWord]);

    return (
        <span ref={ref} className={className} style={{ display: "inline-flex", flexWrap: "wrap", gap: animateByWord ? "0.3em" : 0 }}>
            {tokens.map((token, i) => (
                <motion.span
                    key={token.key}
                    initial={{ opacity: 0, filter: "blur(10px)", ...directionOffset }}
                    animate={
                        isInView
                            ? { opacity: 1, filter: "blur(0px)", x: 0, y: 0 }
                            : { opacity: 0, filter: "blur(10px)", ...directionOffset }
                    }
                    transition={{
                        duration: 0.45,
                        delay: i * (delay / 1000),
                        ease: [0.25, 0.46, 0.45, 0.94],
                    }}
                    style={{ display: "inline-block", willChange: "transform, opacity, filter" }}
                >
                    {token.text}
                </motion.span>
            ))}
        </span>
    );
}
