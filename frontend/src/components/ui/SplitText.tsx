import { useMemo, useRef, useCallback } from "react";
import { motion, useInView, type TargetAndTransition } from "motion/react";

type SplitTextProps = {
    text: string;
    className?: string;
    delay?: number;
    animationFrom?: TargetAndTransition;
    animationTo?: TargetAndTransition;
    threshold?: number;
    rootMargin?: string;
    textAlign?: "left" | "right" | "center" | "justify";
    onLetterAnimationComplete?: () => void;
};

export default function SplitText({
    text,
    className = "",
    delay = 50,
    animationFrom = { opacity: 0, y: 30 },
    animationTo = { opacity: 1, y: 0 },
    threshold = 0.1,
    textAlign = "left",
    onLetterAnimationComplete,
}: SplitTextProps) {
    const ref = useRef<HTMLSpanElement>(null);
    const isInView = useInView(ref, { once: true, amount: threshold });
    const completedRef = useRef(0);

    const letters = useMemo(() => {
        const result: { char: string; key: string }[] = [];
        let index = 0;
        for (const word of text.split(" ")) {
            for (const char of word) {
                result.push({ char, key: `${index}-${char}` });
                index++;
            }
            result.push({ char: "\u00A0", key: `space-${index}` });
            index++;
        }
        return result.slice(0, -1);
    }, [text]);

    const handleComplete = useCallback(() => {
        completedRef.current += 1;
        if (completedRef.current === letters.length && onLetterAnimationComplete) {
            onLetterAnimationComplete();
        }
    }, [letters.length, onLetterAnimationComplete]);

    return (
        <span
            ref={ref}
            className={className}
            style={{
                textAlign,
                display: "inline",
                overflow: "hidden",
            }}
        >
            {letters.map((letter, i) => (
                <motion.span
                    key={letter.key}
                    initial={animationFrom}
                    animate={isInView ? animationTo : animationFrom}
                    transition={{
                        duration: 0.5,
                        delay: i * (delay / 1000),
                        ease: [0.215, 0.61, 0.355, 1],
                    }}
                    onAnimationComplete={handleComplete}
                    style={{
                        display: "inline-block",
                        willChange: "transform, opacity",
                    }}
                >
                    {letter.char}
                </motion.span>
            ))}
        </span>
    );
}
