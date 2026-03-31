import { motion, useSpring, useTransform, type MotionValue } from "motion/react";
import { useEffect, useMemo, type CSSProperties } from "react";
import "./Counter.css";

type PlaceValue = number | ".";

type CounterProps = {
    value: number;
    fontSize?: number;
    padding?: number;
    gap?: number;
    className?: string;
    places?: PlaceValue[];
};

type CSSVarStyles = CSSProperties & {
    "--counter-font-size"?: string;
    "--counter-height"?: string;
    "--counter-gap"?: string;
};

type NumberGlyphProps = {
    motionValue: MotionValue<number>;
    digit: number;
    height: number;
};

function NumberGlyph({ motionValue, digit, height }: NumberGlyphProps) {
    const y = useTransform(motionValue, (latest) => {
        const placeValue = latest % 10;
        const offset = (10 + digit - placeValue) % 10;
        let result = offset * height;
        if (offset > 5) {
            result -= 10 * height;
        }
        return result;
    });

    return (
        <motion.span className="rb-counter__number" style={{ y }}>
            {digit}
        </motion.span>
    );
}

type DigitColumnProps = {
    height: number;
};

function AnimatedDigitColumn({ place, value, height }: DigitColumnProps & { place: number; value: number }) {
    const steppedValue = Math.floor(value / place);
    const animatedValue = useSpring(steppedValue, { stiffness: 130, damping: 20, mass: 0.5 });

    useEffect(() => {
        animatedValue.set(steppedValue);
    }, [animatedValue, steppedValue]);

    return (
        <span className="rb-counter__digit" style={{ height }}>
            {Array.from({ length: 10 }, (_, digit) => (
                <NumberGlyph key={digit} motionValue={animatedValue} digit={digit} height={height} />
            ))}
        </span>
    );
}

function DigitColumn({ place, value, height }: DigitColumnProps & { place: PlaceValue; value: number }) {
    if (place === ".") {
        return (
            <span className="rb-counter__decimal" style={{ height }}>
                .
            </span>
        );
    }

    return <AnimatedDigitColumn place={place} value={value} height={height} />;
}

function derivePlaces(value: number): PlaceValue[] {
    const valueText = value.toString();
    return [...valueText].map((char, index, arr) => {
        if (char === ".") return ".";
        const decimalIndex = arr.indexOf(".");
        const integerOnly = decimalIndex === -1;
        const exponent = integerOnly ? arr.length - index - 1 : index < decimalIndex ? decimalIndex - index - 1 : -(index - decimalIndex);
        return 10 ** exponent;
    });
}

export default function Counter({
    value,
    fontSize = 32,
    padding = 4,
    gap = 3,
    className = "",
    places,
}: CounterProps) {
    const safeValue = Math.max(0, Number.isFinite(value) ? value : 0);
    const resolvedPlaces = useMemo(() => places ?? derivePlaces(safeValue), [places, safeValue]);
    const height = fontSize + padding;

    const styleVars: CSSVarStyles = {
        "--counter-font-size": `${fontSize}px`,
        "--counter-height": `${height}px`,
        "--counter-gap": `${gap}px`,
    };

    return (
        <span className={`rb-counter ${className}`.trim()} style={styleVars}>
            <span className="rb-counter__digits">
                {resolvedPlaces.map((place, index) => (
                    <DigitColumn key={`${place}-${index}`} place={place} value={safeValue} height={height} />
                ))}
            </span>
            <span className="rb-counter__fade rb-counter__fade--top" />
            <span className="rb-counter__fade rb-counter__fade--bottom" />
        </span>
    );
}
