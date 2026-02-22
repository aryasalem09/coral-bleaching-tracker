import { AnimatePresence, motion, type Variants } from "motion/react";
import {
    type ButtonHTMLAttributes,
    Children,
    type HTMLAttributes,
    type JSX,
    type ReactNode,
    type SVGProps,
    useLayoutEffect,
    useRef,
    useState,
} from "react";
import "./Stepper.css";

type StepperProps = HTMLAttributes<HTMLDivElement> & {
    children: ReactNode;
    initialStep?: number;
    onStepChange?: (step: number) => void;
    onFinalStepCompleted?: () => void;
    stepCircleContainerClassName?: string;
    stepContainerClassName?: string;
    contentClassName?: string;
    footerClassName?: string;
    backButtonProps?: ButtonHTMLAttributes<HTMLButtonElement>;
    nextButtonProps?: ButtonHTMLAttributes<HTMLButtonElement>;
    backButtonText?: string;
    nextButtonText?: string;
    disableStepIndicators?: boolean;
    renderStepIndicator?: (props: RenderStepIndicatorProps) => ReactNode;
};

type RenderStepIndicatorProps = {
    step: number;
    currentStep: number;
    onStepClick: (clicked: number) => void;
};

function cx(...parts: Array<string | undefined | null | false>): string {
    return parts.filter(Boolean).join(" ");
}

export default function Stepper({
    children,
    initialStep = 1,
    onStepChange = () => {},
    onFinalStepCompleted = () => {},
    stepCircleContainerClassName = "",
    stepContainerClassName = "",
    contentClassName = "",
    footerClassName = "",
    backButtonProps = {},
    nextButtonProps = {},
    backButtonText = "Back",
    nextButtonText = "Continue",
    disableStepIndicators = false,
    renderStepIndicator,
    ...rest
}: StepperProps) {
    const [currentStep, setCurrentStep] = useState(initialStep);
    const [direction, setDirection] = useState(0);
    const stepsArray = Children.toArray(children);
    const totalSteps = stepsArray.length;
    const isCompleted = currentStep > totalSteps;
    const isLastStep = currentStep === totalSteps;

    const updateStep = (newStep: number) => {
        setCurrentStep(newStep);
        if (newStep > totalSteps) {
            onFinalStepCompleted();
            return;
        }
        onStepChange(newStep);
    };

    const handleBack = () => {
        if (currentStep <= 1) return;
        setDirection(-1);
        updateStep(currentStep - 1);
    };

    const handleNext = () => {
        if (isLastStep) return;
        setDirection(1);
        updateStep(currentStep + 1);
    };

    const handleComplete = () => {
        setDirection(1);
        updateStep(totalSteps + 1);
    };

    const {
        className: backClassName,
        type: backType,
        onClick: backOnClick,
        ...restBackButtonProps
    } = backButtonProps;
    const {
        className: nextClassName,
        type: nextType,
        onClick: nextOnClick,
        ...restNextButtonProps
    } = nextButtonProps;

    return (
        <div className="rb-stepper-outer" {...rest}>
            <div className={cx("rb-stepper-card", stepCircleContainerClassName)}>
                <div className={cx("rb-stepper-indicator-row", stepContainerClassName)}>
                    {stepsArray.map((_, index) => {
                        const stepNumber = index + 1;
                        const isNotLastStep = index < totalSteps - 1;

                        return (
                            <span className="rb-stepper-indicator-group" key={stepNumber}>
                                {renderStepIndicator ? (
                                    renderStepIndicator({
                                        step: stepNumber,
                                        currentStep,
                                        onStepClick: (clicked) => {
                                            setDirection(clicked > currentStep ? 1 : -1);
                                            updateStep(clicked);
                                        },
                                    })
                                ) : (
                                    <StepIndicator
                                        step={stepNumber}
                                        currentStep={currentStep}
                                        disableStepIndicators={disableStepIndicators}
                                        onClickStep={(clicked) => {
                                            setDirection(clicked > currentStep ? 1 : -1);
                                            updateStep(clicked);
                                        }}
                                    />
                                )}
                                {isNotLastStep ? <StepConnector isComplete={currentStep > stepNumber} /> : null}
                            </span>
                        );
                    })}
                </div>

                <StepContentWrapper
                    isCompleted={isCompleted}
                    currentStep={currentStep}
                    direction={direction}
                    className={cx("rb-stepper-content", contentClassName)}
                >
                    {stepsArray[currentStep - 1]}
                </StepContentWrapper>

                {!isCompleted ? (
                    <div className={cx("rb-stepper-footer", footerClassName)}>
                        <div className={cx("rb-stepper-nav", currentStep > 1 ? "rb-stepper-nav--spread" : "rb-stepper-nav--end")}>
                            {currentStep > 1 ? (
                                <button
                                    type={backType ?? "button"}
                                    className={cx("rb-stepper-back cursor-target", backClassName)}
                                    onClick={(event) => {
                                        backOnClick?.(event);
                                        if (event.defaultPrevented) return;
                                        handleBack();
                                    }}
                                    {...restBackButtonProps}
                                >
                                    {backButtonText}
                                </button>
                            ) : null}

                            <button
                                type={nextType ?? "button"}
                                className={cx("rb-stepper-next cursor-target", nextClassName)}
                                onClick={(event) => {
                                    nextOnClick?.(event);
                                    if (event.defaultPrevented) return;
                                    if (isLastStep) {
                                        handleComplete();
                                    } else {
                                        handleNext();
                                    }
                                }}
                                {...restNextButtonProps}
                            >
                                {isLastStep ? "Complete" : nextButtonText}
                            </button>
                        </div>
                    </div>
                ) : null}
            </div>
        </div>
    );
}

type StepContentWrapperProps = {
    isCompleted: boolean;
    currentStep: number;
    direction: number;
    children: ReactNode;
    className?: string;
};

function StepContentWrapper({
    isCompleted,
    currentStep,
    direction,
    children,
    className,
}: StepContentWrapperProps) {
    const [parentHeight, setParentHeight] = useState(0);

    return (
        <motion.div
            className={className}
            style={{ position: "relative", overflow: "hidden" }}
            animate={{ height: isCompleted ? 0 : parentHeight }}
            transition={{ type: "spring", duration: 0.4 }}
        >
            <AnimatePresence initial={false} mode="sync" custom={direction}>
                {!isCompleted ? (
                    <SlideTransition key={currentStep} direction={direction} onHeightReady={(h) => setParentHeight(h)}>
                        {children}
                    </SlideTransition>
                ) : null}
            </AnimatePresence>
        </motion.div>
    );
}

type SlideTransitionProps = {
    children: ReactNode;
    direction: number;
    onHeightReady: (h: number) => void;
};

function SlideTransition({ children, direction, onHeightReady }: SlideTransitionProps) {
    const containerRef = useRef<HTMLDivElement | null>(null);

    useLayoutEffect(() => {
        if (!containerRef.current) return;
        onHeightReady(containerRef.current.offsetHeight);
    }, [children, onHeightReady]);

    return (
        <motion.div
            ref={containerRef}
            custom={direction}
            variants={stepVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.4 }}
            style={{ position: "absolute", left: 0, right: 0, top: 0 }}
        >
            {children}
        </motion.div>
    );
}

const stepVariants: Variants = {
    enter: (dir: number) => ({
        x: dir >= 0 ? "-100%" : "100%",
        opacity: 0,
    }),
    center: {
        x: "0%",
        opacity: 1,
    },
    exit: (dir: number) => ({
        x: dir >= 0 ? "50%" : "-50%",
        opacity: 0,
    }),
};

type StepProps = {
    children: ReactNode;
};

export function Step({ children }: StepProps): JSX.Element {
    return <div className="rb-stepper-step">{children}</div>;
}

type StepIndicatorProps = {
    step: number;
    currentStep: number;
    onClickStep: (step: number) => void;
    disableStepIndicators?: boolean;
};

function StepIndicator({ step, currentStep, onClickStep, disableStepIndicators }: StepIndicatorProps) {
    const status = currentStep === step ? "active" : currentStep < step ? "inactive" : "complete";

    return (
        <motion.button
            type="button"
            className="rb-stepper-indicator cursor-target"
            animate={status}
            initial={false}
            onClick={() => {
                if (disableStepIndicators || step === currentStep) return;
                onClickStep(step);
            }}
            aria-label={`Go to step ${step}`}
        >
            <motion.div
                variants={{
                    inactive: { scale: 1, backgroundColor: "rgba(130,173,198,0.22)", color: "rgba(232,244,255,0.72)" },
                    active: { scale: 1, backgroundColor: "#1FA6A6", color: "#1FA6A6" },
                    complete: { scale: 1, backgroundColor: "#4dbcf5", color: "#4dbcf5" },
                }}
                transition={{ duration: 0.3 }}
                className="rb-stepper-indicator-inner"
            >
                {status === "complete" ? (
                    <CheckIcon className="rb-stepper-check-icon" />
                ) : status === "active" ? (
                    <span className="rb-stepper-active-dot" />
                ) : (
                    <span className="rb-stepper-number">{step}</span>
                )}
            </motion.div>
        </motion.button>
    );
}

type StepConnectorProps = {
    isComplete: boolean;
};

function StepConnector({ isComplete }: StepConnectorProps) {
    const lineVariants: Variants = {
        incomplete: { width: 0, backgroundColor: "transparent" },
        complete: { width: "100%", backgroundColor: "#4dbcf5" },
    };

    return (
        <span className="rb-stepper-connector">
            <motion.span
                className="rb-stepper-connector-inner"
                variants={lineVariants}
                initial={false}
                animate={isComplete ? "complete" : "incomplete"}
                transition={{ duration: 0.35 }}
            />
        </span>
    );
}

type CheckIconProps = SVGProps<SVGSVGElement>;

function CheckIcon(props: CheckIconProps) {
    return (
        <svg {...props} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <motion.path
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ delay: 0.1, type: "tween", ease: "easeOut", duration: 0.25 }}
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M5 13l4 4L19 7"
            />
        </svg>
    );
}
