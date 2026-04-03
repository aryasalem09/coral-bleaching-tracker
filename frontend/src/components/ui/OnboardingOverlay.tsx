import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import Stepper, { Step } from "./Stepper";
import GradientText from "./GradientText";
import BlurText from "./BlurText";
import Counter from "./Counter";
import "./OnboardingOverlay.css";

const STORAGE_KEY = "cbt_onboarding_done";

type OnboardingOverlayProps = {
    onComplete: () => void;
};

type OnboardingParticle = {
    left: string;
    top: string;
    animationDelay: string;
    animationDuration: string;
    width: string;
    height: string;
    opacity: number;
};

function createParticles(count: number): OnboardingParticle[] {
    return Array.from({ length: count }, () => ({
        left: `${Math.random() * 100}%`,
        top: `${Math.random() * 100}%`,
        animationDelay: `${Math.random() * 5}s`,
        animationDuration: `${4 + Math.random() * 6}s`,
        width: `${2 + Math.random() * 4}px`,
        height: `${2 + Math.random() * 4}px`,
        opacity: 0.2 + Math.random() * 0.5,
    }));
}

export default function OnboardingOverlay({ onComplete }: OnboardingOverlayProps) {
    const [visible, setVisible] = useState(() => {
        if (typeof window === "undefined") return false;
        return !localStorage.getItem(STORAGE_KEY);
    });
    const [particles] = useState(() => createParticles(30));

    useEffect(() => {
        if (!visible) {
            onComplete();
        }
    }, [onComplete, visible]);

    const handleFinish = () => {
        localStorage.setItem(STORAGE_KEY, "1");
        setVisible(false);
        setTimeout(onComplete, 600);
    };

    const handleSkip = () => {
        localStorage.setItem(STORAGE_KEY, "1");
        setVisible(false);
        setTimeout(onComplete, 600);
    };

    if (!visible) return null;

    return (
        <AnimatePresence>
            {visible && (
                <motion.div
                    className="onboarding-backdrop"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    {/* Animated background particles */}
                    <div className="onboarding-particles">
                        {particles.map((particle, i) => (
                            <div
                                key={i}
                                className="onboarding-particle"
                                style={particle}
                            />
                        ))}
                    </div>

                    <motion.div
                        className="onboarding-content"
                        initial={{ y: 40, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        exit={{ y: -40, opacity: 0 }}
                        transition={{ duration: 0.6, delay: 0.2, ease: [0.25, 0.46, 0.45, 0.94] }}
                    >
                        <button type="button" className="onboarding-skip" onClick={handleSkip}>
                            Skip intro
                        </button>

                        <Stepper
                            onFinalStepCompleted={handleFinish}
                            nextButtonText="Next"
                            backButtonText="Back"
                        >
                            <Step>
                                <div className="onboarding-step">
                                    <div className="onboarding-step__icon">
                                        <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                                            <circle cx="24" cy="24" r="20" stroke="url(#g1)" strokeWidth="2" opacity="0.5" />
                                            <circle cx="24" cy="24" r="8" fill="url(#g1)" opacity="0.8" />
                                            <defs>
                                                <linearGradient id="g1" x1="4" y1="4" x2="44" y2="44">
                                                    <stop stopColor="#00d4aa" />
                                                    <stop offset="1" stopColor="#3b82f6" />
                                                </linearGradient>
                                            </defs>
                                        </svg>
                                    </div>
                                    <h2 className="onboarding-step__title">
                                        <GradientText as="span" from="#00d4aa" to="#3b82f6" animate speed={5}>
                                            Welcome to the Coral Bleaching Tracker
                                        </GradientText>
                                    </h2>
                                    <p className="onboarding-step__body">
                                        <BlurText
                                            text="Use the map to compare survey results, NOAA heat stress, and a 4-week bleaching risk forecast."
                                            delay={30}
                                        />
                                    </p>
                                </div>
                            </Step>

                            <Step>
                                <div className="onboarding-step">
                                    <div className="onboarding-step__icon">
                                        <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                                            <path d="M24 4L44 14V34L24 44L4 34V14L24 4Z" stroke="#00d4aa" strokeWidth="1.5" opacity="0.5" />
                                            <circle cx="24" cy="24" r="5" fill="#00d4aa" opacity="0.8" />
                                            <circle cx="14" cy="19" r="3" fill="#ff6b6b" opacity="0.6" />
                                            <circle cx="34" cy="19" r="3" fill="#3b82f6" opacity="0.6" />
                                            <circle cx="24" cy="36" r="3" fill="#f0a500" opacity="0.6" />
                                        </svg>
                                    </div>
                                    <h2 className="onboarding-step__title">
                                        Explore the Map
                                    </h2>
                                    <p className="onboarding-step__body">
                                        <BlurText
                                            text="Pan and zoom to find reef sites around the world. Each dot is a monitored site. Click one to open its details."
                                            delay={25}
                                        />
                                    </p>
                                    <div className="onboarding-step__hint">
                                        <div className="onboarding-hint-dot" />
                                        <span>Each dot is a reef site in the dataset</span>
                                    </div>
                                </div>
                            </Step>

                            <Step>
                                <div className="onboarding-step">
                                    <div className="onboarding-step__icon">
                                        <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                                            <rect x="6" y="8" width="36" height="32" rx="4" stroke="#00d4aa" strokeWidth="1.5" opacity="0.4" />
                                            <rect x="10" y="14" width="12" height="4" rx="2" fill="#ff6b6b" opacity="0.6" />
                                            <rect x="10" y="22" width="16" height="4" rx="2" fill="#f0a500" opacity="0.6" />
                                            <rect x="10" y="30" width="10" height="4" rx="2" fill="#3b82f6" opacity="0.6" />
                                        </svg>
                                    </div>
                                    <h2 className="onboarding-step__title">Three Simple Views</h2>
                                    <p className="onboarding-step__body">
                                        <BlurText
                                            text="Observed shows survey results. Heat Stress shows NOAA heat data. Forecast shows the model's estimate for the next 4 weeks."
                                            delay={20}
                                        />
                                    </p>
                                    <div className="onboarding-layers-preview">
                                        <div className="onboarding-layer" style={{ borderColor: "rgba(255, 107, 107, 0.4)" }}>
                                            <div className="onboarding-layer__dot" style={{ background: "#ff6b6b" }} />
                                            <span>Observed</span>
                                        </div>
                                        <div className="onboarding-layer" style={{ borderColor: "rgba(240, 165, 0, 0.4)" }}>
                                            <div className="onboarding-layer__dot" style={{ background: "#f0a500" }} />
                                            <span>Stress</span>
                                        </div>
                                        <div className="onboarding-layer" style={{ borderColor: "rgba(59, 130, 246, 0.4)" }}>
                                            <div className="onboarding-layer__dot" style={{ background: "#3b82f6" }} />
                                            <span>Forecast</span>
                                        </div>
                                    </div>
                                </div>
                            </Step>

                            <Step>
                                <div className="onboarding-step">
                                    <div className="onboarding-step__icon">
                                        <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                                            <path d="M24 4L28 18H44L31 27L35 42L24 33L13 42L17 27L4 18H20L24 4Z" stroke="url(#g2)" strokeWidth="1.5" />
                                            <defs>
                                                <linearGradient id="g2" x1="4" y1="4" x2="44" y2="44">
                                                    <stop stopColor="#00d4aa" />
                                                    <stop offset="1" stopColor="#f0a500" />
                                                </linearGradient>
                                            </defs>
                                        </svg>
                                    </div>
                                    <h2 className="onboarding-step__title">
                                        <GradientText as="span" from="#00d4aa" via="#f0a500" to="#ff6b6b">
                                            Ready to Explore
                                        </GradientText>
                                    </h2>
                                    <p className="onboarding-step__body">
                                        <BlurText
                                            text="Click any reef site to open the survey timeline, NOAA heat history, and the 4-week forecast."
                                            delay={25}
                                        />
                                    </p>
                                    <div className="onboarding-stat-row">
                                        <div className="onboarding-stat">
                                            <Counter value={3700} fontSize={28} />
                                            <span>Reef Sites</span>
                                        </div>
                                        <div className="onboarding-stat">
                                            <Counter value={12} fontSize={28} />
                                            <span>History Weeks</span>
                                        </div>
                                        <div className="onboarding-stat">
                                            <Counter value={3} fontSize={28} />
                                            <span>Views</span>
                                        </div>
                                    </div>
                                </div>
                            </Step>
                        </Stepper>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}
