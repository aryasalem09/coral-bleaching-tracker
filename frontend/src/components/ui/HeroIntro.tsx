import { useEffect, useState } from "react";
import useUniqueDeviceCount from "../../hooks/useUniqueDeviceCount";
import Counter from "./Counter";

const METRIC_ITEMS = [
    {
        id: "metric-dhw",
        title: "Degree Heating Weeks (DHW)",
        description: "DHW tracks cumulative heat stress over the previous 12 weeks above bleaching thresholds.",
        whyMatters: "Sustained thermal stress is strongly linked to bleaching severity and duration.",
    },
    {
        id: "metric-hotspot",
        title: "HotSpot",
        description: "HotSpot measures how much sea surface temperature exceeds local bleaching threshold.",
        whyMatters: "Higher HotSpot values indicate acute thermal stress that can rapidly damage corals.",
    },
    {
        id: "metric-risk-probability",
        title: "Risk Probability",
        description: "Model output from 0 to 1 estimating bleaching risk likelihood at the selected point.",
        whyMatters: "This provides a clear, comparable signal for prioritizing reef monitoring decisions.",
    },
    {
        id: "metric-snap-distance",
        title: "Snap Distance",
        description: "Distance from your clicked point to the nearest valid reef grid cell used for analysis.",
        whyMatters: "It confirms data quality and transparency when clicks land outside reef-supported cells.",
    },
];

export default function HeroIntro() {
    const uniqueDevices = useUniqueDeviceCount();
    const [isMetricsOpen, setMetricsOpen] = useState(false);
    const [expandedMetric, setExpandedMetric] = useState<number | null>(0);

    useEffect(() => {
        const onOpenAboutMetrics = () => {
            setMetricsOpen(true);
            setExpandedMetric((prev) => (prev === null ? 0 : prev));
        };

        window.addEventListener("cbt:open-about-metrics", onOpenAboutMetrics as EventListener);
        return () => {
            window.removeEventListener("cbt:open-about-metrics", onOpenAboutMetrics as EventListener);
        };
    }, []);

    return (
        <section className="hero-intro glass-panel">
            <div className="hero-intro__content">
                <h2 className="hero-intro__title">Estimate Coral Bleaching Risk Worldwide</h2>
                <p className="hero-intro__description">
                    This tool estimates coral bleaching risk using NOAA Degree Heating Weeks (DHW) and HotSpot
                    metrics. Click any reef location to analyze historical stress for a selected date.
                </p>

                <div className="hero-intro__meta">
                    <div className="hero-device-counter cursor-target">
                        <span className="hero-device-counter__label">Unique devices (this browser)</span>
                        <Counter value={uniqueDevices} fontSize={30} />
                    </div>
                </div>

                <section id="about-metrics" className={`metrics-accordion ${isMetricsOpen ? "metrics-accordion--open" : ""}`}>
                    <button
                        type="button"
                        className="metrics-accordion__title cursor-target"
                        aria-expanded={isMetricsOpen}
                        aria-controls="metrics-accordion-panel"
                        onClick={() => setMetricsOpen((prev) => !prev)}
                    >
                        <span>About the metrics</span>
                        <span className="metrics-accordion__chevron" aria-hidden="true">
                            {isMetricsOpen ? "-" : "+"}
                        </span>
                    </button>

                    <div id="metrics-accordion-panel" className="metrics-accordion__panel">
                        {METRIC_ITEMS.map((item, index) => {
                            const isOpen = expandedMetric === index;
                            const contentId = `${item.id}-content`;

                            return (
                                <article key={item.id} className={`metrics-item ${isOpen ? "metrics-item--open" : ""}`}>
                                    <button
                                        type="button"
                                        className="metrics-item__trigger cursor-target"
                                        aria-expanded={isOpen}
                                        aria-controls={contentId}
                                        onClick={() => setExpandedMetric((prev) => (prev === index ? null : index))}
                                    >
                                        <span>{item.title}</span>
                                        <span aria-hidden="true">{isOpen ? "v" : ">"}</span>
                                    </button>
                                    <div id={contentId} className="metrics-item__content">
                                        <p>{item.description}</p>
                                        <p className="metrics-item__why">
                                            <strong>Why it matters:</strong> {item.whyMatters}
                                        </p>
                                    </div>
                                </article>
                            );
                        })}
                    </div>
                </section>
            </div>
        </section>
    );
}
