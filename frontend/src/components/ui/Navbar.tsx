import { createPortal } from "react-dom";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ServerStatus } from "../../types/server";
import StatusIndicator from "./StatusIndicator";

type NavbarProps = {
    serverStatus: ServerStatus;
    onHelpClick: () => void;
    onTutorialClick: () => void;
    onAboutMetricsClick: () => void;
    onWakeServerClick: () => void;
    githubUrl?: string;
};

type MenuPosition = {
    top: number;
    left: number;
};

export default function Navbar({
    serverStatus,
    onHelpClick,
    onTutorialClick,
    onAboutMetricsClick,
    onWakeServerClick,
    githubUrl,
}: NavbarProps) {
    const [isMenuOpen, setMenuOpen] = useState(false);
    const [menuPosition, setMenuPosition] = useState<MenuPosition>({ top: 0, left: 0 });

    const menuButtonRef = useRef<HTMLButtonElement | null>(null);
    const menuPanelRef = useRef<HTMLDivElement | null>(null);

    const canWakeServer = serverStatus !== "ready";

    const updateMenuPosition = useCallback(() => {
        if (!menuButtonRef.current) return;

        const rect = menuButtonRef.current.getBoundingClientRect();
        const margin = 12;
        const menuWidth = 244;
        const fallbackMenuHeight = 260;
        const menuHeight = menuPanelRef.current?.offsetHeight ?? fallbackMenuHeight;
        const offset = 10;
        const left = Math.max(margin, Math.min(rect.right - menuWidth, window.innerWidth - menuWidth - margin));

        let top = rect.bottom + offset;
        if (top + menuHeight > window.innerHeight - margin) {
            top = Math.max(margin, rect.top - menuHeight - offset);
        }

        setMenuPosition({ top, left });
    }, []);

    useLayoutEffect(() => {
        if (!isMenuOpen) return;

        updateMenuPosition();

        const onWindowChange = () => {
            updateMenuPosition();
        };

        window.addEventListener("resize", onWindowChange);
        window.addEventListener("scroll", onWindowChange, true);
        return () => {
            window.removeEventListener("resize", onWindowChange);
            window.removeEventListener("scroll", onWindowChange, true);
        };
    }, [isMenuOpen, updateMenuPosition]);

    useEffect(() => {
        if (!isMenuOpen) return;

        const onPointerDown = (event: PointerEvent) => {
            const targetNode = event.target as Node | null;
            if (!targetNode) return;
            if (menuPanelRef.current?.contains(targetNode)) return;
            if (menuButtonRef.current?.contains(targetNode)) return;
            setMenuOpen(false);
        };

        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                setMenuOpen(false);
            }
        };

        window.addEventListener("pointerdown", onPointerDown);
        window.addEventListener("keydown", onKeyDown);

        return () => {
            window.removeEventListener("pointerdown", onPointerDown);
            window.removeEventListener("keydown", onKeyDown);
        };
    }, [isMenuOpen]);

    const runAndClose = (action: () => void) => {
        action();
        setMenuOpen(false);
    };

    return (
        <header className="top-nav">
            <div className="brand-lockup">
                <h1 className="brand-lockup__title">Coral Bleaching Risk Lab</h1>
                <p className="brand-lockup__subtitle">NOAA DHW + HotSpot Model</p>
            </div>

            <div className="top-nav__actions">
                <StatusIndicator status={serverStatus} />

                <button
                    ref={menuButtonRef}
                    type="button"
                    className="nav-button nav-button--menu cursor-target"
                    aria-expanded={isMenuOpen}
                    aria-haspopup="menu"
                    aria-controls="top-nav-menu"
                    onClick={() => setMenuOpen((prev) => !prev)}
                >
                    Menu
                </button>
            </div>

            {isMenuOpen
                ? createPortal(
                      <div
                          id="top-nav-menu"
                          ref={menuPanelRef}
                          role="menu"
                          aria-label="Navigation menu"
                          className="nav-menu glass-panel"
                          style={{ top: `${menuPosition.top}px`, left: `${menuPosition.left}px` }}
                      >
                          <button
                              type="button"
                              role="menuitem"
                              className="nav-menu__item cursor-target"
                              onClick={() => runAndClose(onHelpClick)}
                          >
                              Help
                          </button>
                          <button
                              type="button"
                              role="menuitem"
                              className="nav-menu__item cursor-target"
                              onClick={() => runAndClose(onTutorialClick)}
                          >
                              Tutorial
                          </button>
                          <button
                              type="button"
                              role="menuitem"
                              className="nav-menu__item cursor-target"
                              onClick={() => runAndClose(onAboutMetricsClick)}
                          >
                              About Metrics
                          </button>
                          <button
                              type="button"
                              role="menuitem"
                              className="nav-menu__item cursor-target"
                              disabled={!canWakeServer}
                              onClick={() => runAndClose(onWakeServerClick)}
                          >
                              Wake server
                          </button>
                          {githubUrl ? (
                              <a
                                  role="menuitem"
                                  className="nav-menu__item cursor-target"
                                  href={githubUrl}
                                  target="_blank"
                                  rel="noreferrer"
                                  onClick={() => setMenuOpen(false)}
                              >
                                  GitHub
                              </a>
                          ) : null}
                      </div>,
                      document.body
                  )
                : null}
        </header>
    );
}
