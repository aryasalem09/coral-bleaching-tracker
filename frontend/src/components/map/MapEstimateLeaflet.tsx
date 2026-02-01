import { useMemo, useState } from "react";
import L from "leaflet";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import iconUrl from "leaflet/dist/images/marker-icon.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";
import { Circle, MapContainer, Marker, Polyline, TileLayer, useMapEvents } from "react-leaflet";
import { apiEstimate } from "../../lib/api";
import type { EstimateResponse } from "../../lib/api";

L.Icon.Default.mergeOptions({
    iconRetinaUrl,
    iconUrl,
    shadowUrl,
});

function ClickHandler(props: { onClick: (lat: number, lon: number) => void }) {
    useMapEvents({
        click(e) {
            props.onClick(e.latlng.lat, e.latlng.lng);
        },
    });
    return null;
}

export default function MapEstimateLeaflet() {
    const [dateStr, setDateStr] = useState("2024-01-01");
    const [loading, setLoading] = useState(false);
    const [err, setErr] = useState("");
    const [res, setRes] = useState<EstimateResponse | null>(null);

    const inputPos = useMemo(() => {
        if (!res) return null;
        return [res.input_lat, res.input_lon] as [number, number];
    }, [res]);

    const usedPos = useMemo(() => {
        if (!res) return null;
        return [res.used_lat, res.used_lon] as [number, number];
    }, [res]);

    const circleRadiusM = useMemo(() => {
        if (!res?.snapped) return 0;
        const km = Number(res.snap_km);
        if (!Number.isFinite(km) || km <= 0) return 0;
        return km * 1000;
    }, [res]);

    const line = useMemo(() => {
        if (!res?.snapped) return null;
        if (!inputPos || !usedPos) return null;
        return [inputPos, usedPos] as [number, number][];
    }, [res, inputPos, usedPos]);

    async function runEstimate(lat: number, lon: number) {
        setErr("");
        setLoading(true);
        setRes(null);

        try {
            const data = await apiEstimate({ lat, lon, date: dateStr });
            setRes(data);
        } catch (e: any) {
            setErr(e?.message ?? "network error calling API");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div style={{ display: "grid", gridTemplateColumns: "1.35fr 1fr", gap: 16 }}>
            <div>
                <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 10 }}>
                    <div style={{ fontWeight: 700 }}>date</div>
                    <input
                        type="date"
                        value={dateStr}
                        onChange={(e) => setDateStr(e.target.value)}
                        style={{
                            padding: "8px 10px",
                            borderRadius: 10,
                            border: "1px solid rgba(0,0,0,0.2)",
                        }}
                    />
                    {loading ? <div style={{ opacity: 0.8 }}>loading...</div> : null}
                </div>

                <div style={{ height: 520, borderRadius: 14, overflow: "hidden", border: "1px solid rgba(0,0,0,0.12)" }}>
                    <MapContainer center={[18.3, -66.5]} zoom={5} style={{ height: "100%", width: "100%" }}>
                        <TileLayer
                            attribution='&copy; OpenStreetMap contributors'
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        />

                        <ClickHandler onClick={runEstimate} />

                        {inputPos ? <Marker position={inputPos} /> : null}
                        {usedPos ? <Marker position={usedPos} /> : null}

                        {res?.snapped && inputPos ? (
                            <Circle
                                center={inputPos}
                                radius={circleRadiusM}
                                pathOptions={{ weight: 2, fillOpacity: 0.12 }}
                            />
                        ) : null}

                        {line ? <Polyline positions={line} pathOptions={{ weight: 2 }} /> : null}
                    </MapContainer>
                </div>

                <div style={{ marginTop: 10, fontSize: 13, opacity: 0.85 }}>
                    click anywhere - if the point is invalid, the API snaps to the nearest valid reef cell and the circle shows the snap distance.
                </div>
            </div>

            <div style={{ padding: 16, borderRadius: 14, border: "1px solid rgba(0,0,0,0.12)" }}>
                <div style={{ fontSize: 18, fontWeight: 900, marginBottom: 8 }}>estimate</div>

                {err ? (
                    <div style={{ padding: 12, borderRadius: 12, background: "rgba(255,0,0,0.08)" }}>
                        <div style={{ fontWeight: 800, marginBottom: 6 }}>error</div>
                        <div style={{ opacity: 0.9 }}>{err}</div>
                    </div>
                ) : null}

                {!res && !err ? <div style={{ opacity: 0.85 }}>click a point on the map to run an estimate.</div> : null}

                {res ? (
                    <div style={{ display: "grid", gap: 10 }}>
                        <div style={{ display: "grid", gap: 6 }}>
                            <div><b>risk_prob:</b> {(res.risk_prob * 100).toFixed(1)}%</div>
                            <div><b>risk_flag:</b> {res.risk_flag === 1 ? "high" : "low"}</div>
                            <div><b>dhw:</b> {Number(res.dhw).toFixed(2)}</div>
                            <div><b>hotspot:</b> {Number(res.hotspot).toFixed(2)}</div>
                        </div>

                        <div style={{ paddingTop: 10, borderTop: "1px solid rgba(0,0,0,0.12)" }}>
                            <div style={{ fontWeight: 900, marginBottom: 6 }}>snap</div>
                            {res.snapped ? (
                                <div>
                                    snapped <b>{Number(res.snap_km).toFixed(2)} km</b> (circle radius)
                                </div>
                            ) : (
                                <div>no snap needed</div>
                            )}
                        </div>

                        <div style={{ paddingTop: 10, borderTop: "1px solid rgba(0,0,0,0.12)", fontSize: 13, opacity: 0.85 }}>
                            <div><b>input</b>: {res.input_lat.toFixed(4)}, {res.input_lon.toFixed(4)}</div>
                            <div><b>used</b>: {res.used_lat.toFixed(4)}, {res.used_lon.toFixed(4)}</div>
                            <div><b>date</b>: {res.date}</div>
                        </div>
                    </div>
                ) : null}
            </div>
        </div>
    );
}
