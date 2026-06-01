import React, { useState, useEffect, useRef, useCallback, useMemo, Fragment } from "react";
import { createChart, CandlestickSeries, HistogramSeries, createSeriesMarkers } from "lightweight-charts";

const API_BASE  = "/api";
const WS_BASE   = `ws://${window.location.host}/api`;
const TZ_OFFSET = -new Date().getTimezoneOffset() * 60; // seconds offset from UTC to local
const toLocal   = (ms) => ms / 1000 + TZ_OFFSET;        // Binance ms → local unix seconds
const toLocalDate = (ts) => {                            // unix seconds → local Date object
  const raw = ts < 4102444800 ? ts * 1000 : ts;
  return new Date(raw);
};

const COLORS = {
  bg: "#080b10",
  panel: "#0d1117",
  border: "#161c26",
  border2: "#1e2830",
  accent: "#00d4aa",
  accentDim: "#003d30",
  gold: "#f59e0b",
  red: "#ef4444",
  green: "#22c55e",
  blue: "#3b82f6",
  purple: "#a855f7",
  text: "#e2e8f0",
  muted: "#64748b",
  muted2: "#334155",
};

const marketPairs = [
  { symbol: "BTCUSDT", name: "Bitcoin",  price: "—", change: "—", up: true },
  { symbol: "ETHUSDT", name: "Ethereum", price: "—", change: "—", up: true },
];

// ── date helpers ───────────────────────────────────────────────────
function newsDate(n) {
  const raw = n.published_ts || n.received_at;
  if (!raw) return null;
  return toLocalDate(raw);
}
function dateKey(d) {
  const y   = d.getFullYear();
  const m   = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}
function itemTime(item) {
  // Always derive time from published_ts (UTC unix seconds → local Date)
  // Never use item.time directly — it's stored as UTC string, not local time
  const raw = item.published_ts || item.received_at;
  if (!raw) return "—";
  const d = new Date(raw * 1000);
  return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
}

function sentimentLabel(s) {
  if (s === "positive") return "Bullish";
  if (s === "negative") return "Bearish";
  return "Neutral";
}
function signalAction(type, modelScore) {
  if (type === "BUY")  return modelScore >= 0.70 ? "Strong Buy"  : "Buy";
  if (type === "SELL") return modelScore >= 0.70 ? "Strong Sell" : "Sell";
  return "Neutral";
}
// Normalize raw model score (0.30–0.48) → 0–1 for old cached items
function clientNormalize(item) {
  if (item.score_normalized) return item; // already normalized by main.py
  const norm = (raw, min, max) => Math.max(0, Math.min(1, (raw - min) / (max - min)));
  return {
    ...item,
    model_score:    norm(Math.abs(item.model_score    || 0), 0.30, 0.48),
    model_score_1h: norm(Math.abs(item.model_score_1h || 0), 0.30, 0.63),
    score_normalized: true,
  };
}
function fmtScore(score) {
  if (score == null) return "—";
  return `${Math.round(Math.abs(score) * 100)}%`;
}
// Estimate predicted BTC % change from model output
// Scale: model_score(0–1) × typical move magnitude × direction sign
function predictedChange(item) {
  const s15  = Math.abs(item.model_score    || 0);
  const s1h  = Math.abs(item.model_score_1h || 0);
  const sign = item.type === "BUY" ? 1 : item.type === "SELL" ? -1 : 0;
  // 15m: when impact occurs, BTC typically moves 0.3–1.5%; scale score linearly
  const p15 = sign * s15 * 1.2;
  // 1h: broader moves, typically 0.5–2.5%
  const p1h = sign * s1h * 2.0;
  return { p15: parseFloat(p15.toFixed(3)), p1h: parseFloat(p1h.toFixed(3)) };
}
function fmtChange(v, prefix = true) {
  if (v == null || v === 0) return "—";
  const n = parseFloat(v);
  if (isNaN(n) || n === 0) return "—";
  return `${n > 0 ? "+" : ""}${n.toFixed(2)}%`;
}
function channelLogo(channel = "") {
  const c = channel.toLowerCase();
  if (c.includes("coinmarketcap")) return "📊";
  if (c.includes("coindesk"))      return "📰";
  if (c.includes("cointelegraph")) return "📡";
  if (c.includes("block"))         return "🔷";
  if (c.includes("porter"))        return "🏦";
  if (c.includes("binance"))       return "🔶";
  return "📌";
}
function fmtAge(ageMin) {
  if (ageMin < 1)  return "just now";
  if (ageMin < 60) return `${Math.round(ageMin)}m ago`;
  return `${Math.round(ageMin / 60)}h ago`;
}

// ── WebSocket hook ─────────────────────────────────────────────────
function useWebSocket(path, onMessage) {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    let retryTimer;
    function connect() {
      const ws = new WebSocket(`${WS_BASE}${path}`);
      wsRef.current = ws;
      ws.onopen    = () => setConnected(true);
      ws.onclose   = () => { setConnected(false); retryTimer = setTimeout(connect, 3000); };
      ws.onerror   = () => ws.close();
      ws.onmessage = (e) => { try { onMessage(JSON.parse(e.data)); } catch {} };
    }
    connect();
    return () => { clearTimeout(retryTimer); wsRef.current?.close(); };
  }, [path]);
  return connected;
}

// ── CalendarPicker ─────────────────────────────────────────────────
function CalendarPicker({ selected, onChange, dbDates = [] }) {
  const today   = new Date();
  // Default view to the month that has the most recent data
  const latestDataDate = dbDates.length
    ? new Date(dbDates[dbDates.length - 1] + "T00:00:00")
    : today;
  const [view, setView] = useState({ y: latestDataDate.getFullYear(), m: latestDataDate.getMonth() });

  const daysInMonth = (y, m) => new Date(y, m + 1, 0).getDate();
  const firstDay    = (y, m) => new Date(y, m, 1).getDay();

  const days    = daysInMonth(view.y, view.m);
  const offset  = firstDay(view.y, view.m);
  const cells   = Array(offset).fill(null).concat(Array.from({ length: days }, (_, i) => i + 1));
  const selKey  = selected ? dateKey(selected) : null;

  const prevMonth = () => setView(v => v.m === 0 ? { y: v.y - 1, m: 11 } : { y: v.y, m: v.m - 1 });
  const nextMonth = () => setView(v => v.m === 11 ? { y: v.y + 1, m: 0 } : { y: v.y, m: v.m + 1 });

  const monthName = new Date(view.y, view.m, 1).toLocaleString("default", { month: "long", year: "numeric" });

  return (
    <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.border2}`, borderRadius: 10, padding: 12, width: 220 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <button onClick={prevMonth} style={{ background: "none", border: "none", color: COLORS.muted, cursor: "pointer", fontSize: 14 }}>‹</button>
        <span style={{ fontSize: 11, fontWeight: 600, color: COLORS.text }}>{monthName}</span>
        <button onClick={nextMonth} style={{ background: "none", border: "none", color: COLORS.muted, cursor: "pointer", fontSize: 14 }}>›</button>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 2, marginBottom: 4 }}>
        {["Su","Mo","Tu","We","Th","Fr","Sa"].map(d => (
          <div key={d} style={{ textAlign: "center", fontSize: 9, color: COLORS.muted, padding: "2px 0" }}>{d}</div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 2 }}>
        {cells.map((day, i) => {
          if (!day) return <div key={i} />;
          const k        = `${view.y}-${String(view.m + 1).padStart(2,"0")}-${String(day).padStart(2,"0")}`;
          const isToday  = k === dateKey(today);
          const isSel    = k === selKey;
          const hasNews  = dbDates.includes(k);
          const isFuture = new Date(view.y, view.m, day) > today;
          return (
            <button key={i} disabled={isFuture} onClick={() => onChange(isSel ? null : new Date(view.y, view.m, day))} style={{
              position: "relative", textAlign: "center", fontSize: 11,
              padding: "4px 2px", borderRadius: 5, border: "none", cursor: isFuture ? "default" : "pointer",
              background: isSel ? COLORS.accent : isToday ? `${COLORS.accent}22` : "transparent",
              color: isFuture ? COLORS.border2 : isSel ? "#000" : isToday ? COLORS.accent : COLORS.text,
              fontWeight: isSel || isToday ? 700 : 400,
            }}>
              {day}
              {hasNews && !isSel && (
                <div style={{ position: "absolute", bottom: 1, left: "50%", transform: "translateX(-50%)", width: 3, height: 3, borderRadius: "50%", background: COLORS.accent }} />
              )}
            </button>
          );
        })}
      </div>
      {selected && (
        <button onClick={() => onChange(null)} style={{
          marginTop: 8, width: "100%", padding: "5px 0", borderRadius: 6, border: `1px solid ${COLORS.border2}`,
          background: "transparent", color: COLORS.muted, fontSize: 10, cursor: "pointer",
        }}>Clear — show today</button>
      )}
    </div>
  );
}

// ── Components ─────────────────────────────────────────────────────
const INTERVAL_MAP = { "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d" };

function BinanceChart({ symbol, interval = "1h", news = [] }) {
  const containerRef = useRef(null);
  const chartRef     = useRef(null);
  const candleRef    = useRef(null);
  const volRef       = useRef(null);
  const wsRef        = useRef(null);
  const [price, setPrice]   = useState(null);
  const [change, setChange] = useState(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Create chart
    const chart = createChart(containerRef.current, {
      layout:     { background: { color: COLORS.bg }, textColor: COLORS.muted },
      grid:       { vertLines: { color: COLORS.panel }, horzLines: { color: COLORS.panel } },
      crosshair:  { mode: 1 },
      rightPriceScale: { borderColor: COLORS.border2 },
      timeScale:  { borderColor: COLORS.border2, timeVisible: true, secondsVisible: false },
      width:  containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
    });
    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e", downColor: "#ef4444",
      borderUpColor: "#22c55e", borderDownColor: "#ef4444",
      wickUpColor: "#22c55e", wickDownColor: "#ef4444",
    });
    candleRef.current = candleSeries;

    const volSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
    });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
    volRef.current = volSeries;

    // Resize observer
    const ro = new ResizeObserver(() => {
      if (containerRef.current)
        chart.applyOptions({ width: containerRef.current.clientWidth, height: containerRef.current.clientHeight });
    });
    ro.observe(containerRef.current);

    // Fetch historical klines from Jan 1 2026 in batches of 1000
    const binSymbol = symbol.replace("BINANCE:", "");
    const START_2026 = 1735689600000; // Jan 1 2026 00:00 UTC in ms

    const intervalMs = { "1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000 };
    const ivMs = intervalMs[interval] || 3600000;

    (async () => {
      const allKlines = [];
      let startTime = START_2026;
      const now = Date.now();
      try {
        while (startTime < now) {
          const url = `${API_BASE}/proxy/klines?symbol=${binSymbol}&interval=${interval}&limit=1000&startTime=${startTime}`;
          const batch = await fetch(url).then(r => r.json());
          if (!Array.isArray(batch) || batch.length === 0) break;
          allKlines.push(...batch);
          const lastTs = batch[batch.length - 1][0];
          if (batch.length < 1000) break;
          startTime = lastTs + ivMs;
        }
        if (!allKlines.length) return;
        // deduplicate by time
        const seen = new Set();
        const unique = allKlines.filter(k => { const t = k[0]; if (seen.has(t)) return false; seen.add(t); return true; });
        const candles = unique.map(k => ({ time: toLocal(k[0]), open: parseFloat(k[1]), high: parseFloat(k[2]), low: parseFloat(k[3]), close: parseFloat(k[4]) }));
        const volumes = unique.map(k => ({ time: toLocal(k[0]), value: parseFloat(k[5]), color: parseFloat(k[4]) >= parseFloat(k[1]) ? "#22c55e33" : "#ef444433" }));
        candleSeries.setData(candles);
        volSeries.setData(volumes);
        chart.timeScale().fitContent();
        const last  = candles[candles.length - 1];
        const first = candles[0];
        setPrice(last.close);
        setChange(((last.close - first.open) / first.open * 100).toFixed(2));
      } catch {}
    })();

    // Live WebSocket via local proxy
    const ws = new WebSocket(`ws://localhost:8000/proxy/stream/${binSymbol}/${interval}`);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const k = JSON.parse(e.data).k;
      const candle = { time: toLocal(k.t), open: parseFloat(k.o), high: parseFloat(k.h), low: parseFloat(k.l), close: parseFloat(k.c) };
      const vol    = { time: toLocal(k.t), value: parseFloat(k.v), color: parseFloat(k.c) >= parseFloat(k.o) ? "#22c55e33" : "#ef444433" };
      candleSeries.update(candle);
      volSeries.update(vol);
      setPrice(parseFloat(k.c));
    };

    return () => {
      ws.close();
      ro.disconnect();
      chart.remove();
    };
  }, [symbol, interval]);

  // Update markers whenever news changes
  useEffect(() => {
    if (!candleRef.current || !news.length) return;
    const intervalSeconds = { "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400 };
    const ivSec = intervalSeconds[interval] || 3600;
    const signals = news
      .filter(n => n.type === "BUY" || n.type === "SELL")
      .map(n => ({
        time:     Math.floor((n.published_ts || 0) / ivSec) * ivSec + TZ_OFFSET,
        position: n.type === "BUY" ? "belowBar" : "aboveBar",
        color:    n.type === "BUY" ? "#22c55e"  : "#ef4444",
        shape:    "circle",
        text:     "",
      }))
      .sort((a, b) => a.time - b.time);
    try {
      createSeriesMarkers(candleRef.current, signals);
    } catch {}
  }, [news, interval]);

  const isUp = change >= 0;
  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      {price && (
        <div style={{ position: "absolute", top: 8, left: 12, zIndex: 10, display: "flex", gap: 12, alignItems: "baseline" }}>
          <span style={{ fontSize: 18, fontFamily: "monospace", fontWeight: 700, color: COLORS.text }}>
            ${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          {change !== null && (
            <span style={{ fontSize: 12, fontFamily: "monospace", color: isUp ? COLORS.green : COLORS.red }}>
              {isUp ? "▲" : "▼"} {Math.abs(change)}%
            </span>
          )}
          <span style={{ fontSize: 10, color: COLORS.muted }}>BINANCE · LIVE</span>
        </div>
      )}
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
    </div>
  );
}

function SentimentGauge({ news }) {
  const WINDOW_SEC = 15 * 60; // 15 minutes

  // Only count news from the last 15 minutes
  const nowSec  = Date.now() / 1000;
  const recent  = news.filter(n => {
    const ts = n.published_ts || n.received_at || 0;
    return (nowSec - ts) <= WINDOW_SEC;
  });

  const hasRecent = recent.length > 0;
  const base    = hasRecent ? recent : [];
  const total   = base.length || 1;
  const bullish = base.filter(n => n.sentiment === "positive").length;
  const bearish = base.filter(n => n.sentiment === "negative").length;
  const neutral = total - bullish - bearish;

  // If no recent news → neutral (50)
  const value   = hasRecent ? Math.round(50 + (bullish - bearish) / total * 50) : 50;
  const bullPct = hasRecent ? Math.round((bullish / total) * 100) : 0;
  const bearPct = hasRecent ? Math.round((bearish / total) * 100) : 0;
  const neuPct  = 100 - bullPct - bearPct;

  const angle    = (value / 100) * 180 - 90;
  const getColor = (v) => v < 30 ? "#ef4444" : v < 50 ? "#f59e0b" : v < 70 ? "#eab308" : "#22c55e";
  const getLabel = (v) => v < 20 ? "Extreme Fear" : v < 40 ? "Fear" : v < 60 ? "Neutral" : v < 80 ? "Greed" : "Extreme Greed";
  const color    = hasRecent ? getColor(value) : COLORS.muted;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", width: "100%", alignItems: "center" }}>
        <span style={{ fontSize: 10, fontWeight: 600, color: COLORS.text }}>Live Sentiment</span>
        <span style={{ fontSize: 9, color: hasRecent ? COLORS.accent : COLORS.muted, fontFamily: "monospace" }}>
          {hasRecent ? `${recent.length} items · 15m` : "no recent news"}
        </span>
      </div>
      <svg width="160" height="90" viewBox="0 0 160 90">
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#ef4444" />
            <stop offset="33%"  stopColor="#f59e0b" />
            <stop offset="66%"  stopColor="#eab308" />
            <stop offset="100%" stopColor="#22c55e" />
          </linearGradient>
        </defs>
        <path d="M 15 80 A 65 65 0 0 1 145 80" fill="none" stroke={COLORS.border2} strokeWidth="12" strokeLinecap="round" />
        <path d="M 15 80 A 65 65 0 0 1 145 80" fill="none"
          stroke={hasRecent ? "url(#gaugeGrad)" : COLORS.border2} strokeWidth="12" strokeLinecap="round" />
        <g transform={`rotate(${angle}, 80, 80)`}>
          <line x1="80" y1="80" x2="80" y2="22" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
          <circle cx="80" cy="80" r="5" fill={color} />
        </g>
        <text x="80" y="68" textAnchor="middle" fill={hasRecent ? COLORS.text : COLORS.muted}
          fontSize="22" fontWeight="700" fontFamily="monospace">{value}</text>
      </svg>
      <span style={{ fontFamily: "monospace", fontSize: 12, color, letterSpacing: 2, textTransform: "uppercase" }}>
        {hasRecent ? getLabel(value) : "NEUTRAL"}
      </span>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6, width: "100%", marginTop: 4 }}>
        {[["Bullish", bullPct, COLORS.green], ["Neutral", neuPct, COLORS.gold], ["Bearish", bearPct, COLORS.red]].map(([label, pct, clr]) => (
          <div key={label}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>{label.toUpperCase()}</span>
              <span style={{ fontSize: 9, color: hasRecent ? clr : COLORS.muted, fontFamily: "monospace" }}>{pct}%</span>
            </div>
            <div style={{ height: 3, background: COLORS.border2, borderRadius: 2 }}>
              <div style={{ height: "100%", width: `${pct}%`, background: hasRecent ? clr : COLORS.border2, borderRadius: 2 }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function NavItem({ icon, label, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
      borderRadius: 8, border: "none", cursor: "pointer", width: "100%", textAlign: "left",
      background: active ? `${COLORS.accent}18` : "transparent",
      color: active ? COLORS.accent : COLORS.muted,
      transition: "all 0.15s",
    }}>
      <span style={{ fontSize: 16 }}>{icon}</span>
      <span style={{ fontSize: 13, fontWeight: active ? 600 : 400, letterSpacing: 0.3 }}>{label}</span>
      {active && <div style={{ marginLeft: "auto", width: 3, height: 16, background: COLORS.accent, borderRadius: 2 }} />}
    </button>
  );
}

// ══════════════════════════════════════════════════════════════════
// INLINE AI EXPLANATION PANEL  (right column of dashboard)
// ══════════════════════════════════════════════════════════════════
function ExplainPanel({ selectedNews: item, onClose }) {
  const [explain, setExplain]       = useState(null);
  const [explainErr, setExplainErr] = useState(null);
  const [loading, setLoading]       = useState(false);

  // Auto-fetch when item changes
  useEffect(() => {
    if (!item) { setExplain(null); setExplainErr(null); return; }
    setExplain(null);
    setExplainErr(null);
    setLoading(true);
    fetch(`${API_BASE}/news/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(item),
    })
      .then(r => r.json())
      .then(data => {
        setLoading(false);
        if (data.error && !data.explanation && !data.similar?.length) {
          setExplainErr(data.error);
        } else {
          // Merge fetched similar into item so it's always fresh
          setExplain({
            ...data,
            similar: data.similar?.length ? data.similar : (item.similar || []),
          });
        }
      })
      .catch(e => { setLoading(false); setExplainErr(String(e)); });
  }, [item?.id]);

  const score      = item ? Math.abs(item.model_score || 0) : 0;
  const impactClr  = score >= 0.67 ? COLORS.red : score >= 0.50 ? COLORS.gold : COLORS.muted;
  const impactLbl  = score >= 0.67 ? "High" : score >= 0.50 ? "Medium" : "Low";
  const sentClr    = { positive: COLORS.green, negative: COLORS.red, neutral: COLORS.muted };
  const sColor     = item ? (sentClr[item.sentiment] || COLORS.muted) : COLORS.muted;

  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Header */}
      <div style={{
        padding: "10px 16px", borderBottom: `1px solid ${COLORS.border}`,
        display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0,
      }}>
        <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: 0.5, color: COLORS.purple }}>
          🧠 Explanation
        </span>
        {item && (
          <button onClick={onClose} style={{
            background: "none", border: "none", color: COLORS.muted,
            fontSize: 14, cursor: "pointer", lineHeight: 1,
          }}>✕</button>
        )}
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "14px 16px" }}>

        {/* Empty state */}
        {!item && (
          <div style={{
            display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
            height: "100%", gap: 10,
          }}>
            <div style={{ fontSize: 28, opacity: 0.3 }}>🧠</div>
            <div style={{ fontSize: 12, color: COLORS.muted, textAlign: "center", lineHeight: 1.6 }}>
              Click any news item<br />to get an AI explanation
            </div>
          </div>
        )}

        {/* Selected item */}
        {item && (
          <>
            {/* Sentiment badge + title */}
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
                <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 4, background: `${sColor}18`, color: sColor, fontFamily: "monospace", fontWeight: 700, letterSpacing: 1 }}>
                  {sentimentLabel(item.sentiment).toUpperCase()}
                </span>
                <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 4, background: `${impactClr}18`, color: impactClr, fontFamily: "monospace", fontWeight: 700, letterSpacing: 1 }}>
                  {impactLbl.toUpperCase()}
                </span>
              </div>
              <div style={{ fontSize: 12, color: COLORS.text, fontWeight: 600, lineHeight: 1.5 }}>
                {item.link
                  ? <a href={item.link} target="_blank" rel="noreferrer" style={{ color: COLORS.text, textDecoration: "none" }}>{item.title}</a>
                  : item.title}
              </div>
              <div style={{ fontSize: 10, color: COLORS.muted, fontFamily: "monospace", marginTop: 4 }}>
                {item.channel || "—"} · {item.time || ""}
              </div>
            </div>

            {/* Stats row */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 10 }}>
              {[
                ["Score", fmtScore(item.model_score), COLORS.gold],
                ["Conf",  `${item.confidence}%`,      COLORS.accent],
                ["BTC 1h", item.btc_change_1h != null && item.btc_change_1h !== 0
                  ? `${item.btc_change_1h > 0 ? "+" : ""}${Number(item.btc_change_1h).toFixed(2)}%`
                  : "—",
                  item.btc_change_1h > 0 ? COLORS.green : item.btc_change_1h < 0 ? COLORS.red : COLORS.muted],
              ].map(([lbl, val, clr]) => (
                <div key={lbl} style={{ background: COLORS.bg, borderRadius: 8, padding: "7px 10px" }}>
                  <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 3 }}>{lbl.toUpperCase()}</div>
                  <div style={{ fontSize: 12, color: clr, fontFamily: "monospace", fontWeight: 700 }}>{val}</div>
                </div>
              ))}
            </div>

            {/* ── Price Prediction Panel ── */}
            {(() => {
              const { p15, p1h } = predictedChange(item);
              const actual15 = parseFloat(item.btc_change_15m || 0);
              const ragAvg   = parseFloat(item.rag_avg_change  || 0);
              const hasActual = Math.abs(actual15) > 0.001;
              const hasRag    = Math.abs(ragAvg)   > 0.001;
              const clr = (v) => v > 0 ? COLORS.green : v < 0 ? COLORS.red : COLORS.muted;
              const arrow = (v) => v > 0 ? "▲" : v < 0 ? "▼" : "—";
              return (
                <div style={{
                  background: COLORS.bg, borderRadius: 10, padding: "10px 12px",
                  border: `1px solid ${COLORS.border2}`, marginBottom: 14,
                }}>
                  <div style={{ fontSize: 9, color: COLORS.accent, fontWeight: 700, letterSpacing: 1, marginBottom: 8 }}>
                    📈 BTC PRICE PREDICTION
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                    {/* Predicted 15m */}
                    <div style={{ background: `${COLORS.panel}`, borderRadius: 7, padding: "7px 10px", border: `1px solid ${COLORS.border}` }}>
                      <div style={{ fontSize: 9, color: COLORS.muted, marginBottom: 4 }}>PREDICTED 15m</div>
                      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
                        <span style={{ fontSize: 16, fontFamily: "monospace", fontWeight: 700, color: clr(p15) }}>
                          {p15 !== 0 ? `${p15 > 0 ? "+" : ""}${p15.toFixed(2)}%` : "—"}
                        </span>
                        {p15 !== 0 && <span style={{ fontSize: 10, color: clr(p15) }}>{arrow(p15)}</span>}
                      </div>
                      <div style={{ fontSize: 9, color: COLORS.muted, marginTop: 3 }}>
                        score {fmtScore(item.model_score)} · {item.type || "NEUTRAL"}
                      </div>
                    </div>
                    {/* Predicted 1h */}
                    <div style={{ background: `${COLORS.panel}`, borderRadius: 7, padding: "7px 10px", border: `1px solid ${COLORS.border}` }}>
                      <div style={{ fontSize: 9, color: COLORS.muted, marginBottom: 4 }}>PREDICTED 1h</div>
                      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
                        <span style={{ fontSize: 16, fontFamily: "monospace", fontWeight: 700, color: clr(p1h) }}>
                          {p1h !== 0 ? `${p1h > 0 ? "+" : ""}${p1h.toFixed(2)}%` : "—"}
                        </span>
                        {p1h !== 0 && <span style={{ fontSize: 10, color: clr(p1h) }}>{arrow(p1h)}</span>}
                      </div>
                      <div style={{ fontSize: 9, color: COLORS.muted, marginTop: 3 }}>
                        score {fmtScore(item.model_score_1h)} · {item.type || "NEUTRAL"}
                      </div>
                    </div>
                    {/* Actual 15m (if available) */}
                    {hasActual && (
                      <div style={{ background: `${clr(actual15)}11`, borderRadius: 7, padding: "7px 10px", border: `1px solid ${clr(actual15)}33` }}>
                        <div style={{ fontSize: 9, color: COLORS.muted, marginBottom: 4 }}>ACTUAL 15m ✓</div>
                        <div style={{ fontSize: 16, fontFamily: "monospace", fontWeight: 700, color: clr(actual15) }}>
                          {fmtChange(actual15)}
                        </div>
                        <div style={{ fontSize: 9, color: COLORS.muted, marginTop: 3 }}>
                          {Math.abs(actual15) >= Math.abs(p15 * 0.7) ? "✅ prediction aligned" : "⚠ diverged"}
                        </div>
                      </div>
                    )}
                    {/* RAG historical average */}
                    {hasRag && (
                      <div style={{ background: `${COLORS.purple}11`, borderRadius: 7, padding: "7px 10px", border: `1px solid ${COLORS.purple}33` }}>
                        <div style={{ fontSize: 9, color: COLORS.muted, marginBottom: 4 }}>HIST. SIMILAR</div>
                        <div style={{ fontSize: 16, fontFamily: "monospace", fontWeight: 700, color: COLORS.purple }}>
                          {fmtChange(ragAvg)}
                        </div>
                        <div style={{ fontSize: 9, color: COLORS.muted, marginTop: 3 }}>avg from RAG memory</div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}

            {/* Loading */}
            {loading && (
              <div style={{
                background: COLORS.bg, borderRadius: 10, padding: "16px",
                border: `1px solid ${COLORS.purple}33`,
                display: "flex", alignItems: "center", gap: 10, marginBottom: 14,
              }}>
                <div style={{ fontSize: 16, animation: "spin 1s linear infinite" }}>⟳</div>
                <span style={{ fontSize: 11, color: COLORS.muted, fontStyle: "italic" }}>Generating explanation...</span>
              </div>
            )}

            {/* Error */}
            {explainErr && !loading && (
              <div style={{ fontSize: 11, color: COLORS.red, padding: "8px 12px", background: `${COLORS.red}11`, borderRadius: 8, marginBottom: 14 }}>
                ⚠ {explainErr}
              </div>
            )}

            {/* CoT steps */}
            {explain && !loading && (
              <div style={{
                background: COLORS.bg, borderRadius: 10, padding: "12px 14px",
                border: `1px solid ${COLORS.purple}33`, marginBottom: 14,
              }}>
                <div style={{ fontSize: 10, color: COLORS.purple, fontWeight: 700, letterSpacing: 0.5, marginBottom: 10 }}>
                  WHY THIS SCORE?
                </div>
                {explain.steps && explain.steps.length > 0
                  ? explain.steps.map((step, i) => (
                    <div key={i} style={{
                      fontSize: 11, color: COLORS.text, lineHeight: 1.65, padding: "5px 0",
                      borderBottom: i < explain.steps.length - 1 ? `1px solid ${COLORS.border}` : "none",
                    }}>
                      {step}
                    </div>
                  ))
                  : <div style={{ fontSize: 11, color: COLORS.text, lineHeight: 1.7, whiteSpace: "pre-wrap" }}>{explain.explanation}</div>
                }
              </div>
            )}

            {/* Similar RAG news — use fetched results from explain response, fall back to item.similar */}
            {((explain?.similar?.length ? explain.similar : item.similar) || []).length > 0 && (
              <>
                <div style={{ fontSize: 10, color: COLORS.accent, fontWeight: 700, letterSpacing: 0.5, marginBottom: 8 }}>📚 SIMILAR PAST NEWS</div>
                {((explain?.similar?.length ? explain.similar : item.similar) || []).slice(0, 3).map((s, i) => {
                  const chg    = s.change ?? 0;
                  const chgClr = chg > 0 ? COLORS.green : chg < 0 ? COLORS.red : COLORS.muted;
                  return (
                    <div key={i} style={{
                      display: "flex", justifyContent: "space-between", alignItems: "flex-start",
                      padding: "8px 10px", marginBottom: 5, background: COLORS.bg, borderRadius: 7,
                      borderLeft: `3px solid ${chgClr}`,
                    }}>
                      <div style={{ flex: 1, fontSize: 10, color: COLORS.text, lineHeight: 1.4, marginRight: 8 }}>{s.title}</div>
                      <div style={{ flexShrink: 0, textAlign: "right" }}>
                        <div style={{ fontSize: 11, color: chgClr, fontFamily: "monospace", fontWeight: 700 }}>
                          {chg > 0 ? "+" : ""}{chg.toFixed(2)}%
                        </div>
                        <div style={{ fontSize: 9, color: COLORS.muted }}>sim {((s.sim ?? 0) * 100).toFixed(0)}%</div>
                      </div>
                    </div>
                  );
                })}
              </>
            )}
          </>
        )}
      </div>
      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════
function NewsModal({ item, onClose }) {
  const label   = sentimentLabel(item.sentiment);
  const colors  = { Bullish: COLORS.green, Bearish: COLORS.red, Neutral: COLORS.muted };
  const color   = colors[label];

  const [explain, setExplain]       = useState(null);
  const [explainErr, setExplainErr] = useState(null);
  const [similar, setSimilar]       = useState(item.similar || []);
  const [simLoading, setSimLoading] = useState(false);

  const score = Math.abs(item.model_score || 0);
  const impactColor = score >= 0.67 ? COLORS.red : score >= 0.50 ? COLORS.gold : COLORS.muted;
  const impactLabel = score >= 0.67 ? "High" : score >= 0.50 ? "Medium" : "Low";

  // Auto-fetch similar news on open if item has none
  useEffect(() => {
    if (similar.length > 0) return;
    setSimLoading(true);
    fetch(`${API_BASE}/news/similar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(item),
    })
      .then(r => r.json())
      .then(data => { setSimLoading(false); if (data.similar?.length) setSimilar(data.similar); })
      .catch(() => setSimLoading(false));
  }, [item?.id]);

  function fetchExplanation() {
    setExplain("loading");
    setExplainErr(null);
    fetch(`${API_BASE}/news/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(item),
    })
      .then(r => r.json())
      .then(data => {
        if (data.error && !data.explanation) {
          setExplainErr(data.error);
          setExplain(null);
        } else {
          setExplain(data);
          // Update similar if explain returned fresh ones
          if (data.similar?.length) setSimilar(data.similar);
        }
      })
      .catch(e => { setExplainErr(String(e)); setExplain(null); });
  }

  return (
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)",
      display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000,
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        background: COLORS.panel, border: `1px solid ${COLORS.border2}`,
        borderRadius: 14, padding: 24, width: 580, maxWidth: "92vw", maxHeight: "85vh", overflowY: "auto",
      }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
          <span style={{ fontSize: 9, padding: "3px 9px", borderRadius: 4, fontFamily: "monospace", background: `${color}18`, color, letterSpacing: 1, fontWeight: 700 }}>
            {label.toUpperCase()}
          </span>
          <button onClick={onClose} style={{ background: "none", border: "none", color: COLORS.muted, fontSize: 18, cursor: "pointer", lineHeight: 1 }}>✕</button>
        </div>

        {/* Title */}
        <div style={{ fontSize: 14, color: COLORS.text, fontWeight: 600, lineHeight: 1.5, marginBottom: 16 }}>
          {item.link
            ? <a href={item.link} target="_blank" rel="noreferrer" style={{ color: COLORS.text, textDecoration: "none" }}>{item.title}</a>
            : item.title}
        </div>

        {/* Stats grid */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
          {[
            ["Channel",     item.channel || "—",    COLORS.text],
            ["Impact",      impactLabel,             impactColor],
            ["Confidence",  `${item.confidence}%`,  COLORS.accent],
            ["Model Score", fmtScore(item.model_score), COLORS.gold],
          ].map(([lbl, val, clr]) => (
            <div key={lbl} style={{ background: COLORS.bg, borderRadius: 8, padding: "8px 10px" }}>
              <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 4 }}>{lbl.toUpperCase()}</div>
              <div style={{ fontSize: 12, color: clr, fontFamily: "monospace", fontWeight: 600 }}>{val}</div>
            </div>
          ))}
        </div>

        {/* ── BTC Price Prediction ─────────────────────────────────── */}
        {(() => {
          const { p15, p1h } = predictedChange(item);
          const actual15 = parseFloat(item.btc_change_15m || 0);
          const actual1h  = parseFloat(item.btc_change_1h  || 0);
          const ragAvg    = parseFloat(item.rag_avg_change  || 0);
          const hasActual15 = Math.abs(actual15) > 0.001;
          const hasActual1h = Math.abs(actual1h) > 0.001;
          const hasRag      = Math.abs(ragAvg) > 0.001;
          const clr = (v) => v > 0 ? COLORS.green : v < 0 ? COLORS.red : COLORS.muted;
          const dirIcon = (v) => v > 0 ? "▲" : v < 0 ? "▼" : "●";
          return (
            <div style={{
              background: COLORS.bg, borderRadius: 12, padding: "14px 16px",
              border: `1px solid ${COLORS.border2}`, marginBottom: 20,
            }}>
              <div style={{ fontSize: 10, color: COLORS.accent, fontWeight: 700, letterSpacing: 1, marginBottom: 12 }}>
                📈 BTC PRICE PREDICTION
              </div>
              {/* Prediction vs Actual comparison */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10, marginBottom: hasRag ? 10 : 0 }}>
                {/* Predicted 15m */}
                <div style={{ background: COLORS.panel, borderRadius: 8, padding: "10px 12px", border: `1px solid ${COLORS.border}` }}>
                  <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 6 }}>PREDICTED 15m</div>
                  <div style={{ fontSize: 20, fontFamily: "monospace", fontWeight: 700, color: clr(p15) }}>
                    {p15 !== 0 ? `${p15 > 0 ? "+" : ""}${p15.toFixed(2)}%` : "—"}
                  </div>
                  <div style={{ fontSize: 9, color: COLORS.muted, marginTop: 4 }}>
                    {dirIcon(p15)} model score {fmtScore(item.model_score)}
                  </div>
                </div>
                {/* Predicted 1h */}
                <div style={{ background: COLORS.panel, borderRadius: 8, padding: "10px 12px", border: `1px solid ${COLORS.border}` }}>
                  <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 6 }}>PREDICTED 1h</div>
                  <div style={{ fontSize: 20, fontFamily: "monospace", fontWeight: 700, color: clr(p1h) }}>
                    {p1h !== 0 ? `${p1h > 0 ? "+" : ""}${p1h.toFixed(2)}%` : "—"}
                  </div>
                  <div style={{ fontSize: 9, color: COLORS.muted, marginTop: 4 }}>
                    {dirIcon(p1h)} model score {fmtScore(item.model_score_1h)}
                  </div>
                </div>
                {/* Actual 15m */}
                <div style={{
                  background: hasActual15 ? `${clr(actual15)}11` : COLORS.panel,
                  borderRadius: 8, padding: "10px 12px",
                  border: `1px solid ${hasActual15 ? clr(actual15)+"44" : COLORS.border}`,
                }}>
                  <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 6 }}>ACTUAL 15m</div>
                  <div style={{ fontSize: 20, fontFamily: "monospace", fontWeight: 700, color: hasActual15 ? clr(actual15) : COLORS.muted }}>
                    {hasActual15 ? fmtChange(actual15) : "—"}
                  </div>
                  {hasActual15 ? (
                    <div style={{ fontSize: 9, marginTop: 4, color: Math.sign(actual15) === Math.sign(p15) ? COLORS.green : COLORS.red }}>
                      {Math.sign(actual15) === Math.sign(p15) ? "✅ direction correct" : "❌ direction missed"}
                    </div>
                  ) : (
                    <div style={{ fontSize: 9, color: COLORS.muted, marginTop: 4 }}>not yet / unavailable</div>
                  )}
                </div>
                {/* Actual 1h */}
                <div style={{
                  background: hasActual1h ? `${clr(actual1h)}11` : COLORS.panel,
                  borderRadius: 8, padding: "10px 12px",
                  border: `1px solid ${hasActual1h ? clr(actual1h)+"44" : COLORS.border}`,
                }}>
                  <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 6 }}>ACTUAL 1h</div>
                  <div style={{ fontSize: 20, fontFamily: "monospace", fontWeight: 700, color: hasActual1h ? clr(actual1h) : COLORS.muted }}>
                    {hasActual1h ? fmtChange(actual1h) : "—"}
                  </div>
                  {hasActual1h ? (
                    <div style={{ fontSize: 9, marginTop: 4, color: Math.sign(actual1h) === Math.sign(p1h) ? COLORS.green : COLORS.red }}>
                      {Math.sign(actual1h) === Math.sign(p1h) ? "✅ direction correct" : "❌ direction missed"}
                    </div>
                  ) : (
                    <div style={{ fontSize: 9, color: COLORS.muted, marginTop: 4 }}>not yet / unavailable</div>
                  )}
                </div>
              </div>
              {/* RAG historical average */}
              {hasRag && (
                <div style={{
                  background: `${COLORS.purple}11`, borderRadius: 8, padding: "9px 12px",
                  border: `1px solid ${COLORS.purple}33`, display: "flex", alignItems: "center", gap: 12,
                }}>
                  <span style={{ fontSize: 14 }}>📚</span>
                  <div>
                    <div style={{ fontSize: 9, color: COLORS.muted }}>SIMILAR HISTORICAL NEWS MOVED BTC BY:</div>
                    <span style={{ fontSize: 14, fontFamily: "monospace", fontWeight: 700, color: COLORS.purple }}>
                      {fmtChange(ragAvg)}
                    </span>
                    <span style={{ fontSize: 10, color: COLORS.muted, marginLeft: 6 }}>on average (from RAG memory)</span>
                  </div>
                </div>
              )}
            </div>
          );
        })()}

        {/* ── AI Explanation panel ─────────────────────────────────── */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
            <span style={{ fontSize: 12, color: COLORS.purple, fontWeight: 700, letterSpacing: 0.5 }}>
              🧠 Explanation
            </span>
            {explain !== "loading" && (
              <button
                onClick={fetchExplanation}
                style={{
                  fontSize: 10, padding: "4px 12px", borderRadius: 6, cursor: "pointer",
                  background: explain ? `${COLORS.purple}22` : `${COLORS.purple}33`,
                  color: COLORS.purple, border: `1px solid ${COLORS.purple}55`,
                  fontWeight: 600, letterSpacing: 0.5,
                }}
              >
                {explain ? "↺ Regenerate" : "✦ Explain"}
              </button>
            )}
          </div>

          {/* Loading state */}
          {explain === "loading" && (
            <div style={{
              background: COLORS.bg, borderRadius: 10, padding: "16px 18px",
              border: `1px solid ${COLORS.purple}33`,
              display: "flex", alignItems: "center", gap: 10,
            }}>
              <div style={{ fontSize: 18, animation: "spin 1s linear infinite" }}>⟳</div>
              <span style={{ fontSize: 12, color: COLORS.muted, fontStyle: "italic" }}>
                Generating explanation...
              </span>
            </div>
          )}

          {/* Error */}
          {explainErr && (
            <div style={{ fontSize: 11, color: COLORS.red, padding: "8px 12px", background: `${COLORS.red}11`, borderRadius: 8 }}>
              ⚠ {explainErr}
            </div>
          )}

          {/* Explanation steps */}
          {explain && explain !== "loading" && (
            <div style={{
              background: COLORS.bg, borderRadius: 10, padding: "14px 16px",
              border: `1px solid ${COLORS.purple}33`,
            }}>
              {/* If we have parsed steps, show them as cards */}
              {explain.steps && explain.steps.length > 0 ? (
                explain.steps.map((step, i) => (
                  <div key={i} style={{
                    fontSize: 12, color: COLORS.text, lineHeight: 1.6,
                    padding: "6px 0",
                    borderBottom: i < explain.steps.length - 1 ? `1px solid ${COLORS.border}` : "none",
                  }}>
                    {step}
                  </div>
                ))
              ) : (
                /* Fallback: raw text */
                <div style={{ fontSize: 12, color: COLORS.text, lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
                  {explain.explanation}
                </div>
              )}

              {/* BTC reaction hint */}
              {(item.btc_change_15m !== 0 || item.btc_change_1h !== 0) && (
                <div style={{
                  marginTop: 12, paddingTop: 10, borderTop: `1px solid ${COLORS.border}`,
                  display: "flex", gap: 16,
                }}>
                  <span style={{ fontSize: 11, color: COLORS.muted }}>
                    BTC 15m: <span style={{ color: item.btc_change_15m > 0 ? COLORS.green : item.btc_change_15m < 0 ? COLORS.red : COLORS.muted, fontFamily: "monospace" }}>
                      {item.btc_change_15m > 0 ? "+" : ""}{(item.btc_change_15m || 0).toFixed(2)}%
                    </span>
                  </span>
                  <span style={{ fontSize: 11, color: COLORS.muted }}>
                    BTC 1h: <span style={{ color: item.btc_change_1h > 0 ? COLORS.green : item.btc_change_1h < 0 ? COLORS.red : COLORS.muted, fontFamily: "monospace" }}>
                      {item.btc_change_1h > 0 ? "+" : ""}{(item.btc_change_1h || 0).toFixed(2)}%
                    </span>
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Placeholder when not yet clicked */}
          {!explain && !explainErr && (
            <div style={{
              background: COLORS.bg, borderRadius: 10, padding: "14px 16px",
              border: `1px dashed ${COLORS.border2}`,
              fontSize: 11, color: COLORS.muted, fontStyle: "italic", textAlign: "center",
            }}>
              Click "✦ Explain" to get an AI step-by-step reasoning for this signal
            </div>
          )}
        </div>

        {/* ── Similar Past News (RAG) ──────────────────────────────── */}
        <div style={{ fontSize: 12, color: COLORS.accent, fontWeight: 600, marginBottom: 10, letterSpacing: 0.5 }}>📚 Similar Past News (RAG)</div>
        {simLoading ? (
          <div style={{ fontSize: 11, color: COLORS.muted, fontStyle: "italic", display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⟳</span> Searching similar news...
          </div>
        ) : similar.length === 0 ? (
          <div style={{ fontSize: 11, color: COLORS.muted, fontStyle: "italic" }}>No similar news found in RAG database.</div>
        ) : similar.map((s, i) => {
          const chg    = s.change ?? 0;
          const chgClr = chg > 0 ? COLORS.green : chg < 0 ? COLORS.red : COLORS.muted;
          return (
            <div key={i} style={{
              display: "flex", justifyContent: "space-between", alignItems: "flex-start",
              padding: "10px 12px", marginBottom: 6, background: COLORS.bg, borderRadius: 8,
              borderLeft: `3px solid ${chgClr}`,
            }}>
              <div style={{ flex: 1, fontSize: 11, color: COLORS.text, lineHeight: 1.4, marginRight: 12 }}>{s.title}</div>
              <div style={{ flexShrink: 0, textAlign: "right" }}>
                <div style={{ fontSize: 12, color: chgClr, fontFamily: "monospace", fontWeight: 700 }}>
                  {chg > 0 ? "+" : ""}{chg.toFixed(2)}%
                </div>
                <div style={{ fontSize: 9, color: COLORS.muted, fontFamily: "monospace" }}>sim {((s.sim ?? 0) * 100).toFixed(0)}%</div>
              </div>
            </div>
          );
        })}
      </div>
      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

function NewsCard({ item, onClick }) {
  const label      = sentimentLabel(item.sentiment);
  const colors     = { Bullish: COLORS.green, Bearish: COLORS.red, Neutral: COLORS.muted };
  const color      = colors[label];
  const logo       = channelLogo(item.channel || item.source);
  const age        = itemTime(item);
  const hasSimilar = item.similar?.length > 0;
  const [bulletHover, setBulletHover] = useState(false);
  return (
    <div onClick={onClick} style={{
      display: "flex", gap: 12, padding: "12px 0",
      borderBottom: `1px solid ${COLORS.border}`, cursor: "pointer",
    }}>
      <div style={{ width: 38, height: 38, borderRadius: 10, background: COLORS.border2, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, flexShrink: 0, position: "relative" }}>
        {logo}
        {item.sentiment === "positive" && (
          <div style={{ position: "absolute", bottom: -4, right: -4, width: 16, height: 16, borderRadius: "50%", background: COLORS.green, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: "#000" }}>▲</div>
        )}
        {item.sentiment === "negative" && (
          <div style={{ position: "absolute", bottom: -4, right: -4, width: 16, height: 16, borderRadius: "50%", background: COLORS.red, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: "#fff" }}>▼</div>
        )}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
          <span style={{ fontSize: 10, color: COLORS.muted, fontFamily: "monospace" }}>{age}</span>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            {hasSimilar && <span style={{ fontSize: 9, color: COLORS.gold, fontFamily: "monospace" }}>📚 {item.similar.length}</span>}
            <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, fontFamily: "monospace", background: `${color}18`, color, letterSpacing: 1, fontWeight: 600 }}>
              {label.toUpperCase()}
            </span>
          </div>
        </div>
        <div style={{ fontSize: 12, color: COLORS.text, lineHeight: 1.4, marginBottom: 4, fontWeight: 500 }}>{item.title}</div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          {(() => {
            const s = Math.abs(item.model_score || 0);
            const isHigh = s >= 0.67;
            const clr = isHigh ? COLORS.red : COLORS.gold;
            const tooltipText = isHigh
              ? `High Impact — Strong likelihood of Bitcoin price movement\nScore: ${fmtScore(item.model_score)}  Conf: ${item.confidence}%`
              : `Medium Impact — Moderate likelihood of Bitcoin price movement\nScore: ${fmtScore(item.model_score)}  Conf: ${item.confidence}%`;
            return (
              <span style={{ fontSize: 10, color: COLORS.muted, display: "flex", alignItems: "center", gap: 3, position: "relative" }}>
                Impact:&nbsp;
                <span
                  onMouseEnter={e => { e.stopPropagation(); setBulletHover(true); }}
                  onMouseLeave={() => setBulletHover(false)}
                  style={{ color: clr, fontSize: 11, letterSpacing: 1, cursor: "help" }}
                >
                  {isHigh ? "●●" : "●"}
                </span>
                <span style={{ color: clr }}>{isHigh ? "High" : "Medium"}</span>
                {bulletHover && (
                  <div style={{
                    position: "absolute", bottom: "calc(100% + 6px)", left: 0,
                    background: COLORS.panel, border: `1px solid ${clr}44`,
                    borderRadius: 8, padding: "8px 12px", zIndex: 999,
                    minWidth: 220, pointerEvents: "none",
                    boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
                  }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: clr, marginBottom: 4 }}>
                      {isHigh ? "●● High Impact" : "● Medium Impact"}
                    </div>
                    <div style={{ fontSize: 10, color: COLORS.text, lineHeight: 1.5, marginBottom: 6 }}>
                      {item.title}
                    </div>
                    <div style={{ fontSize: 10, color: COLORS.muted }}>
                      Score: <span style={{ color: COLORS.gold }}>{fmtScore(item.model_score)}</span>
                      &nbsp;·&nbsp;Conf: <span style={{ color: COLORS.accent }}>{item.confidence}%</span>
                      &nbsp;·&nbsp;<span style={{ color: color }}>{label}</span>
                    </div>
                  </div>
                )}
              </span>
            );
          })()}
          <span style={{ fontSize: 10, color: COLORS.muted }}>Conf: <span style={{ color: COLORS.accent }}>{item.confidence}%</span></span>
          {item.model_score != null && (
            <span style={{ fontSize: 10, color: COLORS.muted }}>Score: <span style={{ color: COLORS.gold }}>{fmtScore(item.model_score)}</span></span>
          )}
        </div>
      </div>
    </div>
  );
}

function SignalCard({ item }) {
  const action = signalAction(item.type, item.model_score);
  const actionColors = { "Strong Buy": COLORS.green, "Buy": COLORS.blue, "Strong Sell": COLORS.red, "Sell": COLORS.red, "Neutral": COLORS.muted };
  const color  = actionColors[action] || COLORS.muted;
  const age    = itemTime(item);
  return (
    <div style={{ border: `1px solid ${COLORS.border2}`, borderRadius: 10, padding: 14, marginBottom: 8, borderLeft: `3px solid ${color}`, background: `${color}06` }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ fontSize: 13, fontWeight: 700, color: COLORS.text, fontFamily: "monospace" }}>{channelLogo(item.channel)} {item.channel || "BTC/USDT"}</span>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontSize: 10, fontFamily: "monospace", padding: "3px 8px", borderRadius: 5, background: `${color}20`, color, fontWeight: 700 }}>{action}</span>
          <span style={{ fontSize: 10, color: COLORS.muted }}>{age}</span>
        </div>
      </div>
      <div style={{ fontSize: 12, color: COLORS.text, lineHeight: 1.4, marginBottom: 10 }}>{item.title}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 10 }}>
        {[
          ["Model Score", fmtScore(item.model_score),                              COLORS.accent],
          ["Pred 15m",   item.pred_15m ? "Impact" : "No Impact", item.pred_15m ? COLORS.green : COLORS.muted],
          ["Weight",     `${item.weight ?? "—"}/10`,                              COLORS.gold],
        ].map(([label, val, clr]) => (
          <div key={label}>
            <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 2 }}>{label.toUpperCase()}</div>
            <div style={{ fontSize: 11, color: clr, fontFamily: "monospace", fontWeight: 600 }}>{val}</div>
          </div>
        ))}
      </div>
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
          <span style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>CONFIDENCE</span>
          <span style={{ fontSize: 9, color, fontFamily: "monospace", fontWeight: 700 }}>{item.confidence}%</span>
        </div>
        <div style={{ height: 3, background: COLORS.border2, borderRadius: 2 }}>
          <div style={{ height: "100%", width: `${item.confidence}%`, background: color, borderRadius: 2, transition: "width 1s ease" }} />
        </div>
      </div>
    </div>
  );
}

function ConnectionDot({ connected }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <div style={{ width: 7, height: 7, borderRadius: "50%", background: connected ? COLORS.green : COLORS.red, boxShadow: connected ? `0 0 6px ${COLORS.green}` : "none" }} />
      <span style={{ fontSize: 10, color: COLORS.muted, fontFamily: "monospace" }}>{connected ? "LIVE" : "OFFLINE"}</span>
    </div>
  );
}

// ── Training Data Analysis ─────────────────────────────────────────
function MetricCard({ label, value, sub, color }) {
  return (
    <div style={{ background: COLORS.panel, borderRadius: 10, padding: "14px 16px", border: `1px solid ${COLORS.border2}` }}>
      <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 6 }}>{label.toUpperCase()}</div>
      <div style={{ fontSize: 22, fontFamily: "monospace", fontWeight: 700, color: color || COLORS.accent }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: COLORS.muted, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function HorizBar({ label, value, max, color, showPct }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div style={{ marginBottom: 9 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 11, color: COLORS.text }}>{label}</span>
        <span style={{ fontSize: 11, fontFamily: "monospace", color }}>
          {value.toLocaleString()}{showPct ? ` (${pct.toFixed(1)}%)` : ""}
        </span>
      </div>
      <div style={{ height: 5, background: COLORS.border2, borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 3 }} />
      </div>
    </div>
  );
}

function ConfusionMatrix({ cm, label }) {
  if (!cm) return null;
  const [[tn, fp], [fn, tp]] = cm;
  const total = tn + fp + fn + tp;
  const cells = [
    { label: "TN", value: tn, color: COLORS.green,  desc: "Correctly no impact" },
    { label: "FP", value: fp, color: COLORS.red,    desc: "False alarm" },
    { label: "FN", value: fn, color: COLORS.gold,   desc: "Missed impact" },
    { label: "TP", value: tp, color: COLORS.accent, desc: "Correctly predicted impact" },
  ];
  return (
    <div>
      <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 8 }}>{label} CONFUSION MATRIX</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        {cells.map(c => (
          <div key={c.label} style={{ background: COLORS.bg, borderRadius: 8, padding: "10px 12px", borderLeft: `3px solid ${c.color}` }}>
            <div style={{ fontSize: 9, color: COLORS.muted, marginBottom: 2 }}>{c.label}</div>
            <div style={{ fontSize: 18, fontFamily: "monospace", fontWeight: 700, color: c.color }}>{c.value.toLocaleString()}</div>
            <div style={{ fontSize: 9, color: COLORS.muted }}>{c.desc}</div>
            <div style={{ fontSize: 9, color: COLORS.muted }}>{((c.value / total) * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrainingAnalysis() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/training/stats`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return (
    <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: COLORS.muted, fontSize: 13 }}>
      Loading training data…
    </div>
  );
  if (error || !data) return (
    <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: COLORS.red, fontSize: 13 }}>
      Failed to load: {error}
    </div>
  );

  const td  = data.training_data || {};
  const mp  = data.model_performance || {};
  const m15 = mp["15_minute"] || {};
  const m1h = mp["1_hour"] || {};
  const dir = mp["direction"] || {};

  const sentMax = Math.max(...Object.values(td.sentiment_counts || {}), 1);
  const typeMax = Math.max(...Object.values(td.news_types || {}), 1);
  const chMax   = Math.max(...Object.values(td.channels || {}), 1);
  const histMax = Math.max(...(td.btc_change_histogram || []).map(b => b.count), 1);

  const perfColor = (v, good) => v >= good ? COLORS.green : v >= good * 0.8 ? COLORS.gold : COLORS.red;

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: 24, display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text }}>
        📊 Training Data Analysis
        <span style={{ fontSize: 11, color: COLORS.muted, fontWeight: 400, marginLeft: 8 }}>
          {(td.total_samples || 0).toLocaleString()} samples · telegram_news_final_clean.csv
        </span>
      </div>

      {/* Dataset KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12 }}>
        <MetricCard label="Total Samples"    value={(td.total_samples || 0).toLocaleString()} color={COLORS.text} />
        <MetricCard label="Avg Confidence"   value={((td.avg_confidence || 0) * 100).toFixed(1) + "%"} color={COLORS.blue} />
        <MetricCard label="Avg Weight"       value={(td.avg_weight || 0).toFixed(1) + "/10"} color={COLORS.gold} />
        <MetricCard label="15m Price Impact" value={td.impact_15m_pct + "%"} sub={`${(td.impact_15m_count || 0).toLocaleString()} items ≥ 0.5%`} color={COLORS.accent} />
        <MetricCard label="1h Price Impact"  value={td.impact_1h_pct + "%"} sub={`${(td.impact_1h_count || 0).toLocaleString()} items ≥ 0.5%`} color={COLORS.purple} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>

        {/* Sentiment distribution */}
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>SENTIMENT DISTRIBUTION</div>
          {[["positive", COLORS.green], ["negative", COLORS.red], ["neutral", COLORS.muted]].map(([s, c]) => (
            <HorizBar key={s} label={`${s.charAt(0).toUpperCase() + s.slice(1)} (${td.sentiment_pcts?.[s] || 0}%)`}
              value={td.sentiment_counts?.[s] || 0} max={sentMax} color={c} showPct={false} />
          ))}
          <div style={{ marginTop: 16, fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>BTC PRICE IMPACT</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {[
              ["Avg 15m Δ", `${(td.avg_btc_change_15m || 0) >= 0 ? "+" : ""}${(td.avg_btc_change_15m || 0).toFixed(3)}%`, (td.avg_btc_change_15m || 0) >= 0 ? COLORS.green : COLORS.red],
              ["Avg 1h Δ",  `${(td.avg_btc_change_1h  || 0) >= 0 ? "+" : ""}${(td.avg_btc_change_1h  || 0).toFixed(3)}%`, (td.avg_btc_change_1h  || 0) >= 0 ? COLORS.green : COLORS.red],
            ].map(([l, v, c]) => (
              <div key={l} style={{ background: COLORS.bg, borderRadius: 8, padding: "8px 10px" }}>
                <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>{l}</div>
                <div style={{ fontSize: 15, fontFamily: "monospace", fontWeight: 700, color: c }}>{v}</div>
              </div>
            ))}
          </div>
        </div>

        {/* News type breakdown */}
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>NEWS TYPES (TOP 10)</div>
          {Object.entries(td.news_types || {}).map(([type, count], i) => (
            <HorizBar key={type} label={type} value={count} max={typeMax}
              color={[COLORS.accent, COLORS.blue, COLORS.purple, COLORS.gold, COLORS.green, COLORS.red][i % 6]} showPct={false} />
          ))}
        </div>

        {/* BTC change histogram */}
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>BTC 15m CHANGE DISTRIBUTION</div>
          {(td.btc_change_histogram || []).map((b, i) => {
            const isNeg = b.label.startsWith("<") || b.label.startsWith("-");
            const isNeu = b.label.includes("0%") || b.label.includes("0.5");
            const color = isNeg ? COLORS.red : isNeu ? COLORS.gold : COLORS.green;
            return <HorizBar key={b.label} label={b.label} value={b.count} max={histMax} color={color} showPct={false} />;
          })}
        </div>
      </div>

      {/* Channel breakdown */}
      <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>CHANNEL BREAKDOWN</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0 24px" }}>
          {Object.entries(td.channels || {}).map(([ch, count], i) => (
            <HorizBar key={ch} label={ch} value={count} max={chMax}
              color={[COLORS.accent, COLORS.blue, COLORS.purple, COLORS.gold, COLORS.green][i % 5]} showPct={false} />
          ))}
        </div>
      </div>

      {/* Model performance */}
      <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 16 }}>MODEL PERFORMANCE (production_system_v5.pt)</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>

          {/* 15m metrics */}
          <div>
            <div style={{ fontSize: 11, color: COLORS.accent, fontWeight: 600, marginBottom: 12 }}>15-MINUTE PREDICTION</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginBottom: 16 }}>
              {[
                ["Accuracy",  ((m15.Accuracy || 0) * 100).toFixed(1) + "%",  perfColor(m15.Accuracy, 0.65)],
                ["ROC AUC",   (m15.ROC_AUC || 0).toFixed(3),                  perfColor(m15.ROC_AUC, 0.70)],
                ["F1 Score",  (m15.F1 || 0).toFixed(3),                        perfColor(m15.F1, 0.50)],
                ["Precision", ((m15.Precision || 0) * 100).toFixed(1) + "%",  perfColor(m15.Precision, 0.50)],
                ["Recall",    ((m15.Recall || 0) * 100).toFixed(1) + "%",     perfColor(m15.Recall, 0.60)],
                ["Dir Acc",   ((m15.DirAcc || 0) * 100).toFixed(1) + "%",     perfColor(m15.DirAcc, 0.53)],
              ].map(([l, v, c]) => (
                <div key={l} style={{ background: COLORS.bg, borderRadius: 8, padding: "8px 10px" }}>
                  <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>{l}</div>
                  <div style={{ fontSize: 15, fontFamily: "monospace", fontWeight: 700, color: c }}>{v}</div>
                </div>
              ))}
            </div>
            <ConfusionMatrix cm={m15.CM} label="15m" />
          </div>

          {/* 1h metrics */}
          <div>
            <div style={{ fontSize: 11, color: COLORS.blue, fontWeight: 600, marginBottom: 12 }}>1-HOUR PREDICTION</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginBottom: 16 }}>
              {[
                ["Accuracy",  ((m1h.Accuracy || 0) * 100).toFixed(1) + "%",  perfColor(m1h.Accuracy, 0.65)],
                ["ROC AUC",   (m1h.ROC_AUC || 0).toFixed(3),                  perfColor(m1h.ROC_AUC, 0.70)],
                ["F1 Score",  (m1h.F1 || 0).toFixed(3),                        perfColor(m1h.F1, 0.50)],
                ["Precision", ((m1h.Precision || 0) * 100).toFixed(1) + "%",  perfColor(m1h.Precision, 0.50)],
                ["Recall",    ((m1h.Recall || 0) * 100).toFixed(1) + "%",     perfColor(m1h.Recall, 0.60)],
                ["Dir Acc",   ((m1h.DirAcc || 0) * 100).toFixed(1) + "%",     perfColor(m1h.DirAcc, 0.53)],
              ].map(([l, v, c]) => (
                <div key={l} style={{ background: COLORS.bg, borderRadius: 8, padding: "8px 10px" }}>
                  <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>{l}</div>
                  <div style={{ fontSize: 15, fontFamily: "monospace", fontWeight: 700, color: c }}>{v}</div>
                </div>
              ))}
            </div>
            <ConfusionMatrix cm={m1h.CM} label="1h" />
          </div>
        </div>

        {/* Direction accuracy */}
        {dir.Acc && (
          <div style={{ marginTop: 16, paddingTop: 16, borderTop: `1px solid ${COLORS.border}`, display: "flex", gap: 12 }}>
            <div style={{ background: COLORS.bg, borderRadius: 8, padding: "8px 14px" }}>
              <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>DIRECTION ACCURACY</div>
              <div style={{ fontSize: 18, fontFamily: "monospace", fontWeight: 700, color: perfColor(dir.Acc, 0.53) }}>{((dir.Acc || 0) * 100).toFixed(1)}%</div>
            </div>
            <div style={{ background: COLORS.bg, borderRadius: 8, padding: "8px 14px" }}>
              <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>DIRECTION F1</div>
              <div style={{ fontSize: 18, fontFamily: "monospace", fontWeight: 700, color: perfColor(dir.F1, 0.50) }}>{(dir.F1 || 0).toFixed(3)}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Model Analysis ─────────────────────────────────────────────────
function StatBox({ label, value, color }) {
  return (
    <div style={{ background: COLORS.bg, borderRadius: 10, padding: "14px 16px", border: `1px solid ${COLORS.border2}` }}>
      <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 6 }}>{label.toUpperCase()}</div>
      <div style={{ fontSize: 22, fontFamily: "monospace", fontWeight: 700, color: color || COLORS.accent }}>{value}</div>
    </div>
  );
}

function MiniBar({ label, value, max, color }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: COLORS.text }}>{label}</span>
        <span style={{ fontSize: 11, fontFamily: "monospace", color }}>{value}</span>
      </div>
      <div style={{ height: 5, background: COLORS.border2, borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 3, transition: "width 0.4s ease" }} />
      </div>
    </div>
  );
}

function ModelAnalysis({ news }) {
  const total = news.length || 1;

  // Score buckets (normalized 0–1, displayed as %)
  const buckets = [
    { label: "0% – 30%",  min: 0,    max: 0.30,  color: COLORS.muted },
    { label: "30% – 50%", min: 0.30, max: 0.50,  color: COLORS.blue },
    { label: "50% – 70%", min: 0.50, max: 0.70,  color: COLORS.gold },
    { label: "70% – 90%", min: 0.70, max: 0.90,  color: COLORS.green },
    { label: "90% – 100%",min: 0.90, max: 1.01,  color: COLORS.accent },
  ].map(b => ({
    ...b,
    count: news.filter(n => {
      const s = Math.abs(n.model_score || 0);
      return s >= b.min && s < b.max;
    }).length,
  }));
  const maxBucket = Math.max(...buckets.map(b => b.count), 1);

  // Sentiment counts
  const bullish = news.filter(n => n.sentiment === "positive").length;
  const bearish = news.filter(n => n.sentiment === "negative").length;
  const neutral = total - bullish - bearish;

  // Signal type counts
  const buys      = news.filter(n => n.type === "BUY").length;
  const sells     = news.filter(n => n.type === "SELL").length;
  const neutralSig = total - buys - sells;

  // Averages
  const avg = (fn) => news.length ? (news.reduce((s, n) => s + (fn(n) || 0), 0) / news.length) : 0;
  const avgScore = avg(n => Math.abs(n.model_score || 0));
  const avgConf  = avg(n => n.confidence || 0);
  const avgWeight = avg(n => n.weight || 0);

  // Pred stats
  const pred15True = news.filter(n => n.pred_15m === 1).length;
  const pred1hTrue = news.filter(n => n.pred_1h  === 1).length;

  // RAG stats
  const ragNews    = news.filter(n => (n.similarity || 0) > 0);
  const avgSim     = ragNews.length ? avg(n => n.similarity || 0) : 0;
  const avgRagChg  = ragNews.length ? avg(n => n.rag_avg_change || 0) : 0;

  // Top scored news
  const topNews = [...news].sort((a, b) => Math.abs(b.model_score || 0) - Math.abs(a.model_score || 0)).slice(0, 5);

  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text, letterSpacing: 0.5 }}>
        🧠 Model Analysis <span style={{ fontSize: 11, color: COLORS.muted, fontWeight: 400 }}>— {news.length} news items</span>
      </div>

      {/* KPI row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <StatBox label="Avg Model Score" value={`${Math.round(avgScore * 100)}%`} color={COLORS.accent} />
        <StatBox label="Avg Confidence"  value={`${avgConf.toFixed(1)}%`} color={COLORS.blue} />
        <StatBox label="Avg Weight"      value={`${avgWeight.toFixed(1)}/10`} color={COLORS.gold} />
        <StatBox label="Total Processed" value={news.length} color={COLORS.text} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>

        {/* Score distribution */}
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>SCORE DISTRIBUTION</div>
          {buckets.map(b => (
            <MiniBar key={b.label} label={b.label} value={b.count} max={maxBucket} color={b.color} />
          ))}
        </div>

        {/* Sentiment & signals */}
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>SENTIMENT BREAKDOWN</div>
          <MiniBar label="Bullish" value={bullish} max={total} color={COLORS.green} />
          <MiniBar label="Bearish" value={bearish} max={total} color={COLORS.red} />
          <MiniBar label="Neutral" value={neutral} max={total} color={COLORS.muted} />
          <div style={{ marginTop: 16, fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>SIGNAL TYPES</div>
          <MiniBar label="BUY"     value={buys}      max={total} color={COLORS.green} />
          <MiniBar label="SELL"    value={sells}      max={total} color={COLORS.red} />
          <MiniBar label="NEUTRAL" value={neutralSig} max={total} color={COLORS.muted} />
        </div>

        {/* Prediction & RAG stats */}
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>MODEL PREDICTIONS</div>
          <MiniBar label={`15m Impact (${pred15True})`} value={pred15True} max={total} color={COLORS.accent} />
          <MiniBar label={`1h Impact  (${pred1hTrue})`}  value={pred1hTrue}  max={total} color={COLORS.blue} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 12 }}>
            {[
              ["15m Rate", `${((pred15True / total) * 100).toFixed(1)}%`, COLORS.accent],
              ["1h Rate",  `${((pred1hTrue  / total) * 100).toFixed(1)}%`, COLORS.blue],
            ].map(([l, v, c]) => (
              <div key={l} style={{ background: COLORS.bg, borderRadius: 8, padding: "8px 10px" }}>
                <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>{l}</div>
                <div style={{ fontSize: 14, fontFamily: "monospace", fontWeight: 700, color: c }}>{v}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 16, fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 10 }}>RAG RETRIEVAL</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {[
              ["Avg Similarity", `${(avgSim * 100).toFixed(1)}%`,         COLORS.purple],
              ["Avg Past Chg",   `${avgRagChg >= 0 ? "+" : ""}${avgRagChg.toFixed(2)}%`, avgRagChg >= 0 ? COLORS.green : COLORS.red],
            ].map(([l, v, c]) => (
              <div key={l} style={{ background: COLORS.bg, borderRadius: 8, padding: "8px 10px" }}>
                <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>{l}</div>
                <div style={{ fontSize: 14, fontFamily: "monospace", fontWeight: 700, color: c }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Top scored news */}
      <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>TOP SCORED NEWS</div>
        {topNews.length === 0 ? (
          <div style={{ color: COLORS.muted, fontSize: 12 }}>No data yet</div>
        ) : topNews.map((item, i) => {
          const score = Math.abs(item.model_score || 0);
          const color = score >= 0.80 ? COLORS.accent : score >= 0.60 ? COLORS.green : score >= 0.40 ? COLORS.gold : COLORS.muted;
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 0", borderBottom: i < topNews.length - 1 ? `1px solid ${COLORS.border}` : "none" }}>
              <div style={{ fontSize: 11, fontFamily: "monospace", fontWeight: 700, color, width: 46, flexShrink: 0 }}>
                {fmtScore(score)}
              </div>
              <div style={{ flex: 1, fontSize: 12, color: COLORS.text, lineHeight: 1.4 }}>{item.title}</div>
              <div style={{ fontSize: 10, color: COLORS.muted, flexShrink: 0 }}>{item.channel}</div>
              <div style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, background: `${item.sentiment === "positive" ? COLORS.green : item.sentiment === "negative" ? COLORS.red : COLORS.muted}18`, color: item.sentiment === "positive" ? COLORS.green : item.sentiment === "negative" ? COLORS.red : COLORS.muted, fontFamily: "monospace", fontWeight: 600 }}>
                {item.sentiment === "positive" ? "BULL" : item.sentiment === "negative" ? "BEAR" : "NEU"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Research / About page ──────────────────────────────────────────
// ── Channel & News-Type Analysis ───────────────────────────────────
const NEWS_CATEGORIES = [
  { key: "etf",        label: "ETF / Institutional",   icon: "🏦", keywords: ["etf", "blackrock", "fidelity", "grayscale", "spot bitcoin", "institutional"] },
  { key: "regulation", label: "Regulation / Legal",    icon: "⚖️", keywords: ["sec", "regulation", "ban", "legal", "lawsuit", "congress", "law", "compliance", "cftc", "doj"] },
  { key: "hack",       label: "Security / Hack",       icon: "🔓", keywords: ["hack", "exploit", "breach", "stolen", "attack", "vulnerability", "scam", "fraud"] },
  { key: "adoption",   label: "Adoption / Partnership",icon: "🤝", keywords: ["adoption", "partnership", "launch", "integrate", "accept", "announces", "listed", "adds"] },
  { key: "macro",      label: "Macro / Economic",      icon: "📊", keywords: ["fed", "interest rate", "inflation", "cpi", "gdp", "recession", "jobs", "fomc", "powell"] },
  { key: "market",     label: "Market / Price",        icon: "📈", keywords: ["rally", "crash", "ath", "all-time high", "bull", "bear", "dump", "surge", "plunge", "drops"] },
  { key: "defi",       label: "DeFi / Protocol",       icon: "⛓️", keywords: ["defi", "protocol", "liquidity", "staking", "yield", "dao", "smart contract", "uniswap", "aave"] },
  { key: "other",      label: "Other",                 icon: "📌", keywords: [] },
];

function categorize(title) {
  const t = (title || "").toLowerCase();
  for (const cat of NEWS_CATEGORIES) {
    if (cat.key === "other") continue;
    if (cat.keywords.some(k => t.includes(k))) return cat.key;
  }
  return "other";
}

function PerformanceBar({ value, max, color, width = 120 }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div style={{ width, height: 5, background: COLORS.border2, borderRadius: 3, display: "inline-block" }}>
      <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 3 }} />
    </div>
  );
}

const NEWS_TYPE_ICONS = {
  etf: "🏦", macro_economic: "📊", institutional: "🏛️", partnership: "🤝",
  market_analysis: "📈", defi: "⛓️", regulatory: "⚖️", exchange: "🔄",
  mining: "⛏️", hack: "🔓", technical: "🔧", unknown: "📌",
};

function ChannelAnalysisPage({ news }) {
  const [sortBy, setSortBy] = useState("count");
  const [catSortBy, setCatSortBy] = useState("test_count");
  const [catData, setCatData] = useState([]);
  const [catLoading, setCatLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/training/category-stats`)
      .then(r => r.json())
      .then(d => { setCatData(Array.isArray(d) ? d : []); setCatLoading(false); })
      .catch(() => setCatLoading(false));
  }, []);

  // ── Channel stats from live feed ───────────────────────────────
  const channelMap = {};
  for (const n of news) {
    const ch = n.channel || "Unknown";
    if (!channelMap[ch]) channelMap[ch] = { count: 0, scores: [], confs: [], btc15: [], btc1h: [], buy: 0, sell: 0, neutral: 0 };
    const c = channelMap[ch];
    c.count++;
    if (n.model_score != null) c.scores.push(Math.abs(n.model_score));
    if (n.confidence != null)  c.confs.push(n.confidence);
    if (n.btc_change_15m)      c.btc15.push(Math.abs(n.btc_change_15m));
    if (n.btc_change_1h)       c.btc1h.push(Math.abs(n.btc_change_1h));
    if (n.type === "BUY")      c.buy++;
    else if (n.type === "SELL")c.sell++;
    else                       c.neutral++;
  }
  const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const channels = Object.entries(channelMap).map(([name, c]) => ({
    name,
    count:    c.count,
    avgScore: avg(c.scores),
    avgConf:  avg(c.confs),
    avgBtc15: avg(c.btc15),
    btcCount: c.btc15.length,
    buyRate:  c.count ? c.buy / c.count : 0,
    sellRate: c.count ? c.sell / c.count : 0,
  })).sort((a, b) => {
    if (sortBy === "count")    return b.count - a.count;
    if (sortBy === "avgScore") return b.avgScore - a.avgScore;
    if (sortBy === "avgConf")  return b.avgConf - a.avgConf;
    if (sortBy === "avgBtc15") return b.avgBtc15 - a.avgBtc15;
    return b.count - a.count;
  });
  const maxCount = Math.max(...channels.map(c => c.count), 1);
  const maxScore = Math.max(...channels.map(c => c.avgScore), 1);

  // ── Category stats from API ────────────────────────────────────
  const categories = [...catData].sort((a, b) => {
    if (catSortBy === "test_count")  return b.test_count - a.test_count;
    if (catSortBy === "train_count") return b.train_count - a.train_count;
    if (catSortBy === "accuracy")    return (b.accuracy ?? -1) - (a.accuracy ?? -1);
    if (catSortBy === "precision")   return (b.precision ?? -1) - (a.precision ?? -1);
    return b.test_count - a.test_count;
  });
  const maxTrain = Math.max(...categories.map(c => c.train_count), 1);
  const maxTest  = Math.max(...categories.map(c => c.test_count), 1);

  const sortBtn = (id, label, active, setter) => (
    <button onClick={() => setter(id)} style={{
      padding: "3px 10px", borderRadius: 5, fontSize: 10, cursor: "pointer", border: "none",
      background: active === id ? COLORS.accent : COLORS.border2,
      color: active === id ? "#000" : COLORS.muted, fontWeight: active === id ? 700 : 400,
    }}>{label}</button>
  );

  const rankColor = (i) => i === 0 ? COLORS.gold : i === 1 ? COLORS.muted : i === 2 ? "#cd7f32" : COLORS.border2;

  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 24 }}>

      {/* ── Channel Performance ── */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text }}>📡 Channel Performance</div>
            <div style={{ fontSize: 11, color: COLORS.muted, marginTop: 2 }}>Which channels produce the highest-impact signals</div>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {sortBtn("count",    "Volume",    sortBy, setSortBy)}
            {sortBtn("avgScore", "Avg Score", sortBy, setSortBy)}
            {sortBtn("avgConf",  "Confidence",sortBy, setSortBy)}
            {sortBtn("avgBtc15", "BTC Impact",sortBy, setSortBy)}
          </div>
        </div>

        <div style={{ background: COLORS.panel, borderRadius: 12, border: `1px solid ${COLORS.border2}`, overflow: "hidden" }}>
          {/* Header row */}
          <div style={{ display: "grid", gridTemplateColumns: "26px 180px 1fr 100px 100px 100px 90px", gap: 0, padding: "8px 16px", borderBottom: `1px solid ${COLORS.border}` }}>
            {["#", "Channel", "Volume", "Avg Score", "Confidence", "BTC 15m Δ", "Signal Split"].map(h => (
              <div key={h} style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, textTransform: "uppercase" }}>{h}</div>
            ))}
          </div>
          {channels.map((ch, i) => (
            <div key={ch.name} style={{ display: "grid", gridTemplateColumns: "26px 180px 1fr 100px 100px 100px 90px", gap: 0, padding: "10px 16px", borderBottom: i < channels.length - 1 ? `1px solid ${COLORS.border}` : "none", alignItems: "center" }}>
              {/* Rank */}
              <div style={{ fontSize: 11, fontWeight: 700, color: rankColor(i), fontFamily: "monospace" }}>{i + 1}</div>
              {/* Name */}
              <div style={{ fontSize: 12, color: COLORS.text, fontWeight: 500 }}>{ch.name}</div>
              {/* Volume bar */}
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <PerformanceBar value={ch.count} max={maxCount} color={COLORS.accent} width={100} />
                <span style={{ fontSize: 11, color: COLORS.muted, fontFamily: "monospace" }}>{ch.count.toLocaleString()}</span>
              </div>
              {/* Avg score */}
              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <span style={{ fontSize: 12, fontFamily: "monospace", fontWeight: 700, color: ch.avgScore >= 0.67 ? COLORS.green : ch.avgScore >= 0.50 ? COLORS.gold : COLORS.muted }}>
                  {Math.round(ch.avgScore * 100)}%
                </span>
                <PerformanceBar value={ch.avgScore} max={maxScore} color={COLORS.gold} width={60} />
              </div>
              {/* Confidence */}
              <div style={{ fontSize: 12, fontFamily: "monospace", color: COLORS.blue }}>{ch.avgConf.toFixed(1)}%</div>
              {/* BTC impact */}
              <div style={{ fontSize: 12, fontFamily: "monospace", color: ch.btcCount > 0 ? COLORS.green : COLORS.border2 }}>
                {ch.btcCount > 0 ? `${ch.avgBtc15.toFixed(3)}%` : "—"}
              </div>
              {/* Signal split mini-bars */}
              <div style={{ display: "flex", gap: 2, height: 14, alignItems: "center" }}>
                <div style={{ height: "100%", width: `${ch.buyRate * 60}px`, background: COLORS.green, borderRadius: 2, maxWidth: 28 }} title={`BUY ${Math.round(ch.buyRate*100)}%`} />
                <div style={{ height: "100%", width: `${ch.sellRate * 60}px`, background: COLORS.red, borderRadius: 2, maxWidth: 28 }} title={`SELL ${Math.round(ch.sellRate*100)}%`} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── News Category Performance ── */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text }}>🗂️ News Category Performance</div>
            <div style={{ fontSize: 11, color: COLORS.muted, marginTop: 2 }}>Which types of news the model predicted most confidently</div>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {sortBtn("test_count",  "Test Vol",   catSortBy, setCatSortBy)}
            {sortBtn("train_count", "Train Vol",  catSortBy, setCatSortBy)}
            {sortBtn("accuracy",    "Accuracy",   catSortBy, setCatSortBy)}
            {sortBtn("precision",   "Precision",  catSortBy, setCatSortBy)}
          </div>
        </div>

        {/* ── Performance ranking bar chart ── */}
        {!catLoading && catData.filter(c => c.accuracy != null && c.test_count >= 3).length > 0 && (() => {
          const ranked = [...catData]
            .filter(c => c.accuracy != null && c.test_count >= 3)
            .sort((a, b) => b.accuracy - a.accuracy);
          const best = ranked[0];
          const worst = ranked[ranked.length - 1];
          return (
            <div style={{ background: COLORS.panel, borderRadius: 12, padding: 18, border: `1px solid ${COLORS.border2}`, marginBottom: 14 }}>
              {/* Summary badges */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.text }}>📊 Accuracy Ranking — Best → Worst</div>
                <div style={{ display: "flex", gap: 8 }}>
                  <div style={{ fontSize: 10, padding: "3px 10px", borderRadius: 20, background: `${COLORS.green}20`, color: COLORS.green, fontFamily: "monospace", border: `1px solid ${COLORS.green}40` }}>
                    🏆 Best: {best.news_type.replace(/_/g, " ")} {Math.round(best.accuracy * 100)}%
                  </div>
                  <div style={{ fontSize: 10, padding: "3px 10px", borderRadius: 20, background: `${COLORS.red}20`, color: COLORS.red, fontFamily: "monospace", border: `1px solid ${COLORS.red}40` }}>
                    ⚠ Worst: {worst.news_type.replace(/_/g, " ")} {Math.round(worst.accuracy * 100)}%
                  </div>
                </div>
              </div>
              {/* Ranked rows */}
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {ranked.map((cat, i) => {
                  const acc      = cat.accuracy;
                  const barColor = acc >= 0.75 ? COLORS.green : acc >= 0.65 ? COLORS.gold : COLORS.red;
                  const tag      = acc >= 0.75 ? "GOOD" : acc >= 0.65 ? "OK" : "POOR";
                  const rColor   = i === 0 ? COLORS.gold : i === 1 ? COLORS.muted : i === 2 ? "#cd7f32" : COLORS.border2;
                  return (
                    <div key={cat.news_type} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      {/* rank */}
                      <div style={{ width: 18, fontSize: 10, fontFamily: "monospace", fontWeight: 700, color: rColor, textAlign: "right", flexShrink: 0 }}>{i + 1}</div>
                      {/* icon + name */}
                      <div style={{ width: 150, display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
                        <span style={{ fontSize: 13 }}>{NEWS_TYPE_ICONS[cat.news_type] ?? "📌"}</span>
                        <span style={{ fontSize: 11, color: COLORS.text, textTransform: "capitalize" }}>{cat.news_type.replace(/_/g, " ")}</span>
                      </div>
                      {/* bar */}
                      <div style={{ flex: 1, height: 9, background: COLORS.border2, borderRadius: 5 }}>
                        <div style={{ height: "100%", width: `${acc * 100}%`, background: barColor, borderRadius: 5, transition: "width 0.5s ease" }} />
                      </div>
                      {/* acc % */}
                      <div style={{ width: 40, fontSize: 12, fontFamily: "monospace", fontWeight: 700, color: barColor, textAlign: "right", flexShrink: 0 }}>{Math.round(acc * 100)}%</div>
                      {/* precision */}
                      <div style={{ width: 52, fontSize: 10, fontFamily: "monospace", color: COLORS.muted, textAlign: "right", flexShrink: 0 }}>
                        P:{cat.precision != null ? Math.round(cat.precision * 100) + "%" : "—"}
                      </div>
                      {/* recall */}
                      <div style={{ width: 52, fontSize: 10, fontFamily: "monospace", color: COLORS.muted, textAlign: "right", flexShrink: 0 }}>
                        R:{cat.recall != null ? Math.round(cat.recall * 100) + "%" : "—"}
                      </div>
                      {/* n tests */}
                      <div style={{ width: 48, fontSize: 9, color: COLORS.border2, textAlign: "right", flexShrink: 0 }}>{cat.test_count}n</div>
                      {/* tag */}
                      <div style={{ width: 34, fontSize: 8, padding: "1px 4px", borderRadius: 3, background: `${barColor}20`, color: barColor, fontFamily: "monospace", fontWeight: 700, textAlign: "center", flexShrink: 0 }}>{tag}</div>
                    </div>
                  );
                })}
              </div>
              {/* Legend */}
              <div style={{ display: "flex", gap: 16, marginTop: 14, paddingTop: 12, borderTop: `1px solid ${COLORS.border}` }}>
                {[[COLORS.green, "≥75% — Good"], [COLORS.gold, "65–74% — OK"], [COLORS.red, "<65% — Poor"]].map(([c, l]) => (
                  <div key={l} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: c }} />
                    <span style={{ fontSize: 10, color: COLORS.muted }}>{l}</span>
                  </div>
                ))}
                <div style={{ marginLeft: "auto", fontSize: 9, color: COLORS.border2 }}>P = Precision · R = Recall · n = test samples</div>
              </div>
            </div>
          );
        })()}

        {catLoading ? (
          <div style={{ padding: 40, textAlign: "center", color: COLORS.muted, fontSize: 12 }}>Loading training data…</div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
            {categories.map((cat, i) => {
              const accColor  = cat.accuracy >= 0.75 ? COLORS.green : cat.accuracy >= 0.65 ? COLORS.gold : COLORS.red;
              const precColor = cat.precision >= 0.4  ? COLORS.green : cat.precision >= 0.25 ? COLORS.gold : COLORS.red;
              return (
                <div key={cat.news_type} style={{ background: COLORS.panel, borderRadius: 10, padding: 16, border: `1px solid ${COLORS.border2}`, position: "relative", overflow: "hidden" }}>
                  {i < 3 && <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: rankColor(i) }} />}

                  {/* Header */}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                    <span style={{ fontSize: 20 }}>{NEWS_TYPE_ICONS[cat.news_type] ?? "📌"}</span>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: 9, fontFamily: "monospace", color: COLORS.muted }}>{cat.train_count.toLocaleString()} train</div>
                      <div style={{ fontSize: 9, fontFamily: "monospace", color: COLORS.accent }}>{cat.test_count} test</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 12, fontWeight: 700, color: COLORS.text, marginBottom: 12, textTransform: "capitalize" }}>
                    {cat.news_type.replace(/_/g, " ")}
                  </div>

                  {/* Training set bar */}
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                      <span style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>TRAIN SET</span>
                      <span style={{ fontSize: 9, fontFamily: "monospace", color: COLORS.muted }}>{cat.train_count.toLocaleString()}</span>
                    </div>
                    <div style={{ height: 4, background: COLORS.border2, borderRadius: 2 }}>
                      <div style={{ height: "100%", width: `${(cat.train_count / maxTrain) * 100}%`, background: COLORS.blue, borderRadius: 2 }} />
                    </div>
                  </div>

                  {/* Test set bar */}
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                      <span style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>TEST SET</span>
                      <span style={{ fontSize: 9, fontFamily: "monospace", color: COLORS.accent }}>{cat.test_count}</span>
                    </div>
                    <div style={{ height: 4, background: COLORS.border2, borderRadius: 2 }}>
                      <div style={{ height: "100%", width: `${(cat.test_count / maxTest) * 100}%`, background: COLORS.accent, borderRadius: 2 }} />
                    </div>
                  </div>

                  {/* Accuracy / Precision / Recall */}
                  {cat.accuracy !== null ? (
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
                      {[
                        ["ACC", cat.accuracy, accColor],
                        ["PREC", cat.precision, precColor],
                        ["REC", cat.recall, COLORS.purple],
                      ].map(([lbl, val, clr]) => (
                        <div key={lbl} style={{ background: COLORS.bg, borderRadius: 6, padding: "6px 8px", textAlign: "center" }}>
                          <div style={{ fontSize: 8, color: COLORS.muted, letterSpacing: 1 }}>{lbl}</div>
                          <div style={{ fontSize: 13, fontFamily: "monospace", fontWeight: 700, color: clr }}>
                            {val != null ? `${Math.round(val * 100)}%` : "—"}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ fontSize: 9, color: COLORS.border2 }}>No test data</div>
                  )}

                  {/* TP/TN/FP/FN mini row */}
                  {cat.test_count > 0 && (
                    <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
                      {[["TP", cat.tp, COLORS.green], ["TN", cat.tn, COLORS.muted], ["FP", cat.fp, COLORS.red], ["FN", cat.fn, COLORS.gold]].map(([lbl, val, clr]) => (
                        <span key={lbl} style={{ fontSize: 9, fontFamily: "monospace", color: clr, background: `${clr}15`, padding: "1px 5px", borderRadius: 3 }}>
                          {lbl}:{val}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Key Insights ── */}
      <div style={{ background: COLORS.panel, borderRadius: 12, padding: 18, border: `1px solid ${COLORS.border2}` }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.accent, marginBottom: 12, letterSpacing: 0.5 }}>⚡ Key Insights</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
          {(() => {
            const topCh    = [...channels].sort((a, b) => b.avgScore - a.avgScore)[0];
            const topBtcCh = channels.filter(c => c.btcCount > 0).sort((a, b) => b.avgBtc15 - a.avgBtc15)[0];
            const topAcc   = [...catData].filter(c => c.accuracy != null).sort((a, b) => b.accuracy - a.accuracy)[0];
            const topPrec  = [...catData].filter(c => c.precision != null).sort((a, b) => b.precision - a.precision)[0];
            return [
              { label: "Highest Avg Score Channel",    value: topCh?.name ?? "—",    sub: `${Math.round((topCh?.avgScore ?? 0) * 100)}% avg model score`, color: COLORS.gold },
              { label: "Best Accuracy Category",        value: topAcc ? topAcc.news_type.replace(/_/g," ") : "—", sub: topAcc ? `${Math.round(topAcc.accuracy*100)}% accuracy on ${topAcc.test_count} test items` : "", color: COLORS.green },
              { label: "Best Precision Category",       value: topPrec ? topPrec.news_type.replace(/_/g," ") : "—", sub: topPrec ? `${Math.round(topPrec.precision*100)}% precision` : "", color: COLORS.accent },
            ].map(({ label, value, sub, color }) => (
              <div key={label} style={{ background: COLORS.bg, borderRadius: 8, padding: "12px 14px" }}>
                <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 6, textTransform: "uppercase" }}>{label}</div>
                <div style={{ fontSize: 14, fontWeight: 700, color, fontFamily: "monospace", textTransform: "capitalize" }}>{value}</div>
                <div style={{ fontSize: 10, color: COLORS.muted, marginTop: 4 }}>{sub}</div>
              </div>
            ));
          })()}
        </div>
      </div>

    </div>
  );
}

// ── Report Page ────────────────────────────────────────────────────
function ReportPage() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/report/summary`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, []);

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 200, color: COLORS.muted, fontSize: 13 }}>
      Loading report…
    </div>
  );
  if (error || !data) return (
    <div style={{ padding: 24, color: COLORS.red, fontSize: 13 }}>Failed to load: {error}</div>
  );

  const { training: tr, cache: ca, results: res, architecture: arch, category_results: catRes = [] } = data;
  const m15 = res?.["15_minute"] || {};
  const m1h  = res?.["1_hour"]   || {};
  const mdir = res?.["direction"] || {};

  const pct  = (v) => v != null ? `${Math.round(v * 100)}%` : "—";
  const num  = (v, d = 3) => v != null ? v.toFixed(d) : "—";
  const perfClr = (v, good) => !v ? COLORS.muted : v >= good ? COLORS.green : v >= good * 0.8 ? COLORS.gold : COLORS.red;

  const Kpi = ({ label, value, sub, color }) => (
    <div style={{ background: COLORS.bg, borderRadius: 10, padding: "14px 16px", border: `1px solid ${COLORS.border2}` }}>
      <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 6 }}>{label.toUpperCase()}</div>
      <div style={{ fontSize: 20, fontFamily: "monospace", fontWeight: 700, color: color || COLORS.accent }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: COLORS.muted, marginTop: 4 }}>{sub}</div>}
    </div>
  );

  const MetRow = ({ label, value, color }) => (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: `1px solid ${COLORS.border}`, fontSize: 12 }}>
      <span style={{ color: COLORS.muted }}>{label}</span>
      <span style={{ fontFamily: "monospace", fontWeight: 700, color: color || COLORS.text }}>{value}</span>
    </div>
  );

  const sentTotal = (tr.sentiment_counts?.positive || 0) + (tr.sentiment_counts?.negative || 0) + (tr.sentiment_counts?.neutral || 0) || 1;
  const caSentTotal = (ca.sentiment_counts?.positive || 0) + (ca.sentiment_counts?.negative || 0) + (ca.sentiment_counts?.neutral || 0) || 1;

  return (
    <div style={{ padding: 28, display: "flex", flexDirection: "column", gap: 24, maxWidth: 1200 }}>

      {/* ── Header ── */}
      <div style={{ borderBottom: `1px solid ${COLORS.border}`, paddingBottom: 18 }}>
        <div style={{ fontSize: 9, letterSpacing: 2, color: COLORS.accent, fontFamily: "monospace", marginBottom: 6 }}>
          DATA & MODEL REPORT
        </div>
        <div style={{ fontSize: 20, fontWeight: 700, color: COLORS.text }}>
          {arch.name} — Training & Deployment Summary
        </div>
        <div style={{ fontSize: 11, color: COLORS.muted, marginTop: 6 }}>
          Training file: <span style={{ color: COLORS.text, fontFamily: "monospace" }}>{tr.file}</span>
          &nbsp;·&nbsp;
          Cache file: <span style={{ color: COLORS.text, fontFamily: "monospace" }}>{ca.file}</span>
          &nbsp;·&nbsp;
          Results: <span style={{ color: COLORS.text, fontFamily: "monospace" }}>production_results_v5.json</span>
        </div>
      </div>

      {/* ── Training Data ── */}
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text }}>📂 Training Data</div>
          <span style={{ fontSize: 10, background: `${COLORS.green}20`, color: COLORS.green, borderRadius: 20, padding: "2px 10px", border: `1px solid ${COLORS.green}40` }}>
            news_cleaned.csv — weight≥5
          </span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 16 }}>
          <Kpi label="Raw Rows"      value={(tr.total_raw || 0).toLocaleString()}      color={COLORS.muted} />
          <Kpi label="After Filter"  value={(tr.total_filtered || 0).toLocaleString()} sub="weight≥5 · no null BTC prices" color={COLORS.text} />
          <Kpi label="Date Range"    value={tr.date_min ? `${tr.date_min}` : "—"}      sub={`to ${tr.date_max || "—"}`} color={COLORS.blue} />
          <Kpi label="Split"         value="70 / 15 / 15"                              sub={`train/val/test · seed=${tr.split?.seed}`} color={COLORS.gold} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>

          {/* Split counts */}
          <div style={{ background: COLORS.panel, borderRadius: 10, padding: 16, border: `1px solid ${COLORS.border2}` }}>
            <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 12 }}>MONTHLY RANDOM SPLIT</div>
            {[
              ["Train (70%)", tr.split?.train_n, COLORS.green],
              ["Val   (15%)", tr.split?.val_n,   COLORS.gold],
              ["Test  (15%)", tr.split?.test_n,  COLORS.accent],
            ].map(([lbl, n, c]) => (
              <div key={lbl} style={{ marginBottom: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 11, color: COLORS.muted }}>{lbl}</span>
                  <span style={{ fontSize: 11, fontFamily: "monospace", color: c }}>{(n || 0).toLocaleString()}</span>
                </div>
                <div style={{ height: 4, background: COLORS.border2, borderRadius: 2 }}>
                  <div style={{ height: "100%", width: `${(n || 0) / (tr.total_filtered || 1) * 100}%`, background: c, borderRadius: 2 }} />
                </div>
              </div>
            ))}
          </div>

          {/* Class balance */}
          <div style={{ background: COLORS.panel, borderRadius: 10, padding: 16, border: `1px solid ${COLORS.border2}` }}>
            <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 12 }}>CLASS BALANCE (IMPACT LABELS)</div>
            {[
              [`BTC|Δ|≥0.3% in 15m`, tr.impactful_15m_count, tr.impactful_15m_pct, COLORS.accent],
              [`BTC|Δ|≥0.5% in 1h`,  tr.impactful_1h_count,  tr.impactful_1h_pct,  COLORS.blue],
            ].map(([lbl, n, p, c]) => (
              <div key={lbl} style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 10, color: COLORS.muted }}>{lbl}</span>
                  <span style={{ fontSize: 11, fontFamily: "monospace", color: c }}>{(n || 0).toLocaleString()} ({p}%)</span>
                </div>
                <div style={{ height: 5, background: COLORS.border2, borderRadius: 2 }}>
                  <div style={{ height: "100%", width: `${p || 0}%`, background: c, borderRadius: 2 }} />
                </div>
              </div>
            ))}
            <div style={{ fontSize: 9, color: COLORS.border2, marginTop: 8 }}>
              Class imbalance: most items are non-impactful → naive accuracy inflated
            </div>
          </div>

          {/* Sentiment */}
          <div style={{ background: COLORS.panel, borderRadius: 10, padding: 16, border: `1px solid ${COLORS.border2}` }}>
            <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 12 }}>SENTIMENT DISTRIBUTION</div>
            {[
              ["Positive", tr.sentiment_counts?.positive || 0, COLORS.green],
              ["Neutral",  tr.sentiment_counts?.neutral  || 0, COLORS.gold],
              ["Negative", tr.sentiment_counts?.negative || 0, COLORS.red],
            ].map(([lbl, n, c]) => (
              <div key={lbl} style={{ marginBottom: 9 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 11, color: COLORS.muted }}>{lbl}</span>
                  <span style={{ fontSize: 11, fontFamily: "monospace", color: c }}>
                    {n.toLocaleString()} ({Math.round(n / sentTotal * 100)}%)
                  </span>
                </div>
                <div style={{ height: 4, background: COLORS.border2, borderRadius: 2 }}>
                  <div style={{ height: "100%", width: `${n / sentTotal * 100}%`, background: c, borderRadius: 2 }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* News type counts */}
        {Object.keys(tr.news_type_counts || {}).length > 0 && (
          <div style={{ background: COLORS.panel, borderRadius: 10, padding: 16, border: `1px solid ${COLORS.border2}`, marginTop: 14 }}>
            <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 12 }}>NEWS TYPES IN TRAINING SET</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0 24px" }}>
              {Object.entries(tr.news_type_counts).map(([nt, n], i) => {
                const max = Math.max(...Object.values(tr.news_type_counts));
                return (
                  <div key={nt} style={{ marginBottom: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                      <span style={{ fontSize: 11, color: COLORS.text, textTransform: "capitalize" }}>
                        {(NEWS_TYPE_ICONS[nt] ?? "📌")} {nt.replace(/_/g, " ")}
                      </span>
                      <span style={{ fontSize: 11, fontFamily: "monospace", color: COLORS.accent }}>{n.toLocaleString()}</span>
                    </div>
                    <div style={{ height: 3, background: COLORS.border2, borderRadius: 2 }}>
                      <div style={{ height: "100%", width: `${n / max * 100}%`, background: [COLORS.accent, COLORS.blue, COLORS.gold, COLORS.green, COLORS.purple, COLORS.red][i % 6], borderRadius: 2 }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* ── Model Architecture ── */}
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text, marginBottom: 14 }}>🧠 Model Architecture — {arch.name}</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 14 }}>
          {arch.towers.map((t, i) => (
            <div key={t.name} style={{ background: COLORS.panel, borderRadius: 10, padding: 14, border: `1px solid ${COLORS.border2}`, borderTop: `3px solid ${[COLORS.accent, COLORS.blue, COLORS.gold, COLORS.purple][i]}` }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: [COLORS.accent, COLORS.blue, COLORS.gold, COLORS.purple][i], marginBottom: 6 }}>
                Tower {i + 1}: {t.name}
              </div>
              <div style={{ fontSize: 10, color: COLORS.muted, lineHeight: 1.6, marginBottom: 8 }}>{t.input}</div>
              <div style={{ fontSize: 10, fontFamily: "monospace", color: COLORS.text }}>→ {t.output}-dim output</div>
            </div>
          ))}
        </div>
        <div style={{ background: COLORS.panel, borderRadius: 10, padding: 16, border: `1px solid ${COLORS.border2}` }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
            {[
              ["Fusion",     arch.fusion],
              ["Loss",       arch.loss],
              ["Optimizer",  arch.optimizer],
              ["Epochs",     `${arch.epochs} (patience=${arch.patience})`],
              ["Batch Size", arch.batch_size],
              ["Embedding",  arch.embedding],
            ].map(([lbl, val]) => (
              <div key={lbl}>
                <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 3 }}>{lbl.toUpperCase()}</div>
                <div style={{ fontSize: 11, color: COLORS.text, fontFamily: "monospace" }}>{val}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Model Evaluation Results ── */}
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text }}>📊 Test Results — Model Evaluation</div>
          <span style={{ fontSize: 10, background: `${COLORS.blue}20`, color: COLORS.blue, borderRadius: 20, padding: "2px 10px", border: `1px solid ${COLORS.blue}40` }}>
            production_results_v5.json · 5,191 test items
          </span>
        </div>

        {/* ── Overall Verdict ── */}
        {(() => {
          const f1_15 = m15.F1 || 0, auc_15 = m15.ROC_AUC || 0;
          const f1_1h = m1h.F1  || 0, auc_1h = m1h.ROC_AUC  || 0;
          const avgAuc = (auc_15 + auc_1h) / 2;
          const verdict = avgAuc >= 0.70 ? { label: "Good", color: COLORS.green, icon: "✅" }
                        : avgAuc >= 0.60 ? { label: "Moderate", color: COLORS.gold, icon: "⚠️" }
                        : { label: "Weak", color: COLORS.red, icon: "❌" };
          return (
            <div style={{ background: `${verdict.color}12`, border: `1px solid ${verdict.color}40`, borderRadius: 12, padding: "16px 20px", marginBottom: 18, display: "flex", alignItems: "center", gap: 20 }}>
              <div style={{ fontSize: 32 }}>{verdict.icon}</div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: verdict.color, marginBottom: 4 }}>
                  Overall Verdict: {verdict.label} Predictive Power
                </div>
                <div style={{ fontSize: 12, color: COLORS.text, lineHeight: 1.7 }}>
                  The model has <strong style={{ color: COLORS.gold }}>genuine but limited</strong> ability to detect impactful news.
                  AUC ≈ {((auc_15 + auc_1h) / 2).toFixed(2)} means it ranks impactful news above non-impactful
                  news better than random ({">"}0.50), but precision is low — many false alarms.
                  The 15m horizon is more reliable; the 1h model trades precision for high recall.
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8, flexShrink: 0 }}>
                {[["15m AUC", auc_15, COLORS.accent], ["1h AUC", auc_1h, COLORS.blue]].map(([l, v, c]) => (
                  <div key={l} style={{ textAlign: "center", background: COLORS.bg, borderRadius: 8, padding: "8px 16px" }}>
                    <div style={{ fontSize: 9, color: COLORS.muted }}>{l}</div>
                    <div style={{ fontSize: 18, fontFamily: "monospace", fontWeight: 700, color: c }}>{v.toFixed(3)}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        })()}

        {/* ── Class Imbalance Warning ── */}
        {m15.CM && (() => {
          const [[tn15, fp15], [fn15, tp15]] = m15.CM;
          const total15  = tn15 + fp15 + fn15 + tp15 || 1;
          const pos15    = fn15 + tp15;   // actual impact events
          const neg15    = tn15 + fp15;   // actual non-impact
          const naive15  = Math.round(neg15 / total15 * 100);
          return (
            <div style={{ background: COLORS.panel, borderRadius: 12, padding: 18, marginBottom: 18, border: `1px solid ${COLORS.border2}`, borderLeft: `4px solid ${COLORS.gold}` }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.gold, marginBottom: 10 }}>⚠️ Why Accuracy Alone Is Misleading — Class Imbalance</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                <div>
                  <div style={{ fontSize: 10, color: COLORS.muted, marginBottom: 6 }}>Test set composition (15m)</div>
                  <div style={{ height: 12, background: COLORS.border2, borderRadius: 6, overflow: "hidden", display: "flex" }}>
                    <div style={{ width: `${neg15 / total15 * 100}%`, background: COLORS.muted, borderRadius: "6px 0 0 6px" }} title={`No impact: ${neg15}`} />
                    <div style={{ width: `${pos15 / total15 * 100}%`, background: COLORS.accent }} title={`Impact: ${pos15}`} />
                  </div>
                  <div style={{ display: "flex", gap: 12, marginTop: 6, fontSize: 10 }}>
                    <span style={{ color: COLORS.muted }}>■ No impact: {neg15.toLocaleString()} ({Math.round(neg15/total15*100)}%)</span>
                    <span style={{ color: COLORS.accent }}>■ Impact: {pos15.toLocaleString()} ({Math.round(pos15/total15*100)}%)</span>
                  </div>
                </div>
                <div style={{ background: COLORS.bg, borderRadius: 8, padding: "12px 14px" }}>
                  <div style={{ fontSize: 10, color: COLORS.muted, marginBottom: 4 }}>Naive baseline accuracy</div>
                  <div style={{ fontSize: 20, fontFamily: "monospace", fontWeight: 700, color: COLORS.red }}>{naive15}%</div>
                  <div style={{ fontSize: 10, color: COLORS.muted, marginTop: 4, lineHeight: 1.5 }}>
                    A model that <em>always</em> says "no impact" would score {naive15}% accuracy — higher than our model's {Math.round((m15.Accuracy || 0) * 100)}%!
                  </div>
                </div>
                <div style={{ background: COLORS.bg, borderRadius: 8, padding: "12px 14px" }}>
                  <div style={{ fontSize: 10, color: COLORS.muted, marginBottom: 4 }}>Why F1 & AUC are better</div>
                  <div style={{ fontSize: 11, color: COLORS.text, lineHeight: 1.6 }}>
                    F1 score penalises missing impact events. AUC measures whether the model <em>ranks</em> impact news higher — regardless of threshold. Both reveal the true signal.
                  </div>
                </div>
              </div>
            </div>
          );
        })()}

        {/* ── Side-by-side horizon comparison ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
          {([
            ["⏱ 15-Minute Prediction", m15, COLORS.accent,
              "Best for short-term impact detection. Balanced threshold (0.39). More conservative — fewer false alarms but misses some events.",
              [["Accuracy", m15.Accuracy, 0.65, "% of all predictions correct. Misleading here due to class imbalance."],
               ["F1 Score", m15.F1, 0.35, "Harmonic mean of precision+recall. Main metric for imbalanced datasets."],
               ["ROC AUC",  m15.ROC_AUC, 0.60, "Ability to rank impactful news above non-impactful. 0.5=random, 1.0=perfect."],
               ["Precision",m15.Precision, 0.30, "Of all 'impact' predictions, how many were real. Low = many false alarms."],
               ["Recall",   m15.Recall, 0.40, "Of all real impact events, how many did we catch. Missing events = cost."],
               ["Dir Acc",  m15.DirAcc, 0.53, "Did the model predict price direction (up/down) correctly?"],
              ]
            ],
            ["🕐 1-Hour Prediction", m1h, COLORS.blue,
              "Higher recall — catches more impact events but at the cost of many false alarms (low precision). Aggressive threshold (0.40).",
              [["Accuracy", m1h.Accuracy, 0.65, "Below 50% — the model predicts many positives, dragging accuracy below naive baseline."],
               ["F1 Score", m1h.F1, 0.35, "Higher F1 than 15m thanks to very high recall. Better overall balance."],
               ["ROC AUC",  m1h.ROC_AUC, 0.60, "Slightly lower AUC than 15m — discriminative power is similar."],
               ["Precision",m1h.Precision, 0.30, "Only 1 in 3.5 impact predictions are correct. High noise."],
               ["Recall",   m1h.Recall, 0.40, "Catches ~79% of all real impact events. Very sensitive detector."],
               ["Dir Acc",  m1h.DirAcc, 0.53, "Slightly better than random for direction — marginal edge."],
              ]
            ],
          ]).map(([title, m, clr, summary, rows]) => (
            <div key={title} style={{ background: COLORS.panel, borderRadius: 12, padding: 18, border: `1px solid ${COLORS.border2}` }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: clr, marginBottom: 6 }}>{title}</div>
              <div style={{ fontSize: 11, color: COLORS.muted, lineHeight: 1.6, marginBottom: 14, borderLeft: `3px solid ${clr}40`, paddingLeft: 10 }}>{summary}</div>
              {rows.map(([lbl, val, good, desc]) => {
                const v    = val || 0;
                const c    = v >= good ? COLORS.green : v >= good * 0.8 ? COLORS.gold : COLORS.red;
                const bar  = Math.min(100, v * 100);
                return (
                  <div key={lbl} style={{ marginBottom: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 3 }}>
                      <span style={{ fontSize: 11, fontWeight: 700, color: COLORS.text }}>{lbl}</span>
                      <span style={{ fontSize: 14, fontFamily: "monospace", fontWeight: 700, color: c }}>
                        {lbl === "F1 Score" || lbl === "ROC AUC" || lbl === "Dir Acc" ? v.toFixed(3) : `${Math.round(v * 100)}%`}
                      </span>
                    </div>
                    <div style={{ height: 5, background: COLORS.border2, borderRadius: 3, marginBottom: 3 }}>
                      <div style={{ height: "100%", width: `${bar}%`, background: c, borderRadius: 3 }} />
                    </div>
                    <div style={{ fontSize: 10, color: COLORS.muted, lineHeight: 1.5 }}>{desc}</div>
                  </div>
                );
              })}

              {/* Confusion Matrix */}
              {m.CM && (() => {
                const [[tn, fp], [fn, tp]] = m.CM;
                const total = tn + fp + fn + tp || 1;
                return (
                  <div style={{ marginTop: 14, paddingTop: 14, borderTop: `1px solid ${COLORS.border}` }}>
                    <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 10 }}>CONFUSION MATRIX — WHAT THE MODEL ACTUALLY DID</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                      {[
                        ["✅ True Negative", "TN", tn, COLORS.green,  "Correctly said no impact.\nNo trade — saved from noise."],
                        ["🔴 False Positive", "FP", fp, COLORS.red,    "Said impact — was wrong.\nFalse alarm, wasted signal."],
                        ["⚠️ False Negative", "FN", fn, COLORS.gold,   "Missed a real impact event.\nLost opportunity."],
                        ["✅ True Positive",  "TP", tp, COLORS.accent, "Correctly caught a real impact.\nSignal was valid."],
                      ].map(([label, tag, n, c, note]) => (
                        <div key={tag} style={{ background: `${c}12`, borderRadius: 8, padding: "10px 12px", border: `1px solid ${c}30` }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 4 }}>
                            <span style={{ fontSize: 10, color: c, fontWeight: 700 }}>{label}</span>
                            <span style={{ fontSize: 9, fontFamily: "monospace", color: COLORS.muted }}>{Math.round(n / total * 100)}%</span>
                          </div>
                          <div style={{ fontSize: 20, fontFamily: "monospace", fontWeight: 700, color: c, marginBottom: 4 }}>{n.toLocaleString()}</div>
                          <div style={{ fontSize: 9, color: COLORS.muted, lineHeight: 1.5, whiteSpace: "pre-line" }}>{note}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </div>
          ))}
        </div>

        {/* ── 15m vs 1h comparison table ── */}
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 18, border: `1px solid ${COLORS.border2}`, marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.text, marginBottom: 14 }}>⚖️ 15m vs 1h — Which Horizon Performs Better?</div>
          <div style={{ display: "grid", gridTemplateColumns: "160px 1fr 1fr 80px", gap: 0 }}>
            {[["", "15-Minute", "1-Hour", "Winner"],
              ["Accuracy",  `${pct(m15.Accuracy)}`, `${pct(m1h.Accuracy)}`,  (m15.Accuracy||0) > (m1h.Accuracy||0) ? "15m ✓" : "1h ✓"],
              ["F1 Score",  `${num(m15.F1)}`,        `${num(m1h.F1)}`,         (m15.F1||0) > (m1h.F1||0) ? "15m ✓" : "1h ✓"],
              ["ROC AUC",   `${num(m15.ROC_AUC)}`,   `${num(m1h.ROC_AUC)}`,   (m15.ROC_AUC||0) > (m1h.ROC_AUC||0) ? "15m ✓" : "1h ✓"],
              ["Precision", `${pct(m15.Precision)}`, `${pct(m1h.Precision)}`, (m15.Precision||0) > (m1h.Precision||0) ? "15m ✓" : "1h ✓"],
              ["Recall",    `${pct(m15.Recall)}`,    `${pct(m1h.Recall)}`,    (m15.Recall||0) > (m1h.Recall||0) ? "15m ✓" : "1h ✓"],
              ["Dir Acc",   `${pct(m15.DirAcc)}`,    `${pct(m1h.DirAcc)}`,   (m15.DirAcc||0) > (m1h.DirAcc||0) ? "15m ✓" : "1h ✓"],
            ].map(([lbl, v15, v1h, winner], i) => (
              <Fragment key={lbl}>
                {[lbl, v15, v1h, winner].map((cell, j) => (
                  <div key={j} style={{
                    padding: "9px 12px",
                    borderBottom: i < 6 ? `1px solid ${COLORS.border}` : "none",
                    fontSize: i === 0 ? 9 : 12,
                    fontFamily: i === 0 ? "inherit" : "monospace",
                    fontWeight: i === 0 ? 400 : j === 3 ? 700 : 400,
                    color: i === 0 ? COLORS.muted
                         : j === 0 ? COLORS.muted
                         : j === 3 ? (winner?.startsWith("15") ? COLORS.accent : COLORS.blue)
                         : COLORS.text,
                    background: i === 0 ? COLORS.bg : "transparent",
                    letterSpacing: i === 0 ? 1 : 0,
                    textTransform: i === 0 ? "uppercase" : "none",
                  }}>{cell}</div>
                ))}
              </Fragment>
            ))}
          </div>
        </div>

        {/* ── Key Findings ── */}
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 18, border: `1px solid ${COLORS.border2}` }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.accent, marginBottom: 14 }}>💡 Key Findings</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {[
              [COLORS.gold,   "⚠️", "Accuracy is inflated by class imbalance",    "Most news has no BTC impact. A model that always says 'no impact' would score higher accuracy than ours. Always read F1 and AUC instead."],
              [COLORS.green,  "✅", "Genuine discriminative ability confirmed",   `AUC ≈ ${((m15.ROC_AUC||0 + m1h.ROC_AUC||0)/2).toFixed(2)} — the model ranks impactful news above non-impactful better than random, proving it has learned real signal.`],
              [COLORS.accent, "🎯", "15m horizon is more trustworthy",            `Higher precision (${pct(m15.Precision)}) and better-calibrated threshold. Fewer false alarms than 1h. Better for automated trading signals.`],
              [COLORS.blue,   "📡", "1h model is a sensitive scanner",            `Recall = ${pct(m1h.Recall)} — catches ${Math.round((m1h.Recall||0)*100)}% of all real impact events, but at the cost of many false positives (${(m1h.CM||[[0,0]])[0][1]} FP). Better for awareness than precision signals.`],
              [COLORS.red,    "❗", "Low precision is the main limitation",       `Only 1 in ${Math.round(1/(m15.Precision||0.25))} predictions for 15m are true positives. The model needs more training data, especially for rare event types.`],
              [COLORS.purple, "🧭", "Direction accuracy is barely above chance", `${pct(mdir.Acc)} accuracy on price direction (up/down). The model knows IF there will be movement better than WHERE it will go. Direction is the harder problem.`],
            ].map(([c, icon, title, body]) => (
              <div key={title} style={{ background: COLORS.bg, borderRadius: 8, padding: "12px 14px", borderLeft: `3px solid ${c}` }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: c, marginBottom: 6 }}>{icon} {title}</div>
                <div style={{ fontSize: 11, color: COLORS.muted, lineHeight: 1.6 }}>{body}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Direction accuracy ── */}
        {mdir.Acc && (
          <div style={{ display: "flex", gap: 12, marginTop: 14 }}>
            <div style={{ background: COLORS.panel, borderRadius: 10, padding: "12px 20px", border: `1px solid ${COLORS.border2}` }}>
              <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>DIRECTION ACCURACY</div>
              <div style={{ fontSize: 20, fontFamily: "monospace", fontWeight: 700, color: perfClr(mdir.Acc, 0.53) }}>{pct(mdir.Acc)}</div>
              <div style={{ fontSize: 10, color: COLORS.muted, marginTop: 2 }}>Barely above 50% coin-flip</div>
            </div>
            <div style={{ background: COLORS.panel, borderRadius: 10, padding: "12px 20px", border: `1px solid ${COLORS.border2}` }}>
              <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>DIRECTION F1</div>
              <div style={{ fontSize: 20, fontFamily: "monospace", fontWeight: 700, color: perfClr(mdir.F1, 0.50) }}>{num(mdir.F1)}</div>
            </div>
            <div style={{ background: COLORS.panel, borderRadius: 10, padding: "12px 20px", border: `1px solid ${COLORS.border2}`, flex: 1 }}>
              <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 4 }}>TRAINED THRESHOLDS (optimised for min precision ≥ 0.25)</div>
              <div style={{ fontSize: 11, color: COLORS.text, fontFamily: "monospace" }}>
                15m: {arch.threshold_15m} &nbsp;·&nbsp; 1h: {arch.threshold_1h}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Per-Category Performance ── */}
      {catRes.filter(c => c.test_count > 0).length > 0 && (() => {
        const cats = catRes.filter(c => c.test_count > 0);
        const ICONS = {
          macro_economic: "🌍", etf: "📊", institutional: "🏦", partnership: "🤝",
          market_analysis: "📈", defi: "⛓️", regulatory: "⚖️", exchange: "🔄",
          mining: "⛏️", hack: "🔓", unknown: "📌",
        };
        const maxTest = Math.max(...cats.map(c => c.test_count), 1);
        const maxTrain = Math.max(...cats.map(c => c.train_count), 1);

        // Sort by 15m accuracy for ranking
        const ranked = [...cats].sort((a, b) => (b["15m"].acc || 0) - (a["15m"].acc || 0));

        return (
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text, marginBottom: 6 }}>
              🏆 Performance by News Category
            </div>
            <div style={{ fontSize: 11, color: COLORS.muted, marginBottom: 16 }}>
              Per-category accuracy on the 732-item test set (ews_ev.csv) — ranked by 15m accuracy
            </div>

            {/* Ranked accuracy bars */}
            <div style={{ background: COLORS.panel, borderRadius: 12, padding: 20, border: `1px solid ${COLORS.border2}`, marginBottom: 14 }}>
              <div style={{ display: "grid", gridTemplateColumns: "140px 1fr 80px 1fr 80px 120px", gap: "0 12px", alignItems: "center", marginBottom: 10 }}>
                <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>CATEGORY</div>
                <div style={{ fontSize: 9, color: COLORS.accent, letterSpacing: 1 }}>15m ACCURACY</div>
                <div style={{ fontSize: 9, color: COLORS.accent, letterSpacing: 1, textAlign: "right" }}>15m ACC</div>
                <div style={{ fontSize: 9, color: COLORS.blue, letterSpacing: 1 }}>1h ACCURACY</div>
                <div style={{ fontSize: 9, color: COLORS.blue, letterSpacing: 1, textAlign: "right" }}>1h ACC</div>
                <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, textAlign: "right" }}>TEST / TRAIN</div>
              </div>
              {ranked.map((cat, i) => {
                const a15 = cat["15m"].acc || 0;
                const a1h = cat["1h"].acc  || 0;
                const c15 = a15 >= 0.70 ? COLORS.green : a15 >= 0.60 ? COLORS.gold : COLORS.red;
                const c1h = a1h >= 0.70 ? COLORS.green : a1h >= 0.60 ? COLORS.gold : COLORS.red;
                const badge = i === 0 ? { label: "BEST", color: COLORS.green }
                            : i === ranked.length - 1 ? { label: "WORST", color: COLORS.red }
                            : null;
                return (
                  <div key={cat.news_type} style={{
                    display: "grid", gridTemplateColumns: "140px 1fr 80px 1fr 80px 120px",
                    gap: "0 12px", alignItems: "center", padding: "8px 0",
                    borderTop: `1px solid ${COLORS.border}`,
                  }}>
                    {/* name */}
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ fontSize: 14 }}>{ICONS[cat.news_type] || "📌"}</span>
                      <div>
                        <span style={{ fontSize: 10, color: COLORS.text, textTransform: "capitalize" }}>
                          {cat.news_type.replace(/_/g, " ")}
                        </span>
                        {badge && (
                          <span style={{ marginLeft: 5, fontSize: 8, fontWeight: 700, color: badge.color,
                            background: `${badge.color}20`, borderRadius: 3, padding: "1px 4px" }}>
                            {badge.label}
                          </span>
                        )}
                      </div>
                    </div>
                    {/* 15m bar */}
                    <div style={{ height: 8, background: COLORS.border2, borderRadius: 4 }}>
                      <div style={{ height: "100%", width: `${a15 * 100}%`, background: c15, borderRadius: 4 }} />
                    </div>
                    <div style={{ fontSize: 12, fontFamily: "monospace", fontWeight: 700, color: c15, textAlign: "right" }}>
                      {Math.round(a15 * 100)}%
                    </div>
                    {/* 1h bar */}
                    <div style={{ height: 8, background: COLORS.border2, borderRadius: 4 }}>
                      <div style={{ height: "100%", width: `${a1h * 100}%`, background: c1h, borderRadius: 4 }} />
                    </div>
                    <div style={{ fontSize: 12, fontFamily: "monospace", fontWeight: 700, color: c1h, textAlign: "right" }}>
                      {Math.round(a1h * 100)}%
                    </div>
                    {/* counts */}
                    <div style={{ fontSize: 10, color: COLORS.muted, textAlign: "right", fontFamily: "monospace" }}>
                      {cat.test_count} / {cat.train_count.toLocaleString()}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Per-category detail cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12 }}>
              {ranked.map((cat) => {
                const r15 = cat["15m"];
                const r1h = cat["1h"];
                const a15 = r15.acc || 0;
                const clr = a15 >= 0.70 ? COLORS.green : a15 >= 0.60 ? COLORS.gold : COLORS.red;
                return (
                  <div key={cat.news_type} style={{ background: COLORS.panel, borderRadius: 10, padding: 14,
                    border: `1px solid ${COLORS.border2}`, borderTop: `3px solid ${clr}` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontSize: 18 }}>{ICONS[cat.news_type] || "📌"}</span>
                        <div>
                          <div style={{ fontSize: 12, fontWeight: 700, color: COLORS.text, textTransform: "capitalize" }}>
                            {cat.news_type.replace(/_/g, " ")}
                          </div>
                          <div style={{ fontSize: 10, color: COLORS.muted }}>
                            {cat.test_count} test samples · {cat.train_count.toLocaleString()} training
                          </div>
                        </div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 9, color: COLORS.muted }}>15m ACC</div>
                        <div style={{ fontSize: 18, fontFamily: "monospace", fontWeight: 700, color: clr }}>
                          {Math.round(a15 * 100)}%
                        </div>
                      </div>
                    </div>
                    {/* 15m vs 1h confusion mini */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      {[["⏱ 15m", r15, COLORS.accent], ["🕐 1h", r1h, COLORS.blue]].map(([lbl, r, c]) => (
                        <div key={lbl} style={{ background: COLORS.bg, borderRadius: 6, padding: "8px 10px" }}>
                          <div style={{ fontSize: 9, color: c, marginBottom: 6, fontWeight: 700 }}>{lbl}</div>
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 3, fontSize: 9 }}>
                            {[
                              ["✅ TP", r.tp, COLORS.accent],
                              ["✔️ TN", r.tn, COLORS.green],
                              ["⚠️ FN", r.fn, COLORS.gold],
                              ["❌ FP", r.fp, COLORS.red],
                            ].map(([tag, n, tc]) => (
                              <div key={tag} style={{ display: "flex", justifyContent: "space-between",
                                background: `${tc}10`, borderRadius: 4, padding: "3px 6px" }}>
                                <span style={{ color: COLORS.muted }}>{tag}</span>
                                <span style={{ fontFamily: "monospace", color: tc, fontWeight: 700 }}>{n ?? 0}</span>
                              </div>
                            ))}
                          </div>
                          {r.prec != null && (
                            <div style={{ marginTop: 6, fontSize: 9, color: COLORS.muted }}>
                              P={Math.round(r.prec*100)}% · R={Math.round((r.rec||0)*100)}%
                              {r.f1 != null && ` · F1=${r.f1.toFixed(2)}`}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })()}

      {/* ── Cache (Deployment) Data ── */}
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text, marginBottom: 14 }}>
          🗃️ Cache / Deployment Data
          <span style={{ fontSize: 11, color: COLORS.muted, fontWeight: 400, marginLeft: 10 }}>news_cache.json</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 16 }}>
          <Kpi label="Total Items"   value={(ca.total || 0).toLocaleString()} color={COLORS.text} />
          <Kpi label="High Score ≥67%" value={(ca.score_high || 0).toLocaleString()} sub={`${ca.score_high_pct}% of cache`} color={COLORS.accent} />
          <Kpi label="Medium 50–67%" value={(ca.score_medium || 0).toLocaleString()} sub={`${ca.score_medium_pct}% of cache`} color={COLORS.gold} />
          <Kpi label="Date Range"    value={ca.date_min || "—"} sub={`to ${ca.date_max || "—"}`} color={COLORS.blue} />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>

          {/* Score + predictions */}
          <div style={{ background: COLORS.panel, borderRadius: 10, padding: 16, border: `1px solid ${COLORS.border2}` }}>
            <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 12 }}>PREDICTIONS IN CACHE</div>
            <MetRow label="Pred 15m = Impact"  value={`${(ca.pred_15m_positive || 0).toLocaleString()} (${ca.pred_15m_positive_pct}%)`} color={COLORS.accent} />
            <MetRow label="Pred 1h = Impact"   value={`${(ca.pred_1h_positive  || 0).toLocaleString()} (${ca.pred_1h_positive_pct}%)`}  color={COLORS.blue} />
            <MetRow label="With BTC price data" value={`${(ca.with_btc_data || 0).toLocaleString()} (${ca.with_btc_pct}%)`} color={COLORS.green} />
            <div style={{ fontSize: 9, color: COLORS.border2, marginTop: 10, lineHeight: 1.5 }}>
              Items without BTC data: news received before price was confirmed (btc_change = 0)
            </div>
          </div>

          {/* Sentiment */}
          <div style={{ background: COLORS.panel, borderRadius: 10, padding: 16, border: `1px solid ${COLORS.border2}` }}>
            <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 12 }}>SENTIMENT IN CACHE</div>
            {[
              ["Positive (Bullish)", ca.sentiment_counts?.positive || 0, COLORS.green],
              ["Neutral",            ca.sentiment_counts?.neutral  || 0, COLORS.gold],
              ["Negative (Bearish)", ca.sentiment_counts?.negative || 0, COLORS.red],
            ].map(([lbl, n, c]) => (
              <div key={lbl} style={{ marginBottom: 9 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 11, color: COLORS.muted }}>{lbl}</span>
                  <span style={{ fontSize: 11, fontFamily: "monospace", color: c }}>
                    {n.toLocaleString()} ({Math.round(n / caSentTotal * 100)}%)
                  </span>
                </div>
                <div style={{ height: 4, background: COLORS.border2, borderRadius: 2 }}>
                  <div style={{ height: "100%", width: `${n / caSentTotal * 100}%`, background: c, borderRadius: 2 }} />
                </div>
              </div>
            ))}
            <div style={{ marginTop: 8, fontSize: 10, color: COLORS.muted, letterSpacing: 1 }}>SIGNALS</div>
            <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
              {[["BUY", COLORS.green], ["SELL", COLORS.red], ["NEUTRAL", COLORS.muted]].map(([s, c]) => (
                <div key={s} style={{ flex: 1, background: `${c}18`, borderRadius: 6, padding: "6px 8px", textAlign: "center", border: `1px solid ${c}30` }}>
                  <div style={{ fontSize: 9, color: c }}>{s}</div>
                  <div style={{ fontSize: 12, fontFamily: "monospace", fontWeight: 700, color: c }}>{(ca.signal_counts?.[s] || 0).toLocaleString()}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Top channels in cache */}
          <div style={{ background: COLORS.panel, borderRadius: 10, padding: 16, border: `1px solid ${COLORS.border2}` }}>
            <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 12 }}>TOP CHANNELS IN CACHE</div>
            {Object.entries(ca.channel_counts || {}).slice(0, 6).map(([ch, n], i) => {
              const max = Math.max(...Object.values(ca.channel_counts || {}));
              return (
                <div key={ch} style={{ marginBottom: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                    <span style={{ fontSize: 10, color: COLORS.text }}>{ch}</span>
                    <span style={{ fontSize: 10, fontFamily: "monospace", color: COLORS.accent }}>{n.toLocaleString()}</span>
                  </div>
                  <div style={{ height: 3, background: COLORS.border2, borderRadius: 2 }}>
                    <div style={{ height: "100%", width: `${n / max * 100}%`, background: [COLORS.accent, COLORS.blue, COLORS.gold, COLORS.green, COLORS.purple, COLORS.red][i % 6], borderRadius: 2 }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

    </div>
  );
}

function ResearchPage() {
  const framework = [
    { icon: "🎯", title: "Calibration", desc: "Does the model's confidence match its actual correctness? Overconfident wrong predictions cause real financial losses." },
    { icon: "🔄", title: "Consistency", desc: "Do similar linguistic inputs yield similar outputs? Unstable predictions across paraphrased headlines signal unreliable models." },
    { icon: "🌐", title: "Cross-Domain", desc: "How does behavior change between macro, crypto, and institutional news? Domain shift is a major source of silent failure." },
    { icon: "🔍", title: "Failure Analysis", desc: "Where exactly does the model make mistakes and why? Identifying failure patterns guides targeted improvement." },
  ];

  const domains = [
    { icon: "₿", label: "Cryptocurrency News", color: COLORS.gold, sources: "CoinDesk · CoinTelegraph", char: "Highly structured, narrow focus — model performs best here." },
    { icon: "🏦", label: "Institutional News", color: COLORS.blue, sources: "Banks · ETFs · Hedge Funds", char: "Formal, nuanced — moderate difficulty for text classifiers." },
    { icon: "📈", label: "Macroeconomic News", color: COLORS.purple, sources: "CPI Reports · Inflation · Interest Rates", char: "Ambiguous, broad economic impact — performance expected to drop." },
  ];

  const pipeline = [
    { step: "01", title: "Data Collection", desc: "Crypto, macro, and institutional financial news from multiple sources" },
    { step: "02", title: "Neural Representation", desc: "Pretrained Sentence Transformers convert headlines to dense semantic vectors" },
    { step: "03", title: "Neural Network Model", desc: "MLP or Logistic Regression — 3-class: Positive / Neutral / Negative impact" },
    { step: "04", title: "Evaluation Engine", desc: "Calibration · Consistency · Cross-Domain · Failure Analysis" },
  ];

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: 28, display: "flex", flexDirection: "column", gap: 24 }}>

      {/* Header */}
      <div style={{ borderBottom: `1px solid ${COLORS.border}`, paddingBottom: 20 }}>
        <div style={{ fontSize: 9, letterSpacing: 2, color: COLORS.accent, fontFamily: "monospace", marginBottom: 8, textTransform: "uppercase" }}>
          Bahçeşehir University · Dept. of Artificial Intelligence Engineering
        </div>
        <div style={{ fontSize: 20, fontWeight: 700, color: COLORS.text, lineHeight: 1.4, maxWidth: 700 }}>
          Trustworthiness and Failure Analysis of Neural Network-Based Financial News Impact Prediction Systems
        </div>
        <div style={{ fontSize: 12, color: COLORS.muted, marginTop: 8 }}>
          Haniye Shakibayi Senobari · <span style={{ color: COLORS.accent, fontFamily: "monospace" }}>haniye.senobari@bahcesehir.edu.tr</span>
        </div>
      </div>

      {/* Abstract */}
      <div style={{ background: COLORS.panel, borderRadius: 12, padding: 20, border: `1px solid ${COLORS.border2}`, borderLeft: `4px solid ${COLORS.accent}` }}>
        <div style={{ fontSize: 10, letterSpacing: 1.5, color: COLORS.accent, fontFamily: "monospace", marginBottom: 10, textTransform: "uppercase" }}>Abstract</div>
        <div style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.75 }}>
          Neural networks are widely used for financial news analysis and market impact prediction, but these models behave like black boxes — unreliable, prone to overconfidence, and sensitive to input domain. This project evaluates <strong style={{ color: COLORS.accent }}>how trustworthy these systems actually are</strong> by testing across crypto, macroeconomic, and institutional news. Beyond accuracy, we examine calibration, consistency, and failure cases to understand when and why these models fail on real financial text classification tasks.
        </div>
        <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
          {["Neural Networks", "Trustworthy AI", "Financial News", "Explainable AI", "Calibration", "Failure Analysis"].map(tag => (
            <span key={tag} style={{ fontSize: 10, padding: "3px 9px", borderRadius: 20, background: `${COLORS.accent}15`, color: COLORS.accent, fontFamily: "monospace", border: `1px solid ${COLORS.accent}30` }}>{tag}</span>
          ))}
        </div>
      </div>

      {/* Problem Statement */}
      <div style={{ background: COLORS.panel, borderRadius: 12, padding: 20, border: `1px solid ${COLORS.border2}` }}>
        <div style={{ fontSize: 10, letterSpacing: 1.5, color: COLORS.gold, fontFamily: "monospace", marginBottom: 10, textTransform: "uppercase" }}>Research Question</div>
        <div style={{ fontSize: 15, color: COLORS.text, fontStyle: "italic", lineHeight: 1.6, borderLeft: `3px solid ${COLORS.gold}`, paddingLeft: 16 }}>
          "How trustworthy are neural network-based financial news prediction models when tested across different financial domains, and under what conditions do these models fail?"
        </div>
      </div>

      {/* Glass Box Framework */}
      <div>
        <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.text, marginBottom: 6 }}>The Glass Box Evaluation Framework</div>
        <div style={{ fontSize: 11, color: COLORS.muted, marginBottom: 14 }}>Moving beyond accuracy requires a four-dimensional evaluation approach</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
          {framework.map(({ icon, title, desc }) => (
            <div key={title} style={{ background: COLORS.panel, borderRadius: 10, padding: 16, border: `1px solid ${COLORS.border2}` }}>
              <div style={{ fontSize: 22, marginBottom: 8 }}>{icon}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.accent, marginBottom: 6 }}>{title}</div>
              <div style={{ fontSize: 11, color: COLORS.muted, lineHeight: 1.6 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Pipeline */}
      <div>
        <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.text, marginBottom: 14 }}>End-to-End Methodology</div>
        <div style={{ display: "flex", gap: 0, alignItems: "stretch" }}>
          {pipeline.map(({ step, title, desc }, i) => (
            <div key={step} style={{ display: "flex", alignItems: "stretch", flex: 1 }}>
              <div style={{ flex: 1, background: COLORS.panel, borderRadius: i === 0 ? "10px 0 0 10px" : i === pipeline.length - 1 ? "0 10px 10px 0" : 0, padding: "16px 14px", border: `1px solid ${COLORS.border2}`, borderRight: i < pipeline.length - 1 ? "none" : `1px solid ${COLORS.border2}` }}>
                <div style={{ fontSize: 11, fontFamily: "monospace", color: COLORS.accent, fontWeight: 700, marginBottom: 6 }}>Stage {step}</div>
                <div style={{ fontSize: 12, fontWeight: 700, color: COLORS.text, marginBottom: 6 }}>{title}</div>
                <div style={{ fontSize: 11, color: COLORS.muted, lineHeight: 1.5 }}>{desc}</div>
              </div>
              {i < pipeline.length - 1 && (
                <div style={{ display: "flex", alignItems: "center", background: COLORS.panel, padding: "0 4px", border: `1px solid ${COLORS.border2}`, borderLeft: "none", borderRight: "none" }}>
                  <span style={{ color: COLORS.accent, fontSize: 14 }}>→</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Data Domains */}
      <div>
        <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.text, marginBottom: 14 }}>Testing Across Distinct Financial Data Ecosystems</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
          {domains.map(({ icon, label, color, sources, char }) => (
            <div key={label} style={{ background: COLORS.panel, borderRadius: 10, padding: 18, border: `1px solid ${COLORS.border2}`, borderTop: `3px solid ${color}` }}>
              <div style={{ fontSize: 26, marginBottom: 10 }}>{icon}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color, marginBottom: 6 }}>{label}</div>
              <div style={{ fontSize: 10, color: COLORS.muted, fontFamily: "monospace", marginBottom: 10 }}>{sources}</div>
              <div style={{ fontSize: 11, color: COLORS.text, lineHeight: 1.6 }}>{char}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Expected Results + Conclusion */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 20, border: `1px solid ${COLORS.border2}` }}>
          <div style={{ fontSize: 10, letterSpacing: 1.5, color: COLORS.blue, fontFamily: "monospace", marginBottom: 12, textTransform: "uppercase" }}>Expected Results</div>
          {[
            ["Crypto news", "Best performance — highly structured domain"],
            ["Macro news", "Performance drop — ambiguous language"],
            ["Social media", "Worst performance — noisy, informal text"],
            ["Overconfidence", "Detected in specific domains"],
            ["Misclassifications", "Systematic patterns to be identified"],
          ].map(([label, val]) => (
            <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: `1px solid ${COLORS.border}`, fontSize: 11 }}>
              <span style={{ color: COLORS.muted }}>{label}</span>
              <span style={{ color: COLORS.text, textAlign: "right", maxWidth: 200 }}>{val}</span>
            </div>
          ))}
        </div>
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 20, border: `1px solid ${COLORS.border2}`, display: "flex", flexDirection: "column" }}>
          <div style={{ fontSize: 10, letterSpacing: 1.5, color: COLORS.green, fontFamily: "monospace", marginBottom: 12, textTransform: "uppercase" }}>Conclusion</div>
          <div style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.75, flex: 1 }}>
            By mapping where neural networks succeed and fail across diverse financial texts, this study provides the foundational analysis needed to build reliable, transparent AI systems for finance.
          </div>
          <div style={{ marginTop: 16, padding: "12px 14px", background: `${COLORS.green}10`, borderRadius: 8, border: `1px solid ${COLORS.green}30` }}>
            <div style={{ fontSize: 10, color: COLORS.green, fontFamily: "monospace", letterSpacing: 1 }}>ADVANCING EXPLAINABLE AI</div>
            <div style={{ fontSize: 11, color: COLORS.muted, marginTop: 4, lineHeight: 1.5 }}>For high-stakes financial environments where overconfident wrong predictions cause real losses.</div>
          </div>
        </div>
      </div>

      {/* References */}
      <div style={{ background: COLORS.panel, borderRadius: 12, padding: 18, border: `1px solid ${COLORS.border2}` }}>
        <div style={{ fontSize: 10, letterSpacing: 1.5, color: COLORS.muted, fontFamily: "monospace", marginBottom: 12, textTransform: "uppercase" }}>References</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {[
            "Guo et al. (2017). On calibration of modern neural networks. ICML.",
            "Ribeiro, Singh & Guestrin (2016). “Why should I trust you?” Explaining classifier predictions. KDD.",
            "Lewis et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.",
            "Devlin et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL.",
          ].map((ref, i) => (
            <div key={i} style={{ fontSize: 11, color: COLORS.muted, lineHeight: 1.5, paddingLeft: 12, borderLeft: `2px solid ${COLORS.border2}` }}>
              [{i + 1}] {ref}
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}

// ── Admin check — show hidden pages only when ?admin=crypto2026 in URL ──
const ADMIN_KEY = "crypto2026";
function useIsAdmin() {
  return new URLSearchParams(window.location.search).get("admin") === ADMIN_KEY;
}

// ── Main Dashboard ─────────────────────────────────────────────────
export default function CryptoDashboard() {
  const isAdmin = useIsAdmin();
  const [activeNav, setActiveNav]         = useState("News & Sentiment");
  const [newsTab, setNewsTab]             = useState("important");
  const [selectedPair, setSelectedPair]   = useState("BINANCE:BTCUSDT");
  const [selectedSymbol, setSelectedSymbol] = useState("BTCUSDT");
  const [chartInterval, setChartInterval] = useState("15m");
  const [time, setTime]                   = useState(new Date());
  const [allNews, setAllNews]             = useState([]);
  const [hotSignals, setHotSignals]       = useState([]);
  const [selectedNews, setSelectedNews]   = useState(null);

  // Calendar state
  const [calendarDate, setCalendarDate]   = useState(null);  // null = today
  const [dbDatesApi, setDbDatesApi]       = useState([]);
  const [dateNews, setDateNews]           = useState([]);
  const [dateLoading, setDateLoading]     = useState(false);

  // Clock
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const sortByTime = (arr) =>
    [...arr].sort((a, b) => (b.published_ts ?? b.received_at ?? 0) - (a.published_ts ?? a.received_at ?? 0));

  // Load history on mount
  useEffect(() => {
    fetch(`${API_BASE}/news/all`)
      .then(r => r.json()).then(data => setAllNews(sortByTime(data.map(clientNormalize).filter(n => {
        const score = Math.abs(n.model_score || 0);
        const conf  = (n.confidence || 0) / 100;
        const hasML = !!n.has_model_score;
        // High-tier (≥67%): always show
        // Medium: ML model score trusted directly; BTC-proxy score needs conf ≥60%
        return score >= 0.67 || (score >= 0.50 && (hasML || conf >= 0.60));
      })))).catch(() => {});
    fetch(`${API_BASE}/news/hot`)
      .then(r => r.json()).then(data => setHotSignals(sortByTime(data.map(clientNormalize)))).catch(() => {});
    fetch(`${API_BASE}/news/dates`)
      .then(r => r.json()).then(data => setDbDatesApi(Array.isArray(data) ? data : [])).catch(() => {});
  }, []);

  // Fetch news for selected date
  useEffect(() => {
    if (!calendarDate) { setDateNews([]); return; }

    setDateLoading(true);
    const start = new Date(calendarDate); start.setHours(0, 0, 0, 0);
    const end   = new Date(calendarDate); end.setHours(23, 59, 59, 999);
    fetch(`${API_BASE}/news/by-date?start=${Math.floor(start.getTime()/1000)}&end=${Math.floor(end.getTime()/1000)}`)
      .then(r => r.json())
      .then(data => {
        const filtered = data.map(clientNormalize).filter(n => {
          const score = Math.abs(n.model_score || 0);
          const conf  = (n.confidence || 0) / 100;
          const hasML = !!n.has_model_score;
          const title = (n.title || "").trim();
          if (title.length < 20) return false;
          return score >= 0.67 || (score >= 0.50 && (hasML || conf >= 0.60));
        });
        setDateNews(sortByTime(filtered));
        setDateLoading(false);
      })
      .catch(() => setDateLoading(false));
  }, [calendarDate]);

  // Live WebSocket feeds
  const allConnected = useWebSocket("/ws/all", (item) => {
    const norm  = clientNormalize(item);
    const score = Math.abs(norm.model_score || 0);
    const conf  = (norm.confidence || 0) / 100;
    const hasML = !!norm.has_model_score;
    if (!(score >= 0.67 || (score >= 0.50 && (hasML || conf >= 0.60)))) return;
    setAllNews(prev => sortByTime([norm, ...prev]).slice(0, 200));
  });
  const hotConnected = useWebSocket("/ws/hot", (item) => {
    setHotSignals(prev => sortByTime([clientNormalize(item), ...prev]).slice(0, 50));
  });

  // Merge API dates + dates derived from locally loaded allNews
  const dbDates = useMemo(() => {
    const localDates = allNews.map(n => {
      const d = newsDate(n);
      return d ? dateKey(d) : null;
    }).filter(Boolean);
    return [...new Set([...dbDatesApi, ...localDates])].sort();
  }, [dbDatesApi, allNews]);

  // ── UTC date helpers (server stores timestamps in UTC) ────────────────────
  function utcDateKey(ts) {
    // ts = unix seconds → "YYYY-MM-DD" in UTC (matches how DB/cache stores dates)
    const d = new Date(ts * 1000);
    return `${d.getUTCFullYear()}-${String(d.getUTCMonth()+1).padStart(2,"0")}-${String(d.getUTCDate()).padStart(2,"0")}`;
  }
  // Local date key — used for "today" display so it matches user's clock
  function localDateKey(ts) {
    const d = new Date(ts * 1000);
    return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")}`;
  }
  const todayUTC   = utcDateKey(Date.now() / 1000);   // today in UTC (for calendar)
  const todayLocal = localDateKey(Date.now() / 1000);  // today in local tz (for "Today" label)

  // ── newsForDate: no date = most recent day in feed, date selected = DB fetch ──
  const isViewingToday = !calendarDate || dateKey(calendarDate) === dateKey(new Date());

  const mostRecentDayNews = useMemo(() => {
    if (!allNews.length) return [];
    // deduplicate by id/published_ts
    const seen = new Set();
    const unique = allNews.filter(n => {
      const key = n.id || n.published_ts;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    // find the most recent UTC date (use UTC so timezone doesn't shift items to wrong day)
    const latestTs  = Math.max(...unique.map(n => n.published_ts || n.received_at || 0));
    const latestKey = utcDateKey(latestTs);
    // filter to that day, sort newest first
    return unique
      .filter(n => { const ts = n.published_ts || n.received_at; return ts ? utcDateKey(ts) === latestKey : false; })
      .sort((a, b) => (b.published_ts || b.received_at || 0) - (a.published_ts || a.received_at || 0));
  }, [allNews]);

  const newsForDate = (filter) => {
    // If a calendar date is explicitly selected → always use dateNews (API fetch).
    // Only fall back to mostRecentDayNews when NO date is selected (live feed mode).
    const base = calendarDate ? dateNews : mostRecentDayNews;
    return filter ? base.filter(filter) : base;
  };

  // Label for the most recent day shown in the news panel
  const mostRecentDateLabel = mostRecentDayNews.length > 0
    ? (() => {
        const ts = mostRecentDayNews[0].published_ts || mostRecentDayNews[0].received_at;
        if (!ts) return "Live";
        const d = new Date(ts * 1000);
        return d.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
      })()
    : "Live";

  // Show "Today" only when the most recent items are actually from today (local time)
  const mostRecentIsToday = mostRecentDayNews.length > 0 &&
    localDateKey(mostRecentDayNews[0].published_ts || mostRecentDayNews[0].received_at || 0) === todayLocal;

  // Tab filters — 3 tiers: Low / Medium / High
  // Thresholds use normalized 0–1 scores (set in main.py)
  const importantTabNews = newsForDate(n => Math.abs(n.model_score || 0) >= 0.50);                         // Medium 50–67%
  const hotTabNews       = newsForDate(n => Math.abs(n.model_score || 0) >= 0.67);                         // High  ≥67%

  const tabNews = newsTab === "important" ? importantTabNews : hotTabNews;

  const navItems = [
    { icon: "◫", label: "News & Sentiment" },
    { icon: "⚡", label: "HOT Signals" },
    { icon: "📄", label: "Analyze" },
    { icon: "📋", label: "Report" },
    ...(isAdmin ? [
      { icon: "🧠", label: "Model Analysis" },
      { icon: "📊", label: "Training Data" },
    ] : []),
  ];

  const dateLabel = calendarDate
    ? calendarDate.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })
    : (mostRecentIsToday ? "Today" : mostRecentDateLabel);

  return (
    <div style={{ display: "flex", height: "100vh", width: "100vw", background: COLORS.bg, color: COLORS.text, fontFamily: "'DM Sans', system-ui, sans-serif", overflow: "hidden" }}>

      {/* ── Sidebar ── */}
      <div style={{ width: 220, flexShrink: 0, background: COLORS.panel, borderRight: `1px solid ${COLORS.border}`, display: "flex", flexDirection: "column", height: "100%" }}>
        <div style={{ padding: "20px 16px", borderBottom: `1px solid ${COLORS.border}` }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text, letterSpacing: 0.5 }}>Crypto Sentiment Analyze</div>
          </div>
        </div>
        <div style={{ flex: 1, padding: "12px 8px", overflowY: "auto" }}>
          {navItems.map(item => (
            <NavItem key={item.label} icon={item.icon} label={item.label}
              active={activeNav === item.label} onClick={() => setActiveNav(item.label)} />
          ))}
        </div>
        <div style={{ padding: "14px 12px", borderTop: `1px solid ${COLORS.border}` }}>
          <SentimentGauge news={allNews} />
        </div>
      </div>

      {/* ── Main area ── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

          {/* ── News & Sentiment full view ── */}
          {activeNav === "News & Sentiment" && (
            <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

              {/* News list */}
              <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

                {/* Header + tabs */}
                <div style={{ padding: "12px 20px", borderBottom: `1px solid ${COLORS.border}`, background: COLORS.panel, flexShrink: 0 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                    <span style={{ fontSize: 13, fontWeight: 600 }}>
                      📰 News — <span style={{ color: COLORS.accent }}>{dateLabel}</span>
                      <span style={{ color: COLORS.muted, fontSize: 11, marginLeft: 8 }}>{tabNews.length} items</span>
                    </span>
                    <ConnectionDot connected={allConnected} />
                  </div>
                  {/* Tabs */}
                  <div style={{ display: "flex", gap: 6 }}>
                    {[
                      { id: "important", label: "Medium", count: importantTabNews.length },
                      { id: "hot",       label: "High",   count: hotTabNews.length },
                    ].map(t => (
                      <button key={t.id} onClick={() => setNewsTab(t.id)} style={{
                        padding: "4px 12px", borderRadius: 6, border: "none", cursor: "pointer", fontSize: 11,
                        background: newsTab === t.id ? COLORS.accent : COLORS.border2,
                        color: newsTab === t.id ? "#000" : COLORS.muted,
                        fontWeight: newsTab === t.id ? 700 : 400,
                      }}>
                        {t.label} <span style={{ fontSize: 10, opacity: 0.8 }}>({t.count})</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* News items */}
                <div style={{ flex: 1, overflowY: "auto", padding: "0 20px" }}>
                  {dateLoading ? (
                    <div style={{ padding: "40px 0", textAlign: "center", color: COLORS.muted, fontSize: 12 }}>Loading…</div>
                  ) : tabNews.length === 0 ? (
                    <div style={{ padding: "40px 0", textAlign: "center", color: COLORS.muted, fontSize: 12 }}>
                      {calendarDate ? `No significant news for ${dateLabel}` : (allConnected ? "No news yet today" : "Bot offline — start api.py + main.py")}
                    </div>
                  ) : (
                    tabNews.map((item, i) => <NewsCard key={item.id ?? i} item={item} onClick={() => setSelectedNews(item)} />)
                  )}
                </div>
              </div>

              {/* Calendar sidebar */}
              <div style={{ width: 248, flexShrink: 0, borderLeft: `1px solid ${COLORS.border}`, padding: 16, overflowY: "auto", background: COLORS.panel }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 10 }}>FILTER BY DATE</div>
                <CalendarPicker selected={calendarDate} onChange={setCalendarDate} dbDates={dbDates} />
              </div>
            </div>
          )}

          {/* ── HOT Signals view ── */}
          {activeNav === "HOT Signals" && (
            <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
              <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
                <div style={{ padding: "12px 20px", borderBottom: `1px solid ${COLORS.border}`, background: COLORS.panel, flexShrink: 0 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 13, fontWeight: 600 }}>
                      🔴 High Signals — <span style={{ color: COLORS.accent }}>{dateLabel}</span>
                      <span style={{ color: COLORS.muted, fontSize: 11, marginLeft: 8 }}>{hotTabNews.length} items</span>
                    </span>
                    <ConnectionDot connected={hotConnected} />
                  </div>
                </div>
                <div style={{ flex: 1, overflowY: "auto", padding: "10px 20px" }}>
                  {dateLoading ? (
                    <div style={{ padding: "40px 0", textAlign: "center", color: COLORS.muted, fontSize: 12 }}>Loading…</div>
                  ) : hotTabNews.length === 0 ? (
                    <div style={{ padding: "40px 0", textAlign: "center", color: COLORS.muted, fontSize: 12 }}>No high signals for {dateLabel}</div>
                  ) : (
                    hotTabNews.map((s, i) => <SignalCard key={s.id ?? i} item={s} />)
                  )}
                </div>
              </div>
              <div style={{ width: 248, flexShrink: 0, borderLeft: `1px solid ${COLORS.border}`, padding: 16, background: COLORS.panel }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 10 }}>FILTER BY DATE</div>
                <CalendarPicker selected={calendarDate} onChange={setCalendarDate} dbDates={dbDates} />
              </div>
            </div>
          )}

          {/* ── Dashboard (main) ── */}
          {activeNav === "Dashboard" && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
              {/* Chart header */}
              <div style={{ padding: "10px 16px", borderBottom: `1px solid ${COLORS.border}`, display: "flex", alignItems: "center", gap: 12, background: COLORS.panel, flexShrink: 0 }}>
                <div style={{ display: "flex", gap: 6 }}>
                  {["BTCUSDT","ETHUSDT","SOLUSDT"].map(s => (
                    <button key={s} onClick={() => { setSelectedSymbol(s); setSelectedPair(`BINANCE:${s}`); }} style={{
                      padding: "4px 10px", borderRadius: 6, fontSize: 11, cursor: "pointer", fontFamily: "monospace",
                      border: `1px solid ${selectedSymbol === s ? COLORS.accent : COLORS.border2}`,
                      background: selectedSymbol === s ? `${COLORS.accent}15` : "transparent",
                      color: selectedSymbol === s ? COLORS.accent : COLORS.muted,
                    }}>{s.replace("USDT","")}</button>
                  ))}
                </div>
                <div style={{ width: 1, height: 16, background: COLORS.border2 }} />
                <div style={{ display: "flex", gap: 4 }}>
                  {["1m","5m","15m","1h","4h","1d"].map(iv => (
                    <button key={iv} onClick={() => setChartInterval(iv)} style={{
                      padding: "3px 8px", borderRadius: 5, fontSize: 10, cursor: "pointer", fontFamily: "monospace",
                      border: "none",
                      background: chartInterval === iv ? COLORS.accent : "transparent",
                      color: chartInterval === iv ? "#000" : COLORS.muted,
                      fontWeight: chartInterval === iv ? 700 : 400,
                    }}>{iv}</button>
                  ))}
                </div>
              </div>

              {/* Binance Real-Time Chart */}
              <div style={{ height: 360, flexShrink: 0, borderBottom: `1px solid ${COLORS.border}` }}>
                <BinanceChart symbol={selectedPair} interval={chartInterval} news={allNews} />
              </div>

              {/* Bottom panels */}
              <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 1fr", overflow: "hidden" }}>

                {/* News feed */}
                <div style={{ borderRight: `1px solid ${COLORS.border}`, display: "flex", flexDirection: "column", overflow: "hidden" }}>
                  <div style={{ padding: "10px 16px", borderBottom: `1px solid ${COLORS.border}`, display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
                    <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: 0.5 }}>⚡ AI News Impact (Live)</span>
                    <ConnectionDot connected={allConnected} />
                  </div>
                  {(() => {
                    // Show TODAY's High + Medium news — use UTC date to avoid
                    // timezone shift showing late-UTC items as "today" in UTC+N zones
                    const todayNews  = allNews.filter(n => {
                      const ts = n.published_ts || n.received_at;
                      return ts ? localDateKey(ts) === todayLocal : false;
                    });
                    // Deduplicate by title
                    const seen = new Set();
                    const uniqueNews = todayNews.filter(n => {
                      if (seen.has(n.title)) return false;
                      seen.add(n.title);
                      return true;
                    });
                    return (
                      <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
                        <div style={{ overflowY: "auto", padding: "0 16px" }}>
                          {uniqueNews.length === 0 ? (
                            <div style={{ padding: "20px 0", textAlign: "center", color: COLORS.muted, fontSize: 12 }}>
                              No news today
                            </div>
                          ) : (
                            uniqueNews.slice(0, 10).map((item, i) => <NewsCard key={item.id ?? i} item={item} onClick={() => setSelectedNews(item)} />)
                          )}
                        </div>
                        {uniqueNews.length > 10 && (
                          <div onClick={() => setActiveNav("News & Sentiment")} style={{ flexShrink: 0, padding: "10px 0", textAlign: "center", color: COLORS.accent, fontSize: 11, cursor: "pointer", fontFamily: "monospace", borderTop: `1px solid ${COLORS.border}` }}>
                            +{uniqueNews.length - 10} more — view all →
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>

                {/* ── AI Explanation Panel (replaces Top Signals) ── */}
                <ExplainPanel selectedNews={selectedNews} onClose={() => setSelectedNews(null)} />

              </div>
            </div>
          )}

          {/* ── Analyze view ── */}
          {activeNav === "Analyze" && (
            <div style={{ flex: 1, overflowY: "auto" }}>
              <ModelAnalysis news={allNews} />
              <div style={{ borderTop: `2px solid ${COLORS.border2}`, margin: "0 24px" }} />
              <ChannelAnalysisPage news={allNews} />
            </div>
          )}

          {/* ── Report view ── */}
          {activeNav === "Report" && (
            <div style={{ flex: 1, overflowY: "auto" }}>
              <ReportPage />
            </div>
          )}

          {/* ── Model Analysis view (admin only) ── */}
          {activeNav === "Model Analysis" && isAdmin && (
            <ModelAnalysis news={allNews} />
          )}

          {/* ── Training Data view (admin only) ── */}
          {activeNav === "Training Data" && isAdmin && (
            <TrainingAnalysis />
          )}

        </div>
      </div>

      {/* Modal only appears on News & Sentiment tab (dashboard tab uses the inline ExplainPanel) */}
      {selectedNews && activeNav !== "Dashboard" && <NewsModal item={selectedNews} onClose={() => setSelectedNews(null)} />}
    </div>
  );
}
