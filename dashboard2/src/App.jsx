import { useState, useEffect, useRef, useCallback, useMemo } from "react";
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

// ── Coin classifier (keyword-based) ───────────────────────────────
function classifyNewsCoin(title) {
  const t = (title || "").toLowerCase();
  const btc = /bitcoin|\bbtc\b|satoshi|halving|lightning|sats\b|michael saylor|grayscale btc/.test(t);
  const eth = /ethereum|\beth\b|ether\b|vitalik|buterin|\bdefi\b|erc-20|erc20|layer.?2|\bl2\b|arbitrum|optimism|uniswap|polygon|staking eth|eth etf/.test(t);
  if (btc && eth) return "both";
  if (btc) return "btc";
  if (eth) return "eth";
  return "both"; // general crypto → show in both
}

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
  const d = toLocalDate(raw); // handles both seconds and milliseconds
  return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
}

function sentimentLabel(s) {
  if (s === "positive") return "Bullish";
  if (s === "negative") return "Bearish";
  return "Neutral";
}
// Impact badges (coloring only — NOT used for display filtering)
// Hot ≥0.50, Medium ≥0.25, uses max(score_15m, score_1h)
const SCORE_HOT  = 0.50;
const SCORE_MED  = 0.25;
const CONF_MIN   = 50;   // minimum confidence to display at all

function scoreTier(score15, conf, score1h) {
  const s = Math.max(Math.abs(score15 || 0), Math.abs(score1h || 0));
  const c = conf || 0;
  if (c < CONF_MIN) return "Hidden";
  if (s >= SCORE_HOT)  return "Hot";
  if (s >= SCORE_MED)  return "Medium";
  return "Show";
}

// ── News Importance (editorial importance — independent of price impact) ──
// Combines confidence, sentiment strength, channel authority, and editorial keywords.
// Returns { tier: "Key"|"Notable"|"Regular", score: 0–100 }
const IMPORTANCE_KEYWORDS = /\b(JUST IN|BREAKING|MASSIVE|BIG|ALERT|NOW|UPDATE|URGENT)\b/i;
const CHANNEL_AUTHORITY = { cointelegraph: 1.0, coindesk: 1.0, the_block_crypto: 0.95, WatcherGuru: 0.85, google_news: 0.7 };

function newsTier(item) {
  const conf = (item.confidence || 0) / 100;                         // 0–1
  const sentStrength = Math.max(
    item.prob_positive || 0, item.prob_negative || 0, item.prob_neutral || 0
  );                                                                  // 0–1 (how decisive)
  const chAuth = CHANNEL_AUTHORITY[item.channel] || 0.5;
  const kwBoost = IMPORTANCE_KEYWORDS.test(item.title || "") ? 0.15 : 0;

  // Weighted importance score (0–1)
  const raw = (conf * 0.35) + (sentStrength * 0.25) + (chAuth * 0.25) + kwBoost;
  const score = Math.min(1, raw);
  const pct = Math.round(score * 100);

  if (pct >= 70) return { tier: "Key",     score: pct };
  if (pct >= 55) return { tier: "Notable", score: pct };
  return               { tier: "Regular", score: pct };
}

function signalAction(type, modelScore, modelScore1h) {
  const s = Math.max(Math.abs(modelScore || 0), Math.abs(modelScore1h || 0));
  if (type === "BUY")  return s >= SCORE_HOT ? "Strong Buy"  : "Buy";
  if (type === "SELL") return s >= SCORE_HOT ? "Strong Sell" : "Sell";
  return "Neutral";
}
// Normalize raw model score → 0–1 for items not yet normalized by main.py
function clientNormalize(item) {
  if (item.score_normalized) return item; // already normalized by main.py / score_historical
  const norm = (raw, min, max) => Math.max(0, Math.min(1, (raw - min) / (max - min)));
  return {
    ...item,
    model_score:    norm(Math.abs(item.model_score    || 0), 0.50, 0.90),
    model_score_1h: item.model_score_1h,
    score_normalized: true,
  };
}
function fmtScore(score) {
  if (score == null) return "—";
  return `${Math.round(Math.abs(score) * 100)}%`;
}
function cleanTitle(title = "") {
  return title
    .replace(/\*\*?/g, "")           // strip ** and *
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1") // strip markdown links [text](url) → text
    .replace(/`([^`]*)`/g, "$1")     // strip inline code
    .trim();
}
const RELIABLE_CHANNELS = new Set(["the_block_crypto", "coindesk", "cointelegraph", "WatcherGuru", "google_news"]);

// Display filter: confidence + reliable channel only (NO score gate — score is for badges, not filtering)
function passesFilter(n) {
  const conf = n.confidence || 0;
  return RELIABLE_CHANNELS.has(n.channel)
    && conf >= CONF_MIN
    && n.sentiment !== "neutral"
    && (n.title || "").trim().length >= 20;
}
function passesChartFilter(n) { return passesFilter(n); }
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

function fmtPct(v, decimals = 2) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  if (isNaN(n)) return "—";
  return `${n > 0 ? "+" : ""}${n.toFixed(decimals)}%`;
}

const NEWS_TYPE_LABELS = {
  regulatory: "⚖️ Regulatory",
  etf:        "📈 ETF",
  hack:       "🔓 Hack",
  macro_economic: "🏦 Macro",
  exchange:   "🔄 Exchange",
  defi:       "🌊 DeFi",
  mining:     "⛏️ Mining",
  institutional: "🏛️ Institutional",
  technical:  "🔧 Technical",
  market_analysis: "📊 Analysis",
};

// ── MacroContext — 5 ML macro signals derived from published_ts ────
function MacroContext({ ts }) {
  if (!ts) return null;
  const d    = new Date(ts * 1000);
  const hour = d.getUTCHours();
  const dow  = d.getUTCDay(); // 0=Sun, 6=Sat

  const flags = [
    { label: "Weekend",   active: dow === 0 || dow === 6,   icon: "📅", color: COLORS.gold   },
    { label: "Low Liq",   active: hour >= 2 && hour < 6,    icon: "🌙", color: COLORS.purple },
    { label: "US Sess",   active: hour >= 13 && hour < 21,  icon: "🗽", color: COLORS.accent },
    { label: "Asia Sess", active: hour >= 0 && hour < 8,    icon: "🌏", color: "#22d3ee"     },
  ];

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 5, fontFamily: "monospace" }}>
        MACRO CONTEXT
      </div>
      <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
        {flags.map(f => (
          <span key={f.label} style={{
            fontSize: 9, padding: "3px 8px", borderRadius: 4, fontFamily: "monospace",
            background: f.active ? `${f.color}22` : `${COLORS.border}66`,
            color: f.active ? f.color : COLORS.muted,
            border: `1px solid ${f.active ? `${f.color}44` : "transparent"}`,
            fontWeight: f.active ? 700 : 400,
            opacity: f.active ? 1 : 0.4,
          }}>
            {f.icon} {f.label}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── ProbBars — stacked sentiment probability bars ──────────────────
function ProbBars({ pos, neg, neu }) {
  const p = Number(pos || 0), n = Number(neg || 0), u = Number(neu || 0);
  const total = p + n + u || 1;
  const bars = [
    { label: "Bullish",  val: p, pct: p / total, color: COLORS.green },
    { label: "Bearish",  val: n, pct: n / total, color: COLORS.red   },
    { label: "Neutral",  val: u, pct: u / total, color: COLORS.muted },
  ];
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 6, fontFamily: "monospace" }}>
        MODEL PROBABILITIES
      </div>
      {bars.map(b => (
        <div key={b.label} style={{ marginBottom: 5 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
            <span style={{ fontSize: 9, color: b.color, fontFamily: "monospace" }}>{b.label}</span>
            <span style={{ fontSize: 9, color: b.color, fontFamily: "monospace", fontWeight: 700 }}>
              {(b.val * 100).toFixed(1)}%
            </span>
          </div>
          <div style={{ height: 4, background: `${COLORS.border}`, borderRadius: 2, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${b.pct * 100}%`, background: b.color, borderRadius: 2, transition: "width 0.4s" }} />
          </div>
        </div>
      ))}
    </div>
  );
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
  const containerRef    = useRef(null);
  const chartRef        = useRef(null);
  const candleRef       = useRef(null);
  const volRef          = useRef(null);
  const wsRef           = useRef(null);
  const markersRef      = useRef(null);
  const candlesDataRef  = useRef([]);   // raw candles for price-signal detection
  const markerNewsRef   = useRef(new Map()); // bucket time → news item, for tooltip
  const [price, setPrice]   = useState(null);
  const [change, setChange] = useState(null);
  const [loading, setLoading] = useState(true);
  const [candlesVer, setCandlesVer] = useState(0); // bumped when candles load
  const [tooltip, setTooltip] = useState(null);    // { x, y, item }

  useEffect(() => {
    if (!containerRef.current) return;
    setLoading(true);

    // Create chart
    const chart = createChart(containerRef.current, {
      layout:     { background: { color: COLORS.bg }, textColor: COLORS.muted },
      grid:       { vertLines: { color: COLORS.panel }, horzLines: { color: COLORS.panel } },
      crosshair:  { mode: 1 },
      rightPriceScale: { borderColor: COLORS.border2 },
      timeScale:  { borderColor: COLORS.border2, timeVisible: true, secondsVisible: false },
      width:  containerRef.current.clientWidth,
      height: containerRef.current.clientHeight || 360,
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

    // Fetch last 150 days of klines in batches of 1000
    const binSymbol = symbol.replace("BINANCE:", "");
    const startMs60d = Date.now() - 150 * 24 * 60 * 60 * 1000; // 150 days ago

    const intervalMs = { "1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000 };
    const ivMs = intervalMs[interval] || 3600000;

    (async () => {
      const allKlines = [];
      let startTime = startMs60d;
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
        if (!allKlines.length) { setLoading(false); return; }
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
        candlesDataRef.current = candles;
        setCandlesVer(v => v + 1);
        setLoading(false);
      } catch { setLoading(false); }
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

    // Crosshair tooltip: show news title when near a marker
    chart.subscribeCrosshairMove(param => {
      if (!param.time || !containerRef.current) { setTooltip(null); return; }
      const markerMap = markerNewsRef.current;
      const ivSec = { "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400 }[interval] || 3600;
      const bucket = Math.floor(param.time / ivSec) * ivSec;
      const item = markerMap.get(bucket);
      if (!item) { setTooltip(null); return; }
      const rect = containerRef.current.getBoundingClientRect();
      setTooltip({ x: (param.point?.x || 0), y: (param.point?.y || 0), item });
    });

    return () => {
      ws.close();
      ro.disconnect();
      chart.remove();
      markersRef.current   = null;
      candleRef.current    = null;
      candlesDataRef.current = [];
    };
  }, [symbol, interval]);

  // Markers: news signals (circles) + price-only signals (arrows) for no-news moves
  useEffect(() => {
    if (!candleRef.current) return;
    const intervalSeconds = { "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400 };
    const ivSec = intervalSeconds[interval] || 3600;

    // ── News markers: deduplicated per candle, max 1 per candle ────
    // High (≥0.67, s1h≥0.60, conf≥60%) = prominent dot on chart
    // Keep only the highest-scoring item per candle bucket
    const buckets = new Map();
    news
      .filter(passesChartFilter)
      .forEach(n => {
        const bucket = Math.floor((n.published_ts || 0) / ivSec) * ivSec;
        const prev = buckets.get(bucket);
        if (!prev || Math.abs(n.model_score) > Math.abs(prev.model_score || 0))
          buckets.set(bucket, n);
      });
    // Save bucket→news for tooltip lookup
    markerNewsRef.current = new Map();
    const newsSignals = [];
    buckets.forEach((n, bucket) => {
      const isHot = Math.abs(n.model_score || 0) >= SCORE_HOT;
      const t     = bucket + TZ_OFFSET;
      const pos   = n.sentiment === "positive" ? "belowBar" : "aboveBar";
      const color = n.sentiment === "positive" ? "#22c55e" : "#ef4444";
      newsSignals.push({ time: t, position: pos, color, shape: "circle", text: "", size: isHot ? 1 : 0.5 });
      markerNewsRef.current.set(t, n);
    });
    newsSignals.sort((a, b) => a.time - b.time);

    const allMarkers = [...newsSignals].sort((a, b) => a.time - b.time);
    try {
      if (markersRef.current) {
        markersRef.current.setMarkers(allMarkers);
      } else {
        markersRef.current = createSeriesMarkers(candleRef.current, allMarkers);
      }
    } catch {}
  }, [news, interval, candlesVer]);

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
      {loading && (
        <div style={{
          position: "absolute", inset: 0, zIndex: 5,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: COLORS.bg, gap: 10,
        }}>
          <div style={{ width: 18, height: 18, border: `2px solid ${COLORS.border2}`, borderTopColor: COLORS.accent, borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
          <span style={{ fontSize: 12, color: COLORS.muted, fontFamily: "monospace" }}>Loading chart…</span>
        </div>
      )}
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
      {tooltip && (
        <div style={{
          position: "absolute",
          left: Math.min(tooltip.x + 12, (containerRef.current?.clientWidth || 400) - 260),
          top:  Math.max(tooltip.y - 80, 8),
          background: COLORS.panel,
          border: `1px solid ${COLORS.border2}`,
          borderRadius: 8,
          padding: "8px 12px",
          zIndex: 100,
          maxWidth: 250,
          pointerEvents: "none",
          boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
        }}>
          <div style={{ fontSize: 10, color: tooltip.item.sentiment === "positive" ? COLORS.green : COLORS.red, fontWeight: 700, marginBottom: 3 }}>
            {tooltip.item.sentiment === "positive" ? "▲ BULLISH" : "▼ BEARISH"}
          </div>
          <div style={{ fontSize: 11, color: COLORS.text, lineHeight: 1.4 }}>
            {cleanTitle(tooltip.item.title)}
          </div>
        </div>
      )}
    </div>
  );
}

function SentimentGauge({ news }) {
  const [momentum, setMomentum] = useState(null); // { change, open, close }

  useEffect(() => {
    const fetchMomentum = async () => {
      try {
        const res = await fetch(`${API_BASE}/proxy/klines?symbol=BTCUSDT&interval=15m&limit=2`);
        const data = await res.json();
        if (data && data.length >= 2) {
          const prev  = data[data.length - 2];
          const curr  = data[data.length - 1];
          const open  = parseFloat(curr[1]);
          const close = parseFloat(curr[4]);
          const change = ((close - open) / open) * 100;
          setMomentum({ change, open, close });
        }
      } catch {}
    };
    fetchMomentum();
    const iv = setInterval(fetchMomentum, 30_000);
    return () => clearInterval(iv);
  }, []);

  // Map price change to gauge value: ±3% → 0–100
  const MAX_CHANGE = 3;
  const value = momentum != null
    ? Math.round(Math.min(100, Math.max(0, 50 + (momentum.change / MAX_CHANGE) * 50)))
    : 50;
  const ready = momentum != null;

  const angle    = (value / 100) * 180 - 90;
  const getColor = (v) => v < 30 ? "#ef4444" : v < 45 ? "#f59e0b" : v < 55 ? "#eab308" : v < 70 ? "#22c55e" : "#16a34a";
  const getLabel = (v) => v < 20 ? "Extreme Bear" : v < 40 ? "Bearish" : v < 60 ? "Neutral" : v < 80 ? "Bullish" : "Extreme Bull";
  const color    = ready ? getColor(value) : COLORS.muted;
  const changeStr = momentum ? `${momentum.change >= 0 ? "+" : ""}${momentum.change.toFixed(2)}%` : "—";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", width: "100%", alignItems: "center" }}>
        <span style={{ fontSize: 10, fontWeight: 600, color: COLORS.text }}>BTC 15m Momentum</span>
        <span style={{ fontSize: 9, color: ready ? color : COLORS.muted, fontFamily: "monospace", fontWeight: 700 }}>
          {changeStr}
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
          stroke={ready ? "url(#gaugeGrad)" : COLORS.border2} strokeWidth="12" strokeLinecap="round" />
        <g transform={`rotate(${angle}, 80, 80)`}>
          <line x1="80" y1="80" x2="80" y2="22" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
          <circle cx="80" cy="80" r="5" fill={color} />
        </g>
        <text x="80" y="68" textAnchor="middle" fill={ready ? COLORS.text : COLORS.muted}
          fontSize="22" fontWeight="700" fontFamily="monospace">{value}</text>
      </svg>
      <span style={{ fontFamily: "monospace", fontSize: 12, color, letterSpacing: 2, textTransform: "uppercase" }}>
        {getLabel(value)}
      </span>
      {/* Price momentum bar */}
      <div style={{ width: "100%", marginTop: 4 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
          <span style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>BEAR</span>
          <span style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>BULL</span>
        </div>
        <div style={{ height: 4, background: COLORS.border2, borderRadius: 2, position: "relative" }}>
          {/* center line */}
          <div style={{ position: "absolute", left: "50%", top: 0, width: 1, height: "100%", background: COLORS.border2 }} />
          {/* fill from center */}
          {ready && (() => {
            const pct = Math.min(50, Math.abs(momentum.change) / MAX_CHANGE * 50);
            const isBull = momentum.change >= 0;
            return <div style={{
              position: "absolute",
              top: 0, height: "100%", borderRadius: 2,
              left: isBull ? "50%" : `${50 - pct}%`,
              width: `${pct}%`,
              background: isBull ? COLORS.green : COLORS.red,
            }} />;
          })()}
        </div>
        {momentum && (
          <div style={{ display: "flex", justifyContent: "center", marginTop: 4 }}>
            <span style={{ fontSize: 9, fontFamily: "monospace", color }}>
              {momentum.change >= 0 ? "▲" : "▼"} {Math.abs(momentum.change).toFixed(3)}% / 15m candle
            </span>
          </div>
        )}
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
  const score1h    = item ? Math.abs(item.model_score_1h || 0) : 0;
  const _tier      = item ? scoreTier(score, item.confidence, score1h) : "Hidden";
  const impactClr  = _tier === "Hot" ? COLORS.red : _tier === "Medium" ? "#f97316" : COLORS.muted;
  const impactLbl  = _tier === "Hot" ? "Hot" : _tier === "Medium" ? "Medium" : _tier === "Show" ? "Show" : "Low";
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
                  ? <a href={item.link} target="_blank" rel="noreferrer" style={{ color: COLORS.text, textDecoration: "none" }}>{cleanTitle(item.title)}</a>
                  : cleanTitle(item.title)}
              </div>
              <div style={{ fontSize: 10, color: COLORS.muted, fontFamily: "monospace", marginTop: 4 }}>
                {item.channel || "—"} · {item.time || ""}
              </div>
            </div>

            {/* Stats rows — all ML fields */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6, marginBottom: 10 }}>
              {[
                ["Score",   fmtScore(item.model_score), COLORS.gold],
                ["Conf",    `${Math.round(item.confidence || 0)}%`, COLORS.accent],
                ["BTC 15m", fmtPct(item.btc_change_15m),
                  (item.btc_change_15m || 0) > 0 ? COLORS.green : (item.btc_change_15m || 0) < 0 ? COLORS.red : COLORS.muted],
                ["Weight",  item.weight != null ? Number(item.weight).toFixed(1) : "—", COLORS.muted],
              ].map(([lbl, val, clr]) => (
                <div key={lbl} style={{ background: COLORS.bg, borderRadius: 7, padding: "6px 8px" }}>
                  <div style={{ fontSize: 8, color: COLORS.muted, letterSpacing: 0.8, marginBottom: 2 }}>{lbl.toUpperCase()}</div>
                  <div style={{ fontSize: 11, color: clr, fontFamily: "monospace", fontWeight: 700 }}>{val}</div>
                </div>
              ))}
            </div>

            {/* News type badge */}
            {item.news_type && (
              <div style={{ marginBottom: 10 }}>
                <span style={{ fontSize: 9, padding: "2px 9px", borderRadius: 4, background: `${COLORS.accent}18`, color: COLORS.accent, fontFamily: "monospace", fontWeight: 600 }}>
                  {NEWS_TYPE_LABELS[item.news_type] || item.news_type}
                </span>
              </div>
            )}

            {/* Probability bars */}
            {(item.prob_positive != null || item.prob_negative != null) && (
              <ProbBars pos={item.prob_positive} neg={item.prob_negative} neu={item.prob_neutral} />
            )}

            {/* Macro context */}
            <MacroContext ts={item.published_ts} />

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

  const score   = Math.abs(item.model_score || 0);
  const score1h = Math.abs(item.model_score_1h || 0);
  const _t    = scoreTier(score, item.confidence, score1h);
  const impactColor = _t === "Hot" ? COLORS.red : _t === "Medium" ? "#f97316" : COLORS.muted;
  const impactLabel = _t === "Hot" ? "Hot" : _t === "Medium" ? "Medium" : _t === "Show" ? "Show" : "Low";

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
            ? <a href={item.link} target="_blank" rel="noreferrer" style={{ color: COLORS.text, textDecoration: "none" }}>{cleanTitle(item.title)}</a>
            : cleanTitle(item.title)}
        </div>

        {/* Full stats grid — all ML fields */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, marginBottom: 14 }}>
          {[
            ["Channel",    item.channel || "—",                           COLORS.text],
            ["Impact",     impactLabel,                                    impactColor],
            ["Confidence", `${Math.round(item.confidence || 0)}%`,        COLORS.accent],
            ["Score",      fmtScore(item.model_score),                    COLORS.gold],
            ["BTC 15m",    fmtPct(item.btc_change_15m),
              (item.btc_change_15m||0)>0 ? COLORS.green : (item.btc_change_15m||0)<0 ? COLORS.red : COLORS.muted],
            ["Weight",     item.weight != null ? Number(item.weight).toFixed(1) : "—", COLORS.muted],
          ].map(([lbl, val, clr]) => (
            <div key={lbl} style={{ background: COLORS.bg, borderRadius: 8, padding: "8px 10px" }}>
              <div style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1, marginBottom: 4 }}>{lbl.toUpperCase()}</div>
              <div style={{ fontSize: 12, color: clr, fontFamily: "monospace", fontWeight: 600 }}>{val}</div>
            </div>
          ))}
        </div>

        {/* News type + source row */}
        <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
          {item.news_type && (
            <span style={{ fontSize: 9, padding: "3px 10px", borderRadius: 4, background: `${COLORS.accent}18`, color: COLORS.accent, fontFamily: "monospace", fontWeight: 600 }}>
              {NEWS_TYPE_LABELS[item.news_type] || item.news_type}
            </span>
          )}
          {item.source && (
            <span style={{ fontSize: 9, padding: "3px 10px", borderRadius: 4, background: `${COLORS.border}`, color: COLORS.muted, fontFamily: "monospace" }}>
              {item.source}
            </span>
          )}
        </div>

        {/* Macro context */}
        <MacroContext ts={item.published_ts} />

        {/* Probability bars */}
        {(item.prob_positive != null || item.prob_negative != null) && (
          <div style={{ marginBottom: 20 }}>
            <ProbBars pos={item.prob_positive} neg={item.prob_negative} neu={item.prob_neutral} />
          </div>
        )}

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
        <div style={{ fontSize: 12, color: COLORS.text, lineHeight: 1.4, marginBottom: 4, fontWeight: 500 }}>{cleanTitle(item.title)}</div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          {(() => {
            const s   = Math.abs(item.model_score || 0);
            const s1h = Math.abs(item.model_score_1h || 0);
            const tier = scoreTier(s, item.confidence, s1h);
            const clr  = tier === "Hot" ? COLORS.red : tier === "Medium" ? "#f97316" : COLORS.muted;
            const dots = tier === "Hot" ? "●●●" : tier === "Medium" ? "●●" : "●";
            return (
              <span style={{ fontSize: 10, color: COLORS.muted, display: "flex", alignItems: "center", gap: 3, position: "relative" }}>
                Impact:&nbsp;
                <span
                  onMouseEnter={e => { e.stopPropagation(); setBulletHover(true); }}
                  onMouseLeave={() => setBulletHover(false)}
                  style={{ color: clr, fontSize: 11, letterSpacing: 1, cursor: "help" }}
                >
                  {dots}
                </span>
                <span style={{ color: clr }}>{tier}</span>
                {bulletHover && (
                  <div style={{
                    position: "absolute", bottom: "calc(100% + 6px)", left: 0,
                    background: COLORS.panel, border: `1px solid ${clr}44`,
                    borderRadius: 8, padding: "8px 12px", zIndex: 999,
                    minWidth: 220, pointerEvents: "none",
                    boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
                  }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: clr, marginBottom: 4 }}>
                      {dots} {tier} Impact
                    </div>
                    <div style={{ fontSize: 10, color: COLORS.text, lineHeight: 1.5, marginBottom: 6 }}>
                      {cleanTitle(item.title)}
                    </div>
                    <div style={{ fontSize: 10, color: COLORS.muted }}>
                      <span style={{ color: color }}>{label}</span>
                    </div>
                  </div>
                )}
              </span>
            );
          })()}
          {(() => {
            const imp = newsTier(item);
            const iClr = imp.tier === "Key" ? COLORS.purple : imp.tier === "Notable" ? COLORS.accent : COLORS.muted;
            const iDots = imp.tier === "Key" ? "◆◆◆" : imp.tier === "Notable" ? "◆◆" : "◆";
            return (
              <span style={{ fontSize: 10, color: COLORS.muted, display: "flex", alignItems: "center", gap: 3 }}>
                News:&nbsp;
                <span style={{ color: iClr, fontSize: 10, letterSpacing: 1 }}>{iDots}</span>
                <span style={{ color: iClr }}>{imp.tier}</span>
              </span>
            );
          })()}
        </div>
      </div>
    </div>
  );
}

function SignalCard({ item }) {
  const action = signalAction(item.type, item.model_score, item.model_score_1h);
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
      <div style={{ fontSize: 12, color: COLORS.text, lineHeight: 1.4, marginBottom: 10 }}>{cleanTitle(item.title)}</div>
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
          {(td.total_samples || 0).toLocaleString()} samples · news_cleaned_filtered_scored.csv
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

      {/* Model comparison table */}
      {data.all_models && Object.keys(data.all_models).length > 0 && (() => {
        const models = data.all_models;
        const names  = Object.keys(models);
        const pColor = (v, good) => v >= good ? COLORS.green : v >= good * 0.85 ? COLORS.gold : COLORS.red;
        const rows = [
          { label: "15m AUC",    key: m => (m["15_minute"]?.ROC_AUC  || 0), fmt: v => v.toFixed(3), good: 0.70 },
          { label: "15m DirAcc", key: m => (m["15_minute"]?.DirAcc   || 0), fmt: v => (v*100).toFixed(1)+"%", good: 0.53 },
          { label: "15m Acc",    key: m => (m["15_minute"]?.Accuracy  || 0), fmt: v => (v*100).toFixed(1)+"%", good: 0.65 },
          { label: "15m F1",     key: m => (m["15_minute"]?.F1        || 0), fmt: v => v.toFixed(3), good: 0.40 },
          { label: "1h AUC",     key: m => (m["1_hour"]?.ROC_AUC     || 0), fmt: v => v.toFixed(3), good: 0.70 },
          { label: "1h DirAcc",  key: m => (m["1_hour"]?.DirAcc      || 0), fmt: v => (v*100).toFixed(1)+"%", good: 0.53 },
          { label: "Dir Acc",    key: m => (m["direction"]?.Acc       || 0), fmt: v => (v*100).toFixed(1)+"%", good: 0.53 },
          { label: "Dir F1",     key: m => (m["direction"]?.F1        || 0), fmt: v => v.toFixed(3), good: 0.50 },
        ];
        return (
          <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>
              MODEL VERSIONS COMPARISON
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "monospace" }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "6px 10px", color: COLORS.muted, fontSize: 9, letterSpacing: 1, fontWeight: 600, borderBottom: `1px solid ${COLORS.border2}` }}>METRIC</th>
                    {names.map(n => (
                      <th key={n} style={{ textAlign: "center", padding: "6px 10px", color: n === "xgboost" ? COLORS.gold : COLORS.accent, fontSize: 10, fontWeight: 700, borderBottom: `1px solid ${COLORS.border2}`, letterSpacing: 0.5 }}>
                        {n.toUpperCase()}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, ri) => {
                    const vals = names.map(n => row.key(models[n]));
                    const best = Math.max(...vals);
                    return (
                      <tr key={row.label} style={{ background: ri % 2 === 0 ? `${COLORS.bg}88` : "transparent" }}>
                        <td style={{ padding: "7px 10px", color: COLORS.muted, fontSize: 9, letterSpacing: 0.8, fontWeight: 600 }}>{row.label}</td>
                        {vals.map((v, i) => (
                          <td key={i} style={{ textAlign: "center", padding: "7px 10px", color: pColor(v, row.good), fontWeight: v === best ? 800 : 500 }}>
                            {row.fmt(v)}
                            {v === best && <span style={{ fontSize: 8, marginLeft: 3, color: COLORS.gold }}>▲</span>}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: 10, fontSize: 9, color: COLORS.muted, fontFamily: "monospace" }}>
              ▲ = best result for that metric &nbsp;·&nbsp; green ≥ target &nbsp;·&nbsp; yellow ≥ 85% of target &nbsp;·&nbsp; red = below target
            </div>
          </div>
        );
      })()}

      {/* Model performance */}
      <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 16 }}>MODEL PERFORMANCE (best version)</div>
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

function CustomAnalyzer() {
  const [title, setTitle]   = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState(null);

  const run = async () => {
    if (!title.trim()) return;
    setLoading(true); setResult(null); setError(null);
    try {
      const res = await fetch(`${API_BASE}/analyze/custom`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title.trim() }),
      });
      if (!res.ok) {
        const text = await res.text();
        let detail;
        try { detail = JSON.parse(text).detail; } catch { detail = text; }
        throw new Error(detail || `Server error (${res.status})`);
      }
      setResult(await res.json());
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const sentClr = { positive: COLORS.green, negative: COLORS.red, neutral: COLORS.muted };
  const sigClr  = { BUY: COLORS.green, SELL: COLORS.red, NEUTRAL: COLORS.muted };

  return (
    <div style={{ padding: "24px 24px 0" }}>
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: 24 }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: COLORS.text, marginBottom: 16, letterSpacing: 0.3 }}>
          Custom News Analyzer
        </div>

        {/* Input */}
        <div style={{ display: "flex", gap: 10 }}>
          <input
            value={title}
            onChange={e => setTitle(e.target.value)}
            onKeyDown={e => e.key === "Enter" && run()}
            placeholder="Enter a crypto news headline..."
            style={{
              flex: 1, background: COLORS.bg, border: `1px solid ${COLORS.border}`,
              borderRadius: 8, padding: "10px 14px", color: COLORS.text,
              fontSize: 13, fontFamily: "monospace", outline: "none",
            }}
          />
          <button
            onClick={run}
            disabled={loading || !title.trim()}
            style={{
              background: loading ? COLORS.border : COLORS.accent,
              color: "#fff", border: "none", borderRadius: 8,
              padding: "10px 22px", fontSize: 13, fontWeight: 600,
              cursor: loading ? "not-allowed" : "pointer", whiteSpace: "nowrap",
            }}
          >
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div style={{ marginTop: 12, color: COLORS.red, fontSize: 12 }}>
            Error: {error}
          </div>
        )}

        {/* Loading hint */}
        {loading && (
          <div style={{ marginTop: 12, color: COLORS.muted, fontSize: 11 }}>
            First run loads BERT models (~30s). Subsequent runs are fast.
          </div>
        )}

        {/* Result */}
        {result && (
          <div style={{ marginTop: 20 }}>
            {/* Headline */}
            <div style={{ fontSize: 12, color: COLORS.muted, marginBottom: 12, fontStyle: "italic" }}>
              "{result.title}"
            </div>

            {/* Impact + Sentiment */}
            {(() => {
              const impClr = result.impact === "Hot" ? COLORS.red : result.impact === "Medium" ? "#f97316" : COLORS.muted;
              const sentColor = result.sentiment === "positive" ? COLORS.green : result.sentiment === "negative" ? COLORS.red : COLORS.muted;
              const sentLabel = result.sentiment === "positive" ? "▲ BULLISH" : result.sentiment === "negative" ? "▼ BEARISH" : "● NEUTRAL";
              return (
                <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 20, flexWrap: "wrap" }}>
                  <div style={{ padding: "10px 24px", borderRadius: 12, border: `2px solid ${impClr}`, background: `${impClr}18`, textAlign: "center" }}>
                    <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 4 }}>IMPACT</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: impClr }}>{result.impact}</div>
                  </div>
                  <div style={{ padding: "10px 24px", borderRadius: 12, border: `2px solid ${sentColor}`, background: `${sentColor}18`, textAlign: "center" }}>
                    <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 4 }}>SENTIMENT</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: sentColor }}>{sentLabel}</div>
                  </div>
                  <div style={{ padding: "10px 24px", borderRadius: 12, border: `1px solid ${COLORS.border2}`, background: COLORS.bg, textAlign: "center" }}>
                    <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: 1, marginBottom: 4 }}>CATEGORY</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: COLORS.text, textTransform: "capitalize" }}>{result.news_type?.replace(/_/g, " ")}</div>
                  </div>
                </div>
              );
            })()}

            {/* Similar news */}
            {result.similar?.length > 0 && (
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.accent, marginBottom: 8, letterSpacing: 0.5 }}>📚 Similar Past News</div>
                {result.similar.slice(0, 3).map((s, i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", background: COLORS.bg, borderRadius: 8, marginBottom: 6 }}>
                    <span style={{ fontSize: 11, color: COLORS.text, flex: 1, marginRight: 10 }}>{s.title?.slice(0, 80)}{s.title?.length > 80 ? "…" : ""}</span>
                    <span style={{ fontSize: 11, fontFamily: "monospace", fontWeight: 700, color: s.change >= 0 ? COLORS.green : COLORS.red, flexShrink: 0 }}>
                      BTC {s.change >= 0 ? "+" : ""}{s.change?.toFixed(2)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ModelAnalysis() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/analyze/full-stats`)
      .then(r => r.json())
      .then(d => { setStats(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div style={{ padding: 40, textAlign: "center", color: COLORS.muted, fontSize: 13 }}>Loading full dataset stats…</div>
  );
  if (!stats || !stats.total) return null;

  const BUCKET_COLORS = [COLORS.muted, COLORS.blue, COLORS.gold, "#f97316", COLORS.green, COLORS.accent];
  const buckets = (stats.score_buckets || []).map((b, i) => ({ ...b, color: BUCKET_COLORS[i] || COLORS.muted }));
  const maxBucket = Math.max(...buckets.map(b => b.count), 1);
  const sent = stats.sentiment || {};
  const sentTotal = (sent.bullish || 0) + (sent.bearish || 0) + (sent.neutral || 0) || 1;

  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text, letterSpacing: 0.5 }}>
        🧠 Model Analysis
        <span style={{ fontSize: 11, color: COLORS.muted, fontWeight: 400, marginLeft: 8 }}>
          — {stats.total.toLocaleString()} news items · {stats.date_from} → {stats.date_to}
        </span>
      </div>

      {/* KPI row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <StatBox label="Avg Model Score"  value={`${Math.round((stats.avg_score || 0) * 100)}%`} color={COLORS.accent} />
        <StatBox label="Avg Confidence"   value={`${Math.round((stats.avg_confidence || 0) * 100)}%`} color={COLORS.blue} />
        <StatBox label="Avg Weight"       value={`${(stats.avg_weight || 0).toFixed(1)}/10`} color={COLORS.gold} />
        <StatBox label="Total Processed"  value={stats.total.toLocaleString()} color={COLORS.text} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Score distribution */}
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>SCORE DISTRIBUTION</div>
          {buckets.map(b => (
            <MiniBar key={b.label} label={b.label} value={b.count} max={maxBucket} color={b.color} />
          ))}
        </div>

        {/* Sentiment breakdown */}
        <div style={{ background: COLORS.panel, borderRadius: 12, padding: 16, border: `1px solid ${COLORS.border2}` }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 14 }}>SENTIMENT BREAKDOWN</div>
          <MiniBar label="Bullish" value={sent.bullish || 0} max={sentTotal} color={COLORS.green} />
          <MiniBar label="Bearish" value={sent.bearish || 0} max={sentTotal} color={COLORS.red} />
          <MiniBar label="Neutral" value={sent.neutral || 0} max={sentTotal} color={COLORS.muted} />
        </div>

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

function ChannelAnalysisPage() {
  const [sortBy, setSortBy] = useState("count");
  const [fullStats, setFullStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/analyze/full-stats`)
      .then(r => r.json())
      .then(d => { setFullStats(d); setStatsLoading(false); })
      .catch(() => setStatsLoading(false));
  }, []);

  const rawChannels = fullStats?.channels || [];
  const channels = [...rawChannels].sort((a, b) => {
    if (sortBy === "count")    return b.count - a.count;
    if (sortBy === "avgScore") return b.avgScore - a.avgScore;
    if (sortBy === "avgConf")  return b.avgConf - a.avgConf;
    if (sortBy === "avgBtc15") return b.avgBtc15 - a.avgBtc15;
    return b.count - a.count;
  });
  const maxCount = Math.max(...channels.map(c => c.count), 1);
  const maxScore = Math.max(...channels.map(c => c.avgScore), 1);

  const sortBtn = (id, label, active, setter) => (
    <button onClick={() => setter(id)} style={{
      padding: "3px 10px", borderRadius: 5, fontSize: 10, cursor: "pointer", border: "none",
      background: active === id ? COLORS.accent : COLORS.border2,
      color: active === id ? "#000" : COLORS.muted, fontWeight: active === id ? 700 : 400,
    }}>{label}</button>
  );

  const rankColor = (i) => i === 0 ? COLORS.gold : i === 1 ? COLORS.muted : i === 2 ? "#cd7f32" : COLORS.border2;

  if (statsLoading) return (
    <div style={{ padding: 40, textAlign: "center", color: COLORS.muted, fontSize: 13 }}>Loading channel stats…</div>
  );

  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 24 }}>

      {/* ── Channel Performance ── */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text }}>📡 Channel Performance</div>
            <div style={{ fontSize: 11, color: COLORS.muted, marginTop: 2 }}>
              All {fullStats?.total?.toLocaleString()} items · {fullStats?.date_from} → {fullStats?.date_to}
            </div>
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
                <span style={{ fontSize: 12, fontFamily: "monospace", fontWeight: 700, color: ch.avgScore >= SCORE_HOT ? COLORS.green : ch.avgScore >= SCORE_MED ? COLORS.gold : COLORS.muted }}>
                  {Math.round(ch.avgScore * 100)}%
                </span>
                <PerformanceBar value={ch.avgScore} max={maxScore} color={COLORS.gold} width={60} />
              </div>
              {/* Confidence */}
              <div style={{ fontSize: 12, fontFamily: "monospace", color: COLORS.blue }}>{(ch.avgConf * 100).toFixed(1)}%</div>
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
  const [activeNav, setActiveNav]         = useState("Dashboard");
  const [newsTab, setNewsTab]             = useState("all");
  const [coinFilter, setCoinFilter]       = useState("all"); // "all" | "btc" | "eth"
  const [selectedPair, setSelectedPair]   = useState("BINANCE:BTCUSDT");
  const [selectedSymbol, setSelectedSymbol] = useState("BTCUSDT");
  const [chartInterval, setChartInterval] = useState("15m");
  const [time, setTime]                   = useState(new Date());
  const [allNews, setAllNews]             = useState([]);
  const [hotSignals, setHotSignals]       = useState([]);
  const [selectedNews, setSelectedNews]   = useState(null);
  const [newsH, setNewsH]                 = useState(450);
  const newsDragRef                        = useRef({ dragging: false, startY: 0, startH: 0 });

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
      .then(r => r.json()).then(data => setAllNews(sortByTime(data.map(clientNormalize).filter(passesFilter)))).catch(() => {});
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
        const filtered = data.map(clientNormalize).filter(passesChartFilter);
        setDateNews(sortByTime(filtered));
        setDateLoading(false);
      })
      .catch(() => setDateLoading(false));
  }, [calendarDate]);

  // Live WebSocket feeds — server pushes new items when bot writes to cache
  const allConnected = useWebSocket("/ws/all", (item) => {
    const norm = clientNormalize(item);
    if (!passesFilter(norm)) return;
    setAllNews(prev => sortByTime([norm, ...prev]).slice(0, 200));
  });
  const hotConnected = useWebSocket("/ws/hot", (item) => {
    setHotSignals(prev => sortByTime([clientNormalize(item), ...prev]).slice(0, 50));
  });

  // Polling fallback — catch items missed by WebSocket (60 s interval)
  useEffect(() => {
    const filterItem = (n) => passesFilter(n);
    const poll = setInterval(() => {
      setAllNews(prev => {
        const latestTs = prev.length ? (prev[0].published_ts || prev[0].received_at || 0) : 0;
        fetch(`${API_BASE}/news/since?ts=${latestTs}`)
          .then(r => r.json())
          .then(fresh => {
            const newItems = fresh.map(clientNormalize).filter(filterItem);
            if (!newItems.length) return;
            setAllNews(p => sortByTime([...newItems, ...p]).slice(0, 500));
          })
          .catch(() => {});
        return prev;
      });
    }, 60_000);
    return () => clearInterval(poll);
  }, []);

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
    // handles both unix seconds and milliseconds
    const raw = ts < 4102444800 ? ts * 1000 : ts;
    const d = new Date(raw);
    return `${d.getUTCFullYear()}-${String(d.getUTCMonth()+1).padStart(2,"0")}-${String(d.getUTCDate()).padStart(2,"0")}`;
  }
  // Local date key — used for "today" display so it matches user's clock
  function localDateKey(ts) {
    const raw = ts < 4102444800 ? ts * 1000 : ts;
    const d = new Date(raw);
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
    // build list of distinct dates newest-first
    const sorted = [...unique].sort((a, b) => (b.published_ts || b.received_at || 0) - (a.published_ts || a.received_at || 0));
    const dateSeen = new Set();
    const orderedDates = [];
    for (const n of sorted) {
      const ts = n.published_ts || n.received_at;
      if (!ts) continue;
      const dk = localDateKey(ts);
      if (!dateSeen.has(dk)) { dateSeen.add(dk); orderedDates.push(dk); }
    }
    // pick the most recent day that has at least one item passing the display filter
    // so we never land on a day where every item is Hidden tier
    const latestKey = orderedDates.find(dk =>
      unique.some(n => {
        const ts = n.published_ts || n.received_at;
        return ts && localDateKey(ts) === dk && passesFilter(n);
      })
    ) || orderedDates[0]; // fallback: show most recent day even if all Hidden
    // filter to that day, sort newest first
    return unique
      .filter(n => { const ts = n.published_ts || n.received_at; return ts ? localDateKey(ts) === latestKey : false; })
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

  // Tab filters — impact badges use max(score_15m, score_1h), importance uses newsTier()
  const hotTabNews       = newsForDate(n => passesFilter(n) && scoreTier(Math.abs(n.model_score || 0), n.confidence, Math.abs(n.model_score_1h || 0)) === "Hot");
  const importantTabNews = newsForDate(n => passesFilter(n) && scoreTier(Math.abs(n.model_score || 0), n.confidence, Math.abs(n.model_score_1h || 0)) === "Medium");
  const keyTabNews       = newsForDate(n => passesFilter(n) && newsTier(n).tier === "Key");   // Editorially important regardless of price impact
  const allTabNews       = newsForDate(passesFilter);

  const tabNewsRaw = newsTab === "hot" ? hotTabNews : allTabNews;
  const tabNews = coinFilter === "all"
    ? tabNewsRaw
    : tabNewsRaw.filter(n => { const c = classifyNewsCoin(n.title); return c === coinFilter || c === "both"; });

  const NavIcon = ({ d, size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      {Array.isArray(d) ? d.map((p, i) => <path key={i} d={p} />) : <path d={d} />}
    </svg>
  );

  const navItems = [
    { icon: <NavIcon d={["M3 3v18h18", "M7 16l4-4 4 4 4-8"]} />, label: "Dashboard" },
    { icon: <NavIcon d={["M4 22h16a2 2 0 002-2V4a2 2 0 00-2-2H8L4 6v14a2 2 0 002 2z", "M8 2v4H4", "M12 12h4", "M12 16h4", "M8 12h.01", "M8 16h.01"]} />, label: "News & Sentiment" },
    { icon: <NavIcon d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />, label: "Analyze" },
    ...(isAdmin ? [
      { icon: <NavIcon d="M12 2a4 4 0 014 4c0 1.5-.8 2.8-2 3.5V12h2a2 2 0 012 2v6H6v-6a2 2 0 012-2h2V9.5C8.8 8.8 8 7.5 8 6a4 4 0 014-4z" />, label: "Model Analysis" },
      { icon: <NavIcon d={["M18 20V10", "M12 20V4", "M6 20v-6"]} />, label: "Training Data" },
    ] : []),
  ];

  const dateLabel = calendarDate
    ? calendarDate.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })
    : (mostRecentIsToday ? "Today" : mostRecentDateLabel);

  return (
    <div style={{ display: "flex", height: "100vh", width: "100vw", background: COLORS.bg, color: COLORS.text, fontFamily: "'DM Sans', system-ui, sans-serif", overflow: "hidden" }}>

      {/* ── Sidebar ── */}
      <div style={{ width: 220, flexShrink: 0, background: COLORS.panel, borderRight: `1px solid ${COLORS.border}`, display: "flex", flexDirection: "column", height: "100%" }}>
        <div style={{ padding: "20px 16px", borderBottom: `1px solid ${COLORS.border}`, display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
          <img src="/logo_center.avif" alt="logo" style={{ width: 50, height: 50, borderRadius: 10, objectFit: "cover" }} />
          <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text, letterSpacing: 0.5, textAlign: "center" }}>Crypto Sentiment Analyze</div>
        </div>
        <div style={{ flex: 1, padding: "12px 8px", overflowY: "auto" }}>
          {navItems.map(item => (
            <NavItem key={item.label} icon={item.icon} label={item.label}
              active={activeNav === item.label} onClick={() => setActiveNav(item.label)} />
          ))}
          {/* Coin filter in sidebar */}
          <div style={{ marginTop: 12, padding: "8px 4px", borderTop: `1px solid ${COLORS.border}` }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: COLORS.muted, letterSpacing: 1, marginBottom: 8, paddingLeft: 4 }}>COIN FILTER</div>
            {[
              { id: "all", label: "All Coins",  icon: "◎", color: COLORS.accent },
              { id: "btc", label: "₿ Bitcoin",  icon: "₿", color: "#F7931A" },
              { id: "eth", label: "Ξ Ethereum", icon: "Ξ", color: "#627EEA" },
            ].map(c => (
              <div key={c.id} onClick={() => { setCoinFilter(c.id); if (c.id !== "all") { setSelectedSymbol(c.id === "btc" ? "BTCUSDT" : "ETHUSDT"); setSelectedPair(`BINANCE:${c.id === "btc" ? "BTCUSDT" : "ETHUSDT"}`); } }} style={{
                display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", borderRadius: 8,
                cursor: "pointer", marginBottom: 2,
                background: coinFilter === c.id ? `${c.color}20` : "transparent",
                border: `1px solid ${coinFilter === c.id ? c.color : "transparent"}`,
              }}>
                <span style={{ fontSize: 13, color: c.color }}>{c.icon}</span>
                <span style={{ fontSize: 12, fontWeight: coinFilter === c.id ? 700 : 400, color: coinFilter === c.id ? c.color : COLORS.muted }}>{c.label}</span>
              </div>
            ))}
          </div>
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
                  {/* Impact tabs + Coin filter */}
                  <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                    <div style={{ display: "flex", gap: 6 }}>
                      {[
                        { id: "hot", label: "🔥 Hot", count: hotTabNews.length },
                        { id: "all", label: "All",   count: allTabNews.length },
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
                    <div style={{ width: 1, height: 16, background: COLORS.border2 }} />
                    <div style={{ display: "flex", gap: 4 }}>
                      {[
                        { id: "all", label: "All",       color: COLORS.accent },
                        { id: "btc", label: "₿ BTC",     color: "#F7931A" },
                        { id: "eth", label: "Ξ ETH",     color: "#627EEA" },
                      ].map(c => (
                        <button key={c.id} onClick={() => setCoinFilter(c.id)} style={{
                          padding: "3px 10px", borderRadius: 12, border: "none", cursor: "pointer", fontSize: 11,
                          background: coinFilter === c.id ? c.color : COLORS.border2,
                          color: coinFilter === c.id ? "#fff" : COLORS.muted,
                          fontWeight: coinFilter === c.id ? 700 : 400,
                        }}>{c.label}</button>
                      ))}
                    </div>
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

          {/* ── News Signals view ── */}
          {activeNav === "News Signals" && (
            <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
              <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
                <div style={{ padding: "12px 20px", borderBottom: `1px solid ${COLORS.border}`, background: COLORS.panel, flexShrink: 0 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 13, fontWeight: 600 }}>
                      {newsTab === "hot" ? "🔥 Hot" : "📰 All"} Signals — <span style={{ color: COLORS.accent }}>{dateLabel}</span>
                      <span style={{ color: COLORS.muted, fontSize: 11, marginLeft: 8 }}>{tabNews.length} items</span>
                    </span>
                    <ConnectionDot connected={hotConnected} />
                  </div>
                </div>
                <div style={{ flex: 1, overflowY: "auto", padding: "10px 20px" }}>
                  {dateLoading ? (
                    <div style={{ padding: "40px 0", textAlign: "center", color: COLORS.muted, fontSize: 12 }}>Loading…</div>
                  ) : tabNews.length === 0 ? (
                    <div style={{ padding: "40px 0", textAlign: "center", color: COLORS.muted, fontSize: 12 }}>No signals for {dateLabel}</div>
                  ) : (
                    tabNews.map((s, i) => <NewsCard key={s.id ?? i} item={s} onClick={() => setSelectedNews(s)} />)
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
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflowY: "auto" }}>
              {/* Chart header — BTC / ETH switcher + interval + clock */}
              <div style={{ padding: "10px 16px", borderBottom: `1px solid ${COLORS.border}`, display: "flex", alignItems: "center", gap: 12, background: COLORS.panel, flexShrink: 0, position: "sticky", top: 0, zIndex: 10 }}>
                <div style={{ display: "flex", gap: 6 }}>
                  {[
                    { sym: "BTCUSDT", label: "₿ BTC", color: "#F7931A" },
                    { sym: "ETHUSDT", label: "Ξ ETH", color: "#627EEA" },
                  ].map(({ sym, label, color }) => (
                    <button key={sym} onClick={() => { setSelectedSymbol(sym); setSelectedPair(`BINANCE:${sym}`); }} style={{
                      padding: "4px 14px", borderRadius: 8, fontSize: 12, cursor: "pointer", fontWeight: 700,
                      border: `2px solid ${selectedSymbol === sym ? color : COLORS.border2}`,
                      background: selectedSymbol === sym ? `${color}20` : "transparent",
                      color: selectedSymbol === sym ? color : COLORS.muted,
                    }}>{label}</button>
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
                <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: 11, fontFamily: "monospace", color: COLORS.muted }}>
                    {time.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })}
                  </span>
                </div>
              </div>

              {/* Single chart — switches between BTC and ETH */}
              <div style={{ height: 520, flexShrink: 0, borderBottom: `1px solid ${COLORS.border}` }}>
                <BinanceChart symbol={selectedPair} interval={chartInterval}
                  news={allNews.filter(n => {
                    const c = classifyNewsCoin(n.title);
                    const sym = selectedSymbol === "ETHUSDT" ? "eth" : "btc";
                    return c === sym || c === "both";
                  })} />
              </div>

              {/* Drag handle */}
              <div
                onMouseDown={e => {
                  const d = newsDragRef.current;
                  d.dragging = true; d.startY = e.clientY; d.startH = newsH;
                  const onMove = ev => { if (!d.dragging) return; setNewsH(Math.max(80, Math.min(800, d.startH + (d.startY - ev.clientY)))); };
                  const onUp   = () => { d.dragging = false; window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
                  window.addEventListener("mousemove", onMove);
                  window.addEventListener("mouseup", onUp);
                }}
                style={{ height: 6, flexShrink: 0, cursor: "ns-resize", background: COLORS.border, display: "flex", alignItems: "center", justifyContent: "center" }}
              >
                <div style={{ width: 32, height: 2, borderRadius: 2, background: COLORS.muted, opacity: 0.5 }} />
              </div>

              {/* Bottom panels */}
              <div style={{ height: newsH, flexShrink: 0, display: "grid", gridTemplateColumns: "1fr 1fr", overflow: "hidden" }}>

                {/* News feed */}
                <div style={{ borderRight: `1px solid ${COLORS.border}`, display: "flex", flexDirection: "column", overflow: "hidden" }}>
                  <div style={{ padding: "10px 16px", borderBottom: `1px solid ${COLORS.border}`, display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
                    <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: 0.5 }}>⚡ AI News Impact (Live)</span>
                    <ConnectionDot connected={allConnected} />
                  </div>
                  {(() => {
                    // Show all passing news from the most recent day
                    const allPassing = allNews.filter(n => passesFilter(n));
                    const latestKey = allPassing.length
                      ? localDateKey(Math.max(...allPassing.map(n => n.published_ts || n.received_at || 0)))
                      : null;
                    const filtered = latestKey ? allPassing.filter(n => localDateKey(n.published_ts || n.received_at) === latestKey) : [];
                    // Deduplicate by title
                    const seen = new Set();
                    const uniqueNews = filtered.filter(n => {
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
              <CustomAnalyzer />
              <div style={{ borderTop: `2px solid ${COLORS.border2}`, margin: "24px 24px 0" }} />
              <ModelAnalysis />
              <div style={{ borderTop: `2px solid ${COLORS.border2}`, margin: "0 24px" }} />
              <ChannelAnalysisPage />
            </div>
          )}

          {/* ── Model Analysis view (admin only) ── */}
          {activeNav === "Model Analysis" && isAdmin && (
            <ModelAnalysis />
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
