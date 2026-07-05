// Shared domain types for the dashboard.
// Import these into App.tsx as you tighten the TypeScript settings:
//   import type { NewsItem, SimilarNews, Candle } from "./types";

export type Sentiment = "positive" | "negative" | "neutral";

export interface SimilarNews {
  title: string;
  change?: number; // % BTC move after the similar headline
  sim?: number;    // cosine similarity 0–1
}

export interface NewsItem {
  id?: string | number;
  title: string;
  link?: string;
  channel?: string;
  time?: string;           // UTC string from API — do not use for display
  published_ts?: number;   // unix seconds (authoritative timestamp)
  received_at?: number;
  sentiment?: Sentiment;
  confidence?: number;     // 0–100
  weight?: number;
  model_score?: number;    // 15m impact score
  model_score_1h?: number; // 1h impact score
  score_normalized?: boolean;
  prob_positive?: number;
  prob_negative?: number;
  prob_neutral?: number;
  news_type?: string;
  btc_change_15m?: number;
  similar?: SimilarNews[];
}

export interface ExplainResponse {
  explanation?: string;
  steps?: string[];
  similar?: SimilarNews[];
  error?: string;
}

export interface Candle {
  time: number; // local unix seconds (lightweight-charts Time)
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface FearGreed {
  value: number; // 0–100
  label: string;
}

export type ImpactTier = "Hot" | "Medium" | "Show" | "Hidden";
export type ImportanceTier = "Key" | "Notable" | "Regular";
export type CoinFilter = "btc" | "eth" | "both";
