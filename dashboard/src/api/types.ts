export interface Episode {
  id: string
  user_id: string
  session_id: string
  role: string
  content: string
  created_at: string
  occurred_at: string
}

export interface SemanticFact {
  id: string
  user_id: string
  fact: string
  confidence: number
  status: string
  source_episode_ids: string[]
  created_at: string
  updated_at: string
  last_reinforced_at: string
}

export interface ScoredEpisode {
  episode: Episode
  similarity: number
  recency_factor: number
  score: number
}

export interface ScoredFact {
  fact: SemanticFact
  similarity: number
  score: number
}

export interface MemoryQueryResult {
  episodes: ScoredEpisode[]
  facts: ScoredFact[]
}

export interface ChatResponse {
  reply: string
  session_id: string
  memory_used: number
  facts_learned: string[]
  strategy_used: string
}

export interface DailyCount {
  date: string
  count: number
}

export interface MemoryStats {
  episodic_total: number
  semantic_active: number
  semantic_merged: number
  semantic_forgotten: number
  episodic_by_day: DailyCount[]
  semantic_by_day: DailyCount[]
}

export interface ReflectionLogEntry {
  id: string
  user_id: string
  action: 'merge' | 'decay' | 'forget'
  fact_ids: string[]
  detail: string
  created_at: string
}

export interface ReflectionRunResult {
  facts_merged_into: number
  facts_decayed: number
  facts_forgotten: number
}

export interface ProceduralStrategyStats {
  strategy_name: string
  uses: number
  successes: number
  success_rate: number
}
