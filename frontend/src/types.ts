export type RerankMode = 'none' | 'cross-encoder' | 'llm'

export interface Chunk {
  source: string
  slug: string
  score: number
  preview: string
  text?: string
  rerank_score?: number
}

export interface DoneMetrics {
  t_ms: number
  in_tokens: number
  out_tokens: number
  embed_tokens?: number
  cost_usd: number
  sources: string[]
  turns?: number
  rerank_ms?: number
  rerank_mode?: RerankMode
}

export interface ToolCallEvent {
  event: 'tool_call'
  tool: string
  args: Record<string, string>
}

export interface GraphNode {
  id: string
  name: string
  type: string
}

export interface GraphEdge {
  src: string
  rel: string
  dst: string
  source?: string
  evidence?: string
}

export interface GraphPayload {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface ToolResultEvent {
  event: 'tool_result'
  tool: string
  args: Record<string, string>
  preview: string
  hits?: Chunk[]
  initial_hits?: Chunk[]
  rerank_mode?: RerankMode
  graph?: GraphPayload
}

export interface RetrievedChunksEvent {
  event: 'retrieved_chunks'
  stage?: 'initial'
  chunks: Chunk[]
  t_ms: number
}

export interface RerankedChunksEvent {
  event: 'reranked_chunks'
  stage: 'reranked'
  mode: 'cross-encoder' | 'llm'
  chunks: Chunk[]
  rerank_ms: number
}

export interface TokenEvent {
  event: 'token'
  text: string
}

export interface DoneEvent extends DoneMetrics {
  event: 'done'
}

export type SSEEvent =
  | RetrievedChunksEvent
  | RerankedChunksEvent
  | ToolCallEvent
  | ToolResultEvent
  | TokenEvent
  | DoneEvent

export interface DemoQuestion {
  id: string
  question: string
  answer: string
  type: 'bridge' | 'comparison'
  level: 'easy' | 'medium' | 'hard'
  supporting_slugs: string[]
}

export interface Verdict {
  correct: boolean
  reason: string
}

export interface PipelineState {
  status: 'idle' | 'streaming' | 'done' | 'error'
  answer: string
  chunks: Chunk[]
  rerankedChunks: Chunk[]
  toolEvents: Array<ToolCallEvent | ToolResultEvent>
  metrics: DoneMetrics | null
  error: string | null
  judging: boolean
  verdict: Verdict | null
}

export type PipelineKey = 'rag' | 'agenticRag' | 'wiki' | 'graphRag'

export const ALL_PIPELINE_KEYS: PipelineKey[] = ['rag', 'agenticRag', 'wiki', 'graphRag']

export const emptyPipelineState = (): PipelineState => ({
  status: 'idle',
  answer: '',
  chunks: [],
  rerankedChunks: [],
  toolEvents: [],
  metrics: null,
  error: null,
  judging: false,
  verdict: null,
})
