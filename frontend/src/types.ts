export interface Chunk {
  source: string
  slug: string
  score: number
  preview: string
  text: string
}

export interface DoneMetrics {
  t_ms: number
  in_tokens: number
  out_tokens: number
  embed_tokens?: number
  cost_usd: number
  sources: string[]
  turns?: number
}

export interface ToolCallEvent {
  event: 'tool_call'
  tool: string
  args: Record<string, string>
}

export interface ToolResultEvent {
  event: 'tool_result'
  tool: string
  args: Record<string, string>
  preview: string
}

export interface RetrievedChunksEvent {
  event: 'retrieved_chunks'
  chunks: Chunk[]
  t_ms: number
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

export interface PipelineState {
  status: 'idle' | 'streaming' | 'done' | 'error'
  answer: string
  chunks: Chunk[]
  toolEvents: Array<ToolCallEvent | ToolResultEvent>
  metrics: DoneMetrics | null
  error: string | null
}

export const emptyPipelineState = (): PipelineState => ({
  status: 'idle',
  answer: '',
  chunks: [],
  toolEvents: [],
  metrics: null,
  error: null,
})
