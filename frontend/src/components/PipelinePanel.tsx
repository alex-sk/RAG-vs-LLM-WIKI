import { cn } from '@/lib/utils'
import type {
  Chunk,
  PipelineState,
  RerankMode,
  ToolCallEvent,
  ToolResultEvent,
} from '@/types'
import {
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  FileText,
  Folder,
  Loader2,
  Search,
  Sparkles,
} from 'lucide-react'

export type PipelineAccent = 'rag' | 'wiki' | 'agentic-rag'
export type PipelineView = 'chunks' | 'trace'

interface Props {
  title: string
  subtitle: string
  accent: PipelineAccent
  view: PipelineView
  state: PipelineState
  collapsed?: boolean
  onToggleCollapse?: () => void
  rerankMode?: RerankMode
  onRerankChange?: (m: RerankMode) => void
  disableRerankControl?: boolean
}

const toolIcon = (tool: string) => {
  if (tool === 'glob') return <Folder className="h-3.5 w-3.5" />
  if (tool === 'read_file') return <FileText className="h-3.5 w-3.5" />
  if (tool === 'grep') return <Search className="h-3.5 w-3.5" />
  if (tool === 'vector_search') return <Sparkles className="h-3.5 w-3.5" />
  return <ChevronRight className="h-3.5 w-3.5" />
}

const toolLabel = (e: ToolCallEvent | ToolResultEvent) => {
  const arg = Object.values(e.args)[0] ?? ''
  return `${e.tool}(${String(arg).slice(0, 60)})`
}

const chunkKey = (c: Chunk) => `${c.source}#${c.slug}`

export function PipelinePanel({
  title,
  subtitle,
  accent,
  view,
  state,
  collapsed,
  onToggleCollapse,
  rerankMode,
  onRerankChange,
  disableRerankControl,
}: Props) {
  const accentRing =
    accent === 'rag'
      ? 'before:bg-blue-500/80'
      : accent === 'agentic-rag'
        ? 'before:bg-violet-500/80'
        : 'before:bg-emerald-500/80'

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={onToggleCollapse}
        title={`Expand ${title}`}
        className={cn(
          'group relative flex h-full w-12 shrink-0 flex-col items-center justify-between rounded-2xl border border-neutral-200 bg-white py-3.5 transition-colors hover:bg-neutral-50',
          'before:absolute before:left-0 before:top-0 before:h-full before:w-1 before:rounded-l-2xl',
          accentRing,
        )}
      >
        <ChevronRight className="h-4 w-4 text-neutral-400 group-hover:text-neutral-700" />
        <span
          className="text-xs font-semibold tracking-tight text-neutral-700"
          style={{ writingMode: 'vertical-rl' }}
        >
          {title}
        </span>
        {state.status === 'streaming' ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-neutral-400" />
        ) : (
          <span className="h-3.5 w-3.5" />
        )}
      </button>
    )
  }

  const hasReranked = accent === 'rag' && state.rerankedChunks.length > 0
  const initialTop5 = state.chunks.slice(0, 5)

  // Map each reranked chunk to its rank in the initial list, so we can show
  // the movement visually (e.g. "#14 → #1").
  const initialRankByKey = new Map(state.chunks.map((c, i) => [chunkKey(c), i + 1]))

  return (
    <section
      className={cn(
        'relative flex h-full min-w-0 flex-1 flex-col rounded-2xl border border-neutral-200 bg-white',
        'before:absolute before:left-0 before:top-0 before:h-full before:w-1 before:rounded-l-2xl',
        accentRing,
      )}
    >
      <header className="flex items-baseline justify-between border-b border-neutral-100 px-5 py-3.5">
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-semibold tracking-tight text-neutral-900">{title}</h2>
          <p className="mt-0.5 text-xs text-neutral-500">{subtitle}</p>
        </div>
        <div className="flex items-center gap-3">
          {(accent === 'rag' || accent === 'agentic-rag') && onRerankChange && (
            <label className="flex items-center gap-1.5 text-[11px] text-neutral-500">
              <span className="uppercase tracking-wider">rerank</span>
              <select
                value={rerankMode}
                onChange={(e) => onRerankChange(e.target.value as RerankMode)}
                disabled={disableRerankControl}
                className={cn(
                  'rounded-md border border-neutral-200 bg-white px-1.5 py-0.5 text-xs font-medium text-neutral-700',
                  'outline-none focus:border-neutral-400 disabled:opacity-50',
                )}
              >
                <option value="none">none</option>
                <option value="cross-encoder">cross-encoder</option>
                <option value="llm">llm</option>
              </select>
            </label>
          )}
          {state.status === 'streaming' && (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-neutral-400" />
          )}
          {onToggleCollapse && (
            <button
              type="button"
              onClick={onToggleCollapse}
              title={`Collapse ${title}`}
              className="flex h-6 w-6 items-center justify-center rounded-md text-neutral-400 hover:bg-neutral-100 hover:text-neutral-700"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
          )}
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-5 py-4">
        {/* Process trace */}
        {(state.chunks.length > 0 || state.toolEvents.length > 0) && (
          <div className="mb-5 space-y-1.5">
            <div className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
              {view === 'chunks'
                ? hasReranked
                  ? `Retrieved → Reranked (top-${state.rerankedChunks.length} of ${state.chunks.length})`
                  : 'Retrieved chunks'
                : 'Agent trace'}
            </div>

            {view === 'chunks' && !hasReranked &&
              initialTop5.map((c, i) => (
                <div
                  key={i}
                  className="group flex items-center gap-2 rounded-md bg-neutral-50 px-2.5 py-1.5 text-xs"
                >
                  <span className="text-neutral-400">#{i + 1}</span>
                  <span className="font-mono text-neutral-700">{c.source}</span>
                  <span className="text-neutral-400">·</span>
                  <span className="text-neutral-500">score {c.score.toFixed(2)}</span>
                </div>
              ))}

            {view === 'chunks' && hasReranked && (
              <div className="grid grid-cols-[1fr_auto_1fr] gap-2">
                <div className="space-y-1">
                  <div className="text-[10px] uppercase tracking-wider text-neutral-400">
                    initial (top 5 of {state.chunks.length})
                  </div>
                  {initialTop5.map((c, i) => (
                    <div
                      key={`init-${i}`}
                      className="flex items-center gap-1.5 rounded-md bg-neutral-50 px-2 py-1 text-[11px]"
                    >
                      <span className="text-neutral-400">#{i + 1}</span>
                      <span className="truncate font-mono text-neutral-700">
                        {c.source}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="flex items-center justify-center pt-5 text-neutral-300">
                  <ArrowRight className="h-3.5 w-3.5" />
                </div>
                <div className="space-y-1">
                  <div className="text-[10px] uppercase tracking-wider text-neutral-400">
                    reranked top 5
                  </div>
                  {state.rerankedChunks.map((c, i) => {
                    const origRank = initialRankByKey.get(chunkKey(c))
                    const moved = origRank !== undefined && origRank !== i + 1
                    const promoted =
                      origRank !== undefined && origRank > i + 1
                    return (
                      <div
                        key={`rr-${i}`}
                        className={cn(
                          'flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px]',
                          moved
                            ? promoted
                              ? 'bg-emerald-50'
                              : 'bg-amber-50'
                            : 'bg-neutral-50',
                        )}
                      >
                        <span className="text-neutral-400">#{i + 1}</span>
                        <span className="truncate font-mono text-neutral-700">
                          {c.source}
                        </span>
                        {origRank !== undefined && moved && (
                          <span
                            className={cn(
                              'ml-auto font-mono text-[10px]',
                              promoted ? 'text-emerald-600' : 'text-amber-600',
                            )}
                          >
                            ←#{origRank}
                          </span>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {view === 'trace' &&
              (() => {
                const calls = state.toolEvents.filter(
                  (e) => e.event === 'tool_call',
                ) as ToolCallEvent[]
                const results = state.toolEvents.filter(
                  (e) => e.event === 'tool_result',
                ) as ToolResultEvent[]
                return calls.map((e, i) => {
                  const result = results[i]
                  const isVecSearch =
                    e.tool === 'vector_search' && result?.hits && result.hits.length > 0
                  const reranked =
                    isVecSearch &&
                    result?.rerank_mode &&
                    result.rerank_mode !== 'none' &&
                    result.initial_hits
                  return (
                    <div
                      key={i}
                      className="rounded-md bg-neutral-50 px-2.5 py-1.5 font-mono text-xs text-neutral-700"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-neutral-400">{toolIcon(e.tool)}</span>
                        <span className="truncate">{toolLabel(e)}</span>
                      </div>
                      {isVecSearch && !reranked && (
                        <div className="mt-1.5 space-y-0.5 pl-5">
                          {result!.hits!.map((c, j) => (
                            <div
                              key={`hit-${j}`}
                              className="flex items-center gap-1.5 text-[11px]"
                            >
                              <span className="text-neutral-400">#{j + 1}</span>
                              <span className="truncate text-neutral-700">{c.source}</span>
                              <span className="text-neutral-400">·</span>
                              <span className="text-neutral-500">
                                {c.score.toFixed(2)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                      {reranked && (() => {
                        const initialRank = new Map(
                          result!.initial_hits!.map((c, idx) => [c.source, idx + 1]),
                        )
                        return (
                          <div className="mt-1.5 pl-5">
                            <div className="grid grid-cols-[1fr_auto_1fr] gap-2">
                              <div className="space-y-0.5">
                                <div className="text-[10px] uppercase tracking-wider text-neutral-400">
                                  initial top {result!.initial_hits!.length}
                                </div>
                                {result!.initial_hits!.map((c, j) => (
                                  <div
                                    key={`ri-${j}`}
                                    className="flex items-center gap-1 rounded bg-white px-1.5 py-0.5 text-[11px]"
                                  >
                                    <span className="text-neutral-400">#{j + 1}</span>
                                    <span className="truncate text-neutral-700">
                                      {c.source}
                                    </span>
                                  </div>
                                ))}
                              </div>
                              <div className="flex items-center justify-center pt-4 text-neutral-300">
                                <ArrowRight className="h-3.5 w-3.5" />
                              </div>
                              <div className="space-y-0.5">
                                <div className="text-[10px] uppercase tracking-wider text-neutral-400">
                                  reranked top {result!.hits!.length}
                                </div>
                                {result!.hits!.map((c, j) => {
                                  const origRank = initialRank.get(c.source)
                                  const moved = origRank !== undefined && origRank !== j + 1
                                  const promoted =
                                    origRank === undefined || origRank > j + 1
                                  return (
                                    <div
                                      key={`rr-${j}`}
                                      className={cn(
                                        'flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px]',
                                        moved || origRank === undefined
                                          ? promoted
                                            ? 'bg-emerald-50'
                                            : 'bg-amber-50'
                                          : 'bg-white',
                                      )}
                                    >
                                      <span className="text-neutral-400">#{j + 1}</span>
                                      <span className="truncate text-neutral-700">
                                        {c.source}
                                      </span>
                                      {origRank !== undefined && moved && (
                                        <span
                                          className={cn(
                                            'ml-auto text-[10px]',
                                            promoted ? 'text-emerald-600' : 'text-amber-600',
                                          )}
                                        >
                                          ←#{origRank}
                                        </span>
                                      )}
                                      {origRank === undefined && (
                                        <span className="ml-auto text-[10px] text-emerald-600">
                                          new
                                        </span>
                                      )}
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                          </div>
                        )
                      })()}
                      {!isVecSearch && result && (
                        <div className="mt-1 pl-5 text-[11px] text-neutral-500">
                          {result.preview}
                        </div>
                      )}
                    </div>
                  )
                })
              })()}
          </div>
        )}

        {/* Answer */}
        {state.answer && (
          <div className="prose prose-sm max-w-none text-neutral-800">
            <div className="whitespace-pre-wrap leading-relaxed">
              {state.answer}
              {state.status === 'streaming' && (
                <span className="ml-0.5 inline-block h-3.5 w-[2px] animate-pulse bg-neutral-400 align-middle" />
              )}
            </div>
          </div>
        )}

        {state.status === 'idle' && (
          <div className="py-12 text-center text-xs text-neutral-400">
            Awaiting query
          </div>
        )}

        {state.error && (
          <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {state.error}
          </div>
        )}
      </div>

      <footer className="border-t border-neutral-100 px-5 py-2.5 text-[11px] text-neutral-500">
        {state.metrics ? (
          <div className="flex items-center justify-between font-mono">
            <span>
              {state.metrics.t_ms} ms
              {state.metrics.rerank_ms !== undefined && state.metrics.rerank_ms > 0 && (
                <span className="ml-1 text-neutral-400">
                  (rerank {state.metrics.rerank_ms})
                </span>
              )}
            </span>
            <span>
              {state.metrics.in_tokens.toLocaleString()} →{' '}
              {state.metrics.out_tokens.toLocaleString()} tok
            </span>
            <span>${state.metrics.cost_usd.toFixed(5)}</span>
            {state.metrics.turns !== undefined && (
              <span>{state.metrics.turns} turns</span>
            )}
          </div>
        ) : (
          <div className="font-mono text-neutral-300">— · — · —</div>
        )}
      </footer>
    </section>
  )
}
