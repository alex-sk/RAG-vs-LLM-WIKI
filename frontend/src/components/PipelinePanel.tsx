import { cn } from '@/lib/utils'
import type { PipelineState, ToolCallEvent, ToolResultEvent } from '@/types'
import { ChevronRight, FileText, Search, Folder, Loader2 } from 'lucide-react'

interface Props {
  title: string
  subtitle: string
  accent: 'rag' | 'wiki'
  state: PipelineState
}

const toolIcon = (tool: string) => {
  if (tool === 'glob') return <Folder className="h-3.5 w-3.5" />
  if (tool === 'read_file') return <FileText className="h-3.5 w-3.5" />
  if (tool === 'grep') return <Search className="h-3.5 w-3.5" />
  return <ChevronRight className="h-3.5 w-3.5" />
}

const toolLabel = (e: ToolCallEvent | ToolResultEvent) => {
  const arg = Object.values(e.args)[0] ?? ''
  return `${e.tool}(${String(arg).slice(0, 60)})`
}

export function PipelinePanel({ title, subtitle, accent, state }: Props) {
  const accentRing =
    accent === 'rag'
      ? 'before:bg-blue-500/80'
      : 'before:bg-emerald-500/80'

  return (
    <section
      className={cn(
        'relative flex h-full flex-col rounded-2xl border border-neutral-200 bg-white',
        'before:absolute before:left-0 before:top-0 before:h-full before:w-1 before:rounded-l-2xl',
        accentRing,
      )}
    >
      <header className="flex items-baseline justify-between border-b border-neutral-100 px-5 py-3.5">
        <div>
          <h2 className="text-sm font-semibold tracking-tight text-neutral-900">{title}</h2>
          <p className="mt-0.5 text-xs text-neutral-500">{subtitle}</p>
        </div>
        {state.status === 'streaming' && (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-neutral-400" />
        )}
      </header>

      <div className="flex-1 overflow-y-auto px-5 py-4">
        {/* Process trace */}
        {(state.chunks.length > 0 || state.toolEvents.length > 0) && (
          <div className="mb-5 space-y-1.5">
            <div className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
              {accent === 'rag' ? 'Retrieved chunks' : 'Agent trace'}
            </div>

            {accent === 'rag' &&
              state.chunks.map((c, i) => (
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

            {accent === 'wiki' &&
              state.toolEvents
                .filter((e) => e.event === 'tool_call')
                .map((e, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 rounded-md bg-neutral-50 px-2.5 py-1.5 font-mono text-xs text-neutral-700"
                  >
                    <span className="text-neutral-400">{toolIcon(e.tool)}</span>
                    <span className="truncate">{toolLabel(e)}</span>
                  </div>
                ))}
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
            <span>{state.metrics.t_ms} ms</span>
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
