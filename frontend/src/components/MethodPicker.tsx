import { cn } from '@/lib/utils'
import { ALL_PIPELINE_KEYS, type PipelineKey } from '@/types'

const STORAGE_KEY = 'rag-vs-wiki:visible-methods'

const META: Record<
  PipelineKey,
  { label: string; activeClass: string; idleClass: string }
> = {
  rag: {
    label: 'RAG',
    activeClass: 'border-blue-500 bg-blue-500/10 text-blue-900',
    idleClass: 'border-blue-200 text-blue-700/70 hover:bg-blue-50',
  },
  agenticRag: {
    label: 'Agentic RAG',
    activeClass: 'border-violet-500 bg-violet-500/10 text-violet-900',
    idleClass: 'border-violet-200 text-violet-700/70 hover:bg-violet-50',
  },
  wiki: {
    label: 'LLM Wiki',
    activeClass: 'border-emerald-500 bg-emerald-500/10 text-emerald-900',
    idleClass: 'border-emerald-200 text-emerald-700/70 hover:bg-emerald-50',
  },
  graphRag: {
    label: 'Graph RAG',
    activeClass: 'border-amber-500 bg-amber-500/10 text-amber-900',
    idleClass: 'border-amber-200 text-amber-700/70 hover:bg-amber-50',
  },
}

interface Props {
  visible: PipelineKey[]
  onChange: (next: PipelineKey[]) => void
  disabled?: boolean
}

export function MethodPicker({ visible, onChange, disabled }: Props) {
  const isOn = (k: PipelineKey) => visible.includes(k)

  const toggle = (k: PipelineKey) => {
    const next = isOn(k)
      ? visible.filter((x) => x !== k)
      : [...visible, k]
    // Always keep at least one method visible.
    if (next.length === 0) return
    onChange(next)
  }

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="text-[11px] uppercase tracking-wider text-neutral-400">
        compare
      </span>
      {ALL_PIPELINE_KEYS.map((k) => {
        const m = META[k]
        const on = isOn(k)
        return (
          <button
            key={k}
            type="button"
            onClick={() => toggle(k)}
            disabled={disabled}
            className={cn(
              'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-medium transition-colors',
              on ? m.activeClass : 'bg-white ' + m.idleClass,
              disabled && 'cursor-not-allowed opacity-60',
            )}
          >
            <span
              className={cn(
                'h-1.5 w-1.5 rounded-full',
                on ? 'bg-current' : 'bg-current/40',
              )}
            />
            {m.label}
          </button>
        )
      })}
    </div>
  )
}

export function loadVisibleMethods(defaults: PipelineKey[]): PipelineKey[] {
  if (typeof window === 'undefined') return defaults
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return defaults
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return defaults
    const filtered = parsed.filter((x): x is PipelineKey =>
      ALL_PIPELINE_KEYS.includes(x as PipelineKey),
    )
    return filtered.length > 0 ? filtered : defaults
  } catch {
    return defaults
  }
}

export function persistVisibleMethods(visible: PipelineKey[]): void {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(visible))
}
