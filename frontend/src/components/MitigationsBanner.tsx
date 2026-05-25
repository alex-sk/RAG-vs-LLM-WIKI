import { useState } from 'react'
import { Info, X } from 'lucide-react'

const STORAGE_KEY = 'mitigations-banner-dismissed-v1'

export function MitigationsBanner() {
  const [dismissed, setDismissed] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem(STORAGE_KEY) === '1'
  })

  if (dismissed) return null

  const dismiss = () => {
    window.localStorage.setItem(STORAGE_KEY, '1')
    setDismissed(true)
  }

  return (
    <aside className="relative rounded-xl border border-amber-200 bg-amber-50/60 px-5 py-4 text-[13px] leading-relaxed text-neutral-700">
      <div className="flex items-start gap-3">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
        <div className="flex-1">
          <p className="font-medium text-neutral-900">
            This PoC compares generation/retrieval styles, not production security.
          </p>
          <p className="mt-1.5">
            Before deploying either pipeline over untrusted content, both would need:
          </p>
          <ul className="mt-1.5 space-y-1 pl-4 [&>li]:list-disc [&>li]:marker:text-neutral-400">
            <li>
              <span className="font-medium text-neutral-900">Source/instruction separation</span>{' '}
              — wrap retrieved content in delimiters the model treats as data, not instructions
            </li>
            <li>
              <span className="font-medium text-neutral-900">Content sanitisation at index time</span>{' '}
              — detect/strip injection patterns on ingest
            </li>
            <li>
              <span className="font-medium text-neutral-900">Provenance + refusal</span>{' '}
              — no answer without a citable source; flag unexpected topic shifts
            </li>
            <li>
              <span className="font-medium text-neutral-900">Red-team eval set</span>{' '}
              — purposeful injection attempts in the test corpus so regressions are caught
            </li>
          </ul>
          <p className="mt-2">
            The agent pipeline (right) has extra surface: tool calls are an action API.
            Production would require sandboxing, allowlisted tools/arguments, and per-tool authorisation.
          </p>
        </div>
        <button
          onClick={dismiss}
          aria-label="Dismiss"
          className="-mr-1 -mt-1 rounded p-1 text-neutral-400 transition-colors hover:bg-amber-100 hover:text-neutral-700"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </aside>
  )
}
