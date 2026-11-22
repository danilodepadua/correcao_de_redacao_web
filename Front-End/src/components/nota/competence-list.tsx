import type { EssayResponse } from "@/lib/api";
import { COMPETENCIAS, nivelCompetencia } from "@/lib/scoring";
import { cn } from "@/lib/utils";
import { TOM_FUNDO, TOM_TEXTO } from "./tone";

export function CompetenceList({ resultado }: { resultado: Omit<EssayResponse, "feedback"> }) {
  return (
    <ol className="divide-y divide-border">
      {COMPETENCIAS.map((c, i) => {
        const nota = resultado[c.key];
        const nivel = nivelCompetencia(nota);
        return (
          <li key={c.key} className="animate-rise py-4" style={{ animationDelay: `${i * 80}ms` }}>
            <div className="flex items-baseline justify-between gap-4">
              <div className="flex items-baseline gap-3">
                <span className="font-display text-sm text-muted-foreground">C{c.numero}</span>
                <span className="font-display text-lg font-medium">{c.nome}</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className={cn("text-xs font-semibold uppercase tracking-wider", TOM_TEXTO[nivel.tom])}>
                  {nivel.rotulo}
                </span>
                <span className="font-display text-2xl font-semibold tabular-nums">{nota}</span>
                <span className="text-xs text-muted-foreground">/200</span>
              </div>
            </div>
            <div className="mt-2 grid grid-cols-5 gap-1" aria-hidden>
              {[40, 80, 120, 160, 200].map((passo) => (
                <div
                  key={passo}
                  className={cn("h-1.5 rounded-full", nota >= passo ? TOM_FUNDO[nivel.tom] : "bg-muted")}
                />
              ))}
            </div>
            <p className="mt-2 text-sm text-muted-foreground">{c.resumo}</p>
          </li>
        );
      })}
    </ol>
  );
}
