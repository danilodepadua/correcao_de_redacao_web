import { AlertTriangle, PenLine, Pilcrow, Sparkles, Info } from "lucide-react";
import type { FeedbackItem } from "@/lib/scoring";
import { cn } from "@/lib/utils";

const ESTILO: Record<FeedbackItem["tipo"], { icone: typeof Info; cor: string; rotulo: string }> = {
  atencao: { icone: AlertTriangle, cor: "text-score-low", rotulo: "Atenção" },
  melhorar: { icone: PenLine, cor: "text-score-mid", rotulo: "Pode melhorar" },
  paragrafo: { icone: Pilcrow, cor: "text-pen", rotulo: "Parágrafo" },
  elogio: { icone: Sparkles, cor: "text-score-great", rotulo: "Muito bem" },
  outro: { icone: Info, cor: "text-muted-foreground", rotulo: "Observação" },
};

export function FeedbackList({ itens }: { itens: FeedbackItem[] }) {
  return (
    <ul className="space-y-4">
      {itens.map((f, i) => {
        const e = ESTILO[f.tipo];
        const Icone = e.icone;
        return (
          <li key={i} className="flex gap-3 border-l-2 border-pen/40 pl-4">
            <Icone className={cn("mt-1 size-4 shrink-0", e.cor)} />
            <div>
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span className={cn("text-xs font-semibold uppercase tracking-wider", e.cor)}>{e.rotulo}</span>
                <span className="font-display font-medium">{f.titulo}</span>
                {f.nota !== undefined && (
                  <span className="text-xs text-muted-foreground tabular-nums">{f.nota}/200</span>
                )}
              </div>
              <p className="mt-1 font-serif text-[1.05rem] leading-relaxed text-foreground/85">{f.texto}</p>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
