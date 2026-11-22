import { Check, X } from "lucide-react";
import type { Verificacao } from "@/lib/text-checks";
import { cn } from "@/lib/utils";

export function ChecksList({ itens }: { itens: Verificacao[] }) {
  return (
    <ul className="grid gap-3 sm:grid-cols-2">
      {itens.map((v) => (
        <li key={v.id} className="rounded-md border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "grid size-5 place-items-center rounded-full",
                v.ok ? "bg-score-great text-pen-foreground" : "bg-pen text-pen-foreground",
              )}
            >
              {v.ok ? <Check className="size-3" /> : <X className="size-3" />}
            </span>
            <span className="font-medium">{v.titulo}</span>
            <span className="ml-auto text-xs text-muted-foreground">C{v.competencia}</span>
          </div>
          <p className="mt-2 text-sm text-muted-foreground">{v.detalhe}</p>
        </li>
      ))}
    </ul>
  );
}
