import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { AlertTriangle, ArrowRight, Copy, Loader2, RefreshCcw, Wand2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { ApiError, reescreverRedacao, type Notas, type RewriteResponse } from "@/lib/api";
import { COMPETENCIAS } from "@/lib/scoring";

function Diferenca({ antes, depois }: { antes: number; depois: number }) {
  const d = depois - antes;
  if (d === 0) return <span className="text-muted-foreground">=</span>;
  return (
    <span className={d > 0 ? "font-semibold text-emerald-700 dark:text-emerald-400" : "font-semibold text-pen"}>
      {d > 0 ? `+${d}` : d}
    </span>
  );
}

function Comparacao({ antes, depois }: { antes: Notas; depois: Notas }) {
  return (
    <div className="rounded-md border border-ink/15 bg-background/60 p-4">
      <p className="text-sm font-medium">Nota estimada pelo modelo</p>
      <p className="mt-1 flex items-baseline gap-3 text-3xl font-semibold">
        <span>{antes.nota_final}</span>
        <ArrowRight className="size-5 self-center text-muted-foreground" />
        <span>{depois.nota_final}</span>
        <span className="text-base">
          <Diferenca antes={antes.nota_final} depois={depois.nota_final} />
        </span>
      </p>
      <ul className="mt-3 grid gap-1 text-sm sm:grid-cols-5">
        {COMPETENCIAS.map((c) => (
          <li key={c.key} className="flex justify-between gap-2 sm:flex-col sm:gap-0">
            <span className="text-muted-foreground">
              C{c.numero} {c.nome}
            </span>
            <span>
              {antes[c.key]} → {depois[c.key]} <Diferenca antes={antes[c.key]} depois={depois[c.key]} />
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function RewritePanel({
  tema,
  redacao,
  onAvaliar,
}: {
  tema: string;
  redacao: string;
  onAvaliar: (texto: string) => void;
}) {
  const [resultado, setResultado] = useState<RewriteResponse | null>(null);
  const mut = useMutation({
    mutationFn: () => reescreverRedacao({ tema, redacao }),
    onSuccess: (r) => setResultado(r),
  });

  const texto = resultado?.redacao_reescrita ?? null;
  const erro = mut.error;
  const naoExiste = erro instanceof ApiError && (erro.status === 404 || erro.status === 405);

  return (
    <div className="rounded-md border border-dashed border-ink/30 bg-card p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-xl">
          <h3 className="text-2xl font-semibold">Refazer a redação</h3>
          <p className="mt-1 text-muted-foreground">
            Gera uma nova versão do seu texto corrigindo os pontos apontados, mantendo suas ideias.
          </p>
        </div>
        <Button variant="pen" onClick={() => mut.mutate()} disabled={mut.isPending}>
          {mut.isPending ? <Loader2 className="animate-spin" /> : <Wand2 />}
          {texto ? "Gerar outra versão" : "Reescrever"}
        </Button>
      </div>

      {erro && (
        <p className="mt-4 rounded-md bg-pen-soft p-3 text-sm text-pen">
          {naoExiste
            ? "O servidor ainda não tem a função de reescrever. Ela precisa existir no back-end como POST /api/v1/reescrever."
            : erro.message}
        </p>
      )}

      {texto && (
        <div className="mt-6 space-y-4">
          {resultado?.notas_original && resultado.notas_reescrita && (
            <Comparacao antes={resultado.notas_original} depois={resultado.notas_reescrita} />
          )}
          <p className="flex gap-2 rounded-md border border-amber-500/40 bg-amber-50 p-3 text-sm text-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <span>
              Esta versão foi gerada por uma IA (Gemini). Confira citações, dados e fatos históricos
              antes de usar: o modelo pode errar o repertório. A nota acima é só uma estimativa do
              nosso modelo e não garante a nota real.
            </span>
          </p>
          <div className="ruled-paper rounded-md whitespace-pre-line">{texto}</div>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              onClick={() => {
                void navigator.clipboard.writeText(texto);
                toast.success("Texto copiado");
              }}
            >
              <Copy /> Copiar
            </Button>
            <Button variant="paper" onClick={() => onAvaliar(texto)}>
              <RefreshCcw /> Corrigir esta versão
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
