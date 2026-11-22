import { useMutation } from "@tanstack/react-query";
import { Loader2, ScanSearch } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ApiError, apontarProblemas } from "@/lib/api";
import { COMPETENCIAS } from "@/lib/scoring";

/** Trechos com problema apontados pelo Gemini, sob demanda. */
export function PointersPanel({ tema, redacao }: { tema: string; redacao: string }) {
  const mut = useMutation({ mutationFn: () => apontarProblemas({ tema, redacao }) });
  const dados = mut.data;
  const erro = mut.error;
  const naoExiste = erro instanceof ApiError && (erro.status === 404 || erro.status === 405);

  return (
    <div>
      <p className="eyebrow">Análise por IA</p>
      <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
        <div className="max-w-2xl">
          <h2 className="text-3xl font-semibold">Trechos com problema</h2>
          <p className="mt-2 text-muted-foreground">
            O Gemini lê a redação e aponta trechos exatos que fazem perder pontos, com a
            competência afetada e uma sugestão. Só aparecem trechos que existem de fato no seu texto.
          </p>
        </div>
        <Button variant="outline" onClick={() => mut.mutate()} disabled={mut.isPending}>
          {mut.isPending ? <Loader2 className="animate-spin" /> : <ScanSearch />}
          {dados ? "Analisar de novo" : "Apontar trechos"}
        </Button>
      </div>

      {erro && (
        <p className="mt-4 rounded-md bg-pen-soft p-3 text-sm text-pen">
          {naoExiste
            ? "O servidor ainda não tem a função de apontar trechos (POST /api/v1/apontar)."
            : erro.message}
        </p>
      )}

      {dados && dados.apontamentos.length === 0 && (
        <p className="mt-6 text-muted-foreground">Nenhum trecho problemático foi apontado.</p>
      )}

      {dados && dados.apontamentos.length > 0 && (
        <ol className="mt-6 space-y-4">
          {dados.apontamentos.map((a, i) => {
            const comp = COMPETENCIAS.find((c) => c.numero === a.competencia);
            return (
              <li key={i} className="rounded-md border border-ink/15 bg-card p-4">
                <p className="text-sm font-medium text-pen">
                  C{a.competencia}
                  {comp ? ` · ${comp.nome}` : ""}
                </p>
                <blockquote className="mt-2 border-l-2 border-pen/60 pl-3 font-serif italic">
                  “{a.trecho}”
                </blockquote>
                <p className="mt-2">{a.problema}</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  <span className="font-medium text-foreground">Sugestão:</span> {a.sugestao}
                </p>
              </li>
            );
          })}
        </ol>
      )}

      {dados && (
        <p className="mt-4 text-xs text-muted-foreground">
          Apontamentos gerados por IA: confira antes de seguir, especialmente sugestões de dados ou
          citações.
          {dados.descartados > 0 &&
            ` ${dados.descartados} apontamento(s) foram descartados por citar trechos que não estão no texto.`}
        </p>
      )}
    </div>
  );
}
