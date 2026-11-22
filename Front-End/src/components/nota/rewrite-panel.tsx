import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Copy, Loader2, RefreshCcw, Wand2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { ApiError, reescreverRedacao } from "@/lib/api";

export function RewritePanel({
  tema,
  redacao,
  onAvaliar,
}: {
  tema: string;
  redacao: string;
  onAvaliar: (texto: string) => void;
}) {
  const [texto, setTexto] = useState<string | null>(null);
  const mut = useMutation({
    mutationFn: () => reescreverRedacao({ tema, redacao }),
    onSuccess: (r) => setTexto(r.redacao_reescrita),
  });

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
