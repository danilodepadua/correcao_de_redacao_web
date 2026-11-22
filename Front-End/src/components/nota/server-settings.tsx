import { useEffect, useState } from "react";
import { Server, Loader2, CheckCircle2, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { DEFAULT_API_URL, setApiUrl, verificarServidor } from "@/lib/api";
import { useApiUrl } from "@/hooks/use-api-url";

type Estado = { tipo: "idle" } | { tipo: "testando" } | { tipo: "ok"; device: string } | { tipo: "erro"; msg: string };

export function ServerSettings() {
  const url = useApiUrl();
  const [aberto, setAberto] = useState(false);
  const [valor, setValor] = useState(url);
  const [estado, setEstado] = useState<Estado>({ tipo: "idle" });

  useEffect(() => {
    if (aberto) {
      setValor(url);
      setEstado({ tipo: "idle" });
    }
  }, [aberto, url]);

  async function testar() {
    setApiUrl(valor);
    setEstado({ tipo: "testando" });
    try {
      const r = await verificarServidor();
      setEstado({ tipo: "ok", device: r.device });
    } catch (e) {
      setEstado({ tipo: "erro", msg: e instanceof Error ? e.message : "Erro desconhecido" });
    }
  }

  return (
    <Dialog open={aberto} onOpenChange={setAberto}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-2 text-muted-foreground">
          <Server />
          <span className="hidden sm:inline">Servidor</span>
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="font-display text-2xl">Servidor de correção</DialogTitle>
          <DialogDescription>
            Endereço onde o back-end em Python (FastAPI) está rodando. Para testar no seu computador,
            rode <code className="rounded bg-muted px-1">python main.py</code> na pasta Back-End.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          <Label htmlFor="api-url">Endereço</Label>
          <Input
            id="api-url"
            value={valor}
            onChange={(e) => setValor(e.target.value)}
            placeholder={DEFAULT_API_URL}
          />
          <p className="text-xs text-muted-foreground">Padrão: {DEFAULT_API_URL}</p>
        </div>
        {estado.tipo === "ok" && (
          <p className="flex items-center gap-2 text-sm text-score-great">
            <CheckCircle2 className="size-4" /> Servidor online ({estado.device}).
          </p>
        )}
        {estado.tipo === "erro" && (
          <p className="flex items-start gap-2 text-sm text-pen">
            <XCircle className="mt-0.5 size-4 shrink-0" /> {estado.msg}
          </p>
        )}
        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={testar} disabled={estado.tipo === "testando"}>
            {estado.tipo === "testando" && <Loader2 className="animate-spin" />}
            Testar conexão
          </Button>
          <Button
            onClick={() => {
              setApiUrl(valor);
              setAberto(false);
            }}
          >
            Salvar
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
