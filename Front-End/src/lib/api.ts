/** Cliente da API Python (FastAPI) do time. Todas as notas vêm do modelo de lá. */

export type EssayRequest = { tema: string; redacao: string };

export type EssayResponse = {
  c1_norma_culta: number;
  c2_compreensao_tema: number;
  c3_projeto_de_texto: number;
  c4_coesao: number;
  c5_proposta_intervencao: number;
  nota_final: number;
  feedback: string[];
};

export type RewriteResponse = { redacao_reescrita: string };

export type HealthResponse = { status: string; device: string };

const STORAGE_KEY = "notamil.apiUrl";
const CHANGE_EVENT = "notamil:api-url";

export const DEFAULT_API_URL: string =
  (import.meta.env["VITE_API_URL"] as string | undefined) ?? "http://localhost:8000";

export function normalizarUrl(url: string): string {
  return url.trim().replace(/\/+$/, "");
}

export function getApiUrl(): string {
  if (typeof window === "undefined") return DEFAULT_API_URL;
  return window.localStorage.getItem(STORAGE_KEY) || DEFAULT_API_URL;
}

export function setApiUrl(url: string) {
  const limpa = normalizarUrl(url);
  if (!limpa || limpa === DEFAULT_API_URL) window.localStorage.removeItem(STORAGE_KEY);
  else window.localStorage.setItem(STORAGE_KEY, limpa);
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

export function onApiUrlChange(cb: () => void) {
  window.addEventListener(CHANGE_EVENT, cb);
  return () => window.removeEventListener(CHANGE_EVENT, cb);
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const base = normalizarUrl(getApiUrl());
  let res: Response;
  try {
    res = await fetch(`${base}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      `Não foi possível falar com o servidor em ${base}. Confira se o back-end está rodando.`,
      0,
    );
  }
  if (!res.ok) {
    let detalhe = `O servidor respondeu com erro ${res.status}.`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detalhe = body.detail;
    } catch {
      /* corpo sem JSON */
    }
    throw new ApiError(detalhe, res.status);
  }
  return (await res.json()) as T;
}

export function avaliarRedacao(payload: EssayRequest) {
  return request<EssayResponse>("/api/v1/avaliar", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function reescreverRedacao(payload: EssayRequest) {
  return request<RewriteResponse>("/api/v1/reescrever", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function verificarServidor() {
  return request<HealthResponse>("/");
}
