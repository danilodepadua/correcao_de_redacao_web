import type { EssayResponse } from "./api";

export const MIN_PALAVRAS = 30;

export type CompetenciaKey = Exclude<keyof EssayResponse, "nota_final" | "feedback">;

export const COMPETENCIAS: {
  key: CompetenciaKey;
  numero: number;
  nome: string;
  resumo: string;
}[] = [
  {
    key: "c1_norma_culta",
    numero: 1,
    nome: "Norma culta",
    resumo: "Gramática, ortografia, pontuação e escolha de palavras formais.",
  },
  {
    key: "c2_compreensao_tema",
    numero: 2,
    nome: "Tema e repertório",
    resumo: "Entender a proposta e usar conhecimentos de outras áreas de forma produtiva.",
  },
  {
    key: "c3_projeto_de_texto",
    numero: 3,
    nome: "Argumentação",
    resumo: "Selecionar, organizar e defender informações e opiniões.",
  },
  {
    key: "c4_coesao",
    numero: 4,
    nome: "Coesão",
    resumo: "Conectivos e retomadas que amarram frases e parágrafos.",
  },
  {
    key: "c5_proposta_intervencao",
    numero: 5,
    nome: "Proposta de intervenção",
    resumo: "Solução com agente, ação, meio, finalidade e detalhamento.",
  },
];

export type Tom = "otimo" | "bom" | "medio" | "ruim";

export function contarPalavras(texto: string): number {
  return texto.split(/\s+/).filter(Boolean).length;
}

/** Mesma regra do back-end: cada quebra de linha não vazia é um parágrafo. */
export function separarParagrafos(texto: string): string[] {
  return texto
    .split("\n")
    .map((p) => p.trim())
    .filter(Boolean);
}

export function validarRedacao(tema: string, redacao: string): string | null {
  if (!tema.trim()) return "Escreva o tema da redação.";
  const palavras = contarPalavras(redacao);
  if (palavras < MIN_PALAVRAS)
    return `A redação precisa de pelo menos ${MIN_PALAVRAS} palavras (agora tem ${palavras}).`;
  return null;
}

export function nivelCompetencia(nota: number): { rotulo: string; tom: Tom } {
  if (nota >= 200) return { rotulo: "Excelente", tom: "otimo" };
  if (nota >= 160) return { rotulo: "Bom", tom: "bom" };
  if (nota >= 120) return { rotulo: "Mediano", tom: "medio" };
  if (nota >= 80) return { rotulo: "Insuficiente", tom: "ruim" };
  if (nota >= 40) return { rotulo: "Precário", tom: "ruim" };
  return { rotulo: "Não atendeu", tom: "ruim" };
}

export function nivelFinal(nota: number): { rotulo: string; tom: Tom } {
  if (nota >= 900) return { rotulo: "Excelente", tom: "otimo" };
  if (nota >= 700) return { rotulo: "Bom", tom: "bom" };
  if (nota >= 500) return { rotulo: "Regular", tom: "medio" };
  return { rotulo: "Precisa de atenção", tom: "ruim" };
}

export type FeedbackTipo = "atencao" | "melhorar" | "paragrafo" | "elogio" | "outro";

export type FeedbackItem = {
  tipo: FeedbackTipo;
  titulo: string;
  texto: string;
  competencia?: number;
  nota?: number;
  paragrafo?: number;
};

/** Lê as mensagens de feedback no formato que o main.py gera. */
export function interpretarFeedback(original: string): FeedbackItem {
  const m = original.match(/^\s*\[([^\]]+)\]\s*([\s\S]*)$/);
  if (!m) {
    const elogio = /excelente/i.test(original);
    return { tipo: elogio ? "elogio" : "outro", titulo: elogio ? "Parabéns" : "Observação", texto: original.trim() };
  }
  const tag = m[1] ?? "";
  const corpo = (m[2] ?? "").trim();

  if (/par\S*grafo/i.test(tag)) {
    const num = corpo.match(/(\d+)\s*\S{0,2}\s*par\S*grafo/i);
    const item: FeedbackItem = { tipo: "paragrafo", titulo: "Parágrafo mais fraco", texto: corpo };
    if (num?.[1]) item.paragrafo = Number(num[1]);
    return item;
  }

  const tipo: FeedbackTipo = /aten/i.test(tag) ? "atencao" : /bom/i.test(tag) ? "melhorar" : "outro";
  const nota = tag.match(/(\d+)\s*\/\s*200/);
  const comp = corpo.match(/compet\S*ncia\s*(\d)/i);
  const sep = corpo.indexOf(":");
  const item: FeedbackItem = {
    tipo,
    titulo: sep > 0 ? corpo.slice(0, sep).trim() : tag,
    texto: sep > 0 ? corpo.slice(sep + 1).trim() : corpo,
  };
  if (nota?.[1]) item.nota = Number(nota[1]);
  if (comp?.[1]) item.competencia = Number(comp[1]);
  return item;
}
