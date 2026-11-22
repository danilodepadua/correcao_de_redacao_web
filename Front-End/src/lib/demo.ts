import type { EssayResponse } from "./api";
import { analisarTexto } from "./text-checks";
import { separarParagrafos } from "./scoring";

/**
 * Resultado de DEMONSTRAÇÃO para ver a tela sem o servidor ligado.
 * As notas são estimadas pelas verificações simples do texto, não pela rede neural.
 */
export function resultadoDemonstracao(redacao: string): EssayResponse {
  const checks = analisarTexto(redacao);
  const falhas = (comp: number) => checks.filter((c) => c.competencia === comp && !c.ok).length;
  const nota = (comp: number) => Math.max(40, 200 - falhas(comp) * 40 - (comp === 2 || comp === 3 ? 40 : 0));
  const n = {
    c1_norma_culta: nota(1),
    c2_compreensao_tema: nota(2),
    c3_projeto_de_texto: nota(3),
    c4_coesao: nota(4),
    c5_proposta_intervencao: nota(5),
  };
  const nota_final = Object.values(n).reduce((a, b) => a + b, 0);
  const feedback: string[] = [];
  if (n.c1_norma_culta <= 160)
    feedback.push(`[${n.c1_norma_culta <= 120 ? "Atenção" : "Bom, mas pode melhorar"} - ${n.c1_norma_culta}/200] Competência 1 (Norma Culta): Revise marcas de oralidade e o uso da primeira pessoa.`);
  if (n.c4_coesao <= 160)
    feedback.push(`[${n.c4_coesao <= 120 ? "Atenção" : "Bom, mas pode melhorar"} - ${n.c4_coesao}/200] Competência 4 (Coesão): Faltam conectivos interparágrafos ou há repetição de palavras.`);
  if (n.c5_proposta_intervencao <= 160)
    feedback.push(`[${n.c5_proposta_intervencao <= 120 ? "Atenção" : "Bom, mas pode melhorar"} - ${n.c5_proposta_intervencao}/200] Competência 5 (Proposta de Intervenção): Garanta Agente, Ação, Meio/Modo, Efeito e Detalhamento.`);
  const ps = separarParagrafos(redacao);
  if (ps.length >= 2) {
    const menor = ps.reduce((idx, p, i, arr) => (p.length < (arr[idx]?.length ?? Infinity) ? i : idx), 0);
    feedback.push(`[Análise de Parágrafo] O ${menor + 1}º parágrafo ("${(ps[menor] ?? "").slice(0, 80)}...") apresentou menor impacto na pontuação (demonstração).`);
  }
  if (feedback.length === 0) feedback.push("Excelente desempenho! Todas as competências atingiram a nota máxima.");
  return { ...n, nota_final, feedback };
}
