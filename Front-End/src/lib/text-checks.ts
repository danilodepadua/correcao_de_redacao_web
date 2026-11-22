import { contarPalavras, separarParagrafos } from "./scoring";

/**
 * Verificações automáticas do texto, independentes da rede neural.
 * Servem para apontar exatamente onde está cada problema.
 */

export type Verificacao = {
  id: string;
  titulo: string;
  ok: boolean;
  detalhe: string;
  competencia: number;
  /** Termos a destacar no texto anotado. */
  termos?: string[];
};

export const CONECTIVOS = [
  "além disso", "ademais", "outrossim", "nesse sentido", "nesse contexto", "dessa forma",
  "desse modo", "sob esse viés", "sob essa ótica", "por conseguinte", "portanto", "logo",
  "assim", "em suma", "por fim", "diante disso", "diante do exposto", "em primeiro lugar",
  "em segundo lugar", "primeiramente", "inicialmente", "contudo", "entretanto", "no entanto",
  "todavia", "por outro lado", "consequentemente", "em síntese", "paralelamente", "nessa perspectiva",
  "com efeito", "posto isso", "isso posto", "dessarte", "destarte", "sob essa perspectiva",
];

const MARCAS_ORALIDADE = ["tipo assim", "né", "a gente", "pra", "tá", "vc", "tb", "aí", "daí", "beleza", "coisa"];
const PRIMEIRA_PESSOA = ["eu", "acho que", "minha opinião", "meu ver", "eu acredito"];

const PALAVRAS_VAZIAS = new Set(
  "porque quando sobre entre mesmo ainda assim pois como para pelo pela pelos pelas sendo ser são está estão isso essa esse esta este essas esses também muito mais menos onde qual quais seus suas dele dela deles delas nossa nosso forma modo sociedade brasil brasileira brasileiro".split(" "),
);

const ELEMENTOS_INTERVENCAO: { nome: string; padrao: RegExp }[] = [
  { nome: "Agente (quem faz)", padrao: /\b(governo|estado|minist[ée]rio|escolas?|m[íi]dia|ongs?|fam[íi]lias?|sociedade civil|poder p[úu]blico|congresso|prefeituras?|empresas?|secretarias?|instituições?)\b/i },
  { nome: "Ação (o que fazer)", padrao: /\b(deve|devem|cabe ao|cabe à|é necess[áa]rio|é preciso|precisa|precisam|compete)\b/i },
  { nome: "Meio (como fazer)", padrao: /\b(por meio d|mediante|atrav[ée]s d|por intermédio d|com o uso d|via )/i },
  { nome: "Finalidade (para quê)", padrao: /\b(a fim de|para que|com o objetivo|com o intuito|com a finalidade|visando|de modo a|com vistas a)/i },
  { nome: "Detalhamento", padrao: /\b(ou seja|isto é|por exemplo|como,|tais como|de modo que|a exemplo d)/i },
];

function escaparRegex(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function encontrarTermos(texto: string, lista: string[]): string[] {
  const achados = new Set<string>();
  for (const termo of lista) {
    const re = new RegExp(`(^|[^\\p{L}])${escaparRegex(termo)}(?=$|[^\\p{L}])`, "iu");
    if (re.test(texto)) achados.add(termo);
  }
  return [...achados];
}

function comecaComConectivo(paragrafo: string): boolean {
  const inicio = paragrafo.slice(0, 70).toLowerCase();
  return CONECTIVOS.some((c) => inicio.includes(c));
}

export function palavrasRepetidas(texto: string, minimo = 4): { palavra: string; vezes: number }[] {
  const contagem = new Map<string, number>();
  for (const bruta of texto.toLowerCase().match(/\p{L}+/gu) ?? []) {
    if (bruta.length < 5 || PALAVRAS_VAZIAS.has(bruta)) continue;
    contagem.set(bruta, (contagem.get(bruta) ?? 0) + 1);
  }
  return [...contagem.entries()]
    .filter(([, n]) => n >= minimo)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([palavra, vezes]) => ({ palavra, vezes }));
}

export function elementosIntervencao(conclusao: string) {
  return ELEMENTOS_INTERVENCAO.map((e) => ({ nome: e.nome, presente: e.padrao.test(conclusao) }));
}

export function analisarTexto(texto: string): Verificacao[] {
  const paragrafos = separarParagrafos(texto);
  const palavras = contarPalavras(texto);
  const resultado: Verificacao[] = [];

  const qtd = paragrafos.length;
  resultado.push({
    id: "paragrafos",
    competencia: 3,
    titulo: "Estrutura em parágrafos",
    ok: qtd >= 4 && qtd <= 5,
    detalhe:
      qtd >= 4 && qtd <= 5
        ? `${qtd} parágrafos: introdução, desenvolvimento e conclusão bem separados.`
        : `${qtd} parágrafo${qtd === 1 ? "" : "s"}. O mais seguro no ENEM é ter de 4 a 5 (introdução, 2 ou 3 de desenvolvimento e conclusão).`,
  });

  resultado.push({
    id: "tamanho",
    competencia: 2,
    titulo: "Tamanho do texto",
    ok: palavras >= 250 && palavras <= 480,
    detalhe:
      palavras < 250
        ? `${palavras} palavras. Textos curtos costumam desenvolver pouco os argumentos; mire em 300 a 400.`
        : palavras > 480
          ? `${palavras} palavras. Cuidado para não passar das 30 linhas da folha oficial.`
          : `${palavras} palavras, dentro do esperado para 30 linhas.`,
  });

  if (qtd >= 2) {
    const semConectivo = paragrafos
      .map((p, i) => ({ p, i }))
      .slice(1)
      .filter(({ p }) => !comecaComConectivo(p))
      .map(({ i }) => i + 1);
    resultado.push({
      id: "conectivos",
      competencia: 4,
      titulo: "Conectivos entre parágrafos",
      ok: semConectivo.length === 0,
      detalhe:
        semConectivo.length === 0
          ? "Todos os parágrafos começam ligados ao anterior."
          : `Parágrafo${semConectivo.length > 1 ? "s" : ""} ${semConectivo.join(", ")} começa${semConectivo.length > 1 ? "m" : ""} sem conectivo. Use "Além disso", "Nesse sentido", "Portanto"...`,
    });
  }

  const repetidas = palavrasRepetidas(texto);
  resultado.push({
    id: "repeticao",
    competencia: 4,
    titulo: "Repetição de palavras",
    ok: repetidas.length === 0,
    detalhe:
      repetidas.length === 0
        ? "Nenhuma palavra repetida em excesso."
        : `Repetidas demais: ${repetidas.map((r) => `"${r.palavra}" (${r.vezes}×)`).join(", ")}. Troque por sinônimos ou pronomes.`,
    termos: repetidas.map((r) => r.palavra),
  });

  const oralidade = encontrarTermos(texto, MARCAS_ORALIDADE);
  resultado.push({
    id: "oralidade",
    competencia: 1,
    titulo: "Marcas de oralidade",
    ok: oralidade.length === 0,
    detalhe:
      oralidade.length === 0
        ? "Linguagem formal, sem gírias ou abreviações."
        : `Encontrado: ${oralidade.map((t) => `"${t}"`).join(", ")}. Prefira a linguagem formal.`,
    termos: oralidade,
  });

  const pessoal = encontrarTermos(texto, PRIMEIRA_PESSOA);
  resultado.push({
    id: "impessoalidade",
    competencia: 1,
    titulo: "Impessoalidade",
    ok: pessoal.length === 0,
    detalhe:
      pessoal.length === 0
        ? "Texto impessoal, sem primeira pessoa do singular."
        : `Encontrado: ${pessoal.map((t) => `"${t}"`).join(", ")}. Evite opinar em primeira pessoa.`,
    termos: pessoal,
  });

  const conclusao = paragrafos.at(-1) ?? "";
  const elementos = elementosIntervencao(conclusao);
  const faltando = elementos.filter((e) => !e.presente).map((e) => e.nome);
  resultado.push({
    id: "intervencao",
    competencia: 5,
    titulo: "Elementos da proposta de intervenção",
    ok: faltando.length === 0,
    detalhe:
      faltando.length === 0
        ? "A conclusão traz agente, ação, meio, finalidade e detalhamento."
        : `Não encontrei na conclusão: ${faltando.join("; ")}.`,
  });

  return resultado;
}
