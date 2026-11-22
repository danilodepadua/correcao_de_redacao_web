import type { EssayResponse } from "@/lib/api";

export type Exemplo = {
  id: string;
  qualidade: "boa" | "mediana" | "fraca";
  titulo: string;
  tema: string;
  redacao: string;
  notas: Omit<EssayResponse, "feedback">;
  comentarios: string[];
};

/** Redações ilustrativas escritas para o site, com notas de referência pela grade do ENEM. */
export const EXEMPLOS: Exemplo[] = [
  {
    id: "leitura-boa",
    qualidade: "boa",
    titulo: "Repertório, conectivos e proposta completa",
    tema: "Desafios para a formação de leitores no Brasil",
    redacao: `Em "Fahrenheit 451", Ray Bradbury descreve uma sociedade em que os livros são queimados e o pensamento crítico é substituído pelo entretenimento imediato. Embora a obra seja ficcional, ela dialoga com a realidade brasileira, na qual a formação de leitores ainda enfrenta obstáculos significativos. Nesse contexto, destacam-se a desigualdade no acesso aos livros e a ausência de estímulo à leitura no ambiente escolar.

Em primeiro lugar, é importante considerar que o acesso ao livro no país é desigual. Segundo a pesquisa Retratos da Leitura no Brasil, grande parte dos municípios não possui bibliotecas públicas em funcionamento, o que restringe o contato da população mais pobre com obras literárias. Dessa forma, o hábito da leitura torna-se privilégio de poucos, contrariando o direito à educação previsto na Constituição Federal de 1988.

Além disso, a escola, muitas vezes, trata a leitura apenas como obrigação avaliativa. Conforme defendia o educador Paulo Freire, "a leitura do mundo precede a leitura da palavra", isto é, o estudante precisa enxergar sentido naquilo que lê. Quando os textos são impostos sem conexão com a vivência dos alunos, cria-se uma relação de rejeição que se prolonga na vida adulta.

Portanto, é necessário enfrentar esses desafios. Para isso, o Ministério da Educação deve ampliar a rede de bibliotecas escolares e comunitárias, por meio de investimentos específicos e de parcerias com prefeituras, a fim de democratizar o acesso aos livros. Ademais, as escolas devem promover clubes de leitura com obras escolhidas pelos próprios estudantes, de modo que a leitura deixe de ser obrigação e se torne prazer. Assim, o Brasil se afastará da distopia de Bradbury.`,
    notas: {
      c1_norma_culta: 200,
      c2_compreensao_tema: 200,
      c3_projeto_de_texto: 160,
      c4_coesao: 200,
      c5_proposta_intervencao: 200,
      nota_final: 960,
    },
    comentarios: [
      "Abre com repertório legitimado (Bradbury) ligado diretamente ao tema.",
      "Todo parágrafo começa com conectivo: “Em primeiro lugar”, “Além disso”, “Portanto”.",
      "A proposta tem agente, ação, meio, finalidade e detalhamento.",
      "Ponto de melhora: o segundo argumento poderia trazer um dado concreto.",
    ],
  },
  {
    id: "leitura-mediana",
    qualidade: "mediana",
    titulo: "Estrutura correta, argumentos rasos",
    tema: "Desafios para a formação de leitores no Brasil",
    redacao: `A leitura é muito importante para a sociedade. No Brasil muitas pessoas não leem livros e isso é um problema muito grande que precisa ser resolvido. Existem vários motivos para isso acontecer.

Um dos motivos é que os livros são caros. Muitas pessoas não tem dinheiro para comprar livros e por isso não leem. Os livros no Brasil são muito caros e as pessoas preferem gastar com outras coisas.

Outro motivo é a tecnologia. Hoje em dia as pessoas ficam muito tempo no celular e nas redes sociais e não tem tempo para ler livros. A tecnologia atrapalha muito a leitura dos jovens.

Portanto, o governo deve fazer algo para melhorar a leitura no Brasil, para que as pessoas leiam mais livros.`,
    notas: {
      c1_norma_culta: 120,
      c2_compreensao_tema: 120,
      c3_projeto_de_texto: 120,
      c4_coesao: 120,
      c5_proposta_intervencao: 80,
      nota_final: 560,
    },
    comentarios: [
      "Não há repertório: as afirmações não são sustentadas por dados ou referências.",
      "Repete “livros”, “muito” e “pessoas” várias vezes.",
      "Erro de concordância: “as pessoas não tem” → “não têm”.",
      "Proposta vaga: “o governo deve fazer algo” não diz o quê, nem como.",
    ],
  },
  {
    id: "leitura-fraca",
    qualidade: "fraca",
    titulo: "Oralidade e fuga parcial do tema",
    tema: "Desafios para a formação de leitores no Brasil",
    redacao: `Eu acho que ler é legal mas a gente não tem tempo pra isso né. Hoje em dia tudo é no celular, tipo assim, ninguém mais pega um livro de papel e eu também não pego. Na escola os professores mandam ler uns livros chatos que ninguém entende e aí ninguém lê mesmo. Eu prefiro ver vídeo que é mais rápido e da pra aprender as coisas também, então não acho que é um problema tão grande assim.`,
    notas: {
      c1_norma_culta: 40,
      c2_compreensao_tema: 40,
      c3_projeto_de_texto: 40,
      c4_coesao: 40,
      c5_proposta_intervencao: 0,
      nota_final: 160,
    },
    comentarios: [
      "Um único parágrafo, sem introdução, desenvolvimento e conclusão.",
      "Marcas de oralidade: “a gente”, “pra”, “né”, “tipo assim”, “aí”.",
      "Escrito em primeira pessoa e com opinião sem argumento.",
      "Não há proposta de intervenção, o que zera a Competência 5.",
    ],
  },
];

export function buscarExemplo(id: string | undefined) {
  return EXEMPLOS.find((e) => e.id === id);
}
