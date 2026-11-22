"""Código compartilhado do avaliador de redações (BERTimbau ajustado no Essay-BR).

Módulos:
  caminhos      onde ficam os dados, o modelo treinado e os resultados
  modelo        arquitetura (EnemEssayEvaluator), dataset e predição
  dados         leitura do Essay-BR e divisão treino/validação/teste
  notas         conversão da previsão em notas 0–200, calibração dos cortes e QWK
  perturbacoes  cópias alteradas das redações para o treino contra vieses

Usado pela API (main.py), pelos scripts de treino (treino/) e pelos experimentos
(experimentos/). Os módulos são importados sob demanda, para a API não carregar
pandas e scikit-learn sem precisar.
"""
