# Filtro de Spam Inteligente — Projeto Base

Estrutura inicial do projeto de filtro de spam/phishing com redes neurais de NLP.
Segue a abordagem em etapas: **baseline simples → fine-tuning de Transformer → adição de metadados**.

## Estrutura de pastas

```
spam-filter-project/
├── data/
│   ├── raw/            # dados brutos baixados (não versionar datasets grandes)
│   └── processed/       # dados limpos/tokenizados, prontos para treino
├── models/               # checkpoints de modelos treinados
├── notebooks/             # exploração e prototipagem (EDA, testes rápidos)
├── src/
│   ├── baseline.py       # baseline TF-IDF + Naive Bayes
│   └── finetune_distilbert.py  # fine-tuning de DistilBERT via Hugging Face
├── requirements.txt
└── README.md
```

## Como rodar

### 1. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Rodar o baseline (rápido, roda em CPU)

```bash
python src/baseline.py
```

Isso vai:
- Baixar automaticamente um dataset público de exemplo (SMS Spam Collection, via Hugging Face `datasets`)
- Treinar um modelo TF-IDF + Naive Bayes
- Imprimir métricas (precisão, recall, F1, matriz de confusão)

Use esse resultado como **linha de base**: qualquer modelo mais complexo (Transformer) só vale a pena se superar isso de forma relevante.

### 3. Rodar o fine-tuning do DistilBERT (recomendado usar GPU)

```bash
python src/finetune_distilbert.py
```

Isso vai:
- Carregar o mesmo dataset
- Fazer fine-tuning de um `distilbert-base-uncased` (troque por `neuralmind/bert-base-portuguese-cased` se for focar em PT-BR)
- Avaliar com as mesmas métricas do baseline, para comparação direta
- Salvar o modelo treinado em `models/`

### 4. Interface desktop (todas as versões de modelo)

```bash
python src/app.py
```

O menu **Modelo** lista tudo o que existe em `models/`: o baseline e cada fine-tuning do DistilBERT
(um item por checkpoint de época, mais o modelo final). "melhor F1" marca a versão com o maior F1 do treino.

- **Analisar:** cola uma mensagem e mostra veredito e probabilidade. O limiar de decisão é ajustável
  (subir o limiar reduz falsos positivos). O baseline também lista as palavras que mais pesaram.
- **Comparar:** roda a mesma mensagem em todas as versões disponíveis, lado a lado.
- **Métricas:** números da versão escolhida e uma tabela com todas as versões.

**Pesos fora do Git.** O `model.safetensors` do DistilBERT não fica no repositório (ver `.gitignore`).
Versões sem pesos aparecem como "indisponível" na lista, mas o histórico de métricas
(`trainer_state.json`) continua visível. Para rodar o DistilBERT, gere os pesos com
`python src/finetune_distilbert.py` ou copie a pasta `models/distilbert-spam-filter/` completa (com os pesos)
da máquina onde o treino foi feito. Use o botão **Atualizar lista** depois de copiar.

**Baseline.** Na primeira execução, clique em **Treinar agora** (ou rode `python src/baseline.py`), que salva
`models/baseline/baseline.joblib`. Esse arquivo é um pickle: só carregue arquivos que você mesmo treinou.

**Comparando métricas.** O baseline é avaliado com um teste estratificado e o DistilBERT com a divisão do
Hugging Face. As duas usam semente 42 e 20% de teste, mas os conjuntos de teste não são os mesmos, então
diferenças pequenas entre elas não são conclusivas.

**Requisitos.** Python 3.10 ou superior. No Linux, o `tkinter` pode precisar de instalação à parte
(`sudo apt install python3-tk`). O DistilBERT precisa de `torch` e `transformers`; o baseline roda sem eles.

A lógica de inferência fica em `src/predict.py`, separada da interface.

## Próximos passos sugeridos

1. Trocar o dataset de exemplo por um mais próximo do seu caso real (SpamAssassin, Enron, ou dados internos anonimizados)
2. Comparar baseline vs. DistilBERT nas mesmas métricas
3. Adicionar features de metadados (remetente, links, cabeçalhos) como um segundo input ao modelo
4. Ajustar o *threshold* de decisão focando em reduzir falsos positivos
5. Montar pipeline de feedback do usuário para re-treino incremental

## Dataset de exemplo usado

Os scripts usam o dataset `sms_spam` do Hugging Face Hub apenas para deixar tudo rodando de imediato,
sem necessidade de baixar arquivos manualmente. Para o projeto real de e-mail, troque a função
`carregar_dataset()` em cada script pela leitura do seu corpus de e-mails (SpamAssassin/Enron ou dados próprios).
