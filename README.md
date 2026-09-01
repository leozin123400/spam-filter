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
