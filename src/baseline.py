"""
Baseline do Filtro de Spam Inteligente
========================================
Modelo simples (TF-IDF + Naive Bayes) para servir como linha de base
antes de partir para modelos neurais mais pesados (Transformers).

Rodar:
    python src/baseline.py
"""

import time

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def carregar_dataset() -> pd.DataFrame:
    """
    Carrega um dataset público de exemplo (SMS Spam Collection).

    TODO: substituir por um dataset real de e-mails (SpamAssassin, Enron,
    ou seu corpus interno). O DataFrame final deve ter as colunas:
        - "texto": conteúdo da mensagem
        - "label": 0 (legítimo) ou 1 (spam/phishing)
    """
    print("Carregando dataset de exemplo (sms_spam)...")
    ds = load_dataset("ucirvine/sms_spam")["train"]
    df = ds.to_pandas().rename(columns={"sms": "texto", "label": "label"})
    return df


def treinar_baseline(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print(f"Treino: {len(X_train)} exemplos | Teste: {len(X_test)} exemplos")

    # Vetorização TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",  # trocar para lista de stopwords em PT-BR se aplicável
        max_features=5000,
        ngram_range=(1, 2),
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Modelo
    modelo = MultinomialNB()

    inicio = time.time()
    modelo.fit(X_train_vec, y_train)
    duracao = time.time() - inicio
    print(f"Treinamento concluído em {duracao:.2f}s")

    # Avaliação
    y_pred = modelo.predict(X_test_vec)

    print("\n=== Métricas do Baseline ===")
    print(f"Acurácia:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precisão:  {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")

    print("\n=== Relatório completo ===")
    print(classification_report(y_test, y_pred, target_names=["Legítimo", "Spam"]))

    print("=== Matriz de Confusão ===")
    print("            Previsto Legítimo | Previsto Spam")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Real Legítimo:     {cm[0][0]:>6}       |    {cm[0][1]:>6}")
    print(f"Real Spam:         {cm[1][0]:>6}       |    {cm[1][1]:>6}")

    print(
        "\nAtenção: cm[0][1] (falsos positivos) é a métrica mais crítica do projeto — "
        "e-mails legítimos classificados erroneamente como spam."
    )

    return modelo, vectorizer


if __name__ == "__main__":
    dados = carregar_dataset()
    treinar_baseline(dados)
