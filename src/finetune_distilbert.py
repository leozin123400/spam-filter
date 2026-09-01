"""
Fine-tuning de Encoder Transformer (DistilBERT) para o Filtro de Spam Inteligente
====================================================================================
Depois de validar o baseline (src/baseline.py), este script faz o fine-tuning
de um modelo Transformer pré-treinado para a mesma tarefa de classificação.

Para foco em português, troque MODEL_NAME por:
    "neuralmind/bert-base-portuguese-cased"

Rodar:
    python src/finetune_distilbert.py

Recomendado rodar com GPU disponível (bem mais lento em CPU).
"""

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

MODEL_NAME = "distilbert-base-uncased"  # troque para BERTimbau se for PT-BR
OUTPUT_DIR = "models/distilbert-spam-filter"
NUM_EPOCHS = 3
BATCH_SIZE = 16


def carregar_dataset():
    """
    TODO: substituir por dataset real de e-mails (colunas 'texto' e 'label').
    Por padrão, usa o dataset público sms_spam para deixar o pipeline funcional
    de imediato.
    """
    ds = load_dataset("ucirvine/sms_spam")["train"]
    ds = ds.rename_column("sms", "texto")
    ds = ds.train_test_split(test_size=0.2, seed=42)
    return ds


def tokenizar_dataset(dataset, tokenizer):
    def tokenize_fn(batch):
        return tokenizer(batch["texto"], truncation=True, max_length=256)

    return dataset.map(tokenize_fn, batched=True)


def calcular_metricas(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    resultado = {}
    resultado.update(accuracy.compute(predictions=preds, references=labels))
    resultado.update(precision.compute(predictions=preds, references=labels))
    resultado.update(recall.compute(predictions=preds, references=labels))
    resultado.update(f1.compute(predictions=preds, references=labels))
    return resultado


def main():
    print(f"Carregando tokenizer e modelo base: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    print("Carregando dataset...")
    dataset = carregar_dataset()

    print("Tokenizando...")
    dataset_tok = tokenizar_dataset(dataset, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=modelo,
        args=args,
        train_dataset=dataset_tok["train"],
        eval_dataset=dataset_tok["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=calcular_metricas,
    )

    print("Iniciando fine-tuning...")
    trainer.train()

    print("\n=== Avaliação final ===")
    metricas = trainer.evaluate()
    for chave, valor in metricas.items():
        print(f"{chave}: {valor:.4f}" if isinstance(valor, float) else f"{chave}: {valor}")

    print(f"\nSalvando modelo em: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(
        "\nCompare essas métricas com as do baseline (src/baseline.py) para "
        "decidir se o ganho justifica a complexidade extra do Transformer."
    )


if __name__ == "__main__":
    main()
