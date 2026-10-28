"""
Módulo de inferência do Filtro de Spam Inteligente
====================================================
Independente de interface: a GUI (src/app.py), uma API ou qualquer outro
cliente só precisam de duas funções:

    modelos = descobrir_modelos()          # lista todas as versões em models/
    clf = carregar_classificador(modelos[i])
    resultado = clf.classificar("texto", limiar=0.5)

Versões descobertas automaticamente
-----------------------------------
- models/baseline/baseline.joblib                -> TF-IDF + Naive Bayes
- models/<nome>/                                 -> modelo final de um fine-tuning
- models/<nome>/checkpoint-N/                    -> cada checkpoint salvo pelo Trainer

Um modelo Transformer só roda se a pasta tiver config.json, o tokenizer e os
pesos (model.safetensors ou pytorch_model.bin). Os pesos NÃO ficam no Git
(ver .gitignore); quando faltam, o modelo aparece como indisponível, mas o
histórico de treino (trainer_state.json) continua sendo lido.

Segurança: o baseline é um arquivo joblib (pickle). Só carregue modelos que
você mesmo treinou ou em que confia.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = RAIZ / "models"
CAMINHO_BASELINE = PASTA_MODELOS / "baseline" / "baseline.joblib"

ARQUIVOS_PESOS = ("model.safetensors", "pytorch_model.bin")
ARQUIVOS_TOKENIZER = ("tokenizer.json", "vocab.txt")
MAX_TOKENS = 256  # mesmo valor usado em src/finetune_distilbert.py

_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")
_NOMES_TIPO = {"distilbert": "DistilBERT", "bert": "BERT", "roberta": "RoBERTa"}

# Chaves do trainer_state.json -> chaves usadas pelo projeto
_CHAVES_AVALIACAO = {
    "eval_accuracy": "acuracia",
    "eval_precision": "precisao",
    "eval_recall": "recall",
    "eval_f1": "f1",
    "eval_loss": "perda",
}


class ModeloIndisponivel(RuntimeError):
    """O modelo foi descoberto, mas faltam arquivos para executá-lo."""


class DependenciaAusente(RuntimeError):
    """Falta uma biblioteca (torch/transformers) para executar o modelo."""


# --------------------------------------------------------------------------
# Estruturas de dados
# --------------------------------------------------------------------------
@dataclass
class Resultado:
    prob_spam: float
    eh_spam: bool
    limiar: float
    palavras_suspeitas: list = field(default_factory=list)  # [(palavra, peso)]
    palavras_legitimas: list = field(default_factory=list)


@dataclass
class ModeloInfo:
    id: str
    nome: str
    tipo: str  # "baseline" | "transformer"
    caminho: Path
    disponivel: bool
    motivo: str = ""  # por que está indisponível
    epoca: float | None = None
    step: int | None = None
    melhor: bool = False  # melhor F1 do treino (checkpoint) ou equivalente
    observacao: str = ""
    metricas: dict = field(default_factory=dict)
    fonte_metricas: str = ""


# --------------------------------------------------------------------------
# Baseline: salvar / checar
# --------------------------------------------------------------------------
def salvar_baseline(modelo, vetorizador, metricas=None, caminho=CAMINHO_BASELINE) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"modelo": modelo, "vetorizador": vetorizador, "metricas": metricas or {}},
        caminho,
    )
    return caminho


def baseline_existe(caminho=CAMINHO_BASELINE) -> bool:
    return Path(caminho).is_file()


# --------------------------------------------------------------------------
# Descoberta de versões
# --------------------------------------------------------------------------
def _ler_json(caminho: Path):
    try:
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _faltando_para_executar(pasta: Path) -> str:
    """Devolve '' se a pasta tem tudo para rodar, ou o motivo de não rodar."""
    if not any((pasta / nome).is_file() for nome in ARQUIVOS_PESOS):
        return (
            "Pesos do modelo não encontrados (model.safetensors). Eles não ficam "
            "no Git: gere com `python src/finetune_distilbert.py` ou copie da "
            "máquina onde o treino foi feito."
        )
    if not any((pasta / nome).is_file() for nome in ARQUIVOS_TOKENIZER):
        return "Tokenizer não encontrado (tokenizer.json ou vocab.txt)."
    return ""


def _metricas_do_estado(estado: dict, step: int) -> dict:
    """Métricas de avaliação registradas no trainer_state.json para um step."""
    for entrada in estado.get("log_history", []):
        if entrada.get("step") == step and "eval_f1" in entrada:
            m = {novo: float(entrada[antigo]) for antigo, novo in _CHAVES_AVALIACAO.items() if antigo in entrada}
            m["epoca"] = float(entrada.get("epoch", 0.0))
            return m
    return {}


def _ler_metrics_json(pasta: Path) -> dict:
    """metrics.json é opcional: se existir na pasta do modelo, tem prioridade sobre o
    trainer_state.json. Chaves aceitas: acuracia, precisao, recall, f1, perda,
    matriz_confusao ([[VN, FP], [FN, VP]]), n_teste."""
    dados = _ler_json(pasta / "metrics.json")
    return dados if isinstance(dados, dict) else {}


def _info_baseline(caminho: Path) -> ModeloInfo:
    info = ModeloInfo(
        id="baseline",
        nome="Baseline (TF-IDF + Naive Bayes)",
        tipo="baseline",
        caminho=caminho,
        disponivel=caminho.is_file(),
    )
    if not info.disponivel:
        info.motivo = "Modelo ainda não treinado. Use o botão de treino ou `python src/baseline.py`."
        return info
    try:
        info.metricas = dict(joblib.load(caminho).get("metricas", {}))
        info.fonte_metricas = "baseline.joblib (teste estratificado, random_state=42)"
    except Exception as e:  # arquivo corrompido / versão incompatível
        info.disponivel = False
        info.motivo = f"Não foi possível ler o arquivo do baseline ({e}). Treine novamente."
    return info


def _infos_transformer(pasta: Path) -> list:
    """Uma entrada por checkpoint + uma para o modelo final da pasta."""
    config = _ler_json(pasta / "config.json")
    checkpoints = sorted(
        (int(m.group(1)), p)
        for p in pasta.iterdir()
        if p.is_dir() and (m := _RE_CHECKPOINT.match(p.name)) and (p / "config.json").is_file()
    )
    if not isinstance(config, dict) and not checkpoints:
        return []

    if not isinstance(config, dict):
        config = _ler_json(checkpoints[-1][1] / "config.json") or {}
    tipo_modelo = str(config.get("model_type", "transformer"))
    tipo_legivel = _NOMES_TIPO.get(tipo_modelo, tipo_modelo.capitalize())

    # Qual checkpoint teve o melhor F1 (lido do estado do último checkpoint)
    melhor_step = None
    estado_final = _ler_json(checkpoints[-1][1] / "trainer_state.json") if checkpoints else None
    if isinstance(estado_final, dict):
        melhor_step = estado_final.get("best_global_step")

    infos = []
    for step, p in checkpoints:
        estado = _ler_json(p / "trainer_state.json")
        epoca = estado.get("epoch") if isinstance(estado, dict) else None
        rotulo_epoca = f"época {epoca:g}" if isinstance(epoca, (int, float)) else f"step {step}"
        motivo = _faltando_para_executar(p)
        info = ModeloInfo(
            id=f"{pasta.name}/{p.name}",
            nome=f"{tipo_legivel} — {rotulo_epoca} ({p.name})",
            tipo="transformer",
            caminho=p,
            disponivel=not motivo,
            motivo=motivo,
            epoca=float(epoca) if isinstance(epoca, (int, float)) else None,
            step=step,
            melhor=(melhor_step == step),
        )
        if isinstance(estado, dict):
            info.metricas = _metricas_do_estado(estado, step)
            if info.metricas:
                info.fonte_metricas = "trainer_state.json (avaliação ao fim da época)"
        extra = _ler_metrics_json(p)
        if extra:
            info.metricas.update(extra)
            info.fonte_metricas = "metrics.json (na pasta do modelo)"
        infos.append(info)

    if (pasta / "config.json").is_file():
        motivo = _faltando_para_executar(pasta)
        final = ModeloInfo(
            id=pasta.name,
            nome=f"{tipo_legivel} — final",
            tipo="transformer",
            caminho=pasta,
            disponivel=not motivo,
            motivo=motivo,
        )
        # O script de fine-tuning usa load_best_model_at_end=True: o modelo salvo
        # na raiz é o melhor checkpoint. Isto é uma inferência, não um dado gravado.
        equivalente = next((i for i in infos if i.melhor), None)
        if equivalente is not None:
            final.epoca = equivalente.epoca
            final.step = equivalente.step
            final.melhor = True
            final.metricas = {k: v for k, v in equivalente.metricas.items()}
            final.fonte_metricas = f"estimado a partir do melhor checkpoint ({equivalente.caminho.name})"
            final.observacao = (
                f"Equivale ao {equivalente.caminho.name} se foi treinado com "
                "load_best_model_at_end=True (padrão de src/finetune_distilbert.py)."
            )
        extra = _ler_metrics_json(pasta)
        if extra:
            final.metricas.update(extra)
            final.fonte_metricas = "metrics.json (na pasta do modelo)"
        infos.append(final)

    # Rótulo do melhor checkpoint
    for i in infos:
        if i.melhor and i.step is not None and i.caminho.name.startswith("checkpoint-"):
            i.observacao = "Melhor F1 do treino."
    return infos


def descobrir_modelos(pasta=PASTA_MODELOS) -> list:
    """Lista todas as versões: baseline primeiro, depois cada fine-tuning
    (checkpoints em ordem crescente e o modelo final por último)."""
    pasta = Path(pasta)
    modelos = [_info_baseline(pasta / "baseline" / "baseline.joblib")]
    if pasta.is_dir():
        for sub in sorted(p for p in pasta.iterdir() if p.is_dir() and p.name != "baseline"):
            modelos.extend(_infos_transformer(sub))

    # Nomes repetidos (dois fine-tunings do mesmo tipo): acrescenta a pasta
    contagem = {}
    for m in modelos:
        contagem[m.nome] = contagem.get(m.nome, 0) + 1
    for m in modelos:
        if contagem[m.nome] > 1:
            pasta_pai = m.id.split("/")[0]
            m.nome = f"{m.nome} [{pasta_pai}]"
    return modelos


# --------------------------------------------------------------------------
# Classificadores
# --------------------------------------------------------------------------
class ClassificadorBaseline:
    """TF-IDF + Naive Bayes carregado do disco."""

    nome = "Baseline (TF-IDF + Naive Bayes)"
    explica_palavras = True

    def __init__(self, caminho=CAMINHO_BASELINE):
        dados = joblib.load(caminho)
        self.modelo = dados["modelo"]
        self.vetorizador = dados["vetorizador"]
        self.metricas = dados.get("metricas", {})
        self._vocab = self.vetorizador.get_feature_names_out()
        # Diferença de log-probabilidade (spam - legítimo) por termo:
        # positivo empurra para spam, negativo para legítimo.
        lp = self.modelo.feature_log_prob_
        self._peso_termo = lp[1] - lp[0]

    def classificar(self, texto: str, limiar: float = 0.5, top_n: int = 5) -> Resultado:
        X = self.vetorizador.transform([texto])
        prob_spam = float(self.modelo.predict_proba(X)[0][1])

        indices = X.nonzero()[1]
        contribuicoes = [
            (str(self._vocab[i]), float(self._peso_termo[i] * X[0, i])) for i in indices
        ]
        suspeitas = sorted((c for c in contribuicoes if c[1] > 0), key=lambda c: -c[1])
        legitimas = sorted((c for c in contribuicoes if c[1] < 0), key=lambda c: c[1])

        return Resultado(
            prob_spam=prob_spam,
            eh_spam=prob_spam >= limiar,
            limiar=limiar,
            palavras_suspeitas=suspeitas[:top_n],
            palavras_legitimas=legitimas[:top_n],
        )


class ClassificadorTransformer:
    """Modelo Transformer (ex.: DistilBERT) ajustado para spam, carregado de uma pasta.

    Importa torch/transformers só aqui, para o baseline funcionar sem eles.
    O índice 1 é a classe spam (mesma convenção do dataset sms_spam).
    """

    explica_palavras = False

    def __init__(self, caminho, metricas=None, nome=None):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise DependenciaAusente(
                "Para usar o DistilBERT instale as dependências: pip install torch transformers"
            ) from e

        caminho = Path(caminho)
        self._torch = torch
        self.nome = nome or caminho.name
        self.metricas = metricas or {}
        self.tokenizer = AutoTokenizer.from_pretrained(str(caminho))
        self.modelo = AutoModelForSequenceClassification.from_pretrained(str(caminho))
        if self.modelo.config.num_labels != 2:
            raise ValueError(
                f"O modelo em {caminho} tem {self.modelo.config.num_labels} classes; "
                "o filtro espera 2 (legítimo, spam)."
            )
        self.modelo.eval()
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo.to(self.dispositivo)

        # Índice da classe spam: por padrão 1; respeita nomes explícitos no config.
        self._idx_spam = 1
        for idx, rotulo in (self.modelo.config.id2label or {}).items():
            if str(rotulo).strip().lower() in ("spam", "phishing"):
                self._idx_spam = int(idx)

    def classificar(self, texto: str, limiar: float = 0.5, top_n: int = 5) -> Resultado:
        torch = self._torch
        entrada = self.tokenizer(
            texto, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        )
        entrada = {k: v.to(self.dispositivo) for k, v in entrada.items()}
        with torch.inference_mode():
            logits = self.modelo(**entrada).logits
        prob_spam = float(torch.softmax(logits, dim=-1)[0, self._idx_spam])
        return Resultado(prob_spam=prob_spam, eh_spam=prob_spam >= limiar, limiar=limiar)


def carregar_classificador(info: ModeloInfo):
    """Cria o classificador de uma versão descoberta por `descobrir_modelos`."""
    if not info.disponivel:
        raise ModeloIndisponivel(info.motivo or "Modelo indisponível.")
    if info.tipo == "baseline":
        clf = ClassificadorBaseline(info.caminho)
        clf.nome = info.nome
        if not info.metricas:
            info.metricas = clf.metricas
        return clf
    return ClassificadorTransformer(info.caminho, metricas=info.metricas, nome=info.nome)
