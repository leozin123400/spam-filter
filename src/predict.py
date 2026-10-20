"""
Módulo de inferência do Filtro de Spam Inteligente
====================================================
Independente de interface: a GUI (src/app.py), uma API ou qualquer outro
cliente só precisam chamar `ClassificadorBaseline.classificar(texto)`.

Quando o DistilBERT for integrado, basta criar uma classe com a mesma
interface (`classificar` e `metricas`) e oferecê-la na interface.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_BASELINE = RAIZ / "models" / "baseline" / "baseline.joblib"


@dataclass
class Resultado:
    prob_spam: float
    eh_spam: bool
    limiar: float
    palavras_suspeitas: list = field(default_factory=list)  # [(palavra, peso)]
    palavras_legitimas: list = field(default_factory=list)


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


class ClassificadorBaseline:
    """TF-IDF + Naive Bayes carregado do disco."""

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

        # Termos presentes no texto, ordenados pela influência
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
