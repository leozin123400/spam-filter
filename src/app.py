"""
Interface desktop do Filtro de Spam Inteligente (CustomTkinter)
================================================================
Usa todas as versões de modelo encontradas em `models/`:

- Baseline (TF-IDF + Naive Bayes), em `models/baseline/baseline.joblib`
- Cada fine-tuning de Transformer (ex.: DistilBERT): os checkpoints por época
  e o modelo final

Abas:
- Analisar: cola uma mensagem e vê o veredito do modelo escolhido.
- Comparar: roda a mesma mensagem em todas as versões disponíveis.
- Métricas: números do modelo escolhido e tabela com todas as versões.

Rodar:
    python src/app.py

Os pesos dos Transformers (model.safetensors) não ficam no Git. Versões sem
pesos aparecem marcadas como "indisponível", mas o histórico de métricas
delas continua visível.
"""

import gc
import os
import queue
import sys
import threading
import time
import tkinter as tk
from collections import OrderedDict
from dataclasses import replace

# Em execução sem console (ex.: pythonw no Windows) sys.stdout pode ser None
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import customtkinter as ctk

from predict import (
    DependenciaAusente,
    ModeloIndisponivel,
    carregar_classificador,
    descobrir_modelos,
    salvar_baseline,
)

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

COR_SPAM = "#d9534f"
COR_OK = "#2e9e5b"
COR_NEUTRA = "#7a7a7a"
COR_AVISO = ("#fff3cd", "#4a3f12")

CACHE_MAX = 2  # quantos classificadores ficam na memória ao mesmo tempo

EXEMPLOS = {
    "Exemplo: spam": (
        "WINNER!! As a valued network customer you have been selected to receive "
        "a £900 prize reward! To claim call 09061701461. Claim code KL341. "
        "Valid 12 hours only."
    ),
    "Exemplo: legítimo": (
        "Hey, are we still on for dinner tonight? I can pick you up around 7 "
        "if that works for you."
    ),
}


def _pct(valor, casas=2):
    """0.9693 -> '96,93%' (formato brasileiro)."""
    if valor is None:
        return "—"
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Filtro de Spam Inteligente")
        self.geometry("1120x780")
        self.minsize(960, 700)

        # Estado dos modelos
        self.modelos = []  # lista de ModeloInfo
        self._por_rotulo = {}  # texto do menu -> ModeloInfo
        self.info_atual = None
        self.classificador = None
        self._cache = OrderedDict()  # id -> classificador (acesso sob _lock_modelo)
        self._invalidar = set()  # ids a descartar do cache (ex.: baseline retreinado)
        self._lock_modelo = threading.Lock()

        # Estado da interface
        self._fila = queue.Queue()
        self._debounce_id = None
        self._carga_id = 0
        self._analise_id = 0
        self._ultimo_resultado = None
        self._treinando = False
        self._comparando = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._construir_banner()
        self._construir_barra_modelo()
        self._construir_abas()
        self._construir_barra_status()

        self._atualizar_previa_comparacao("")
        self.after(100, self._drenar_fila)
        self._atualizar_lista()

    # ------------------------------------------------------------ Threads
    def _em_thread(self, trabalho, ao_terminar):
        """Roda `trabalho()` fora da interface; `ao_terminar(resultado, erro)` volta na interface."""

        def alvo():
            try:
                resultado, erro = trabalho(), None
            except Exception as e:  # noqa: BLE001 - qualquer falha vira mensagem na tela
                resultado, erro = None, e
            self._fila.put((ao_terminar, (resultado, erro)))

        threading.Thread(target=alvo, daemon=True).start()

    def _drenar_fila(self):
        try:
            while True:
                funcao, args = self._fila.get_nowait()
                try:
                    funcao(*args)
                except tk.TclError:
                    return  # janela fechada
                except Exception as e:  # noqa: BLE001
                    self._status(f"Erro interno: {e}")
        except queue.Empty:
            pass
        try:
            self.after(100, self._drenar_fila)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ UI
    def _construir_banner(self):
        self.banner = ctk.CTkFrame(self, fg_color=COR_AVISO)
        self.banner.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 0))
        self.banner.grid_columnconfigure(0, weight=1)

        self.lbl_banner = ctk.CTkLabel(self.banner, text="", anchor="w", justify="left", wraplength=900)
        self.lbl_banner.grid(row=0, column=0, sticky="ew", padx=12, pady=10)

        self.btn_treinar_banner = ctk.CTkButton(
            self.banner, text="Treinar agora", width=130, command=self._iniciar_treino
        )
        self.btn_treinar_banner.grid(row=0, column=1, padx=12, pady=10)

        self.progresso_treino = ctk.CTkProgressBar(self.banner, mode="indeterminate")
        self.progresso_treino.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))
        self.progresso_treino.grid_remove()

        self.banner.grid_remove()

    def _mostrar_banner(self, texto, treinar=False):
        self.lbl_banner.configure(text=texto)
        if treinar:
            self.btn_treinar_banner.grid()
        else:
            self.btn_treinar_banner.grid_remove()
        self.banner.grid()

    def _esconder_banner(self):
        if not self._treinando:
            self.banner.grid_remove()

    def _construir_barra_modelo(self):
        barra = ctk.CTkFrame(self, fg_color="transparent")
        barra.grid(row=1, column=0, sticky="ew", padx=12, pady=(10, 0))
        barra.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(barra, text="Modelo:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=(0, 8))

        self.menu_modelo = ctk.CTkOptionMenu(
            barra, values=["—"], width=430, dynamic_resizing=False, command=self._ao_escolher_modelo
        )
        self.menu_modelo.grid(row=0, column=1, padx=(0, 8))

        self.lbl_info_modelo = ctk.CTkLabel(barra, text="", anchor="w", text_color=COR_NEUTRA)
        self.lbl_info_modelo.grid(row=0, column=2, sticky="ew", padx=(4, 8))

        self.progresso_modelo = ctk.CTkProgressBar(barra, mode="indeterminate", width=110)
        self.progresso_modelo.grid(row=0, column=3, padx=(0, 8))
        self.progresso_modelo.grid_remove()

        ctk.CTkButton(
            barra,
            text="Atualizar lista",
            width=120,
            fg_color="transparent",
            border_width=1,
            text_color=("gray20", "gray85"),
            command=lambda: self._atualizar_lista(),
        ).grid(row=0, column=4)

    def _construir_abas(self):
        self.abas = ctk.CTkTabview(self)
        self.abas.grid(row=2, column=0, sticky="nsew", padx=12, pady=12)
        self.abas.add("Analisar")
        self.abas.add("Comparar")
        self.abas.add("Métricas")
        self._construir_aba_analisar(self.abas.tab("Analisar"))
        self._construir_aba_comparar(self.abas.tab("Comparar"))
        self._construir_aba_metricas(self.abas.tab("Métricas"))

    # ---- Aba Analisar
    def _construir_aba_analisar(self, aba):
        aba.grid_columnconfigure(0, weight=3, uniform="col")
        aba.grid_columnconfigure(1, weight=2, uniform="col")
        aba.grid_rowconfigure(0, weight=1)

        esq = ctk.CTkFrame(aba, fg_color="transparent")
        esq.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        esq.grid_columnconfigure(0, weight=1)
        esq.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(esq, text="Mensagem", font=ctk.CTkFont(size=15, weight="bold"), anchor="w").grid(
            row=0, column=0, sticky="ew", pady=(0, 6)
        )

        self.caixa = ctk.CTkTextbox(esq, wrap="word", font=ctk.CTkFont(size=14))
        self.caixa.grid(row=1, column=0, sticky="nsew")
        self.caixa.bind("<KeyRelease>", self._ao_digitar)
        self.caixa.bind("<<Paste>>", lambda e: self.after(50, self._ao_digitar))
        self.caixa.bind("<Control-Return>", self._atalho_analisar)
        self.caixa.focus_set()

        linha = ctk.CTkFrame(esq, fg_color="transparent")
        linha.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        self.btn_analisar = ctk.CTkButton(linha, text="Analisar", width=110, command=self._analisar)
        self.btn_analisar.grid(row=0, column=0, padx=(0, 8))
        ctk.CTkButton(
            linha, text="Limpar", width=90, fg_color="gray40", hover_color="gray30", command=self._limpar
        ).grid(row=0, column=1, padx=(0, 8))

        for i, (rotulo, texto) in enumerate(EXEMPLOS.items()):
            ctk.CTkButton(
                linha,
                text=rotulo,
                width=130,
                fg_color="transparent",
                border_width=1,
                text_color=("gray20", "gray85"),
                command=lambda t=texto: self._usar_exemplo(t),
            ).grid(row=0, column=2 + i, padx=(0, 8))

        self.var_tempo_real = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(esq, text="Analisar enquanto digito", variable=self.var_tempo_real).grid(
            row=3, column=0, sticky="w", pady=(10, 0)
        )

        dir_ = ctk.CTkScrollableFrame(aba)
        dir_.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        dir_.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(dir_, text="Resultado", font=ctk.CTkFont(size=15, weight="bold"), anchor="w").grid(
            row=0, column=0, sticky="ew", padx=8, pady=(6, 6)
        )

        self.lbl_veredito = ctk.CTkLabel(
            dir_, text="Aguardando mensagem", font=ctk.CTkFont(size=26, weight="bold"), text_color=COR_NEUTRA
        )
        self.lbl_veredito.grid(row=1, column=0, padx=8, pady=(4, 2))

        self.lbl_prob = ctk.CTkLabel(dir_, text="Probabilidade de spam: —")
        self.lbl_prob.grid(row=2, column=0, padx=8)

        self.barra_prob = ctk.CTkProgressBar(dir_, height=14)
        self.barra_prob.set(0)
        self.barra_prob.grid(row=3, column=0, sticky="ew", padx=8, pady=(8, 16))

        ctk.CTkLabel(dir_, text="Limiar de decisão", anchor="w", font=ctk.CTkFont(weight="bold")).grid(
            row=4, column=0, sticky="ew", padx=8
        )
        self.lbl_limiar = ctk.CTkLabel(dir_, text="Spam se probabilidade ≥ 50%", anchor="w")
        self.lbl_limiar.grid(row=5, column=0, sticky="ew", padx=8)
        self.slider = ctk.CTkSlider(dir_, from_=0.05, to=0.95, number_of_steps=90, command=self._ao_mudar_limiar)
        self.slider.set(0.5)
        self.slider.grid(row=6, column=0, sticky="ew", padx=8, pady=(4, 2))
        ctk.CTkLabel(
            dir_,
            text="Limiar mais alto = menos falsos positivos (legítimos marcados como spam), "
            "mas mais spam passa.",
            wraplength=280,
            justify="left",
            anchor="w",
            text_color=COR_NEUTRA,
            font=ctk.CTkFont(size=11),
        ).grid(row=7, column=0, sticky="ew", padx=8, pady=(0, 14))

        ctk.CTkLabel(dir_, text="Palavras que indicam spam", anchor="w", font=ctk.CTkFont(weight="bold")).grid(
            row=8, column=0, sticky="ew", padx=8
        )
        self.lbl_suspeitas = ctk.CTkLabel(dir_, text="—", wraplength=280, justify="left", anchor="w")
        self.lbl_suspeitas.grid(row=9, column=0, sticky="ew", padx=8, pady=(0, 10))

        ctk.CTkLabel(dir_, text="Palavras que indicam legítimo", anchor="w", font=ctk.CTkFont(weight="bold")).grid(
            row=10, column=0, sticky="ew", padx=8
        )
        self.lbl_legitimas = ctk.CTkLabel(dir_, text="—", wraplength=280, justify="left", anchor="w")
        self.lbl_legitimas.grid(row=11, column=0, sticky="ew", padx=8, pady=(0, 10))

        ctk.CTkLabel(
            dir_,
            text="Modelos treinados com SMS em inglês (dataset de exemplo). "
            "Textos em português terão resultado pouco confiável.",
            wraplength=280,
            justify="left",
            anchor="w",
            text_color=COR_NEUTRA,
            font=ctk.CTkFont(size=11),
        ).grid(row=12, column=0, sticky="ew", padx=8, pady=(4, 8))

    # ---- Aba Comparar
    def _construir_aba_comparar(self, aba):
        aba.grid_columnconfigure(0, weight=1)
        aba.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            aba,
            text="Roda a mensagem da aba Analisar em todas as versões disponíveis, usando o limiar atual.",
            anchor="w",
            text_color=COR_NEUTRA,
        ).grid(row=0, column=0, sticky="ew", pady=(4, 6))

        self.lbl_comp_texto = ctk.CTkLabel(aba, text="", anchor="w", justify="left", wraplength=980)
        self.lbl_comp_texto.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        linha = ctk.CTkFrame(aba, fg_color="transparent")
        linha.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.btn_comparar = ctk.CTkButton(
            linha, text="Comparar versões disponíveis", width=220, command=self._comparar
        )
        self.btn_comparar.grid(row=0, column=0, padx=(0, 12))
        self.lbl_comp_status = ctk.CTkLabel(linha, text="", text_color=COR_NEUTRA)
        self.lbl_comp_status.grid(row=0, column=1)

        self.quadro_comparacao = ctk.CTkScrollableFrame(aba)
        self.quadro_comparacao.grid(row=3, column=0, sticky="nsew")
        for col, peso in enumerate((5, 3, 1, 1)):
            self.quadro_comparacao.grid_columnconfigure(col, weight=peso)
        self._linhas_comparacao = []

        ctk.CTkLabel(
            aba,
            text="O modelo final é salvo com load_best_model_at_end=True, então costuma ser idêntico ao "
            "checkpoint marcado como \"melhor F1\". O tempo medido é só o da classificação, sem a carga do modelo.",
            anchor="w",
            justify="left",
            wraplength=980,
            text_color=COR_NEUTRA,
            font=ctk.CTkFont(size=11),
        ).grid(row=4, column=0, sticky="ew", pady=(8, 0))

    # ---- Aba Métricas
    def _construir_aba_metricas(self, aba):
        aba.grid_columnconfigure(0, weight=1)
        aba.grid_rowconfigure(0, weight=1)
        cont = ctk.CTkScrollableFrame(aba, fg_color="transparent")
        cont.grid(row=0, column=0, sticky="nsew")
        cont.grid_columnconfigure(0, weight=1)

        self.lbl_met_titulo = ctk.CTkLabel(cont, text="", font=ctk.CTkFont(size=15, weight="bold"))
        self.lbl_met_titulo.grid(row=0, column=0, pady=(4, 2))
        self.lbl_fonte = ctk.CTkLabel(
            cont, text="", text_color=COR_NEUTRA, wraplength=900, justify="center", font=ctk.CTkFont(size=11)
        )
        self.lbl_fonte.grid(row=1, column=0, pady=(0, 8))

        self.lbl_sem_metricas = ctk.CTkLabel(cont, text="Sem métricas registradas para esta versão.", text_color=COR_NEUTRA)
        self.lbl_sem_metricas.grid(row=2, column=0, pady=10)

        self.quadro_metricas = ctk.CTkFrame(cont, fg_color="transparent")
        self.quadro_metricas.grid(row=3, column=0, sticky="ew")
        self.valores_metricas = {}
        for i, (chave, nome) in enumerate(
            [("acuracia", "Acurácia"), ("precisao", "Precisão"), ("recall", "Recall"), ("f1", "F1-score")]
        ):
            self.quadro_metricas.grid_columnconfigure(i, weight=1, uniform="m")
            card = ctk.CTkFrame(self.quadro_metricas)
            card.grid(row=0, column=i, padx=6, sticky="ew")
            val = ctk.CTkLabel(card, text="—", font=ctk.CTkFont(size=26, weight="bold"))
            val.pack(padx=10, pady=(10, 0))
            ctk.CTkLabel(card, text=nome, text_color=COR_NEUTRA).pack(padx=10, pady=(0, 10))
            self.valores_metricas[chave] = val

        self.lbl_matriz_titulo = ctk.CTkLabel(cont, text="Matriz de confusão", font=ctk.CTkFont(size=15, weight="bold"))
        self.lbl_matriz_titulo.grid(row=4, column=0, pady=(18, 6))

        self.lbl_sem_matriz = ctk.CTkLabel(
            cont,
            text="A matriz de confusão não foi registrada para esta versão: o histórico de treino "
            "guarda só as métricas globais.",
            text_color=COR_NEUTRA,
            wraplength=700,
        )
        self.lbl_sem_matriz.grid(row=5, column=0, pady=(0, 6))

        self.quadro_matriz = ctk.CTkFrame(cont, fg_color="transparent")
        self.quadro_matriz.grid(row=6, column=0)
        self.celulas_matriz = {}
        ctk.CTkLabel(self.quadro_matriz, text="Previsto →", text_color=COR_NEUTRA).grid(row=0, column=0, padx=8)
        ctk.CTkLabel(self.quadro_matriz, text="Legítimo", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, padx=8)
        ctk.CTkLabel(self.quadro_matriz, text="Spam", font=ctk.CTkFont(weight="bold")).grid(row=0, column=2, padx=8)
        for r, nome in enumerate(["Real: Legítimo", "Real: Spam"], start=1):
            ctk.CTkLabel(self.quadro_matriz, text=nome, font=ctk.CTkFont(weight="bold")).grid(
                row=r, column=0, padx=8, pady=4, sticky="e"
            )
            for c in range(2):
                cor = ("gray85", "gray25")
                if (r - 1, c) == (0, 1):  # falsos positivos
                    cor = ("#f5c6c4", "#6b2a28")
                cel = ctk.CTkLabel(
                    self.quadro_matriz,
                    text="—",
                    width=130,
                    height=54,
                    corner_radius=8,
                    fg_color=cor,
                    font=ctk.CTkFont(size=20, weight="bold"),
                )
                cel.grid(row=r, column=c + 1, padx=4, pady=4)
                self.celulas_matriz[(r - 1, c)] = cel

        self.lbl_nota_matriz = ctk.CTkLabel(
            cont,
            text="Destaque em vermelho: falsos positivos (legítimos classificados como spam), "
            "a métrica mais crítica do projeto.",
            text_color=COR_NEUTRA,
            wraplength=700,
            font=ctk.CTkFont(size=11),
        )
        self.lbl_nota_matriz.grid(row=7, column=0, pady=(4, 0))

        self.lbl_amostra = ctk.CTkLabel(cont, text="", text_color=COR_NEUTRA)
        self.lbl_amostra.grid(row=8, column=0, pady=(2, 0))

        ctk.CTkLabel(cont, text="Todas as versões", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=9, column=0, pady=(22, 6)
        )
        self.quadro_versoes = ctk.CTkFrame(cont, fg_color="transparent")
        self.quadro_versoes.grid(row=10, column=0, sticky="ew", padx=4)

        ctk.CTkLabel(
            cont,
            text="Atenção ao comparar: o baseline é avaliado com um teste estratificado e o DistilBERT com a "
            "divisão do Hugging Face. As duas usam semente 42 e 20% de teste, mas os conjuntos de teste "
            "não são os mesmos. Diferenças pequenas entre elas não são conclusivas. \"Melhor F1\" marca "
            "a versão com o maior F1 de cada treino.",
            text_color=COR_NEUTRA,
            wraplength=900,
            justify="left",
            font=ctk.CTkFont(size=11),
        ).grid(row=11, column=0, pady=(8, 0))

        self.btn_retreinar = ctk.CTkButton(cont, text="Retreinar baseline", width=170, command=self._iniciar_treino)
        self.btn_retreinar.grid(row=12, column=0, pady=18)

    def _construir_barra_status(self):
        self.lbl_status = ctk.CTkLabel(self, text="", anchor="w", text_color=COR_NEUTRA, font=ctk.CTkFont(size=11))
        self.lbl_status.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 8))

    # ------------------------------------------------------ Lista de modelos
    @staticmethod
    def _rotulo(info):
        marca = " · melhor F1" if info.melhor else ""
        sufixo = "" if info.disponivel else "  (indisponível)"
        return f"{info.nome}{marca}{sufixo}"

    def _atualizar_lista(self, selecionar_id=None):
        """Relê models/ e refaz o menu. Mantém a escolha atual quando possível."""
        anterior = selecionar_id or (self.info_atual.id if self.info_atual else None)
        self.modelos = descobrir_modelos()

        self._por_rotulo = {}
        for info in self.modelos:
            rotulo = self._rotulo(info)
            if rotulo in self._por_rotulo:
                rotulo = f"{rotulo} [{info.id}]"
            self._por_rotulo[rotulo] = info
        self.menu_modelo.configure(values=list(self._por_rotulo))

        escolha = next((i for i in self.modelos if i.id == anterior), None)
        if escolha is None:  # padrão: baseline se existir; senão a primeira versão disponível
            escolha = next((i for i in self.modelos if i.id == "baseline" and i.disponivel), None) or next(
                (i for i in self.modelos if i.disponivel), self.modelos[0]
            )
        self._selecionar(escolha)
        self._atualizar_tabela_versoes()

    def _rotulo_de(self, info):
        return next((r for r, i in self._por_rotulo.items() if i is info), self._rotulo(info))

    def _ao_escolher_modelo(self, rotulo):
        info = self._por_rotulo.get(rotulo)
        if info is not None:
            self._selecionar(info)

    def _resumo_modelo(self, info):
        partes = ["Baseline" if info.tipo == "baseline" else "Transformer"]
        if info.epoca is not None:
            partes.append(f"época {info.epoca:g}")
        if info.metricas.get("f1") is not None:
            partes.append(f"F1 {_pct(info.metricas['f1'])}")
        if info.tipo == "baseline":
            partes.append("treinado" if info.disponivel else "não treinado")
        else:
            partes.append("pesos presentes" if info.disponivel else "sem pesos")
        return " · ".join(partes)

    def _selecionar(self, info):
        """Troca o modelo ativo; a carga acontece em segundo plano."""
        self.info_atual = info
        self._carga_id += 1
        carga = self._carga_id
        self.classificador = None
        self._ultimo_resultado = None
        self.menu_modelo.set(self._rotulo_de(info))
        self.lbl_info_modelo.configure(text=self._resumo_modelo(info))
        self._resetar_resultado()
        self._atualizar_metricas(info)

        if not info.disponivel:
            self.btn_analisar.configure(state="disabled")
            self.progresso_modelo.stop()
            self.progresso_modelo.grid_remove()
            self._mostrar_banner(info.motivo or "Modelo indisponível.", treinar=(info.tipo == "baseline"))
            self._status(f"{info.nome}: indisponível. As métricas do histórico continuam visíveis.")
            return

        self._esconder_banner()
        self.btn_analisar.configure(state="disabled")
        self.progresso_modelo.grid()
        self.progresso_modelo.start()
        self._status(f"Carregando {info.nome}…")
        self._em_thread(
            lambda: self._obter_classificador(info),
            lambda clf, erro: self._ao_carregar(carga, info, clf, erro),
        )

    def _obter_classificador(self, info):
        """Roda em thread. Devolve o classificador do cache ou carrega do disco."""
        with self._lock_modelo:
            for id_ in list(self._invalidar):
                self._cache.pop(id_, None)
            self._invalidar.clear()

            clf = self._cache.get(info.id)
            if clf is not None:
                self._cache.move_to_end(info.id)
                return clf
            clf = carregar_classificador(info)
            self._cache[info.id] = clf
            while len(self._cache) > CACHE_MAX:
                self._cache.popitem(last=False)
            gc.collect()
            return clf

    def _ao_carregar(self, carga, info, clf, erro):
        if carga != self._carga_id:
            return  # o usuário já escolheu outro modelo
        self.progresso_modelo.stop()
        self.progresso_modelo.grid_remove()
        if erro is not None:
            self.btn_analisar.configure(state="disabled")
            if isinstance(erro, (DependenciaAusente, ModeloIndisponivel)):
                self._mostrar_banner(str(erro))
            else:
                self._mostrar_banner(f"Falha ao carregar {info.nome}: {erro}", treinar=(info.tipo == "baseline"))
            self._status(f"Não foi possível carregar {info.nome}.")
            return

        self.classificador = clf
        if info.tipo == "baseline" and clf.metricas and not info.metricas:
            info.metricas = clf.metricas
            self._atualizar_metricas(info)
            self.lbl_info_modelo.configure(text=self._resumo_modelo(info))
        self.btn_analisar.configure(state="normal")
        dispositivo = getattr(clf, "dispositivo", None)
        extra = f" (em {dispositivo})" if dispositivo else ""
        self._status(f"{info.nome} carregado{extra}.")
        if self.caixa.get("1.0", "end").strip():
            self._analisar()

    # ------------------------------------------------------------- Treino
    def _iniciar_treino(self):
        if self._treinando:
            return
        self._treinando = True
        self._mostrar_banner("Treinando o baseline (a primeira vez baixa o dataset)…")
        self.btn_treinar_banner.configure(state="disabled")
        self.btn_retreinar.configure(state="disabled")
        self.progresso_treino.grid()
        self.progresso_treino.start()
        self._status("Treinando o baseline em segundo plano…")

        def trabalho():
            from baseline import carregar_dataset, treinar_baseline

            df = carregar_dataset()
            modelo, vetorizador, metricas = treinar_baseline(df)
            salvar_baseline(modelo, vetorizador, metricas)

        self._em_thread(trabalho, self._ao_treinar)

    def _ao_treinar(self, _resultado, erro):
        self._treinando = False
        self.progresso_treino.stop()
        self.progresso_treino.grid_remove()
        self.btn_treinar_banner.configure(state="normal")
        self.btn_retreinar.configure(state="normal")
        if erro is not None:
            self._mostrar_banner(f"Falha no treino: {erro}", treinar=True)
            self._status("Falha no treino.")
            return
        self._invalidar.add("baseline")  # descarta a versão antiga do cache
        self.banner.grid_remove()
        self._atualizar_lista(selecionar_id="baseline")
        self._status("Treino concluído e baseline salvo.")

    # ------------------------------------------------------------ Análise
    def _ao_digitar(self, _evento=None):
        if not self.var_tempo_real.get():
            return
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(350, self._analisar)

    def _atalho_analisar(self, _evento=None):
        self._analisar()
        return "break"

    def _analisar(self):
        self._debounce_id = None
        texto = self.caixa.get("1.0", "end").strip()
        self._atualizar_previa_comparacao(texto)
        if not texto:
            self._resetar_resultado()
            return
        if self.classificador is None:
            self._status("Escolha um modelo disponível para analisar.")
            return

        self._analise_id += 1
        analise = self._analise_id
        clf, limiar = self.classificador, self.slider.get()

        def trabalho():
            with self._lock_modelo:
                if analise != self._analise_id:
                    return None  # chegou uma análise mais nova
                return clf.classificar(texto, limiar=limiar)

        self._em_thread(trabalho, lambda r, e: self._ao_analisar(analise, clf, r, e))

    def _ao_analisar(self, analise, clf, resultado, erro):
        if analise != self._analise_id or clf is not self.classificador:
            return
        if erro is not None:
            self._status(f"Erro ao analisar: {erro}")
            return
        if resultado is None:
            return
        self._ultimo_resultado = resultado
        self._mostrar_resultado(resultado)

    def _mostrar_resultado(self, r):
        cor = COR_SPAM if r.eh_spam else COR_OK
        self.lbl_veredito.configure(text="SPAM" if r.eh_spam else "Legítimo", text_color=cor)
        self.lbl_prob.configure(text=f"Probabilidade de spam: {_pct(r.prob_spam, 1)}")
        self.barra_prob.configure(progress_color=cor)
        self.barra_prob.set(r.prob_spam)
        if getattr(self.classificador, "explica_palavras", False):
            self.lbl_suspeitas.configure(text=", ".join(p for p, _ in r.palavras_suspeitas) or "—")
            self.lbl_legitimas.configure(text=", ".join(p for p, _ in r.palavras_legitimas) or "—")
        else:
            self.lbl_suspeitas.configure(text="Este modelo (Transformer) não aponta palavras individuais.")
            self.lbl_legitimas.configure(text="—")

    def _resetar_resultado(self):
        self.lbl_veredito.configure(text="Aguardando mensagem", text_color=COR_NEUTRA)
        self.lbl_prob.configure(text="Probabilidade de spam: —")
        self.barra_prob.configure(progress_color=("#3B8ED0", "#1F6AA5"))
        self.barra_prob.set(0)
        self.lbl_suspeitas.configure(text="—")
        self.lbl_legitimas.configure(text="—")
        self._ultimo_resultado = None

    def _ao_mudar_limiar(self, valor):
        self.lbl_limiar.configure(text=f"Spam se probabilidade ≥ {valor * 100:.0f}%")
        # Mudar o limiar não exige reclassificar: só reaplicar o corte à probabilidade.
        if self._ultimo_resultado is not None:
            r = replace(self._ultimo_resultado, eh_spam=self._ultimo_resultado.prob_spam >= valor, limiar=valor)
            self._ultimo_resultado = r
            self._mostrar_resultado(r)

    def _limpar(self):
        self.caixa.delete("1.0", "end")
        self._analise_id += 1  # descarta análises em andamento
        self._resetar_resultado()
        self._atualizar_previa_comparacao("")
        self.caixa.focus_set()

    def _usar_exemplo(self, texto):
        self.caixa.delete("1.0", "end")
        self.caixa.insert("1.0", texto)
        self._analisar()

    # ----------------------------------------------------------- Comparação
    def _atualizar_previa_comparacao(self, texto):
        if texto:
            previa = texto if len(texto) <= 220 else texto[:220] + "…"
            self.lbl_comp_texto.configure(text=f"Mensagem: {previa}", text_color=("gray10", "gray90"))
        else:
            self.lbl_comp_texto.configure(
                text="Nenhuma mensagem. Digite ou cole um texto na aba Analisar.", text_color=COR_NEUTRA
            )

    def _comparar(self):
        if self._comparando:
            return
        texto = self.caixa.get("1.0", "end").strip()
        self._atualizar_previa_comparacao(texto)
        if not texto:
            self.lbl_comp_status.configure(text="Escreva uma mensagem na aba Analisar primeiro.")
            return

        limiar = self.slider.get()
        infos = list(self.modelos)
        self._comparando = True
        self.btn_comparar.configure(state="disabled")
        self.lbl_comp_status.configure(text="Comparando…")

        for w in self.quadro_comparacao.winfo_children():
            w.destroy()
        self._linhas_comparacao = []
        for col, titulo in enumerate(["Versão", "Veredito", "Prob. spam", "Tempo"]):
            ctk.CTkLabel(self.quadro_comparacao, text=titulo, font=ctk.CTkFont(weight="bold"), anchor="w").grid(
                row=0, column=col, sticky="w", padx=8, pady=(2, 6)
            )
        for i, info in enumerate(infos, start=1):
            nome = ctk.CTkLabel(self.quadro_comparacao, text=self._rotulo(info), anchor="w")
            veredito = ctk.CTkLabel(self.quadro_comparacao, text="aguardando…", anchor="w", text_color=COR_NEUTRA)
            prob = ctk.CTkLabel(self.quadro_comparacao, text="—", anchor="w")
            tempo = ctk.CTkLabel(self.quadro_comparacao, text="—", anchor="w")
            for col, w in enumerate((nome, veredito, prob, tempo)):
                w.grid(row=i, column=col, sticky="w", padx=8, pady=3)
            self._linhas_comparacao.append((veredito, prob, tempo))

        def trabalho():
            for idx, info in enumerate(infos):
                if not info.disponivel:
                    self._fila.put((self._linha_comparacao, (idx, None, None, None, "indisponível (sem pesos)")))
                    continue
                self._fila.put((self._linha_comparacao, (idx, None, None, None, "carregando…")))
                try:
                    clf = self._obter_classificador(info)
                    with self._lock_modelo:
                        t0 = time.perf_counter()
                        r = clf.classificar(texto, limiar=limiar)
                        ms = (time.perf_counter() - t0) * 1000
                    self._fila.put((self._linha_comparacao, (idx, r.eh_spam, r.prob_spam, ms, "")))
                except DependenciaAusente:
                    self._fila.put((self._linha_comparacao, (idx, None, None, None, "falta instalar torch/transformers")))
                except Exception as e:  # noqa: BLE001
                    self._fila.put((self._linha_comparacao, (idx, None, None, None, f"erro: {e}")))

        self._em_thread(trabalho, self._ao_comparar)

    def _linha_comparacao(self, idx, eh_spam, prob, ms, obs):
        if idx >= len(self._linhas_comparacao):
            return
        veredito, prob_lbl, tempo_lbl = self._linhas_comparacao[idx]
        if prob is None:
            veredito.configure(text=obs, text_color=COR_NEUTRA)
            prob_lbl.configure(text="—")
            tempo_lbl.configure(text="—")
            return
        veredito.configure(text="SPAM" if eh_spam else "Legítimo", text_color=COR_SPAM if eh_spam else COR_OK)
        prob_lbl.configure(text=_pct(prob, 1))
        tempo_lbl.configure(text=f"{ms:.0f} ms")

    def _ao_comparar(self, _resultado, erro):
        self._comparando = False
        self.btn_comparar.configure(state="normal")
        self.lbl_comp_status.configure(text="Falha na comparação." if erro else "Concluído.")

    # ------------------------------------------------------------ Métricas
    def _atualizar_metricas(self, info):
        met = info.metricas or {}
        self.lbl_met_titulo.configure(text=info.nome)
        fonte = f"Fonte: {info.fonte_metricas}" if info.fonte_metricas else ""
        if info.observacao:
            fonte = f"{fonte}\n{info.observacao}".strip()
        self.lbl_fonte.configure(text=fonte)

        tem = any(met.get(k) is not None for k in self.valores_metricas)
        if tem:
            self.lbl_sem_metricas.grid_remove()
            self.quadro_metricas.grid()
        else:
            self.lbl_sem_metricas.grid()
            self.quadro_metricas.grid_remove()
        for chave, lbl in self.valores_metricas.items():
            lbl.configure(text=_pct(met.get(chave)))

        cm = met.get("matriz_confusao")
        if cm:
            self.lbl_sem_matriz.grid_remove()
            self.quadro_matriz.grid()
            self.lbl_nota_matriz.grid()
            for (r, c), cel in self.celulas_matriz.items():
                cel.configure(text=str(cm[r][c]))
        else:
            self.lbl_sem_matriz.grid()
            self.quadro_matriz.grid_remove()
            self.lbl_nota_matriz.grid_remove()

        n_teste = met.get("n_teste")
        n_treino = met.get("n_treino")
        if n_teste is not None:
            texto = f"Teste: {n_teste} exemplos"
            if n_treino is not None:
                texto = f"Treino: {n_treino} exemplos | {texto}"
            self.lbl_amostra.configure(text=texto)
        else:
            self.lbl_amostra.configure(text="")

    def _atualizar_tabela_versoes(self):
        for w in self.quadro_versoes.winfo_children():
            w.destroy()
        colunas = ["Versão", "Época", "Acurácia", "Precisão", "Recall", "F1", "Situação"]
        for col, titulo in enumerate(colunas):
            self.quadro_versoes.grid_columnconfigure(col, weight=3 if col == 0 else 1)
            ctk.CTkLabel(self.quadro_versoes, text=titulo, font=ctk.CTkFont(weight="bold"), anchor="w").grid(
                row=0, column=col, sticky="w", padx=6, pady=(0, 4)
            )
        for i, info in enumerate(self.modelos, start=1):
            m = info.metricas or {}
            situacao = ("pesos presentes" if info.tipo != "baseline" else "treinado") if info.disponivel else (
                "sem pesos" if info.tipo != "baseline" else "não treinado"
            )
            valores = [
                self._rotulo(info),
                f"{info.epoca:g}" if info.epoca is not None else "—",
                _pct(m.get("acuracia")),
                _pct(m.get("precisao")),
                _pct(m.get("recall")),
                _pct(m.get("f1")),
                situacao,
            ]
            cor = None if info.disponivel else COR_NEUTRA
            for col, v in enumerate(valores):
                ctk.CTkLabel(self.quadro_versoes, text=v, anchor="w", text_color=cor).grid(
                    row=i, column=col, sticky="w", padx=6, pady=2
                )

    # -------------------------------------------------------------- Status
    def _status(self, texto):
        self.lbl_status.configure(text=texto)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
