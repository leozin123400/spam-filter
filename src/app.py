"""
Interface desktop do Filtro de Spam Inteligente (CustomTkinter)
================================================================
Versão inicial: usa apenas o baseline (TF-IDF + Naive Bayes).

Rodar:
    python src/app.py

Na primeira execução, se ainda não existir modelo salvo, a interface oferece
um botão para treinar (em segundo plano, sem travar a janela).
"""

import os
import queue
import sys
import threading

# Em execução sem console (ex.: pythonw no Windows) sys.stdout pode ser None
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import customtkinter as ctk

from predict import ClassificadorBaseline, baseline_existe, salvar_baseline

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

COR_SPAM = "#d9534f"
COR_OK = "#2e9e5b"
COR_NEUTRA = "#7a7a7a"

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


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Filtro de Spam Inteligente")
        self.geometry("1000x700")
        self.minsize(860, 620)

        self.classificador = None
        self._debounce_id = None
        self._fila = queue.Queue()
        self._treinando = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._construir_banner()
        self._construir_abas()
        self._construir_barra_status()

        self._carregar_modelo()

    # ------------------------------------------------------------------ UI
    def _construir_banner(self):
        self.banner = ctk.CTkFrame(self, fg_color=("#fff3cd", "#4a3f12"))
        self.banner.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 0))
        self.banner.grid_columnconfigure(0, weight=1)

        self.lbl_banner = ctk.CTkLabel(
            self.banner,
            text="Nenhum modelo treinado encontrado. Treine o baseline para começar.",
            anchor="w",
        )
        self.lbl_banner.grid(row=0, column=0, sticky="ew", padx=12, pady=10)

        self.btn_treinar_banner = ctk.CTkButton(
            self.banner, text="Treinar agora", width=130, command=self._iniciar_treino
        )
        self.btn_treinar_banner.grid(row=0, column=1, padx=12, pady=10)

        self.progresso_treino = ctk.CTkProgressBar(self.banner, mode="indeterminate")
        self.progresso_treino.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))
        self.progresso_treino.grid_remove()

        self.banner.grid_remove()

    def _construir_abas(self):
        self.abas = ctk.CTkTabview(self)
        self.abas.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        self.abas.add("Analisar")
        self.abas.add("Métricas")
        self._construir_aba_analisar(self.abas.tab("Analisar"))
        self._construir_aba_metricas(self.abas.tab("Métricas"))

    def _construir_aba_analisar(self, aba):
        aba.grid_columnconfigure(0, weight=3, uniform="col")
        aba.grid_columnconfigure(1, weight=2, uniform="col")
        aba.grid_rowconfigure(0, weight=1)

        # ---- Coluna esquerda: entrada
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
        linha.grid_columnconfigure(4, weight=1)

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

        # ---- Coluna direita: resultado
        dir_ = ctk.CTkFrame(aba)
        dir_.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        dir_.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(dir_, text="Resultado", font=ctk.CTkFont(size=15, weight="bold"), anchor="w").grid(
            row=0, column=0, sticky="ew", padx=16, pady=(14, 6)
        )

        self.lbl_veredito = ctk.CTkLabel(
            dir_, text="Aguardando mensagem", font=ctk.CTkFont(size=26, weight="bold"), text_color=COR_NEUTRA
        )
        self.lbl_veredito.grid(row=1, column=0, padx=16, pady=(4, 2))

        self.lbl_prob = ctk.CTkLabel(dir_, text="Probabilidade de spam: —")
        self.lbl_prob.grid(row=2, column=0, padx=16)

        self.barra_prob = ctk.CTkProgressBar(dir_, height=14)
        self.barra_prob.set(0)
        self.barra_prob.grid(row=3, column=0, sticky="ew", padx=16, pady=(8, 16))

        ctk.CTkLabel(dir_, text="Limiar de decisão", anchor="w", font=ctk.CTkFont(weight="bold")).grid(
            row=4, column=0, sticky="ew", padx=16
        )
        self.lbl_limiar = ctk.CTkLabel(dir_, text="Spam se probabilidade ≥ 50%", anchor="w")
        self.lbl_limiar.grid(row=5, column=0, sticky="ew", padx=16)
        self.slider = ctk.CTkSlider(dir_, from_=0.05, to=0.95, number_of_steps=90, command=self._ao_mudar_limiar)
        self.slider.set(0.5)
        self.slider.grid(row=6, column=0, sticky="ew", padx=16, pady=(4, 2))
        ctk.CTkLabel(
            dir_,
            text="Limiar mais alto = menos falsos positivos (legítimos marcados como spam), "
            "mas mais spam passa.",
            wraplength=300,
            justify="left",
            anchor="w",
            text_color=COR_NEUTRA,
            font=ctk.CTkFont(size=11),
        ).grid(row=7, column=0, sticky="ew", padx=16, pady=(0, 14))

        ctk.CTkLabel(dir_, text="Palavras que indicam spam", anchor="w", font=ctk.CTkFont(weight="bold")).grid(
            row=8, column=0, sticky="ew", padx=16
        )
        self.lbl_suspeitas = ctk.CTkLabel(dir_, text="—", wraplength=300, justify="left", anchor="w")
        self.lbl_suspeitas.grid(row=9, column=0, sticky="ew", padx=16, pady=(0, 10))

        ctk.CTkLabel(dir_, text="Palavras que indicam legítimo", anchor="w", font=ctk.CTkFont(weight="bold")).grid(
            row=10, column=0, sticky="ew", padx=16
        )
        self.lbl_legitimas = ctk.CTkLabel(dir_, text="—", wraplength=300, justify="left", anchor="w")
        self.lbl_legitimas.grid(row=11, column=0, sticky="ew", padx=16, pady=(0, 10))

        ctk.CTkLabel(
            dir_,
            text="Modelo treinado com SMS em inglês (dataset de exemplo). "
            "Textos em português terão resultado pouco confiável.",
            wraplength=300,
            justify="left",
            anchor="w",
            text_color=COR_NEUTRA,
            font=ctk.CTkFont(size=11),
        ).grid(row=12, column=0, sticky="ew", padx=16, pady=(4, 14))

    def _construir_aba_metricas(self, aba):
        aba.grid_columnconfigure(0, weight=1)

        self.lbl_sem_metricas = ctk.CTkLabel(aba, text="Treine o modelo para ver as métricas.", text_color=COR_NEUTRA)
        self.lbl_sem_metricas.grid(row=0, column=0, pady=30)

        self.quadro_metricas = ctk.CTkFrame(aba, fg_color="transparent")
        self.quadro_metricas.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.valores_metricas = {}
        for i, (chave, nome) in enumerate(
            [("acuracia", "Acurácia"), ("precisao", "Precisão"), ("recall", "Recall"), ("f1", "F1-score")]
        ):
            self.quadro_metricas.grid_columnconfigure(i, weight=1, uniform="m")
            card = ctk.CTkFrame(self.quadro_metricas)
            card.grid(row=0, column=i, padx=6, sticky="ew")
            val = ctk.CTkLabel(card, text="—", font=ctk.CTkFont(size=28, weight="bold"))
            val.pack(padx=10, pady=(12, 0))
            ctk.CTkLabel(card, text=nome, text_color=COR_NEUTRA).pack(padx=10, pady=(0, 12))
            self.valores_metricas[chave] = val

        ctk.CTkLabel(aba, text="Matriz de confusão", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=2, column=0, pady=(24, 8)
        )
        self.quadro_matriz = ctk.CTkFrame(aba, fg_color="transparent")
        self.quadro_matriz.grid(row=3, column=0)
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
                    height=60,
                    corner_radius=8,
                    fg_color=cor,
                    font=ctk.CTkFont(size=20, weight="bold"),
                )
                cel.grid(row=r, column=c + 1, padx=4, pady=4)
                self.celulas_matriz[(r - 1, c)] = cel

        ctk.CTkLabel(
            aba,
            text="Destaque em vermelho: falsos positivos (e-mails legítimos classificados como spam) — "
            "a métrica mais crítica do projeto.",
            text_color=COR_NEUTRA,
            wraplength=600,
            font=ctk.CTkFont(size=11),
        ).grid(row=4, column=0, pady=(8, 0))

        self.lbl_amostra = ctk.CTkLabel(aba, text="", text_color=COR_NEUTRA)
        self.lbl_amostra.grid(row=5, column=0, pady=(6, 0))

        self.btn_retreinar = ctk.CTkButton(aba, text="Retreinar modelo", width=160, command=self._iniciar_treino)
        self.btn_retreinar.grid(row=6, column=0, pady=20)

    def _construir_barra_status(self):
        self.lbl_status = ctk.CTkLabel(self, text="", anchor="w", text_color=COR_NEUTRA, font=ctk.CTkFont(size=11))
        self.lbl_status.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))

    # -------------------------------------------------------------- Modelo
    def _carregar_modelo(self):
        if not baseline_existe():
            self.banner.grid()
            self._status("Modelo não encontrado.")
            self._atualizar_metricas()
            return
        try:
            self.classificador = ClassificadorBaseline()
        except Exception as e:  # arquivo corrompido, versão incompatível etc.
            self.classificador = None
            self.lbl_banner.configure(text=f"Falha ao carregar o modelo ({e}). Treine novamente.")
            self.banner.grid()
            self._status("Erro ao carregar o modelo.")
            self._atualizar_metricas()
            return
        self.banner.grid_remove()
        self._status("Modelo baseline (TF-IDF + Naive Bayes) carregado.")
        self._atualizar_metricas()
        if self.caixa.get("1.0", "end").strip():
            self._analisar()

    # ------------------------------------------------------------- Treino
    def _iniciar_treino(self):
        if self._treinando:
            return
        self._treinando = True
        self.banner.grid()
        self.lbl_banner.configure(text="Treinando o baseline (a primeira vez baixa o dataset)…")
        self.btn_treinar_banner.configure(state="disabled")
        self.btn_retreinar.configure(state="disabled")
        self.progresso_treino.grid()
        self.progresso_treino.start()
        self._status("Treinando em segundo plano…")
        threading.Thread(target=self._trabalho_treino, daemon=True).start()
        self.after(200, self._verificar_fila)

    def _trabalho_treino(self):
        try:
            from baseline import carregar_dataset, treinar_baseline

            df = carregar_dataset()
            modelo, vetorizador, metricas = treinar_baseline(df)
            salvar_baseline(modelo, vetorizador, metricas)
            self._fila.put(("ok", None))
        except Exception as e:
            self._fila.put(("erro", str(e)))

    def _verificar_fila(self):
        try:
            tipo, dado = self._fila.get_nowait()
        except queue.Empty:
            self.after(200, self._verificar_fila)
            return

        self._treinando = False
        self.progresso_treino.stop()
        self.progresso_treino.grid_remove()
        self.btn_treinar_banner.configure(state="normal")
        self.btn_retreinar.configure(state="normal")

        if tipo == "ok":
            self._carregar_modelo()
            self._status("Treino concluído e modelo salvo.")
        else:
            self.lbl_banner.configure(text=f"Falha no treino: {dado}")
            self.banner.grid()
            self._status("Falha no treino.")

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
        if not texto:
            self._resetar_resultado()
            return
        if self.classificador is None:
            self._status("Treine o modelo antes de analisar.")
            return
        resultado = self.classificador.classificar(texto, limiar=self.slider.get())
        self._mostrar_resultado(resultado)

    def _mostrar_resultado(self, r):
        cor = COR_SPAM if r.eh_spam else COR_OK
        self.lbl_veredito.configure(text="SPAM" if r.eh_spam else "Legítimo", text_color=cor)
        self.lbl_prob.configure(text=f"Probabilidade de spam: {r.prob_spam * 100:.1f}%")
        self.barra_prob.configure(progress_color=cor)
        self.barra_prob.set(r.prob_spam)
        self.lbl_suspeitas.configure(text=", ".join(p for p, _ in r.palavras_suspeitas) or "—")
        self.lbl_legitimas.configure(text=", ".join(p for p, _ in r.palavras_legitimas) or "—")

    def _resetar_resultado(self):
        self.lbl_veredito.configure(text="Aguardando mensagem", text_color=COR_NEUTRA)
        self.lbl_prob.configure(text="Probabilidade de spam: —")
        self.barra_prob.set(0)
        self.lbl_suspeitas.configure(text="—")
        self.lbl_legitimas.configure(text="—")

    def _ao_mudar_limiar(self, valor):
        self.lbl_limiar.configure(text=f"Spam se probabilidade ≥ {valor * 100:.0f}%")
        if self.classificador is not None and self.caixa.get("1.0", "end").strip():
            self._analisar()

    def _limpar(self):
        self.caixa.delete("1.0", "end")
        self._resetar_resultado()
        self.caixa.focus_set()

    def _usar_exemplo(self, texto):
        self.caixa.delete("1.0", "end")
        self.caixa.insert("1.0", texto)
        self._analisar()

    # ------------------------------------------------------------ Métricas
    def _atualizar_metricas(self):
        met = getattr(self.classificador, "metricas", None) if self.classificador else None
        if not met:
            self.lbl_sem_metricas.grid()
            self.quadro_metricas.grid_remove()
            self.quadro_matriz.grid_remove()
            self.lbl_amostra.configure(text="")
            return
        self.lbl_sem_metricas.grid_remove()
        self.quadro_metricas.grid()
        self.quadro_matriz.grid()
        for chave, lbl in self.valores_metricas.items():
            lbl.configure(text=f"{met.get(chave, 0) * 100:.1f}%")
        cm = met.get("matriz_confusao")
        if cm:
            for (r, c), cel in self.celulas_matriz.items():
                cel.configure(text=str(cm[r][c]))
        self.lbl_amostra.configure(
            text=f"Treino: {met.get('n_treino', '?')} exemplos | Teste: {met.get('n_teste', '?')} exemplos"
        )

    # -------------------------------------------------------------- Status
    def _status(self, texto):
        self.lbl_status.configure(text=texto)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
