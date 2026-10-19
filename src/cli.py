"""
cli.py — dna-predict CLI Bot
An animated, Cline-style terminal agent wrapping the DNA mutation predictor.

Launch with:  predict.bat   (or: python -m src.cli)

Features
--------
* animated boot sequence + live step checklist during every prediction
* Config Agent (AI #2) parses loose input, flags gaps, fills them interactively
* self-predict: type a gene and get its variants; 'all <gene>' batch-predicts
* fuzzy gene suggestions, autocomplete, session summary
* interactive protein graph (matplotlib) after predictions
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config_agent import (analyze, interactive_fix, nearby_variants,
                              suggest_genes, variants_for_gene)
from src.paths import data_path

BANNER_LINES = [
    r" ____  _   _    _    ____  _____   _   ____  _______ ",
    r"|  _ \| \ | |  / \  |  _ \|_   _| / \ |  _ \| ____|",
    r"| | | |  \| | / _ \ | |_) | | |  / _ \| |_) |  _|  ",
    r"| |_| | |\  |/ ___ \|  _ <  | | / ___ \  _ <| |___ ",
    r"|____/|_| \_/_/   \_\_| \_\ |_|/_/   \_\_| \_\_____|",
]
BANNER_COLORS = ["#00e5ff", "#00b0ff", "#7c4dff", "#b388ff", "#ff80ab"]

COMMAND_NAMES = ["predict", "all", "suggest", "info", "genes", "graph",
                 "3d", "graph3d", "summary", "help", "clear", "exit"]

HELP_TABLE_ROWS = [
    ("predict <gene> <variant>", "Predict one mutation — loose formats OK "
     "(e.g. predict TP53 R175H)"),
    ("<gene> <variant>", "Same thing; free text works too (TP53 p.Arg175His)"),
    ("all <gene> [limit]", "Self-predict every known variant of a gene, "
     "ranked (default limit 20)"),
    ("suggest <gene>", "Show known ClinVar variants recorded for a gene"),
    ("info <gene>", "Gene card: UniProt ID, protein length, variant stats"),
    ("genes [filter] [n]", "List the most-documented genes in the index"),
    ("graph <gene> <variant>", "Interactive 2D protein graph (click residues)"),
    ("3d <gene> <variant>", "Interactive 3D structure — drag to rotate, "
     "scroll to zoom (AlphaFold DB)"),
    ("summary", "Session statistics"),
    ("clear", "Clear the screen"),
    ("exit", "Quit (with session summary)"),
]


def type_out(console: Console, text: str, delay: float = 0.006,
             style: str = "cyan"):
    """Cline-style typewriter effect for bot chatter."""
    for ch in text:
        console.print(ch, style=style, end="", markup=False,
                      highlight=False)
        time.sleep(delay)
    console.print()


# ---------------------------------------------------------------------------
# Bot mascot — degrades gracefully on legacy consoles (codepage 437, etc.)
# ---------------------------------------------------------------------------
_UNICODE_OK = None


def _supports_box_chars() -> bool:
    """Best-effort: can this Windows console render Unicode glyphs?"""
    global _UNICODE_OK
    if _UNICODE_OK is not None:
        return _UNICODE_OK
    ok = True
    if os.name == "nt":
        try:
            import ctypes
            cp = ctypes.windll.kernel32.GetConsoleOutputCP()
            modern = (os.environ.get("WT_SESSION")
                      or os.environ.get("TERM_PROGRAM"))
            ok = (cp == 65001) or bool(modern)
        except Exception:
            ok = True
    _UNICODE_OK = ok
    return ok


def _eye_open() -> str:
    return "◉" if _supports_box_chars() else "O"


def _eye_blink() -> str:
    return "─" if _supports_box_chars() else "-"


def _mouth(idx: int) -> str:
    seq = (["‿", "_", "‿", "ᗜ", "‿"] if _supports_box_chars()
           else ["_", "-", "_", "o", "_"])
    return seq[idx % len(seq)]


def bot_face(eye: str = None, mouth: str = None) -> str:
    """Round bot face — Unicode bands when possible, pure ASCII otherwise."""
    e = eye or _eye_open()
    m = mouth or _mouth(0)
    if _supports_box_chars():
        return f"ᕙ({e}{m}{e})ᕗ"
    return f"({e}{m}{e})"


def bot_frames() -> list:
    if _supports_box_chars():
        return ["ᕙ(◉‿◉)ᕗ", "ᕙ(‿‿◉)ᕗ", "ᕙ(◉‿‿)ᕗ", "ᕙ(◉‿◉)ᕗ", "ᕙ(◉ᗜ◉)ᕗ"]
    return ["(O_O)", "(O_-)", "(-_O)", "(O_O)", "(O v O)"]


# register the animated bot spinner so every status line has the mascot
try:  # rich >= 10 exposes the table privately
    from rich._spinners import SPINNERS as _SPINNERS
except ImportError:  # pragma: no cover
    from rich.spinners import SPINNERS as _SPINNERS
_SPINNERS["bot"] = {"interval": 180, "frames": bot_frames()}

_ACTIVE_BOT = None  # set by DNABot.__init__ for the bottom toolbar


def bot_mascot_message() -> str:
    """Prompt-line bot: blinks, mouths, and follows the typing caret.
    Never raises — pure-ASCII fallback so it always appears."""
    import time as _t
    col = 0
    try:
        from prompt_toolkit.application import get_app
        app = get_app()
        buf = getattr(app, "current_buffer", None)
        if buf is not None:
            col = buf.cursor_position
    except Exception:
        col = 0
    try:
        eye = _eye_blink() if int(_t.time() * 3) % 19 == 0 else _eye_open()
        mouth = _mouth(int(_t.time() * 2.5))
        pad = min(max(col, 0), 22)
        return f"{' ' * pad}{bot_face(eye, mouth)} ❯ "
    except Exception:
        return "dna-bot ❯ "


def _bottom_toolbar() -> str:
    """Cline-style persistent status bar under the input line."""
    try:
        bot = _ACTIVE_BOT
        runs = bot.stats["runs"] if bot and bot.stats else 0
    except Exception:
        runs = 0
    return (" dna-bot v2.0 "
            f"· {runs} prediction(s) this session "
            "· Tab=complete · all <gene> = batch self-predict · "
            "3d <gene> <variant> · exit")


def _type_line(console: Console, text: str, style: str, delay: float = 0.003):
    """Typewriter-reveal a single line, centered + padded to console width."""
    text = text.center(max(20, console.width))
    for ch in text:
        console.print(ch, style=style, end="", markup=False, highlight=False)
        time.sleep(delay)
    console.print()


def risk_note(prob: float) -> str:
    if prob >= 0.9:
        return "very high confidence pathogenic — likely deleterious"
    if prob >= 0.75:
        return "likely pathogenic"
    if prob >= 0.6:
        return "weak evidence for pathogenicity"
    if prob >= 0.4:
        return "uncertain significance (grey zone)"
    if prob >= 0.25:
        return "leaning benign"
    return "confidently benign — likely tolerated"


class DNABot:
    """The terminal agent: Cline-style REPL around the prediction engine."""

    def __init__(self):
        global _ACTIVE_BOT
        self.console = Console()
        self.engine = None      # dict: model/selector/scaler/tokenizer/...
        self.history = []       # session predictions
        self.stats = {"runs": 0, "pathogenic": 0, "benign": 0, "skipped": 0}
        _ACTIVE_BOT = self

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------
    def boot(self):
        c = self.console
        c.clear()
        from rich.live import Live
        from rich.text import Text

        # ---- 1) animated ASCII logo reveal (typewriter, like Cline) ----
        for i, line in enumerate(BANNER_LINES):
            _type_line(c, line, f"bold {BANNER_COLORS[i]}", delay=0.003)
        c.print()

        # ---- 2) live startup strip: bot puppeteers a loading bar ----
        frames = bot_frames()
        t0 = time.time()
        dur = 1.6
        try:
            with Live(console=c, refresh_per_second=18,
                      transient=False) as live:
                while True:
                    el = time.time() - t0
                    if el >= dur:
                        break
                    face = frames[int(el * 5) % len(frames)]
                    pct = min(el / dur, 1.0)
                    bar = "█" * int(26 * pct) + "░" * (26 - int(26 * pct))
                    txt = Text.from_markup(
                        f"[bold cyan]{face}[/] [bold white]DNA-BOT v2.0[/] — "
                        f"mutation effect AI · [bold green]boot "
                        f"{int(pct*100)}%[/]\n"
                        f"[{'green' if pct > 0.5 else 'yellow'}]{bar}[/]")
                    live.update(txt)
                    time.sleep(0.055)
                live.update(Text.from_markup(
                    f"[bold cyan]{frames[-1]}[/] "
                    f"[bold white]DNA-BOT v2.0[/] — [green]✓[/] "
                    f"[dim]all systems nominal[/]"))
            c.print()
        except Exception:
            # any terminal that can't do Live still gets a static header
            c.print(Text.from_markup(
                f"[bold cyan]{frames[-1]}[/] [bold white]DNA-BOT v2.0[/] — "
                f"[bold green]boot complete ✓[/]"))
            c.print()

        # ---- 3) pinned header stays on screen above the working area ----
        c.print(Panel(
            Text.from_markup(
                f"[bold cyan]{frames[-1]}[/] [bold white]DNA-BOT v2.0[/] "
                f"[dim]·[/] type a gene or variant — [bold]TP53 R175H[/] · "
                f"[bold]all TP53[/] · [bold]3d TP53 R175H[/] · [bold]help[/]"),
            border_style="bright_cyan", box=box.ROUNDED), justify="center")
        c.print()

        def _ok(label, detail=""):
            time.sleep(0.1)
            c.print(f"  [green]✓[/green] [bold]{label}[/bold] "
                    f"[dim]{detail}[/dim]")

        _ok("terminal online",
            f"python {sys.version.split()[0]} on {sys.platform}")
        _ok("config agent ready", "gap-analysis AI standing by")

        with c.status(f"[cyan]{bot_face()} indexing ClinVar variants…[/]",
                      spinner="bot"):
            from src.config_agent import gene_index
            known, by_gene = gene_index()
        c.print(f"  [green]✓[/green] [bold]variant index built[/bold] "
                f"[dim]{len(known)} genes · "
                f"{sum(len(v) for v in by_gene.values())} missense variants"
                f"[/dim]")

        for f, label in ((data_path("scaler.pkl"), "XGBoost head"),
                         (data_path("esm_model.onnx"), "ESM-2 ONNX encoder"),
                         (data_path("variants.csv"), "ClinVar variants")):
            _ok(label, "found" if os.path.exists(f) else "MISSING")

        _ok("prediction engine", "lazy — wakes on first prediction")
        c.print()
        type_out(c, f"  dna-bot ❯ hi! {frames[-1]} here. type a gene or "
                    "variant and I'll take it from there.",
                 delay=0.004)
        c.print()

    # ------------------------------------------------------------------
    # Engine loading (lazy, heavy)
    # ------------------------------------------------------------------
    def ensure_engine(self):
        if self.engine is not None:
            return
        c = self.console
        import warnings
        import logging
        from src.predict import ESM_MODEL_NAME, ONNX_PATH, OnnxEsm
        with c.status("[cyan]ᕙ(◉‿◉)ᕗ waking prediction engine (first time "
                      "only)…[/]", spinner="bot"):
            warnings.filterwarnings("ignore")
            logging.disable(logging.WARNING)
            import transformers
            transformers.logging.set_verbosity_error()
            transformers.logging.disable_progress_bar()
            from transformers import AutoTokenizer
            from src.predict import load_model
            model, selector, scaler, sc_extra, base_models = load_model()
            tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
            if os.path.exists(ONNX_PATH):
                esm_model = OnnxEsm(ONNX_PATH)
                device = None
            else:
                import torch
                from transformers import EsmModel
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                esm_model = EsmModel.from_pretrained(
                    ESM_MODEL_NAME).to(device)
                esm_model.eval()
            logging.disable(logging.NOTSET)
            warnings.resetwarnings()
            self.engine = {
                "model": model, "selector": selector, "scaler": scaler,
                "tokenizer": tokenizer, "esm_model": esm_model,
                "device": device, "base_models": base_models,
                "sc_extra": sc_extra,
            }
        if getattr(self.engine["esm_model"], "onnx_mode", False):
            detail = "ONNX encoder · CPU fast path"
        else:
            detail = ("GPU (CUDA) · " if "cuda" in str(
                self.engine["device"]) else "CPU · ") + "torch encoder"
        c.print(f"  [green]✓[/green] engine online [dim]{detail} · "
                f"XGBoost head[/dim]")

    # ------------------------------------------------------------------
    # Config card display
    # ------------------------------------------------------------------
    def config_card(self, st, title="config agent — parsed request"):
        tbl = Table(show_header=False, box=box.SIMPLE_HEAD, pad_edge=False,
                    show_edge=False, header_style="bold")
        tbl.add_column("k", style="cyan", width=9)
        tbl.add_column("v")
        tbl.add_row("gene", f"[bold white]{st.gene}[/]" if st.gene
                    else "[yellow]— missing —[/]")
        tbl.add_row("variant", f"[bold white]{st.hgvsp}[/]" if st.hgvsp
                    else "[yellow]— missing —[/]")
        if st.complete:
            tbl.add_row("change", f"{st.wt_1} → {st.mt_1} at position "
                                  f"{st.pos} (1-letter)")
            tbl.add_row("source", {"parsed": "understood your input",
                                   "picked": "picked from known variants",
                                   "manual": "you provided it"}[st.source])
        for code, msg in st.issues:
            tbl.add_row("issue", f"[yellow]⚠ {msg}[/]")
        for code, msg in st.suggestions:
            tbl.add_row("idea", f"[dim]💡 {msg}[/]")
        self.console.print(Panel(tbl, title=f"[bold cyan]{title}[/]",
                                 border_style="cyan", box=box.ROUNDED))

    # ------------------------------------------------------------------
    # Prediction with animated Cline-style checklist
    # ------------------------------------------------------------------
    def run_prediction(self, st):
        """Run the prediction, streaming step checkmarks. Returns
        (label, prob, msg)."""
        self.ensure_engine()
        from src.predict import predict_with_steps
        gen = predict_with_steps(st.gene, st.hgvsp, **self.engine)

        q = queue.Queue()

        def worker():
            try:
                while True:
                    q.put(("step", next(gen)))
            except StopIteration as e:
                q.put(("done", e.value))
            except Exception as ex:
                q.put(("error", f"{type(ex).__name__}: {ex}"))

        th = threading.Thread(target=worker, daemon=True)
        th.start()

        c = self.console
        c.print(f"  [magenta]🧠[/magenta] [bold]task:[/bold] predict "
                f"[bold white]{st.hgvsp}[/bold white] in "
                f"[bold white]{st.gene}[/bold white]")
        t0 = time.time()
        result, error = None, None
        with c.status("[magenta]▐▌ thinking…[/]", spinner="dots"):
            while True:
                kind, payload = q.get()
                if kind == "done":
                    result = payload
                    break
                if kind == "error":
                    error = payload
                    break
                name, detail = payload
                c.print(f"  [green]✓[/green] [bold]{name}[/bold] "
                        f"[dim]— {detail} ({time.time() - t0:.1f}s)[/dim]")
        if error:
            c.print(f"  [red]✗[/red] [red]failed[/red] [dim]{error}[/dim]")
            return None, 0.0, error
        label, prob, msg = result
        if label is None:
            c.print(f"  [red]✗[/red] [red]{msg}[/red]")
        return label, prob, msg

    # ------------------------------------------------------------------
    # Verdict panel
    # ------------------------------------------------------------------
    @staticmethod
    def _evo2_parts(msg):
        """Parse '|esm=..|evo2=..|llr=..|dnabert=..' segments."""
        info = {}
        for seg in str(msg or "").split("|"):
            if "=" in seg:
                k, _, v = seg.partition("=")
                try:
                    info[k.strip()] = float(v)
                except ValueError:
                    pass
        return info

    def verdict_panel(self, st, label, prob, elapsed, msg=""):
        c = self.console
        patho = label == 1
        verdict = "PATHOGENIC" if patho else "BENIGN"
        col = "red" if patho else "green"
        filled = int(round(prob * 24))
        bar = "█" * filled + "░" * (24 - filled)

        tbl = Table(show_header=False, box=box.SIMPLE, pad_edge=False,
                    show_edge=False)
        tbl.add_column("k", style="dim", width=12)
        tbl.add_column("v")
        tbl.add_row("variant", f"[bold white]{st.gene} {st.hgvsp}[/]")
        tbl.add_row("change", f"{st.wt_1} → {st.mt_1} @ position {st.pos}")
        scores = self._evo2_parts(msg)
        if scores:
            if "esm" in scores:
                tbl.add_row("esm-2",
                            f"{scores.get('esm', prob):.4f} (protein view)")
            if "tab" in scores:
                tbl.add_row("population",
                            f"{scores['tab']} (gnomAD/SIFT/PolyPhen)")
            if "evo2" in scores:
                tbl.add_row("evo2-7b",
                            f"{scores['evo2']:.4f} (DNA LLR {scores.get('llr', 0.0):+.2f})")
            if "dnabert" in scores:
                tbl.add_row("dnabert-2",
                            f"{scores['dnabert']:.4f} (DNA cosine sim)")
            tbl.add_row("blended", f"[bold]{prob:.4f}[/]")
        else:
            tbl.add_row("model", "[dim]esm-2 only[/dim]")
        tbl.add_row("verdict",
                    Panel(Text(verdict, style=f"bold white on {col}",
                               justify="center"), box=box.ROUNDED))
        tbl.add_row("probability",
                    f"[{col}]{bar}[/{col}] [bold]{prob:.4f}[/]")
        tbl.add_row("reading", f"[italic]{risk_note(prob)}[/]")
        tbl.add_row("elapsed", f"{elapsed:.1f}s")
        c.print(Panel(tbl, title="[bold]🧬 prediction result[/]",
                      border_style=col, box=box.ROUNDED, padding=(0, 1)))

    # ------------------------------------------------------------------
    # Post-prediction suggestions + graph offer
    # ------------------------------------------------------------------
    def post_prediction(self, st):
        from rich.prompt import Confirm
        c = self.console

        near = nearby_variants(st.gene, st.pos, radius=10, n=5)
        known_ct = len(variants_for_gene(st.gene))
        lines = []
        if near:
            pretty = ", ".join(
                f"{hg} ({'patho' if lab else 'benign'})" for hg, lab in near)
            lines.append(f"known ClinVar variants within ±10 positions: "
                         f"{pretty}")
        lines.append(f"batch self-predict all {known_ct} known variants: "
                     f"[bold]all {st.gene}[/]")
        lines.append(f"re-render graph any time: "
                     f"[bold]graph {st.gene} {st.hgvsp}[/]")
        body = "\n".join(f"  💡 {ln}" for ln in lines)
        c.print(Panel(body, title="[bold magenta]suggestions[/]",
                      border_style="magenta", box=box.ROUNDED))

        from rich.prompt import Prompt
        view = Prompt.ask(
            "  [magenta]🧬 visualize?[/magenta]",
            choices=["2d", "3d", "skip"], default="3d", console=c).lower()
        if view == "2d":
            self.cmd_graph(st.gene, st.hgvsp)
            if Confirm.ask("  [magenta]also rotate it in 3D?[/]",
                           default=False, console=c):
                self.cmd_3d(st.gene, st.hgvsp)
        elif view == "3d":
            self.cmd_3d(st.gene, st.hgvsp)
        self.console.print()

    # ------------------------------------------------------------------
    # Query handler (free text / predict)
    # ------------------------------------------------------------------
    def handle_query(self, raw):
        from rich.prompt import Confirm
        st = analyze(raw)
        self.config_card(st, "config agent — what I understood")

        if not st.complete:
            st = interactive_fix(st, self.console)
            if st.complete:
                self.config_card(st, "config agent — after gap filling")

        if st.complete and any(code == "synonymous" for code, _ in st.issues):
            self.console.print(
                "  [yellow]⚠[/yellow] refusing to predict a synonymous "
                "change. Try a different residue.\n")
            return
        if not st.complete:
            type_out(self.console,
                     "  dna-bot ❯ I still need a gene + variant to work "
                     "with. Example: TP53 R175H", delay=0.003)
            self.console.print()
            return

        if not Confirm.ask("  [cyan]▶ proceed with prediction?[/]",
                           default=True, console=self.console):
            return

        t0 = time.time()
        label, prob, msg = self.run_prediction(st)
        if label is None:
            self.stats["skipped"] += 1
            if "UniProt" in str(msg):
                self.console.print(
                    "  💡 [dim]UniProt unreachable — check your internet "
                    "connection and retry.[/dim]")
            return
        self.stats["runs"] += 1
        self.stats["pathogenic" if label == 1 else "benign"] += 1
        self.history.append((st.gene, st.hgvsp, label, prob))
        self.verdict_panel(st, label, prob, time.time() - t0, msg)
        self.post_prediction(st)
        self.console.print()

    def _quiet_predict(self, st):
        """Silent prediction (no step stream) for batch runs."""
        from src.predict import predict_with_steps
        gen = predict_with_steps(st.gene, st.hgvsp, **self.engine)
        while True:
            try:
                next(gen)
            except StopIteration as e:
                return e.value

    # ------------------------------------------------------------------
    # Batch self-prediction: all <gene> [limit]
    # ------------------------------------------------------------------
    def cmd_all(self, gene, limit=20):
        from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                   SpinnerColumn, TextColumn,
                                   TimeElapsedColumn)
        from rich.prompt import Confirm

        gene = (gene or "").upper()
        variants = variants_for_gene(gene)
        if not variants:
            self.console.print(f"  [red]✗[/red] no known variants for "
                               f"'{gene}'.")
            hits = suggest_genes(gene)
            if hits:
                self.console.print(f"  💡 did you mean: "
                                   f"[bold]{', '.join(hits)}[/bold]?")
            return

        states = []
        for hg, _lab in variants:
            s = analyze(f"{gene} {hg}")
            if s.complete:
                states.append(s)
        states = states[:limit]

        type_out(self.console,
                 f"  dna-bot ❯ self-predicting {len(states)} variants of "
                 f"{gene} (of {len(variants)} on record)…", delay=0.003)
        self.ensure_engine()

        results, failed = [], 0
        with Progress(SpinnerColumn(),
                      TextColumn("[progress.description]{task.description}"),
                      BarColumn(), MofNCompleteColumn(),
                      TimeElapsedColumn(), console=self.console) as prog:
            task = prog.add_task(f"predicting {gene} variants",
                                 total=len(states))
            for s in states:
                prog.update(task, description=f"predicting {s.hgvsp}")
                try:
                    label, prob, msg = self._quiet_predict(s)
                except Exception as ex:
                    label, msg = None, str(ex)
                if label is None:
                    failed += 1
                else:
                    evo = self._evo2_parts(msg)
                    results.append({"hgvsp": s.hgvsp, "pos": s.pos,
                                    "wt": s.wt_1, "mt": s.mt_1,
                                    "label": label, "prob": prob,
                                    "esm": evo.get("esm"),
                                    "pop": evo.get("tab"),
                                    "evo2": evo.get("evo2"),
                                    "llr": evo.get("llr"),
                                    "dnabert": evo.get("dnabert")})
                prog.advance(task)

        results.sort(key=lambda r: -r["prob"])
        self.stats["runs"] += len(results)
        self.stats["skipped"] += failed
        for r in results:
            self.stats["pathogenic" if r["label"] == 1 else "benign"] += 1
            self.history.append((gene, r["hgvsp"], r["label"], r["prob"]))
        self._last_batch = (gene, results)

        tbl = Table(title=f"[bold]🧬 {gene} — ranked predictions "
                          f"({len(results)} ok, {failed} skipped)[/]",
                    box=box.ROUNDED, header_style="bold cyan")
        tbl.add_column("#", justify="right", style="dim", width=4)
        tbl.add_column("variant", style="white")
        tbl.add_column("change", style="dim")
        tbl.add_column("esm-2", justify="right", style="dim")
        tbl.add_column("pop", justify="right", style="dim")
        tbl.add_column("evo2", justify="right", style="dim")
        tbl.add_column("dnabert", justify="right", style="dim")
        tbl.add_column("blended", justify="right")
        tbl.add_column("verdict", justify="center")
        for i, r in enumerate(results, 1):
            col = "red" if r["label"] == 1 else "green"
            esm_s = f"{r['esm']:.3f}" if r.get("esm") is not None else "—"
            pop_s = f"{r['pop']:.3f}" if r.get("pop") else "—"
            evo_s = (f"{r['evo2']:.3f} ({r['llr']:+.1f})"
                     if r.get("evo2") is not None else "—")
            dna_s = f"{r['dnabert']:.3f}" if r.get("dnabert") is not None else "—"
            tbl.add_row(str(i), r["hgvsp"], f"{r['wt']}{r['pos']}{r['mt']}",
                        esm_s, pop_s, evo_s, dna_s, f"{r['prob']:.4f}",
                        f"[{col}]{'PATHO' if r['label'] else 'benign'}[/]")
        self.console.print(tbl)

        if results and Confirm.ask(
                "  [magenta]🧬 open the interactive graph colored by these "
                "predictions?[/]", default=False, console=self.console):
            self.cmd_graph(gene, results[0]["hgvsp"], results=results)
        if results and Confirm.ask(
                "  [magenta]🧬 rotate the {g} structure in interactive 3D?"
                "[/]".replace("{g}", gene),
                default=False, console=self.console):
            self.cmd_3d(gene, results[0]["hgvsp"], results=results)
        self.console.print()

    # ------------------------------------------------------------------
    # Interactive protein graphs (2D + 3D)
    # ------------------------------------------------------------------
    def _fetch_protein(self, gene):
        """Return (uniprot_id, sequence) or (None, None)."""
        with self.console.status("[cyan]ᕙ(◉‿◉)ᕗ fetching protein "
                                 "sequence…[/]", spinner="bot"):
            from src.engine import get_protein_sequence, get_uniprot_id
            uniprot = get_uniprot_id(gene)
            seq = get_protein_sequence(uniprot) if uniprot else None
        return uniprot, seq

    def cmd_graph(self, gene=None, hgvsp=None, results=None):
        gene = (gene or "").upper()
        if not gene or not hgvsp:
            self.console.print("  [red]✗[/red] usage: graph <gene> <variant>")
            return
        uniprot, seq = self._fetch_protein(gene)
        if not seq:
            self.console.print(
                "  [red]✗[/red] could not fetch the {g} protein sequence "
                "from UniProt — is the internet up?".replace("{g}", gene))
            return
        st = analyze(f"{gene} {hgvsp}")
        if not st.complete:
            self.console.print(f"  [red]✗[/red] could not parse variant "
                               f"'{hgvsp}'.")
            return
        verdict = None
        if results:
            best = next((r for r in results if r["hgvsp"] == st.hgvsp), None)
            if best:
                verdict = "PATHOGENIC" if best["label"] == 1 else "benign"
        self.console.print(
            f"  [green]✓[/green] {gene}: UniProt [dim]{uniprot}[/dim], "
            f"{len(seq)} residues — close the graph window to return.")
        from src.protein_graph import show_protein_graph
        show_protein_graph(gene, seq, st.wt_1, st.pos, st.mt_1,
                           results=results, verdict=verdict)

    def cmd_3d(self, gene=None, hgvsp=None, results=None):
        """Interactive rotatable/zoomable 3D structure (AlphaFold DB)."""
        gene = (gene or "").upper()
        if not gene or not hgvsp:
            self.console.print("  [red]✗[/red] usage: 3d <gene> <variant>")
            return
        uniprot, seq = self._fetch_protein(gene)
        if not seq:
            self.console.print(
                "  [red]✗[/red] could not fetch the {g} protein sequence "
                "from UniProt — is the internet up?".replace("{g}", gene))
            return
        st = analyze(f"{gene} {hgvsp}")
        if not st.complete:
            self.console.print(f"  [red]✗[/red] could not parse variant "
                               f"'{hgvsp}'.")
            return
        verdict = None
        if results:
            best = next((r for r in results if r["hgvsp"] == st.hgvsp), None)
            if best:
                verdict = "PATHOGENIC" if best["label"] == 1 else "benign"
        with self.console.status("[cyan]ᕙ(◉‿◉)ᕗ loading 3D structure "
                                 "(AlphaFold DB)…[/]", spinner="bot"):
            from src.protein_graph import show_protein_3d
        self.console.print(
            "  [green]✓[/green] 3D view opening — [dim]drag to rotate, "
            "scroll to zoom; falls back to a synthetic backbone if "
            "AlphaFold is unreachable.[/dim]")
        show_protein_3d(gene, seq, st.wt_1, st.pos, st.mt_1, uniprot=uniprot,
                        results=results, verdict=verdict)

    # ------------------------------------------------------------------
    # Misc commands
    # ------------------------------------------------------------------
    def cmd_suggest(self, gene):
        gene = (gene or "").upper()
        variants = variants_for_gene(gene)
        if not variants:
            hits = suggest_genes(gene)
            self.console.print(f"  [red]✗[/red] no records for '{gene}'."
                               + (f" Did you mean [bold]{hits[0]}[/bold]?"
                                  if hits else ""))
            return
        patho = sum(1 for _, lab in variants if lab == 1)
        tbl = Table(title=f"[bold]📋 {gene} — {len(variants)} known variants "
                          f"({patho} pathogenic, {len(variants) - patho} "
                          f"benign)[/]",
                    box=box.ROUNDED, header_style="bold cyan")
        tbl.add_column("#", justify="right", style="dim", width=4)
        tbl.add_column("HGVSp", style="white")
        tbl.add_column("ClinVar label", justify="center")
        for i, (hg, lab) in enumerate(variants[:25], 1):
            tbl.add_row(str(i), hg,
                        "[red]Pathogenic[/red]" if lab else "[green]Benign[/green]")
        if len(variants) > 25:
            tbl.caption = (f"showing 25 of {len(variants)} — try "
                           f"[bold]all {gene}[/bold] to batch-predict")
        self.console.print(tbl)
        self.console.print()

    def cmd_info(self, gene):
        gene = (gene or "").upper()
        if gene not in suggest_genes(gene, 1):
            hits = suggest_genes(gene)
            self.console.print(f"  [yellow]?[/yellow] '{gene}' not in index."
                               + (f" Did you mean [bold]{hits[0]}[/bold]?"
                                  if hits else ""))
            return
        from src.config_agent import gene_index
        _, by_gene = gene_index()
        variants = variants_for_gene(gene)
        patho = sum(1 for _, lab in variants if lab == 1)
        tbl = Table(show_header=False, box=box.ROUNDED, pad_edge=False)
        tbl.add_column("k", style="cyan", width=14)
        tbl.add_column("v")
        tbl.add_row("gene", f"[bold white]{gene}[/]")
        tbl.add_row("known variants", str(len(variants)))
        tbl.add_row("pathogenic", str(patho))
        tbl.add_row("benign", str(len(variants) - patho))
        tbl.add_row("top variant", variants[0][0] if variants else "—")
        tbl.add_row("batch predict", f"[bold]all {gene}[/]")
        self.console.print(Panel(tbl, title="[bold cyan]🧾 gene card[/]",
                                 border_style="cyan", box=box.ROUNDED))

    def cmd_genes(self, args):
        from src.config_agent import gene_index
        _, by_gene = gene_index()
        filt = args[0].upper() if args else ""
        n = int(args[1]) if len(args) > 1 and args[1].isdigit() else 15
        ranked = sorted(by_gene.items(), key=lambda kv: -len(kv[1]))
        if filt:
            ranked = [(g, v) for g, v in ranked if filt in g]
        title = ("[bold]🔎 most-documented genes[/]" if not filt
                 else f"[bold]🔎 genes matching {filt}[/]")
        tbl = Table(title=title, box=box.ROUNDED, header_style="bold cyan")
        tbl.add_column("#", justify="right", style="dim", width=4)
        tbl.add_column("gene", style="bold white")
        tbl.add_column("variants", justify="right")
        tbl.add_column("pathogenic", justify="right", style="red")
        for i, (g, v) in enumerate(ranked[:n], 1):
            p = sum(1 for _, lab in v if lab == 1)
            tbl.add_row(str(i), g, str(len(v)), str(p))
        self.console.print(tbl)
        self.console.print()

    def cmd_summary(self):
        s = self.stats
        tbl = Table(show_header=False, box=box.ROUNDED, pad_edge=False)
        tbl.add_column("k", style="cyan", width=16)
        tbl.add_column("v")
        tbl.add_row("predictions run", str(s["runs"]))
        tbl.add_row("pathogenic", f"[red]{s['pathogenic']}[/red]")
        tbl.add_row("benign", f"[green]{s['benign']}[/green]")
        tbl.add_row("skipped/failed", str(s["skipped"]))
        if self.history:
            last = self.history[-1]
            tbl.add_row("last", f"{last[0]} {last[1]} → "
                                f"{'PATHO' if last[2] else 'benign'} "
                                f"({last[3]:.3f})")
        self.console.print(Panel(tbl, title="[bold]📊 session summary[/]",
                                 border_style="cyan", box=box.ROUNDED))

    def cmd_help(self):
        tbl = Table(title="[bold]dna-bot command reference[/]",
                    box=box.ROUNDED, header_style="bold cyan")
        tbl.add_column("command", style="bold white")
        tbl.add_column("what happens")
        for cmd, desc in HELP_TABLE_ROWS:
            tbl.add_row(cmd, desc)
        self.console.print(tbl)
        self.console.print()

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------
    def repl(self):
        session = None
        try:
            if sys.stdin is not None and sys.stdin.isatty():
                session = _make_prompt_session()
        except Exception:
            session = None

        while True:
            try:
                if session is not None:
                    line = session.prompt()
                else:
                    self.console.print("[bold cyan]dna-bot ❯[/bold cyan]",
                                       end="")
                    line = input(" ")
            except (KeyboardInterrupt, EOFError):
                line = "exit"
            except Exception:
                line = input("dna-bot ❯ ")
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()
            rest = parts[1:]

            try:
                if cmd in ("exit", "quit", "q", "bye"):
                    self.cmd_summary()
                    type_out(self.console,
                             "  dna-bot ❯ goodbye! stay sequenced. 🧬",
                             delay=0.004)
                    return
                elif cmd in ("help", "?", "h"):
                    self.cmd_help()
                elif cmd in ("clear", "cls"):
                    self.boot()
                elif cmd == "summary":
                    self.cmd_summary()
                elif cmd == "predict" and rest:
                    self.handle_query(" ".join(rest))
                elif cmd == "all" and rest:
                    self.cmd_all(rest[0],
                                 int(rest[1]) if len(rest) > 1
                                 and rest[1].isdigit() else 20)
                elif cmd == "all":
                    self.console.print("  [red]✗[/red] usage: all <gene> "
                                       "[limit]")
                elif cmd == "suggest" and rest:
                    self.cmd_suggest(rest[0])
                elif cmd == "info" and rest:
                    self.cmd_info(rest[0])
                elif cmd == "genes":
                    self.cmd_genes(rest)
                elif cmd == "graph" and len(rest) >= 2:
                    self.cmd_graph(rest[0], " ".join(rest[1:]))
                elif cmd == "graph":
                    self.console.print("  [red]✗[/red] usage: graph <gene> "
                                       "<variant>")
                elif cmd in ("3d", "graph3d") and len(rest) >= 2:
                    self.cmd_3d(rest[0], " ".join(rest[1:]))
                elif cmd in ("3d", "graph3d"):
                    self.console.print("  [red]✗[/red] usage: 3d <gene> "
                                       "<variant>")
                else:
                    # anything else = free text for the config agent
                    self.handle_query(line)
            except KeyboardInterrupt:
                self.console.print("\n  [yellow]⚠ interrupted[/yellow]")
            except Exception as ex:
                self.console.print(
                    f"  [red]✗[/red] unexpected error: "
                    f"{type(ex).__name__}: {ex}")
                self.console.print(
                    "  💡 [dim]try 'help' — and if it persists, rerun "
                    "predict.bat[/dim]")


# ----------------------------------------------------------------------
# Autocomplete (prompt_toolkit, optional)
# ----------------------------------------------------------------------
def _make_prompt_session():
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import InMemoryHistory

    class BotCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            words = text.split()
            from src.config_agent import suggest_genes
            if len(words) <= 1 and not text.endswith(" "):
                frag = words[0] if words else ""
                for w in COMMAND_NAMES:
                    if w.startswith(frag.lower()):
                        yield Completion(w, start_position=-len(frag))
                if frag and frag == frag.upper() and len(frag) >= 2:
                    for g in suggest_genes(frag, n=5):
                        yield Completion(g, start_position=-len(frag))
            elif words:
                frag = "" if text.endswith(" ") else words[-1]
                if (frag and frag == frag.upper() and len(frag) >= 2
                        and words[0].lower() in COMMAND_NAMES):
                    for g in suggest_genes(frag, n=5):
                        yield Completion(g, start_position=-len(frag))

    return PromptSession(history=InMemoryHistory(), completer=BotCompleter(),
                         message=bot_mascot_message, mouse_support=True,
                         refresh_interval=0.1, bottom_toolbar=_bottom_toolbar,
                         complete_style="readline")


# ----------------------------------------------------------------------
# Self-test (no model load): python -m src.cli --selftest
# ----------------------------------------------------------------------
def _selftest():
    import tempfile

    console = Console()
    ok = True

    cases = [
        ("TP53 p.Arg175His", True, None),
        ("BRCA1 K50Q", True, None),
        ("what about PIK3CA E545K", True, None),
        ("TP35 R175H", False, None),          # fuzzy suggestion expected
        ("GLA p.Gln301Ter", False, "unsupported_type"),
    ]
    from src.config_agent import analyze
    for raw, want_complete, want_issue in cases:
        st = analyze(raw)
        got = st.complete
        issue_ok = want_issue is None or any(
            c == want_issue for c, _ in st.issues)
        good = got == want_complete and issue_ok
        if not good:
            ok = False
        console.print(f"  [{'green' if good else 'red'}]"
                      f"{'✓' if good else '✗'}[/] analyze({raw!r}) -> "
                      f"gene={st.gene} hgvsp={st.hgvsp} complete={got}")

    # graph smoke test (headless save)
    try:
        from src.protein_graph import show_protein_graph
        seq = ("MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED"
               "LQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN") * 2
        fig = show_protein_graph(
            "INS", seq, "C", 30, "G",
            results=[{"hgvsp": "p.Cys30Gly", "prob": 0.91, "label": 1},
                     {"hgvsp": "p.Leu35Pro", "prob": 0.12, "label": 0}],
            verdict="PATHOGENIC", block=False)
        out = os.path.join(tempfile.gettempdir(),
                           "dna_predict_graph_selftest.png")
        fig.savefig(out, dpi=80)
        console.print(f"  [green]✓[/green] 2D graph smoke test saved: {out}")
    except Exception as ex:
        ok = False
        console.print(f"  [red]✗[/red] 2D graph smoke test failed: "
                      f"{type(ex).__name__}: {ex}")

    # 3D structure smoke test (offline synthetic backbone, no network)
    try:
        from src.protein_graph import show_protein_3d, synthetic_structure
        seq3d = ("MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT") * 3
        fig = show_protein_3d(
            "INS", seq3d, "C", 30, "G", uniprot=None,
            results=[{"hgvsp": "p.Cys30Gly", "prob": 0.91, "label": 1}],
            verdict="PATHOGENIC", block=False)
        out = os.path.join(tempfile.gettempdir(),
                           "dna_predict_graph3d_selftest.png")
        fig.savefig(out, dpi=80)
        console.print(f"  [green]✓[/green] 3D graph smoke test saved: {out}")
        _ = synthetic_structure(seq3d)
        console.print(f"  [green]✓[/green] mascot spinner frames: "
                      f"{_SPINNERS['bot']['frames']}")
    except Exception as ex:
        ok = False
        console.print(f"  [red]✗[/red] 3D graph smoke test failed: "
                      f"{type(ex).__name__}: {ex}")

    console.print(Panel(
        "[bold green]SELFTEST PASSED[/]" if ok
        else "[bold red]SELFTEST FAILED[/]",
        border_style="green" if ok else "red"))
    sys.exit(0 if ok else 1)


def _demo():
    """Scripted visual walkthrough — renders the whole bot UI (boot, config
    card, verdict panels, suggestions, tables, summary) *without* waking the
    ML engine. Use for screenshots / visual comparisons:

        python -m src.cli --demo
    """
    import time as _t

    bot = DNABot()
    bot.boot()
    c = bot.console

    def pause(s=0.35):
        _t.sleep(s)

    # ---- sample human query --------------------------------------------
    type_out(c, "  dna-bot: you said \u201cwhat about TP53 \u2014 the "
                "R175H hotspot? any of the other variants alarming?\u201d",
             delay=0.004)
    pause(0.6)
    type_out(c, "  dna-bot \u276f parsing your request and checking what "
                "you might mean\u2026", delay=0.004)
    pause(0.5)

    from src.config_agent import analyze

    # ---- config agent card ----------------------------------------------
    st = analyze("what about TP53 R175H?")
    bot.config_card(st, "config agent \u2014 what I understood")
    pause(0.7)

    # ---- proceed prompt (simulated) --------------------------------------
    type_out(c, "  dna-bot \u276f looks complete. Running the classifier\u2026",
             delay=0.004)
    pause(0.5)

    # ---- animated step checklist (simulated) -----------------------------
    c.print("  [magenta]\U0001f9e8[/magenta] [bold]task:[/bold] predict "
            "[bold white]p.Arg175His[/bold white] in "
            "[bold white]TP53[/bold white]")
    for _name, _detail, _t0 in (
            ("fetch UniProt sequence", "TP53 = a UniProt P04637", 2.0),
            ("ESM-2 embed (wild-type)", "64 residues \u00b7 ONNX", 2.05),
            ("ESM-2 embed (mutant)", "Arg175His \u00b7 ONNX", 2.1),
            ("feature select + scale", "2048 dims -> selector", 2.15),
            ("classifier", "XGBoost \u00b7 500 trees", 2.2)):
        pause(0.1)
        c.print(f"  [green]\u2713[/green] [bold]{_name}[/bold] "
                f"[dim]\u2014 {_detail} ({_t0:.1f}s)[/dim]")

    # ---- verdict panels --------------------------------------------------
    st_patho = analyze("TP53 p.Arg175His")
    bot.verdict_panel(st_patho, 1, 0.9906, 3.1)
    pause(0.8)
    st_benign = analyze("BRCA1 p.Lys50Gln")
    bot.verdict_panel(st_benign, 0, 0.032, 2.8)
    pause(0.8)

    # ---- suggestions panel (static, no confirm prompt) --------------------
    near = ", ".join(f"{hg} ({'patho' if lab else 'benign'})"
                     for hg, lab in
                     [("p.Gln144Ter", 1), ("p.Tyr220Cys", 1),
                      ("p.Gly245Ser", 1), ("p.Pro278Leu", 0)][:4])
    body = "\n".join([
        "  \U0001f4a1 known ClinVar variants within \u00b110 positions: "
        f"{near}",
        "  \U0001f4a1 batch self-predict all 2144 known variants: "
        "[bold]all TP53[/]",
        "  \U0001f4a1 re-render graph any time: [bold]graph TP53 "
        "p.Arg175His[/]",
        "  \U0001f4a1 view the mutation in context: [bold]3d TP53 R175H[/]",
    ])
    c.print(Panel(body, title="[bold magenta]suggestions[/]",
                  border_style="magenta", box=box.ROUNDED))
    pause(0.8)

    # ---- ranked batch table ------------------------------------------------
    tbl = Table(title="[bold]\U0001f9ec TP53 \u2014 ranked predictions "
                       "(id: demo, 4 results)[/]",
                box=box.ROUNDED, header_style="bold cyan")
    tbl.add_column("#", justify="right", style="dim", width=4)
    tbl.add_column("variant", style="white")
    tbl.add_column("change", style="dim")
    tbl.add_column("probability", justify="right")
    tbl.add_column("verdict", justify="center")
    for i, (hg, ch, p) in enumerate([
            ("p.Arg175His", "R175H", 0.9906),
            ("p.Gly245Ser", "G245S", 0.9512),
            ("p.Pro278Leu", "P278L", 0.6130),
            ("p.Arg181Cys", "R181C", 0.0240)], 1):
        col = "red" if p >= 0.5 else "green"
        tbl.add_row(str(i), hg, ch, f"{p:.4f}",
                    f"[{col}]{'PATHO' if p >= 0.5 else 'benign'}[/]")
    c.print(tbl)
    pause(1.0)

    # ---- gene card ---------------------------------------------------------
    bot.cmd_info("TP53")
    pause(0.8)

    # ---- session summary -----------------------------------------------------
    bot.stats.update({"runs": 6, "pathogenic": 4, "benign": 2, "skipped": 0})
    bot.history.append(("TP53", "p.Arg175His", 1, 0.9906))
    bot.cmd_summary()

    c.print()
    type_out(c, "  dna-bot \u276f that was a demo \u2014 your real commands work "
                "the same way. TP53 R175H to start!", delay=0.004)
    pause(0.3)
    c.print()


def main():
    if "--selftest" in sys.argv:
        _selftest()
        return
    if "--demo" in sys.argv:
        _demo()
        return
    bot = DNABot()
    bot.boot()
    try:
        bot.repl()
    except Exception as ex:
        bot.console.print(f"[red]fatal: {type(ex).__name__}: {ex}[/red]")
        raise


if __name__ == "__main__":
    main()

