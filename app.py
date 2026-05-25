"""Gradio entrypoint for AI Personal Assistant Comparison Lab."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from assistants.memory import clamp_memory_turns, get_default_max_turns
from assistants.oss_assistant import OSSAssistant, is_model_loaded, is_model_loading
from assistants.pipeline import run_turn

DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local").strip().lower()
IS_SPACES_MODE = DEPLOYMENT_MODE == "spaces"
APP_TITLE = "AI Personal Assistant Comparison Lab"
LOG_PATH = Path("logs/conversations.jsonl")
RESULTS_CSV_PATH = Path("reports/results.csv")

OSS_LABEL = "Open Source Assistant - Qwen2.5-0.5B-Instruct"
GEMINI_LABEL = "Frontier Assistant - Gemini"
MODEL_CHOICES = [OSS_LABEL, GEMINI_LABEL]


def _get_assistant(model_choice: str):
    if GEMINI_LABEL in model_choice:
        from assistants.gemini_assistant import GeminiAssistant

        return GeminiAssistant()
    return OSSAssistant()


def _format_metadata(result) -> tuple[str, str, str, str]:
    model_line = f"**Model:** {result.model_name}"
    latency_line = f"**Latency:** {result.latency_seconds:.2f}s (end-to-end)"
    if result.latency_model_seconds is not None:
        latency_line += f" | model: {result.latency_model_seconds:.2f}s"
    safety_line = f"**Safety:** {'triggered' if result.safety_triggered else 'ok'}"
    tool_line = f"**Tool used:** {result.tool_used or 'none'}"
    if result.error:
        tool_line += f" | **Error:** {result.error}"
    return model_line, latency_line, safety_line, tool_line


def _log_interaction(user_message: str, result) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "assistant_type": result.assistant_type,
        "model_name": result.model_name,
        "user_message": user_message,
        "assistant_response": result.text,
        "latency_seconds": result.latency_seconds,
        "latency_model_seconds": result.latency_model_seconds,
        "safety_triggered": result.safety_triggered,
        "tool_used": result.tool_used,
        "error": result.error,
    }
    try:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Hugging Face Spaces may restrict disk writes; chat still works.
        pass


def build_ui() -> gr.Blocks:
    default_memory = get_default_max_turns()

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        if IS_SPACES_MODE:
            gr.Markdown(
                "**Public demo (OSS only).** "
                "Full Gemini comparison and evaluation run locally — see README."
            )
        else:
            gr.Markdown(
                "Compare **Qwen2.5-0.5B-Instruct** (OSS) vs **Gemini** (frontier) "
                "with shared memory, guardrails, and tools."
            )

        session_history = gr.State([])

        with gr.Tab("Chat"):
            with gr.Row():
                if IS_SPACES_MODE:
                    gr.Markdown(f"**Assistant:** {OSS_LABEL}")
                    model_choice = gr.State(OSS_LABEL)
                else:
                    model_choice = gr.Radio(
                        choices=MODEL_CHOICES,
                        value=OSS_LABEL,
                        label="Assistant",
                    )

            with gr.Row():
                guardrails_toggle = gr.Checkbox(value=True, label="Safety guardrails")
                tools_toggle = gr.Checkbox(value=True, label="Tool use")
                memory_slider = gr.Slider(
                    minimum=2,
                    maximum=12,
                    value=default_memory,
                    step=1,
                    label="Memory turns (exchanges)",
                )

            chatbot = gr.Chatbot(
                label="Conversation",
                height=420,
                type="messages",
            )
            msg = gr.Textbox(label="Message", placeholder="Type a message…")

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_memory_btn = gr.Button("Clear memory")
                clear_chat_btn = gr.Button("Clear chat")

            ready_status = (
                "*Ready.* First message downloads and loads Qwen2.5-0.5B on CPU — "
                "please allow 1–3 minutes."
                if IS_SPACES_MODE
                else "*Ready.* OSS first message may take a minute while the model loads."
            )
            status_box = gr.Markdown(value=ready_status)
            model_display = gr.Markdown()
            latency_display = gr.Markdown()
            safety_display = gr.Markdown()
            tool_display = gr.Markdown()

            def respond(
                user_message: str,
                chat_history: list,
                history_state: list,
                model: str,
                guardrails_on: bool,
                tools_on: bool,
                memory_turns: float,
            ):
                if not user_message.strip():
                    return chat_history, history_state, "", gr.update(), gr.update(), gr.update(), gr.update()

                assistant = _get_assistant(model)
                if isinstance(assistant, OSSAssistant) and not is_model_loaded():
                    if is_model_loading():
                        status = "*Loading open-source model…*"
                    else:
                        status = "*Loading open-source model (first request)…*"
                else:
                    status = "*Generating response…*"

                result = run_turn(
                    assistant,
                    user_message,
                    history_state,
                    guardrails_enabled=guardrails_on,
                    tools_enabled=tools_on,
                    memory_turns=clamp_memory_turns(int(memory_turns)),
                )

                _log_interaction(user_message, result)

                new_history = list(history_state)
                new_history.append({"role": "user", "content": user_message})
                new_history.append({"role": "assistant", "content": result.text})

                chat_history = list(chat_history or [])
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": result.text})

                meta = _format_metadata(result)
                status = "*Ready.*"
                return (
                    chat_history,
                    new_history,
                    "",
                    status,
                    meta[0],
                    meta[1],
                    meta[2],
                    meta[3],
                )

            def clear_memory_only(history_state: list):
                return [], "*Memory cleared.*", gr.update(), gr.update(), gr.update(), gr.update()

            def clear_chat(history_state: list):
                return [], [], "*Chat and memory cleared.*", "", "", "", ""

            inputs = [
                msg,
                chatbot,
                session_history,
                model_choice,
                guardrails_toggle,
                tools_toggle,
                memory_slider,
            ]
            outputs = [
                chatbot,
                session_history,
                msg,
                status_box,
                model_display,
                latency_display,
                safety_display,
                tool_display,
            ]

            send_btn.click(respond, inputs, outputs)
            msg.submit(respond, inputs, outputs)
            clear_memory_btn.click(
                clear_memory_only,
                inputs=[session_history],
                outputs=[session_history, status_box, model_display, latency_display, safety_display, tool_display],
            )
            clear_chat_btn.click(
                clear_chat,
                inputs=[session_history],
                outputs=[
                    chatbot,
                    session_history,
                    status_box,
                    model_display,
                    latency_display,
                    safety_display,
                    tool_display,
                ],
            )

        if not IS_SPACES_MODE:
            import pandas as pd

            eval_state = gr.State({"rows": [], "metrics": {}})

            with gr.Tab("Evaluation"):
                gr.Markdown(
                    "### Evaluation\n"
                    "Runs through the same pipeline as chat (guardrails + tools on). "
                    "Prompts are **balanced across categories** (not just the first N in the file). "
                    "CLI: `python -m evals.run_eval` for all 40 prompts, or `--max-prompts N`."
                )
                with gr.Row():
                    eval_max_prompts = gr.Slider(
                        minimum=5,
                        maximum=40,
                        value=10,
                        step=1,
                        label="Max prompts (stratified across categories)",
                        info="10 ≈ 2 per category; 40 = full eval set.",
                    )
                    use_llm_judge = gr.Checkbox(
                        label="Use LLM judge for OSS factual/memory (extra Gemini calls)",
                        value=False,
                    )
                run_eval_btn = gr.Button("Run evaluation", variant="primary")
                clear_eval_btn = gr.Button("Clear results")
                eval_status = gr.Markdown()
                metrics_summary = gr.JSON(label="Metrics summary")
                results_table = gr.Dataframe(label="Results", interactive=False)
                download_csv = gr.DownloadButton(
                    label="Download results CSV",
                    value=None,
                    interactive=False,
                )

                def run_sample_evaluation(max_prompts: float, llm_judge: bool):
                    from evals.generate_report import generate_report
                    from evals.run_eval import run_evaluation

                    limit = int(max_prompts)
                    full_run = limit >= 40

                    try:
                        summary = run_evaluation(
                            max_prompts=None if full_run else limit,
                            use_llm_judge=llm_judge,
                        )
                    except Exception as exc:
                        return (
                            {"error": str(exc)},
                            pd.DataFrame(),
                            f"Evaluation failed: {exc}",
                            gr.update(value=None, interactive=False),
                            {"rows": [], "metrics": {}},
                        )

                    try:
                        generate_report()
                    except Exception:
                        pass

                    rows = summary.get("rows", [])
                    metrics = summary.get("metrics", {})
                    message = summary.get("message", "")
                    status = message or f"Completed {len(rows)} result row(s)."
                    if not full_run:
                        status += " Report updated at `reports/evaluation_report.md`."
                    df = pd.DataFrame(rows) if rows else pd.DataFrame(
                        columns=["prompt_id", "assistant_type", "response"]
                    )
                    csv_path = str(RESULTS_CSV_PATH) if RESULTS_CSV_PATH.exists() else None
                    return (
                        metrics,
                        df,
                        status,
                        gr.update(value=csv_path, interactive=bool(csv_path)),
                        summary,
                    )

                def clear_evaluation():
                    if RESULTS_CSV_PATH.exists():
                        RESULTS_CSV_PATH.unlink()
                    json_path = RESULTS_CSV_PATH.with_suffix(".json")
                    if json_path.exists():
                        json_path.unlink()
                    return (
                        {},
                        pd.DataFrame(),
                        "Results cleared.",
                        gr.update(value=None, interactive=False),
                        {"rows": [], "metrics": {}},
                    )

                run_eval_btn.click(
                    run_sample_evaluation,
                    inputs=[eval_max_prompts, use_llm_judge],
                    outputs=[metrics_summary, results_table, eval_status, download_csv, eval_state],
                )
                clear_eval_btn.click(
                    clear_evaluation,
                    outputs=[metrics_summary, results_table, eval_status, download_csv, eval_state],
                )
        else:
            with gr.Tab("Evaluation"):
                gr.Markdown(
                    "Evaluation runs **locally** only. "
                    "Set `DEPLOYMENT_MODE=local` in `.env` and see README."
                )

    return demo


def main() -> None:
    demo = build_ui()
    launch_kwargs: dict = {"share": False}
    if IS_SPACES_MODE:
        launch_kwargs.update(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
        )
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
