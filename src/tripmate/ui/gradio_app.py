"""Gradio front end. A genuine API client: it only talks HTTP to the FastAPI service, so the same
authentication, validation, rate limits and idempotency rules apply to the UI and to any other client."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import gradio as gr
import httpx

from tripmate.ui import render

POLL_SECONDS = 1.0
MAX_WAIT_SECONDS = 240
SETTLED = {"awaiting_approval", "completed", "blocked", "failed", "needs_clarification"}
EXAMPLES = [
    "5 days in Tokyo from Karachi, budget 3000 USD, love food and temples",
    "Weekend in Istanbul for two, relaxed pace, vegetarian",
    "A week in Lisbon from London, mid-range, 2500 EUR",
]


class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    async def call(
        self, method: str, path: str, token: str | None = None, **kw: Any
    ) -> tuple[int, Any]:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        headers.update(kw.pop("headers", {}))
        async with httpx.AsyncClient(base_url=self.base, timeout=30) as client:
            try:
                r = await client.request(method, path, headers=headers, **kw)
            except httpx.HTTPError:
                return 503, {"detail": "The planning service is unreachable."}
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, None


def build_ui(api_base: str) -> gr.Blocks:
    api = Api(api_base)

    with gr.Blocks(title="TripMate", theme=gr.themes.Soft()) as demo:
        token = gr.State(None)
        plan_id = gr.State(None)
        version = gr.State(None)
        gr.Markdown(
            "# TripMate\nAn AI travel planner that shows its sources, flags estimates and asks before it finalises."
        )

        with gr.Group() as auth_box:
            with gr.Row():
                email = gr.Textbox(label="Email", scale=2)
                password = gr.Textbox(label="Password (10+ characters)", type="password", scale=2)
                login_btn = gr.Button("Sign in", variant="primary")
                register_btn = gr.Button("Create account")
            auth_msg = gr.Markdown()

        with gr.Column(visible=False) as app_box:
            with gr.Row():
                request_box = gr.Textbox(
                    label="Where do you want to go?", lines=2, scale=4, placeholder=EXAMPLES[0]
                )
                plan_btn = gr.Button("Plan my trip", variant="primary", scale=1)
            gr.Examples([[x] for x in EXAMPLES], request_box)
            status_md = gr.Markdown()
            question_md = gr.Markdown(visible=False)
            with gr.Row(visible=False) as answer_row:
                answer_box = gr.Textbox(label="Your answer", scale=4)
                answer_btn = gr.Button("Send", scale=1)

            with gr.Tabs(visible=False) as result_tabs:
                with gr.Tab("Itinerary"):
                    itinerary_md = gr.Markdown()
                    feedback = gr.Textbox(label="Want changes? Say what to change", lines=2)
                    with gr.Row():
                        approve_btn = gr.Button("Approve", variant="primary")
                        revise_btn = gr.Button("Request changes")
                with gr.Tab("Budget"):
                    budget_md = gr.Markdown()
                    budget_df = gr.Dataframe(
                        headers=["Category", "Amount", "Type", "Basis"],
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Alternatives"):
                    bundle_radio = gr.Radio(
                        label="Flight + hotel combinations (Pareto-optimal on cost, time, quality)"
                    )
                    switch_btn = gr.Button("Use this option")
                with gr.Tab("Edit trip"):
                    with gr.Row():
                        e_start = gr.Textbox(label="Start (YYYY-MM-DD)")
                        e_end = gr.Textbox(label="End (YYYY-MM-DD)")
                        e_trav = gr.Number(label="Travelers", precision=0, minimum=1, maximum=12)
                    with gr.Row():
                        e_budget = gr.Number(label="Budget (total)", minimum=0)
                        e_style = gr.Dropdown(
                            ["", "budget", "mid_range", "luxury"], label="Style", value=""
                        )
                        e_pace = gr.Dropdown(
                            ["", "relaxed", "balanced", "packed"], label="Pace", value=""
                        )
                    e_interests = gr.Textbox(label="Interests (comma separated)")
                    e_apply = gr.Button("Re-plan with these changes")
                with gr.Tab("Sources & notes"):
                    sources_md = gr.Markdown()
                    problems_md = gr.Markdown()

        # ---------------------------------------------------------------- handlers
        # One output layout for every handler (12 components), so each handler yields a flat tuple.
        OUT = [
            plan_id,
            version,
            result_tabs,
            itinerary_md,
            budget_md,
            budget_df,
            problems_md,
            bundle_radio,
            sources_md,
            question_md,
            answer_row,
            status_md,
        ]
        keep = gr.update()

        def message_only(text: str) -> tuple[Any, ...]:
            return (keep,) * 11 + (text,)

        def frame(plan: dict[str, Any], progress: str) -> tuple[Any, ...]:
            has = bool(plan.get("itinerary"))
            asking = plan["status"] == "needs_clarification"
            shown = progress + (f"\n\n**{plan['error']}**" if plan.get("error") else "")
            return (
                plan["id"],
                plan["itinerary"]["version"] if has else None,
                gr.update(visible=has),
                render.itinerary_markdown(plan["itinerary"]["itinerary"] if has else None),
                render.budget_summary(plan.get("budget")),
                render.budget_rows(plan.get("budget")),
                render.problems_markdown(plan),
                gr.update(
                    choices=render.bundle_choices(plan.get("bundles")),
                    value=plan.get("selected_label"),
                ),
                render.sources_markdown(plan.get("sources", [])),
                gr.update(
                    visible=asking, value=f"**{plan.get('clarification')}**" if asking else ""
                ),
                gr.update(visible=asking),
                shown,
            )

        async def sign_in(mail: str, pw: str, path: str) -> tuple[Any, ...]:
            code, body = await api.call("POST", path, json={"email": mail, "password": pw})
            if code >= 300:
                return None, gr.update(), gr.update(), render.problem_message(code, body)
            return body["access_token"], gr.update(visible=False), gr.update(visible=True), ""

        async def do_login(mail: str, pw: str) -> tuple[Any, ...]:
            return await sign_in(mail, pw, "/v1/auth/login")

        async def do_register(mail: str, pw: str) -> tuple[Any, ...]:
            return await sign_in(mail, pw, "/v1/auth/register")

        async def follow(tok: str, pid: str) -> AsyncIterator[tuple[Any, ...]]:
            """Poll until the plan needs the person again, streaming progress and partial results."""
            seen, events, waited = 0, [], 0.0
            while waited < MAX_WAIT_SECONDS:
                code, plan = await api.call("GET", f"/v1/plans/{pid}", tok)
                if code >= 300:
                    yield message_only(render.problem_message(code, plan))
                    return
                _, new = await api.call("GET", f"/v1/plans/{pid}/events?after={seen}", tok)
                for e in new or []:
                    seen = e["seq"]
                    events.append(e)
                yield frame(plan, render.progress_markdown(events, plan["status"]))
                if plan["status"] in SETTLED:
                    return
                await asyncio.sleep(POLL_SECONDS)
                waited += POLL_SECONDS
            yield message_only("Still working - refresh in a moment to see the result.")

        async def submit(
            tok: str, method: str, path: str, **kw: Any
        ) -> AsyncIterator[tuple[Any, ...]]:
            code, body = await api.call(method, path, tok, **kw)
            if code >= 300:
                yield message_only(render.problem_message(code, body))
                return
            async for out in follow(tok, body["id"] if "id" in body else path.split("/")[3]):
                yield out

        async def start(tok: str, text: str) -> AsyncIterator[tuple[Any, ...]]:
            if not text or len(text.strip()) < 3:
                yield message_only("Please describe your trip.")
                return
            async for o in submit(tok, "POST", "/v1/plans", json={"message": text}):
                yield o

        async def decide(
            tok: str, pid: str, ver: int, approved: bool, note: str
        ) -> AsyncIterator[tuple[Any, ...]]:
            if not pid or ver is None:
                yield message_only("Plan a trip first.")
                return
            key = (
                f"ui-{uuid.uuid4().hex}"  # one key per click: a double click is a no-op server side
            )
            code, body = await api.call(
                "POST",
                f"/v1/plans/{pid}/decision",
                tok,
                headers={"Idempotency-Key": key},
                json={
                    "itinerary_version": int(ver),
                    "approved": approved,
                    "feedback": note or None,
                },
            )
            if code >= 300:
                yield message_only(render.problem_message(code, body))
                return
            async for o in follow(tok, pid):
                yield o

        async def approve(
            tok: str, pid: str, ver: int, note: str
        ) -> AsyncIterator[tuple[Any, ...]]:
            async for o in decide(tok, pid, ver, True, note):
                yield o

        async def revise(tok: str, pid: str, ver: int, note: str) -> AsyncIterator[tuple[Any, ...]]:
            async for o in decide(tok, pid, ver, False, note):
                yield o

        async def switch(tok: str, pid: str, label: str | None) -> AsyncIterator[tuple[Any, ...]]:
            if not pid or not label:
                yield message_only("Pick an option first.")
                return
            async for o in submit(
                tok, "POST", f"/v1/plans/{pid}/alternative", json={"label": label}
            ):
                yield o

        async def edit(
            tok: str,
            pid: str,
            s: str,
            e: str,
            t: float | None,
            b: float | None,
            st: str,
            pc: str,
            ints: str,
        ) -> AsyncIterator[tuple[Any, ...]]:
            if not pid:
                yield message_only("Plan a trip first.")
                return
            changes: dict[str, Any] = {}
            if s.strip():
                changes["start_date"] = s.strip()
            if e.strip():
                changes["end_date"] = e.strip()
            if t:
                changes["travelers"] = int(t)
            if b:
                _, plan = await api.call("GET", f"/v1/plans/{pid}", tok)
                currency = (plan or {}).get("currency") or "USD"
                changes["budget"] = {"amount": str(b), "currency": currency}
            if st:
                changes["style"] = st
            if pc:
                changes["pace"] = pc
            if ints.strip():
                changes["interests"] = [i.strip() for i in ints.split(",") if i.strip()]
            if not changes:
                yield message_only("Change at least one field.")
                return
            async for o in submit(tok, "PATCH", f"/v1/plans/{pid}/constraints", json=changes):
                yield o

        async def answer(tok: str, pid: str, text: str) -> AsyncIterator[tuple[Any, ...]]:
            async for o in submit(tok, "POST", f"/v1/plans/{pid}/answer", json={"answer": text}):
                yield o

        login_btn.click(do_login, [email, password], [token, auth_box, app_box, auth_msg])
        register_btn.click(do_register, [email, password], [token, auth_box, app_box, auth_msg])
        plan_btn.click(start, [token, request_box], OUT)
        approve_btn.click(approve, [token, plan_id, version, feedback], OUT)
        revise_btn.click(revise, [token, plan_id, version, feedback], OUT)
        switch_btn.click(switch, [token, plan_id, bundle_radio], OUT)
        e_apply.click(
            edit,
            [token, plan_id, e_start, e_end, e_trav, e_budget, e_style, e_pace, e_interests],
            OUT,
        )
        answer_btn.click(answer, [token, plan_id, answer_box], OUT)
    return demo


def mount_ui(app: Any, api_base_url: str | None) -> None:
    base = api_base_url or os.environ.get("TRIPMATE_UI_API_URL", "http://127.0.0.1:8000")
    gr.mount_gradio_app(app, build_ui(base), path="/ui")


def main() -> None:  # python -m tripmate.ui.gradio_app  (stand-alone UI against any API)
    build_ui(os.environ.get("TRIPMATE_UI_API_URL", "http://127.0.0.1:8000")).queue().launch(
        server_name="0.0.0.0",  # noqa: S104 - container deployment binds all interfaces
        server_port=int(os.environ.get("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
