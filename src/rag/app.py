import gradio_client.utils as _gcu
import gradio as gr
import time
import json
from src.rag.agent import LangChainAgentModel
from mlflow.types.responses import ResponsesAgentRequest
from mlflow.types.responses_helpers import Message

# ── PATCH GRADIO CLIENT ──
_original_get_type = _gcu.get_type

def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "any"
    return _original_get_type(schema)

_gcu.get_type = _safe_get_type

_original_j2p = _gcu._json_schema_to_python_type

def _safe_j2p(schema, defs=None):
    if not isinstance(schema, dict):
        return "any"
    try:
        return _original_j2p(schema, defs)
    except TypeError:
        return "any"

_gcu._json_schema_to_python_type = _safe_j2p
# ─────────────────────────

chatbot_model = LangChainAgentModel()
chatbot_model.load_context(None)

PRESET_PROMPTS = [
    "⚔️  Qui sont les ennemis d'Abaddon le Fléau ?",
    "🛡️  À quelle faction appartient Roboute Guilliman ?",
    "☠️  Qui Horus Lupercal a-t-il trahi ?",
    "🔱  Quelles reliques le Primarque Sanguinius maniait-il ?",
    "🤝  Quelles sont les alliances des Space Wolves ?",
    "Quelle entité est vénérée par l'Adeptus Mechanicus ?",
]

CSS = """
/* ── Imports typographiques ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Cinzel:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');

/* ── Variables ── */
:root {
    --bg-void:       #04060a;
    --bg-panel:      #080d14;
    --bg-card:       #0c1420;
    --bg-input:      #06090f;
    --green-phos:    #00ff88;
    --green-dim:     #00cc66;
    --green-dark:    #003322;
    --red-alert:     #ff2244;
    --red-dim:       #cc1133;
    --amber-warn:    #ffaa00;
    --gold-mech:     #c8922a;
    --gold-bright:   #f0c060;
    --text-primary:  #d4e8d4;
    --text-muted:    #556655;
    --text-bright:   #eeffee;
    --border-glow:   #00ff8844;
    --border-dim:    #1a2a1a;
    --scan-speed:    4s;
}

/* ── Reset & Base ── */
* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-void) !important;
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
}

/* ── Fond animé (grille Noosphère) ── */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,136,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,136,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
    animation: gridPulse 8s ease-in-out infinite;
}

@keyframes gridPulse {
    0%, 100% { opacity: 0.4; }
    50%       { opacity: 1;   }
}

/* ── Ligne de scan ── */
.gradio-container::after {
    content: '';
    position: fixed;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--green-phos), transparent);
    animation: scanLine var(--scan-speed) linear infinite;
    pointer-events: none;
    z-index: 1;
    opacity: 0.5;
}

@keyframes scanLine {
    0%   { top: -2px; }
    100% { top: 100vh; }
}

/* ── En-tête principal ── */
#main-header {
    position: relative;
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border-glow);
    margin-bottom: 1rem;
    background: linear-gradient(180deg, rgba(0,255,136,0.05) 0%, transparent 100%);
}

#main-header::before {
    content: '◈ NOOSPHÈRE ACTIVE ◈';
    display: block;
    font-size: 0.6rem;
    letter-spacing: 0.4em;
    color: var(--green-dim);
    margin-bottom: 0.5rem;
    animation: blink 2s step-end infinite;
}

@keyframes blink {
    50% { opacity: 0; }
}

#cogitator-title {
    font-family: 'Cinzel', serif !important;
    font-size: clamp(1.4rem, 3vw, 2.4rem) !important;
    font-weight: 900 !important;
    color: var(--gold-bright) !important;
    text-shadow:
        0 0 20px rgba(200,146,42,0.8),
        0 0 60px rgba(200,146,42,0.3);
    letter-spacing: 0.15em;
    margin: 0;
    line-height: 1.2;
}

#cogitator-subtitle {
    font-size: 0.7rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.3em;
    margin-top: 0.4rem;
}

#status-bar {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1rem;
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
}

.status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green-phos);
    margin-right: 0.3rem;
    animation: statusPulse 1.5s ease-in-out infinite;
    vertical-align: middle;
}

@keyframes statusPulse {
    0%, 100% { box-shadow: 0 0 4px var(--green-phos); opacity: 1; }
    50%       { box-shadow: 0 0 12px var(--green-phos); opacity: 0.7; }
}

/* ── Fenêtre de chat ── */
#chatbot-window {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 4px !important;
    box-shadow:
        0 0 30px rgba(0,255,136,0.06),
        inset 0 0 60px rgba(0,0,0,0.5) !important;
}

#chatbot-window .wrap {
    background: transparent !important;
}

/* Messages utilisateur */
#chatbot-window .message.user {
    background: linear-gradient(135deg, #0a1a2a, #0d2235) !important;
    border: 1px solid rgba(0,255,136,0.2) !important;
    border-radius: 2px 12px 12px 2px !important;
    color: var(--text-bright) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    position: relative;
    padding-left: 1.2rem !important;
}

#chatbot-window .message.user::before {
    content: '▶';
    position: absolute;
    left: 0.4rem;
    top: 50%;
    transform: translateY(-50%);
    color: var(--green-dim);
    font-size: 0.5rem;
}

/* Messages assistant */
#chatbot-window .message.bot {
    background: linear-gradient(135deg, #080f08, #0a140a) !important;
    border: 1px solid rgba(0,255,136,0.12) !important;
    border-left: 3px solid var(--gold-mech) !important;
    border-radius: 12px 2px 2px 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
}

/* ── Zone de saisie ── */
#input-row {
    margin-top: 0.5rem;
    background: var(--bg-panel);
    border: 1px solid var(--border-glow);
    border-radius: 4px;
    padding: 0.75rem;
}

#user-input textarea {
    background: var(--bg-input) !important;
    border: 1px solid #1a3322 !important;
    border-radius: 3px !important;
    color: var(--text-bright) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    caret-color: var(--green-phos) !important;
    transition: border-color 0.3s;
    resize: none !important;
}

#user-input textarea:focus {
    border-color: var(--green-dim) !important;
    box-shadow: 0 0 12px rgba(0,255,136,0.15) !important;
    outline: none !important;
}

#user-input textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic;
}

/* ── Boutons ── */
#send-btn {
    background: linear-gradient(135deg, #003322, #004433) !important;
    border: 1px solid var(--green-dim) !important;
    color: var(--green-phos) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 3px !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 10px rgba(0,255,136,0.1) !important;
}

#send-btn:hover {
    background: linear-gradient(135deg, #004433, #006644) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.3) !important;
    transform: translateY(-1px) !important;
}

#send-btn:active { transform: translateY(0) !important; }

#clear-btn {
    background: linear-gradient(135deg, #1a0005, #220008) !important;
    border: 1px solid rgba(255,34,68,0.4) !important;
    color: var(--red-alert) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 3px !important;
    transition: all 0.2s !important;
}

#clear-btn:hover {
    background: linear-gradient(135deg, #220008, #330010) !important;
    box-shadow: 0 0 16px rgba(255,34,68,0.25) !important;
    transform: translateY(-1px) !important;
}

/* ── Prompts préparés ── */
#presets-label {
    font-size: 0.65rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.25em !important;
    margin-bottom: 0.5rem !important;
    display: block;
}

.preset-btn button {
    background: var(--bg-card) !important;
    border: 1px solid #1a2a1a !important;
    border-left: 2px solid var(--gold-mech) !important;
    color: #9ab89a !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    text-align: left !important;
    border-radius: 2px !important;
    padding: 0.5rem 0.8rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

.preset-btn button:hover {
    background: #0f1f0f !important;
    border-color: var(--green-dim) !important;
    border-left-color: var(--gold-bright) !important;
    color: var(--green-phos) !important;
    box-shadow: 0 0 10px rgba(0,255,136,0.1) !important;
    transform: translateX(3px) !important;
}

/* ── Panneau latéral Archivum ── */
#archivum-panel {
    background: var(--bg-panel);
    border: 1px solid #1a2a1a;
    border-top: 2px solid var(--gold-mech);
    border-radius: 4px;
    padding: 1rem;
    font-size: 0.7rem;
    color: var(--text-muted);
    line-height: 1.8;
}

#archivum-panel .panel-title {
    font-family: 'Cinzel', serif;
    color: var(--gold-mech);
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    border-bottom: 1px solid #1a2a1a;
    padding-bottom: 0.5rem;
    margin-bottom: 0.75rem;
}

#archivum-panel .relation-tag {
    display: inline-block;
    background: #0a1a0a;
    border: 1px solid #1a3a1a;
    color: var(--green-dim);
    padding: 0.1rem 0.4rem;
    border-radius: 2px;
    margin: 0.15rem;
    font-size: 0.62rem;
}

/* ── Footer ── */
#footer {
    text-align: center;
    font-size: 0.6rem;
    color: #334433;
    letter-spacing: 0.3em;
    padding: 1rem;
    border-top: 1px solid #0d1a0d;
    margin-top: 1rem;
}

/* ── Scrollbar Mechanicus ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb {
    background: var(--green-dark);
    border-radius: 2px;
}
::-webkit-scrollbar-thumb:hover { background: var(--green-dim); }

/* ── Markdown dans les messages ── */
#chatbot-window table {
    border-collapse: collapse;
    font-size: 0.78rem;
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-primary);
    width: 100%;
    margin: 0.5rem 0;
}

#chatbot-window th {
    background: #0a1a0a;
    border: 1px solid var(--green-dark);
    color: var(--gold-bright);
    padding: 0.3rem 0.6rem;
    text-align: left;
}

#chatbot-window td {
    border: 1px solid #1a2a1a;
    padding: 0.25rem 0.6rem;
}

#chatbot-window tr:hover td {
    background: rgba(0,255,136,0.04);
}

#chatbot-window code:not(pre code) {
    background: #0a1a0a !important;
    color: var(--green-dim) !important;
    border: 1px solid #1a3a1a !important;
    padding: 0.1rem 0.3rem;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace !important;
}

#chatbot-window strong {
    color: var(--gold-bright) !important;
}

#chatbot-window em {
    color: var(--amber-warn) !important;
    font-style: normal !important;
}

/* ── En-têtes des Tools (h3 = Call, h4 = Output) ── */
#chatbot-window h3 {
    color: var(--amber-warn);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin: 0.8rem 0 0.2rem 0;
    animation: toolPulse 1.2s ease-in-out 3;
}

#chatbot-window h4 {
    color: var(--green-dim);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin: 0.8rem 0 0.2rem 0;
}

/* ── Arguments d'appel (Blockquote) ── */
#chatbot-window blockquote {
    background: linear-gradient(135deg, #0d1a0d, #0a150a);
    border: 1px solid rgba(255,170,0,0.35);
    border-left: 3px solid var(--amber-warn);
    border-radius: 3px;
    padding: 0.5rem 0.8rem;
    margin: 0.2rem 0 0.6rem 0;
    color: #aaccaa;
    font-size: 0.72rem;
}

/* ── Contenu d'output (Pre / Code blocks) ── */
#chatbot-window pre {
    background: linear-gradient(135deg, #070f07, #080f08) !important;
    border: 1px solid rgba(0,255,136,0.18) !important;
    border-left: 3px solid var(--green-dim) !important;
    padding: 0.5rem 0.8rem !important;
    margin: 0.2rem 0 0.6rem 0 !important;
    border-radius: 3px !important;
    max-height: 280px;
    overflow-y: auto;
}

#chatbot-window pre code {
    color: #88bb88 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    white-space: pre-wrap !important;
    background: transparent !important;
    border: none !important;
}

/* ── Séparateur (hr) ── */
#chatbot-window hr {
    border: none;
    border-top: 1px dashed #1a3a1a;
    margin: 1rem 0;
    opacity: 0.6;
}

@keyframes toolPulse {
    0%, 100% { opacity: 0.7; }
    50%       { opacity: 1; border-color: rgba(255,170,0,0.6); }
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-10px); }
    to   { opacity: 1; transform: translateY(0); }
}

#main-header  { animation: fadeInDown 0.8s ease; }
#chatbot-window { animation: fadeInDown 0.9s ease 0.1s both; }
#input-row    { animation: fadeInDown 1s ease 0.2s both; }
"""

def format_tool_call(tool_name: str, args: str) -> str:
    """Formate un appel d'outil en Markdown."""
    icons = {
        "get_entity_id":         "🔍 RÉSOLUTION_ENTITÉ",
        "predict_relation_tail": "⚙️ INFÉRENCE_GRAPHE",
    }
    label = icons.get(tool_name, f"🛠️ {tool_name.upper()}")
    try:
        parsed = json.loads(args)
        args_str = " | ".join(f"**{k}**={v}" for k, v in parsed.items())
    except Exception:
        args_str = args
    return f"### ▶ APPEL · {label}\n> {args_str}\n"

def format_tool_output(output: str) -> str:
    """Formate la réponse de l'outil en Markdown."""
    lines = output.strip().splitlines()
    if len(lines) > 25:
        preview = "\n".join(lines[:25])
        preview += f"\n… [{len(lines) - 25} lignes supplémentaires]"
    else:
        preview = output.strip()
    return f"#### ◀ DONNÉES REÇUES DE LA NOOSPHÈRE\n```text\n{preview}\n```\n"

# ── CHANGEMENT GRADIO 6 ──────────────────────────────────────────────────────
# L'historique est maintenant une liste de dicts {"role": ..., "content": ...}
# au lieu de tuples [user_msg, bot_msg].
# ─────────────────────────────────────────────────────────────────────────────
def respond(message: str, history: list):
    if not message.strip():
        yield history, ""
        return

    # Reconstruction des messages MLflow depuis le nouvel format dict
    mlflow_messages = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            mlflow_messages.append(Message(role=role, content=content))

    mlflow_messages.append(Message(role="user", content=message))

    request = ResponsesAgentRequest(input=mlflow_messages)

    # Ajout du message utilisateur + emplacement pour la réponse assistant
    new_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield new_history, ""

    accumulated_response = ""

    for event in chatbot_model.predict_stream(request=request):
        item = event.item

        if item["type"] == "function_call":
            tool_block = format_tool_call(item["name"], item.get("arguments", "{}"))
            accumulated_response += f"\n{tool_block}\n"
            new_history[-1]["content"] = accumulated_response
            yield new_history, ""
            time.sleep(0.05)

        elif item["type"] == "function_call_output":
            output_block = format_tool_output(item.get("output", ""))
            accumulated_response += f"\n{output_block}\n"
            new_history[-1]["content"] = accumulated_response
            yield new_history, ""
            time.sleep(0.05)

        elif item["type"] == "message":
            content_blocks = item.get("content", [])
            for block in content_blocks:
                if block.get("type") == "output_text":
                    text = block.get("text", "")

                    if accumulated_response.strip():
                        # Séparateur Markdown au lieu du HTML
                        accumulated_response += '\n\n---\n\n'

                    words = text.split(" ")
                    for i, word in enumerate(words):
                        accumulated_response += word + (" " if i < len(words) - 1 else "")
                        new_history[-1]["content"] = accumulated_response
                        yield new_history, ""
                        time.sleep(0.015)

    yield new_history, ""

def fill_preset(prompt: str):
    return prompt

def clear_chat():
    return [], ""

with gr.Blocks(
    title="COGITATOR-SIGMA-VII | Adeptus Mechanicus",
) as demo:

    gr.HTML("""
    <div id="main-header">
        <h1 id="cogitator-title">COGITATOR-SIGMA-VII</h1>
        <p id="cogitator-subtitle">ARCHIVUM NOOSPHÉRIQUE ✦ ADEPTUS MECHANICUS ✦ 41ème MILLÉNAIRE</p>
        <div id="status-bar">
            <span><span class="status-dot"></span>NOOSPHÈRE EN LIGNE</span>
            <span><span class="status-dot"></span>GRAPHE DE CONNAISSANCE ACTIF</span>
            <span><span class="status-dot"></span>MODÈLE ROTATE CHARGÉ</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=240):

            gr.HTML("""
            <div id="archivum-panel">
                <div class="panel-title">◈ ARCHIVUM SIGMA</div>
                <div style="margin-bottom:0.8rem; color:#556655;">
                    Ce Cogitateur accède au Graphe de Connaissance Impérial
                    via deux rites binariques sacrés.
                </div>
                <div style="color:#7a9a7a; margin-bottom:0.4rem;">RELATIONS MAÎTRISÉES</div>
                <div>
                    <span class="relation-tag">memberOf</span>
                    <span class="relation-tag">leads</span>
                    <span class="relation-tag">enemyOf</span>
                    <span class="relation-tag">allyOf</span>
                    <span class="relation-tag">participatedIn</span>
                    <span class="relation-tag">tookPlaceAt</span>
                    <span class="relation-tag">locatedIn</span>
                    <span class="relation-tag">wields</span>
                    <span class="relation-tag">createdBy</span>
                    <span class="relation-tag">worships</span>
                    <span class="relation-tag">controls</span>
                    <span class="relation-tag">subgroupOf</span>
                    <span class="relation-tag">originatesFrom</span>
                    <span class="relation-tag">defeated</span>
                    <span class="relation-tag">betrayed</span>
                </div>
            </div>
            """)

            gr.HTML("""
            <div style="margin-top:1rem; margin-bottom:0.4rem;">
                <span id="presets-label">◈ REQUÊTES LITURGIQUES PRÉ-APPROUVÉES</span>
            </div>
            """)

            preset_buttons = []
            for prompt in PRESET_PROMPTS:
                btn = gr.Button(prompt, elem_classes=["preset-btn"])
                preset_buttons.append((btn, prompt))

        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                label="",
                elem_id="chatbot-window",
                height=520,
                allow_tags=True,
                reasoning_tags=[("<thinking>", "</thinking>")],
                avatar_images=(None, None),
                render_markdown=True,
                sanitize_html=False,
                buttons=[]

            )

            with gr.Row(elem_id="input-row"):
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="Formulez votre Requête Liturgique, Citoyen… L'Omnimessie attend.",
                    lines=2,
                    max_lines=6,
                    elem_id="user-input",
                    scale=5,
                    container=False,
                )
                with gr.Column(scale=1, min_width=120):
                    send_btn  = gr.Button("▶ TRANSMETTRE", elem_id="send-btn", variant="primary")
                    clear_btn = gr.Button("✕ PURGER",      elem_id="clear-btn", variant="secondary")

    gr.HTML("""
    <div id="footer">
        COGITATOR-SIGMA-VII ✦ ADEPTUS MECHANICUS ✦ PROPRIÉTÉ DE L'IMPERIUM DE L'HUMANITÉ
        ✦ TOUTE HÉRÉSIE TECHNOLOGIQUE SERA PURGÉE ✦ LOUÉ SOIT LE DIEU-MACHINE
    </div>
    """)

    send_btn.click(
        fn=respond,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input],
        queue=True,
    )

    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input],
        queue=True,
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, user_input],
    )

    def make_fill(p):
        def _fill():
            return p
        return _fill

    for btn, prompt in preset_buttons:
        btn.click(
            fn=make_fill(prompt),
            inputs=[],
            outputs=[user_input],
        )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,

        footer_links=["gradio", "settings"],

        css=CSS,
        theme=gr.themes.Base(
            primary_hue="green",
            secondary_hue="orange",
            neutral_hue="slate",
            font=[
                gr.themes.GoogleFont("Share Tech Mono"),
                "monospace",
            ],
        )
    )