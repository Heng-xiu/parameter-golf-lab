import {
  ArrowRight,
  Binary,
  Bot,
  BrainCircuit,
  Check,
  Clipboard,
  Gauge,
  GitBranch,
  Layers3,
  LineChart,
  Play,
  Workflow,
} from "lucide-react";
import { useMemo, useState } from "react";

type Step = {
  id: string;
  title: string;
  lesson: string;
  icon: typeof Play;
  purpose: string;
  command: string;
  expected: string;
};

const steps: Step[] = [
  {
    id: "setup",
    title: "Setup",
    lesson: "Foundation",
    icon: Workflow,
    purpose: "Create the local environment and generate the tiny teaching corpus.",
    command: "make setup\nmake smoke",
    expected: "A checkpoint appears in outputs/baseline with loss, BPB, and size report logs.",
  },
  {
    id: "baseline",
    title: "Tiny LM",
    lesson: "Lesson 1",
    icon: BrainCircuit,
    purpose: "Train the baseline byte-level causal Transformer and inspect BPB.",
    command:
      "python scripts/make_tiny_corpus.py --out data/tiny.txt\npython scripts/train.py --config configs/baseline.yaml",
    expected: "The log prints step loss and bits per byte for a small overfit run.",
  },
  {
    id: "context",
    title: "Context Growth",
    lesson: "Lesson 2",
    icon: GitBranch,
    purpose: "Watch sequence length grow as training progress increases.",
    command: "python scripts/train.py --config configs/progressive_context.yaml",
    expected: "The seq_len field moves through 32, 64, and 128.",
  },
  {
    id: "gqa",
    title: "GQA",
    lesson: "Lesson 3",
    icon: Layers3,
    purpose: "Compare query heads with fewer key/value heads.",
    command: "python scripts/train.py --config configs/gqa.yaml",
    expected: "Training runs with n_heads=4 and n_kv_heads=2.",
  },
  {
    id: "quant",
    title: "Quantization",
    lesson: "Lessons 4-5",
    icon: Binary,
    purpose: "Measure size and loss changes from int8 and low-rank correction.",
    command:
      "python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8\npython scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8-lowrank --rank 4",
    expected: "The report shows artifact bytes, loss delta, and reconstruction error.",
  },
  {
    id: "ttt",
    title: "Score-First TTT",
    lesson: "Lesson 6",
    icon: Gauge,
    purpose: "Verify each chunk is scored before LoRA updates happen.",
    command:
      "python scripts/run_ttt.py --checkpoint outputs/baseline/checkpoint.pt --config configs/ttt_lora.yaml",
    expected: "Each chunk log includes scored_before_update=true.",
  },
  {
    id: "chat",
    title: "Chat Data",
    lesson: "Lesson 7",
    icon: Bot,
    purpose: "Train on dialogue-formatted data and compare it with FineWeb continuation.",
    command:
      "python -m pip install -r requirements-hf.txt\npython scripts/export_hf_chat_corpus.py --out data/hf_chat.txt --max-examples 1200\npython scripts/train.py --config configs/hf_chat_mps.yaml --device mps",
    expected: "The model learns User/Assistant structure, but remains a tiny byte-level demo.",
  },
];

const comparisons = [
  ["Tiny synthetic", "Fast smoke tests", "Not conversational"],
  ["FineWeb sample", "Better prose continuation", "Still not assistant chat"],
  ["HF chat corpus", "Dialogue structure", "Needs capacity and assistant-only loss"],
];

export function App() {
  const [selectedId, setSelectedId] = useState(steps[0].id);
  const [copied, setCopied] = useState(false);
  const selected = useMemo(
    () => steps.find((step) => step.id === selectedId) ?? steps[0],
    [selectedId],
  );
  const Icon = selected.icon;

  async function copyCommand() {
    await navigator.clipboard.writeText(selected.command);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <main>
      <section className="hero" aria-labelledby="page-title">
        <nav className="topbar" aria-label="Tutorial sections">
          <span className="brand">mini-parameter-golf</span>
          <a href="#guide">Guide</a>
          <a href="#matrix">Matrix</a>
          <a href="#chat">Chat</a>
        </nav>
        <div className="hero-grid">
          <div className="hero-copy">
            <p className="eyebrow">Hands-on classroom path</p>
            <h1 id="page-title">Train, compress, and adapt a tiny language model.</h1>
            <p className="intro">
              Follow the repository in the order it was built: baseline Transformer, GQA,
              progressive context, quantization, low-rank correction, TTT, and chat data.
            </p>
            <a className="primary-link" href="#guide">
              Start with setup <ArrowRight size={18} aria-hidden="true" />
            </a>
          </div>
          <div className="terminal" aria-label="Smoke test preview">
            <div className="terminal-bar">
              <span />
              <span />
              <span />
            </div>
            <pre>{`$ make smoke
step=20 seq_len=64 loss=4.8463 bpb=6.9917
saved=outputs/baseline/checkpoint.pt
parameters=115008
checkpoint_bytes=467106`}</pre>
          </div>
        </div>
      </section>

      <section id="guide" className="guide" aria-labelledby="guide-title">
        <div className="section-heading">
          <p className="eyebrow">Guided lab</p>
          <h2 id="guide-title">One command rail, seven concepts.</h2>
        </div>
        <div className="workspace">
          <aside className="step-list" aria-label="Tutorial steps">
            {steps.map((step, index) => {
              const StepIcon = step.icon;
              return (
                <button
                  className={step.id === selected.id ? "step active" : "step"}
                  key={step.id}
                  onClick={() => setSelectedId(step.id)}
                  type="button"
                >
                  <span className="step-index">{String(index + 1).padStart(2, "0")}</span>
                  <StepIcon size={18} aria-hidden="true" />
                  <span>
                    <strong>{step.title}</strong>
                    <small>{step.lesson}</small>
                  </span>
                </button>
              );
            })}
          </aside>
          <article className="detail" aria-live="polite">
            <div className="detail-title">
              <Icon size={28} aria-hidden="true" />
              <div>
                <p className="eyebrow">{selected.lesson}</p>
                <h3>{selected.title}</h3>
              </div>
            </div>
            <p className="purpose">{selected.purpose}</p>
            <div className="code-block">
              <button onClick={copyCommand} type="button">
                {copied ? <Check size={16} /> : <Clipboard size={16} />}
                {copied ? "Copied" : "Copy"}
              </button>
              <pre>{selected.command}</pre>
            </div>
            <p className="expected">
              <strong>Expected:</strong> {selected.expected}
            </p>
          </article>
        </div>
      </section>

      <section id="matrix" className="matrix" aria-labelledby="matrix-title">
        <div>
          <p className="eyebrow">Experiment matrix</p>
          <h2 id="matrix-title">Use the right data for the behavior you want.</h2>
        </div>
        <div className="comparison-grid">
          {comparisons.map(([name, teaches, limitation]) => (
            <article key={name} className="comparison">
              <LineChart size={20} aria-hidden="true" />
              <h3>{name}</h3>
              <p>{teaches}</p>
              <small>{limitation}</small>
            </article>
          ))}
        </div>
      </section>

      <section id="chat" className="chat-note" aria-labelledby="chat-title">
        <div>
          <p className="eyebrow">Visual rule</p>
          <h2 id="chat-title">Chat quality starts with dialogue format.</h2>
        </div>
        <p>
          A FineWeb checkpoint continues prose. A dialogue checkpoint learns turn structure. The
          tutorial keeps both visible so students can connect data, objective, and behavior instead
          of treating chat quality as a mystery.
        </p>
      </section>
    </main>
  );
}
