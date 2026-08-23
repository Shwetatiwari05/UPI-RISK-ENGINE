import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BadgeIndianRupee,
  BarChart3,
  Blocks,
  Clock3,
  ChevronRight,
  Cpu,
  Database,
  Download,
  FileText,
  GitBranch,
  Gauge,
  Layers3,
  ListChecks,
  LineChart,
  MapPin,
  Network,
  Radar,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  WalletCards,
} from "lucide-react";
import "./styles.css";

const API_BASE = "/api";

const defaultForm = {
  amount: 650,
  transaction_type: "TRANSFER",
  device_type: "Android",
  merchant_category: "Personal",
  timestamp: new Date().toISOString().slice(0, 16),
  sender_id: "user_1024",
  receiver_id: "user_2048",
  location: "Mumbai",
};

function App() {
  const [page, setPage] = useState("simulation");
  const [form, setForm] = useState(defaultForm);
  const [status, setStatus] = useState(null);
  const [presets, setPresets] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [result, setResult] = useState(null);
  const [selectedLog, setSelectedLog] = useState(null);
  const [logs, setLogs] = useState(() => {
    const saved = localStorage.getItem("upi_prediction_logs");
    return saved ? JSON.parse(saved) : [];
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function bootstrap() {
      try {
        const [statusResponse, presetResponse, analyticsResponse] = await Promise.all([
          fetch(`${API_BASE}/models/status`),
          fetch(`${API_BASE}/presets`),
          fetch(`${API_BASE}/analytics/data`),
        ]);
        setStatus(await statusResponse.json());
        setPresets(await presetResponse.json());
        setAnalytics(await analyticsResponse.json());
      } catch {
        setError("Local API is not reachable. Start it with: uvicorn api.main:app --reload");
      }
    }
    bootstrap();
  }, []);

  useEffect(() => {
    localStorage.setItem("upi_prediction_logs", JSON.stringify(logs.slice(0, 25)));
  }, [logs]);

  function updateField(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  function applyPreset(preset) {
    const timestamp = new Date();
    timestamp.setHours(preset.hour, 0, 0, 0);
    setForm({
      amount: preset.amount,
      transaction_type: preset.transaction_type,
      device_type: preset.device_type,
      merchant_category: preset.merchant_category,
      timestamp: timestamp.toISOString().slice(0, 16),
      sender_id: preset.sender_id,
      receiver_id: preset.receiver_id,
      location: preset.location,
    });
    setResult(null);
  }

  async function runPrediction(event) {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          amount: Number(form.amount),
          timestamp: new Date(form.timestamp).toISOString(),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }
      setResult(data);
      const logEntry = {
        id: data.report?.transaction?.transaction_id || data.transaction_id,
        time: new Date().toLocaleTimeString(),
        amount: Number(form.amount),
        type: form.transaction_type,
        fraud: data.supervised.fraud_probability,
        anomaly: data.anomaly.anomaly_score,
        confidence: data.anomaly.anomaly_confidence,
        label: data.anomaly.anomaly_label,
        resolution: data.fusion?.resolution,
        transaction: data.report?.transaction || { ...form, amount: Number(form.amount) },
        supervised: data.supervised,
        anomalyOutput: data.anomaly,
        fusion: data.fusion,
        report: data.report,
      };
      setSelectedLog(logEntry);
      setLogs((current) => [logEntry, ...current.filter((item) => item.id !== logEntry.id)]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <div className="eyebrow">
            <Sparkles size={16} />
            Post Transaction Fraud Analysis
          </div>
          <h1>UPI Fraud Lab</h1>
          <p>
            A hybrid machine learning framework for post-facto UPI fraud analysis that combines supervised fraud detection and anomaly detection to identify suspicious and unusual transaction behavior.
          </p>
        </div>
        <StatusPanel status={status} />
      </section>

      <nav className="page-tabs">
        <TabButton active={page === "simulation"} icon={<Radar size={18} />} label="Simulation" onClick={() => setPage("simulation")} />
        <TabButton active={page === "workflow"} icon={<GitBranch size={18} />} label="Workflow" onClick={() => setPage("workflow")} />
        <TabButton active={page === "analytics"} icon={<BarChart3 size={18} />} label="Analytics" onClick={() => setPage("analytics")} />
        <TabButton active={page === "report"} icon={<FileText size={18} />} label="Finance Report" onClick={() => setPage("report")} />
        <TabButton active={page === "logs"} icon={<ListChecks size={18} />} label="Prediction Logs" onClick={() => setPage("logs")} />
      </nav>

      {error && (
        <div className="notice">
          <AlertTriangle size={18} />
          {error}
        </div>
      )}

      {page === "simulation" && (
        <SimulationPage
          form={form}
          result={result}
          logs={logs}
          loading={loading}
          presets={presets}
          updateField={updateField}
          applyPreset={applyPreset}
          runPrediction={runPrediction}
        />
      )}
      {page === "workflow" && <WorkflowPage />}
      {page === "analytics" && <AnalyticsPage analytics={analytics} />}
      {page === "report" && <FinanceResearchReportPage report={result?.report || selectedLog?.report} analytics={analytics} />}
      {page === "logs" && <PredictionLogsPage logs={logs} selectedLog={selectedLog} onSelect={setSelectedLog} />}
    </main>
  );
}

function SimulationPage({ form, result, loading, presets, updateField, applyPreset, runPrediction }) {
  const fraudProbability = result?.supervised?.fraud_probability ?? 0;
  const anomalyConfidence = result?.anomaly?.anomaly_confidence ?? 0;
  const supervisedReady = result?.supervised?.signal_status === "calculated";

  return (
    <section className="workspace">
      <form className="control-panel" onSubmit={runPrediction}>
        <div className="panel-head">
          <div>
            <span>Manual Transaction</span>
            <h2>Simulation Input</h2>
          </div>
          <button type="submit" className="run-button" disabled={loading}>
            {loading ? <RefreshCw className="spin" size={18} /> : <Radar size={18} />}
            {loading ? "Testing" : "Run Test"}
          </button>
        </div>

        <PresetStrip presets={presets} onApply={applyPreset} />

        <div className="field-grid">
          <NumberField icon={<BadgeIndianRupee size={18} />} label="Amount" value={form.amount} onChange={(value) => updateField("amount", value)} />
          <SelectField icon={<WalletCards size={18} />} label="Transaction Type" value={form.transaction_type} options={["PAYMENT", "TRANSFER", "CASH_IN", "CASH_OUT", "DEBIT", "UPI", "TOP_UP"]} onChange={(value) => updateField("transaction_type", value)} />
          <SelectField icon={<Cpu size={18} />} label="Device Type" value={form.device_type} options={["Android", "iOS", "Web", "POS", "Unknown"]} onChange={(value) => updateField("device_type", value)} />
          <TextField label="Merchant Category" value={form.merchant_category} onChange={(value) => updateField("merchant_category", value)} />
          <TextField label="Sender ID" value={form.sender_id} onChange={(value) => updateField("sender_id", value)} />
          <TextField label="Receiver ID" value={form.receiver_id} onChange={(value) => updateField("receiver_id", value)} />
          <TextField icon={<MapPin size={18} />} label="Location" value={form.location} onChange={(value) => updateField("location", value)} />
          <DateField icon={<Clock3 size={18} />} label="Transaction Time" value={form.timestamp} onChange={(value) => updateField("timestamp", value)} />
        </div>
      </form>

      <section className="result-column">
        <section className="result-panel">
        <div className="panel-head">
          <div>
            <span>Model Outputs</span>
            <h2>Separate Signals</h2>
          </div>
          <Gauge size={28} />
        </div>

        <div className="signal-grid">
          <SignalCard title="Supervised Model" value={supervisedReady ? formatPercent(fraudProbability) : "Waiting"} label={supervisedReady ? (result.supervised.fraud_prediction ? "Fraud" : "Legitimate") : "Not calculated"} meta={supervisedReady ? result.supervised.model_name : "Run Test to calculate"} accent="red" progress={supervisedReady ? fraudProbability : 0} />
          <SignalCard title="Anomaly Model" value={result ? result.anomaly.anomaly_score.toFixed(4) : "Waiting"} label={result?.anomaly?.anomaly_label ?? "Waiting"} meta={result ? "Isolation Forest" : "Run Test to calculate"} accent="teal" progress={anomalyConfidence} />
        </div>

        <ComparisonChart amount={Number(form.amount)} fraudProbability={fraudProbability} anomalyConfidence={anomalyConfidence} />
        <PersonalizationSignals supervised={result?.supervised} />
        <EvidencePanel diagnostics={result?.diagnostics} />
        <FusionResolution fusion={result?.fusion} />
        <CompactTransactionReport report={result?.report} />
        </section>
      </section>
    </section>
  );
}

function FinanceResearchReportPage({ report, analytics }) {
  return (
    <section className="finance-report-page">
      <div className="section-head report-page-heading">
        <span>One-page analysis summary</span>
        <h2>Finance Research Report</h2>
        <p>Trace how the submitted transaction moved through preprocessing, model scoring, evidence checks, and the final research resolution.</p>
      </div>
      <FullReportPanel report={report} analytics={analytics} />
    </section>
  );
}

function WorkflowPage() {
  const steps = [
    ["Data Loading", "Kaggle and Zenodo files are loaded from data/raw.", Database],
    ["Schema Mapping", "Dataset-specific columns are normalized into one transaction schema.", Blocks],
    ["Preprocessing", "Missing values, scaling, encoding, and outlier handling prepare the model matrix.", Layers3],
    ["Feature Engineering", "Behavioral, velocity, risk, and temporal signals are generated offline.", Activity],
    ["Supervised Models", "XGBoost and Random Forest produce fraud probability scores.", ShieldCheck],
    ["Anomaly Models", "Isolation Forest and LOF produce separate anomaly scores.", Network],
    ["Research Fusion", "A weighted score retains both signals while disagreement and uncertainty identify ambiguous review cases.", GitBranch],
    ["Transaction Report", "The dashboard produces a transparent, downloadable post-transaction research report.", FileText],
  ];

  return (
    <section className="workflow-page">
      <div className="section-head">
        <span>Project Pipeline</span>
        <h2>Offline Batch Workflow</h2>
      </div>
      <div className="workflow-canvas">
        <div className="flow-line" />
        {steps.map(([title, body, Icon], index) => (
          <article className="flow-node" style={{ "--delay": `${index * 120}ms` }} key={title}>
            <div className="node-index">{index + 1}</div>
            <Icon size={24} />
            <h3>{title}</h3>
            <p>{body}</p>
          </article>
        ))}
      </div>
      <div className="workflow-band">
        <MetricTile label="Model families" value="2 independent signals" />
        <MetricTile label="Fusion method" value="Transparent scoring" />
        <MetricTile label="Final band" value="Ambiguous review" />
      </div>
    </section>
  );
}

function AnalyticsPage({ analytics }) {
  if (!analytics?.ready) {
    return (
      <section className="result-panel solo-panel">
        <div className="panel-head">
          <div>
            <span>Imported Data</span>
            <h2>Analytics Waiting</h2>
          </div>
          <Database size={28} />
        </div>
        <p className="muted">{analytics?.message || "Analytics are loading from the local API."}</p>
      </section>
    );
  }

  const summary = analytics.summary;
  return (
    <section className="analytics-page">
      <div className="section-head">
        <span>Imported Data</span>
        <h2>Dataset Analytics</h2>
      </div>
      <div className="metric-grid">
        <MetricTile label="Mapped Transactions" value={compactNumber(summary.total_rows)} />
        <MetricTile label="Fraud Rows" value={compactNumber(summary.fraud_rows)} />
        <MetricTile label="Fraud Rate" value={formatPercent(summary.fraud_rate)} />
        <MetricTile label="Average Amount" value={currency(summary.average_amount)} />
      </div>

      <div className="analytics-grid">
        <ChartPanel title="Fraud vs Legitimate">
          <DonutChart fraud={summary.fraud_rows} legitimate={summary.legitimate_rows} />
        </ChartPanel>
        <ChartPanel title="Amount Distribution">
          <BarList data={analytics.amount_bins} />
        </ChartPanel>
        <ChartPanel title="Transaction Types">
          <BarList data={analytics.transaction_types} />
        </ChartPanel>
        <ChartPanel title="Hourly Volume">
          <SparkBars data={analytics.hourly_volume} />
        </ChartPanel>
        <ChartPanel title="Device Types">
          <BarList data={analytics.device_types} />
        </ChartPanel>
        <ChartPanel title="Top Locations">
          <BarList data={analytics.locations} />
        </ChartPanel>
      </div>
    </section>
  );
}

function StatusPanel({ status }) {
  const ready = status?.ready;
  return (
    <aside className={`status-panel ${ready ? "ready" : "waiting"}`}>
      <ShieldCheck size={24} />
      <div className={`status-content ${!ready ? "missing" : ""}`}>
        <span>Model Status</span>
        <strong>{ready ? "Ready for Phase 1 testing" : "Training artifacts missing"}</strong>
      </div>
      <div className="artifact-dots">
        {Object.entries(status?.artifacts || {}).map(([key, value]) => (
          <span key={key} title={key} className={value ? "on" : ""} />
        ))}
      </div>
    </aside>
  );
}

function TabButton({ active, icon, label, onClick }) {
  return (
    <button className={active ? "active" : ""} type="button" onClick={onClick}>
      {icon}
      {label}
    </button>
  );
}

function PresetStrip({ presets, onApply }) {
  return (
    <div className="presets">
      {presets.map((preset) => (
        <button type="button" key={preset.name} onClick={() => onApply(preset)}>
          {preset.name}
        </button>
      ))}
    </div>
  );
}

function FieldWrap({ icon, label, children }) {
  return (
    <label className="field">
      <span>
        {icon}
        {label}
      </span>
      {children}
    </label>
  );
}

function NumberField({ icon, label, value, onChange }) {
  return (
    <FieldWrap icon={icon} label={label}>
      <input type="number" min="0" value={value} onChange={(event) => onChange(event.target.value)} />
    </FieldWrap>
  );
}

function TextField({ icon, label, value, onChange }) {
  return (
    <FieldWrap icon={icon} label={label}>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </FieldWrap>
  );
}

function DateField({ icon, label, value, onChange }) {
  return (
    <FieldWrap icon={icon} label={label}>
      <input type="datetime-local" value={value} onChange={(event) => onChange(event.target.value)} />
    </FieldWrap>
  );
}

function SelectField({ icon, label, value, options, onChange }) {
  return (
    <FieldWrap icon={icon} label={label}>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </FieldWrap>
  );
}

function SignalCard({ title, value, label, meta, accent, progress }) {
  return (
    <article className={`signal-card ${accent}`}>
      <span>{title}</span>
      <strong>{value}</strong>
      <em>{label}</em>
      <small>{meta}</small>
      <div className="meter">
        <i style={{ width: `${clampPercent(progress * 100)}%` }} />
      </div>
    </article>
  );
}

function ComparisonChart({ amount, fraudProbability, anomalyConfidence }) {
  const scaledAmount = Math.min(100, amount / 1000);
  const rows = [
    ["Amount intensity", scaledAmount],
    ["Fraud probability", fraudProbability * 100],
    ["Anomaly percentile", anomalyConfidence * 100],
  ];
  return (
    <div className="comparison">
      <div className="mini-title">
        <Activity size={17} />
        Transaction Comparison
      </div>
      {rows.map(([label, value]) => (
        <div className="bar-row" key={label}>
          <span>{label}</span>
          <div><i style={{ width: `${clampPercent(value)}%` }} /></div>
          <b>{value.toFixed(1)}</b>
        </div>
      ))}
    </div>
  );
}

function PersonalizationSignals({ supervised }) {
  const personalization = supervised?.personalization;
  if (!personalization) {
    return (
      <div className="evidence-panel">
        <div className="mini-title">Personalization Signals</div>
        <p>Run a transaction to see how this sender's live history shaped the behavioral features.</p>
      </div>
    );
  }

  const count = supervised.live_history_count ?? 0;
  const minutes = personalization.minutes_since_previous_sender_txn;
  const flags = [
    ["Amount Spike", personalization.amount_spike],
    ["New Payee", personalization.new_payee_flag],
    ["Unusual Location", personalization.unusual_location_flag],
    ["Rapid (<5 min)", personalization.rapid_transactions],
    ["Above ₹20L Bound", supervised.exceeds_realistic_amount_bound],
  ];

  return (
    <div className="evidence-panel">
      <div className="mini-title">Personalization Signals</div>
      <div className="evidence-summary">
        <span>Live history compared: <strong>{count} prior transaction{count === 1 ? "" : "s"}</strong></span>
        <span>Sender baseline avg: <strong>₹{Math.round(personalization.avg_transaction_amount ?? 0).toLocaleString("en-IN")}</strong></span>
      </div>
      <div className="sensitivity-list">
        {flags.map(([label, value]) => (
          <div className="sensitivity-row" key={label}>
            <span>{label}</span>
            <em className={value ? "flag-on" : "flag-off"}>{value ? "Yes" : "No"}</em>
          </div>
        ))}
      </div>
      <small>
        {minutes == null
          ? "No previous scored transaction for this sender, so velocity signals are inactive."
          : `${minutes.toFixed(1)} min since this sender's previous scored transaction.`}
      </small>
    </div>
  );
}

function EvidencePanel({ diagnostics }) {
  if (!diagnostics) {
    return (
      <div className="evidence-panel">
        <div className="mini-title">Model Evidence</div>
        <p>No evidence run yet. Submit a transaction to see repeatability and input sensitivity checks.</p>
      </div>
    );
  }

  const repeatFraud = diagnostics.deterministic.repeat_fraud_delta;
  const repeatAnomaly = diagnostics.deterministic.repeat_anomaly_delta;
  const sensitivity = diagnostics.sensitivity || [];

  return (
    <div className="evidence-panel">
      <div className="mini-title">Model Evidence</div>
      <div className="evidence-summary">
        <span>Same input repeat fraud delta: <strong>{formatDelta(repeatFraud)}</strong></span>
        <span>Same input repeat anomaly delta: <strong>{formatDelta(repeatAnomaly)}</strong></span>
      </div>
      <div className="sensitivity-list">
        {sensitivity.map((item) => (
          <div className="sensitivity-row" key={item.name}>
            <span>{item.name}</span>
            <em>Fraud {formatSignedPercent(item.fraud_delta)}</em>
            <small>Anomaly {formatSignedPercent(item.anomaly_delta)}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

function FusionResolution({ fusion }) {
  if (!fusion) {
    return (
      <div className="fusion-panel waiting">
        <div className="mini-title"><GitBranch size={17} /> Research Resolution</div>
        <p>Run a transaction to calculate the fusion score and ambiguity band.</p>
      </div>
    );
  }

  const tone = resolutionTone(fusion.resolution);
  return (
    <section className={`fusion-panel ${tone}`}>
      <div className="fusion-head">
        <div>
          <div className="mini-title"><GitBranch size={17} /> Research Resolution</div>
          <h3>{formatResolution(fusion.resolution)}</h3>
          <p>{fusion.resolution_text}</p>
        </div>
        <span className="resolution-badge">{formatPercent(fusion.ambiguity_score)} ambiguity</span>
      </div>
      <div className="fusion-metrics">
        <MetricReadout label="Fusion score" value={formatPercent(fusion.fusion_score)} />
        <MetricReadout label="Signal disagreement" value={formatPercent(fusion.signal_disagreement)} />
        <MetricReadout label="Supervised uncertainty" value={formatPercent(fusion.supervised_uncertainty)} />
      </div>
      {fusion.velocity_rule?.triggered && (
        <div className="rule-override-badge">
          Velocity-abuse rule override ({fusion.velocity_rule.flags_true.length}/{fusion.velocity_rule.min_flags} flags: {fusion.velocity_rule.flags_true.join(", ")})
        </div>
      )}
      <small>Research output only. It does not approve, decline, block, or prove fraud for a transaction.</small>
    </section>
  );
}

function FullReportPanel({ report, analytics }) {
  if (!report) {
    return (
      <section className="full-report-panel report-waiting">
        <div className="report-head">
          <div>
            <div className="mini-title"><FileText size={17} /> Finance Research Report</div>
            <h2>Report Waiting</h2>
            <p>Run a transaction to generate the full post-transaction report and population comparison.</p>
          </div>
          <LineChart size={28} />
        </div>
      </section>
    );
  }

  const transaction = report.transaction || {};
  const fusion = report.fusion_resolution || {};
  const supervised = report.supervised_model_output || {};
  const anomaly = report.unsupervised_model_output || {};
  const reasoning = report.reasoning || {};

  return (
    <section className="full-report-panel">
      <div className="report-head">
        <div>
          <div className="mini-title"><FileText size={17} /> Finance Research Report</div>
          <h2>{formatResolution(fusion.resolution)}</h2>
          <p>Post-transaction analysis record for {transaction.transaction_id || "this transaction"}.</p>
          <p className="report-resolution-text">{fusion.resolution_text}</p>
        </div>
        <button type="button" className="download-button" onClick={() => downloadJsonReport(report)}>
          <Download size={16} /> Download Report
        </button>
      </div>

      <div className="report-score-strip">
        <ReportItem label="Final resolution" value={formatResolution(fusion.resolution)} />
        <ReportItem label="Fusion score" value={formatPercent(fusion.fusion_score)} />
        <ReportItem label="Ambiguity score" value={formatPercent(fusion.ambiguity_score)} />
      </div>

      <ReportSection title="Why this resolution?">
        <div className="reasoning-copy">
          <p>{reasoning.resolution_reason || report.interpretation}</p>
          <p>{reasoning.supervised_reason}</p>
          <p>{reasoning.anomaly_reason}</p>
          <p>{reasoning.sensitivity_note}</p>
        </div>
        <div className="reasoning-factors">
          <strong>Observed input factors</strong>
          <ul>
            {(reasoning.input_factors || ["No input factors recorded"]).map((factor) => <li key={factor}>{factor}</li>)}
          </ul>
        </div>
      </ReportSection>

      <div className="full-report-grid">
        <ReportSection title="Transaction details">
          <ReportItem label="Amount" value={currency(transaction.amount)} />
          <ReportItem label="Transaction type" value={transaction.transaction_type} />
          <ReportItem label="Sender" value={transaction.sender_id} />
          <ReportItem label="Receiver" value={transaction.receiver_id} />
          <ReportItem label="Device" value={transaction.device_type} />
          <ReportItem label="Location" value={transaction.location} />
          <ReportItem label="Merchant" value={transaction.merchant_category} />
          <ReportItem label="Timestamp" value={formatTimestamp(transaction.timestamp)} />
        </ReportSection>
        <ReportSection title="Model evidence">
          <ReportItem label="Fraud probability" value={formatPercent(supervised.fraud_probability)} />
          <ReportItem label="Fraud prediction" value={supervised.fraud_prediction ? "Fraud signal" : "Legitimate signal"} />
          <ReportItem label="Supervised model" value={supervised.model_name} />
          <ReportItem label="Supervised status" value={supervised.signal_status} />
          <ReportItem label="Unusualness percentile" value={formatPercent(anomaly.anomaly_percentile)} />
          <ReportItem label="Raw anomaly score" value={Number(anomaly.anomaly_score || 0).toFixed(4)} />
          <ReportItem label="Signal disagreement" value={formatPercent(fusion.signal_disagreement)} />
          <ReportItem label="Supervised uncertainty" value={formatPercent(fusion.supervised_uncertainty)} />
        </ReportSection>
      </div>

      <ReportSection title="Research method and scope">
        <ReportItem label="Analysis scope" value={report.analysis_scope} />
        <ReportItem label="Generated at" value={formatTimestamp(report.generated_at_utc)} />
        <ReportItem label="Fusion method" value={fusion.method} />
        <ReportItem label="Supervised weight" value={formatPercent(fusion.weights?.supervised_fraud_probability)} />
        <ReportItem label="Unsupervised weight" value={formatPercent(fusion.weights?.unsupervised_unusualness_percentile)} />
        <ReportItem label="Calibration" value={anomaly.calibration_method} />
        <ReportItem label="Decision thresholds" value="50.0% supervised fraud; 60.0% fusion fraud-likely; 38.0% ambiguity" />
      </ReportSection>

      <TransactionDistributionChart
        distribution={analytics?.transaction_distribution && {
          ...analytics.transaction_distribution,
          input_amount: transaction.amount,
        }}
      />
      <ReportEvidenceTable evidence={report.model_evidence} />
      <p className="report-disclaimer">Research interpretation only. This report does not approve, decline, block, or prove fraud.</p>
    </section>
  );
}

function CompactTransactionReport({ report }) {
  if (!report) return null;
  const transaction = report.transaction || {};
  const fusion = report.fusion_resolution || {};
  return (
    <section className="compact-report">
      <div className="mini-title"><FileText size={17} /> Resolution Summary</div>
      <div className="report-grid compact">
        <ReportItem label="Resolution" value={formatResolution(fusion.resolution)} />
        <ReportItem label="Fusion score" value={formatPercent(fusion.fusion_score)} />
        <ReportItem label="Transaction" value={transaction.transaction_id} />
        <ReportItem label="Amount" value={currency(transaction.amount)} />
      </div>
    </section>
  );
}

function ReportSection({ title, children }) {
  return (
    <section className="report-section">
      <h3>{title}</h3>
      <div className="report-grid">{children}</div>
    </section>
  );
}

function TransactionDistributionChart({ distribution }) {
  if (!distribution?.ready) {
    return <div className="distribution-empty">Population distribution is available after imported-data analytics are ready.</div>;
  }

  const points = distribution.line_points || [];
  const amountRange = Math.max(distribution.maximum - distribution.minimum, 1);
  const toY = (amount) => 150 - (((amount - distribution.minimum) / amountRange) * 130);
  const toX = (index) => points.length <= 1 ? 50 : (index / (points.length - 1)) * 100;
  const linePoints = points.map((item, index) => `${toX(index)},${toY(item.amount)}`).join(" ");
  const medianY = toY(distribution.median);
  const greyTop = toY(distribution.grey_area_high);
  const greyHeight = Math.max(0, toY(distribution.grey_area_low) - greyTop);
  const inputAmount = Number(distribution.input_amount);
  const nearestIndex = Number.isFinite(inputAmount) && points.length
    ? points.reduce((closest, item, index) => Math.abs(item.amount - inputAmount) < Math.abs(points[closest].amount - inputAmount) ? index : closest, 0)
    : null;
  const inputX = nearestIndex === null ? null : toX(nearestIndex);
  const inputY = nearestIndex === null ? null : toY(points[nearestIndex].amount);

  return (
    <section className="distribution-section">
      <div className="report-section-head">
        <div>
          <div className="mini-title"><LineChart size={17} /> Population comparison</div>
          <h3>All imported transactions by amount rank</h3>
          <p>{compactNumber(distribution.line_represents_transactions)} transactions shape this ranked line. {compactNumber(distribution.sampled_transactions)} deterministic points are plotted for browser performance.</p>
        </div>
        <div className="distribution-stats">
          <span>Median <strong>{currency(distribution.median)}</strong></span>
          <span>Grey band <strong>{currency(distribution.grey_area_low)} - {currency(distribution.grey_area_high)}</strong></span>
        </div>
      </div>
      <div className="line-chart-wrap">
        <div className="line-chart-axis y-axis"><span>{currency(distribution.maximum)}</span><span>{currency(distribution.median)}</span><span>{currency(distribution.minimum)}</span></div>
        <svg className="transaction-line-chart" viewBox="0 0 100 160" preserveAspectRatio="none" role="img" aria-label="Ranked transaction amounts with median and grey area">
          <defs>
            <linearGradient id="transactionLineGradient" x1="0" x2="0" y1="1" y2="0">
              <stop offset="0%" stopColor="#128a70" />
              <stop offset={`${clampPercent(((distribution.grey_area_low - distribution.minimum) / amountRange) * 100)}%`} stopColor="#128a70" />
              <stop offset={`${clampPercent(((distribution.grey_area_high - distribution.minimum) / amountRange) * 100)}%`} stopColor="#737d78" />
              <stop offset="100%" stopColor="#d84a3a" />
            </linearGradient>
          </defs>
          <rect x="0" y={greyTop} width="100" height={greyHeight} className="line-grey-area" />
          <line x1="0" x2="100" y1={medianY} y2={medianY} className="line-median" />
          <polyline points={linePoints} className="transaction-polyline" stroke="url(#transactionLineGradient)" />
          {inputX !== null && (
            <g>
              <circle cx={inputX} cy={inputY} r="2" className="line-input-marker" />
              <text x={Math.min(inputX + 2, 82)} y={Math.max(inputY - 4, 8)} className="line-input-label">Tested input</text>
            </g>
          )}
        </svg>
        <div className="line-chart-axis x-axis"><span>Lowest amount</span><span>Transaction rank</span><span>Highest amount</span></div>
      </div>
      <div className="distribution-legend">
        <span><i className="legend-green" /> Lower amount zone</span>
        <span><i className="legend-grey" /> Grey review band</span>
        <span><i className="legend-red" /> Upper amount zone</span>
        <span><i className="legend-input" /> Tested input</span>
      </div>
    </section>
  );
}

function ReportEvidenceTable({ evidence }) {
  const sensitivity = evidence?.sensitivity || [];
  if (!sensitivity.length) return null;
  return (
    <section className="report-section evidence-table-section">
      <h3>Input sensitivity evidence</h3>
      <div className="report-evidence-table">
        <div className="report-evidence-row report-evidence-header"><strong>Controlled variation</strong><strong>Fraud delta</strong><strong>Unusualness delta</strong></div>
        {sensitivity.map((item) => (
          <div className="report-evidence-row" key={item.name}>
            <span>{item.name}</span>
            <b>{formatSignedPercent(item.fraud_delta)}</b>
            <b>{formatSignedPercent(item.anomaly_delta)}</b>
          </div>
        ))}
      </div>
    </section>
  );
}

function downloadJsonReport(report) {
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${report.transaction?.transaction_id || "transaction"}_research_report.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function MetricReadout({ label, value }) {
  return <span><small>{label}</small><strong>{value}</strong></span>;
}

function ReportItem({ label, value }) {
  return <span><small>{label}</small><strong>{value ?? "Not available"}</strong></span>;
}

function PredictionLogsPage({ logs, selectedLog, onSelect }) {
  const activeLog = selectedLog || logs[0];

  return (
    <section className="logs-page">
      <div className="section-head">
        <span>Audit Trail</span>
        <h2>Prediction Logs</h2>
      </div>
      <div className="logs-layout">
        <section className="log-list-panel">
          <div className="log-list-head">
            <div className="mini-title"><ListChecks size={17} /> Stored transactions</div>
            <span>{logs.length} saved</span>
          </div>
          {logs.length === 0 ? (
            <p className="muted">No prediction runs yet. Submit a transaction from Simulation.</p>
          ) : (
            <div className="log-list">
              {logs.map((log) => (
                <button type="button" className={`log-list-item ${activeLog?.id === log.id ? "selected" : ""}`} key={log.id || log.time} onClick={() => onSelect(log)}>
                  <span>
                    <strong>{log.type}</strong>
                    <small>{log.time} · {currency(log.amount)}</small>
                  </span>
                  <span className="log-list-score">
                    <b>{formatPercent(log.fraud)}</b>
                    <small>{formatResolution(log.resolution) || log.label}</small>
                  </span>
                  <ChevronRight size={17} />
                </button>
              ))}
            </div>
          )}
        </section>

        <LogDetail log={activeLog} />
      </div>
    </section>
  );
}

function LogDetail({ log }) {
  if (!log) {
    return (
      <section className="log-detail-panel log-detail-empty">
        <Search size={28} />
        <h3>Select a prediction</h3>
        <p>Click a saved transaction to inspect its input fields and model response.</p>
      </section>
    );
  }

  const transaction = log.transaction || {};
  const supervised = log.supervised || {};
  const anomaly = log.anomalyOutput || {};
  const fusion = log.fusion || {};
  return (
    <section className="log-detail-panel">
      <div className="log-detail-head">
        <div>
          <div className="mini-title"><FileText size={17} /> Transaction detail</div>
          <h2>{transaction.transaction_id || log.id}</h2>
          <p>{formatTimestamp(transaction.timestamp)} · {transaction.transaction_type}</p>
        </div>
        {log.report && (
          <button type="button" className="download-button" onClick={() => downloadJsonReport(log.report)}>
            <Download size={16} /> Download
          </button>
        )}
      </div>
      <div className="log-detail-scores">
        <ReportItem label="Resolution" value={formatResolution(fusion.resolution)} />
        <ReportItem label="Fusion score" value={formatPercent(fusion.fusion_score)} />
        <ReportItem label="Ambiguity score" value={formatPercent(fusion.ambiguity_score)} />
      </div>
      <ReportSection title="Submitted transaction">
        <ReportItem label="Amount" value={currency(transaction.amount)} />
        <ReportItem label="Sender" value={transaction.sender_id} />
        <ReportItem label="Receiver" value={transaction.receiver_id} />
        <ReportItem label="Device" value={transaction.device_type} />
        <ReportItem label="Merchant" value={transaction.merchant_category} />
        <ReportItem label="Location" value={transaction.location} />
      </ReportSection>
      <ReportSection title="Model response">
        <ReportItem label="Fraud probability" value={formatPercent(supervised.fraud_probability)} />
        <ReportItem label="Fraud output" value={supervised.fraud_prediction ? "Fraud signal" : "Legitimate signal"} />
        <ReportItem label="Anomaly score" value={Number(anomaly.anomaly_score || 0).toFixed(4)} />
        <ReportItem label="Unusualness percentile" value={formatPercent(anomaly.anomaly_percentile ?? anomaly.anomaly_confidence)} />
        <ReportItem label="Model label" value={anomaly.anomaly_label} />
        <ReportItem label="Logged at" value={log.time} />
      </ReportSection>
    </section>
  );
}

function MetricTile({ label, value }) {
  return (
    <article className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function ChartPanel({ title, children }) {
  return (
    <article className="chart-panel">
      <h3>{title}</h3>
      {children}
    </article>
  );
}

function BarList({ data }) {
  const maxValue = Math.max(...data.map((item) => item.value), 1);
  return (
    <div className="bar-list">
      {data.map((item) => (
        <div className="bar-row wide" key={item.label}>
          <span>{item.label}</span>
          <div><i style={{ width: `${(item.value / maxValue) * 100}%` }} /></div>
          <b>{compactNumber(item.value)}</b>
        </div>
      ))}
    </div>
  );
}

function SparkBars({ data }) {
  const maxValue = Math.max(...data.map((item) => item.value), 1);
  return (
    <div className="spark-bars">
      {data.map((item) => (
        <span key={item.label} title={`${item.label}:00 - ${compactNumber(item.value)}`}>
          <i style={{ height: `${Math.max(6, (item.value / maxValue) * 100)}%` }} />
        </span>
      ))}
    </div>
  );
}

function DonutChart({ fraud, legitimate }) {
  const total = Math.max(fraud + legitimate, 1);
  const fraudPercent = (fraud / total) * 100;
  return (
    <div className="donut-wrap">
      <div className="donut" style={{ "--fraud": `${fraudPercent}%` }}>
        <strong>{fraudPercent.toFixed(2)}%</strong>
        <span>Fraud</span>
      </div>
      <div className="legend">
        <span><i className="fraud-dot" /> Fraud {compactNumber(fraud)}</span>
        <span><i className="ok-dot" /> Legitimate {compactNumber(legitimate)}</span>
      </div>
    </div>
  );
}

function formatPercent(value) {
  const numericValue = Number(value);
  return `${((Number.isFinite(numericValue) ? numericValue : 0) * 100).toFixed(1)}%`;
}

function formatResolution(value) {
  return value ? value.replaceAll("_", " ") : "";
}

function resolutionTone(value) {
  if (value === "AMBIGUOUS_REVIEW") return "ambiguous";
  if (value === "FRAUD_LIKELY") return "elevated";
  return "legitimate";
}

function formatTimestamp(value) {
  if (!value) return "Not available";
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime()) ? String(value) : timestamp.toLocaleString();
}

function formatSignedPercent(value) {
  const percent = (value * 100).toFixed(1);
  return `${value >= 0 ? "+" : ""}${percent}%`;
}

function formatDelta(value) {
  return value < 0.000001 ? "0.000000" : value.toFixed(6);
}

function compactNumber(value) {
  return Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 1 }).format(value || 0);
}

function currency(value) {
  return Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(value || 0);
}

function clampPercent(value) {
  return Math.min(100, Math.max(0, value || 0));
}

createRoot(document.getElementById("root")).render(<App />);
