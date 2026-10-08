import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertCircle,
  ArrowDown,
  ArrowUp,
  Brain,
  CheckCircle2,
  Cpu,
  Database,
  FlaskConical,
  Leaf,
  LoaderCircle,
  Play,
  RotateCw,
  Sprout,
  TrendingUp,
  Zap,
} from "lucide-react";
import { getHealth, getModels, getResults, predict } from "./services/api";
import "./App.css";

const EMPTY_FORM = {
  fertilizer: "",
  temp: "",
  N: "",
  P: "",
  K: "",
};
const EMPTY_MODELS = [];

const MODEL_PREDICTIONS = [
  {
    id: "linear_regression",
    name: "Linear Regression",
    group: "Classical",
    tone: "classical",
  },
  {
    id: "linear_regression_quantum",
    name: "Linear Regression + Quantum",
    group: "Quantum-integrated",
    tone: "quantum",
  },
  {
    id: "random_forest",
    name: "Random Forest",
    group: "Classical",
    tone: "classical",
  },
  {
    id: "random_forest_quantum",
    name: "Random Forest + Quantum",
    group: "Quantum-integrated",
    tone: "quantum",
  },
];

const METRICS = [
  { key: "r2", label: "R²", direction: "higher", color: "#2f795b" },
  { key: "mae", label: "MAE", direction: "lower", color: "#b87835" },
  { key: "rmse", label: "RMSE", direction: "lower", color: "#4f6e9c" },
  { key: "mape", label: "MAPE", direction: "lower", color: "#8263a3" },
];

const FIELD_CONFIG = [
  { key: "fertilizer", label: "Fertilizer", detail: "Application amount" },
  { key: "temp", label: "Temperature", detail: "Ambient temperature" },
  { key: "N", label: "Nitrogen (N)", detail: "Soil nutrient" },
  { key: "P", label: "Phosphorus (P)", detail: "Soil nutrient" },
  { key: "K", label: "Potassium (K)", detail: "Soil nutrient" },
];

function formatMetric(value, key) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return "—";
  return key === "mape" ? `${numericValue.toFixed(4)}%` : numericValue.toFixed(4);
}

function getModelShortName(name) {
  return name
    .replace("Linear Regression + Quantum", "LR + Quantum")
    .replace("Random Forest + Quantum", "RF + Quantum")
    .replace("Linear Regression", "Linear Regression")
    .replace("Random Forest", "Random Forest");
}

function LoadingPanel({ message = "Loading model results..." }) {
  return (
    <div className="state-panel" role="status" aria-live="polite">
      <LoaderCircle className="spin" size={25} />
      <span>{message}</span>
    </div>
  );
}

function ApiError({ message, onRetry }) {
  return (
    <div className="error-banner" role="alert">
      <AlertCircle size={19} />
      <div>
        <strong>Unable to connect to the prediction API.</strong>
        <p>{message || "Make sure the FastAPI backend is running."}</p>
      </div>
      <button className="retry-button" type="button" onClick={onRetry}>
        <RotateCw size={15} />
        Retry
      </button>
    </div>
  );
}

function MetricChart({ metric, models }) {
  const data = models.map((model) => ({
    name: getModelShortName(model.name),
    value: Number(model[metric.key]),
    type: model.type,
  }));

  return (
    <article className="chart-card">
      <div className="chart-heading">
        <div>
          <span className="eyebrow">MODEL METRIC</span>
          <h3>{metric.label} comparison</h3>
        </div>
        <span className="metric-direction" title={metric.direction === "higher" ? "Higher is better" : "Lower is better"}>
          {metric.direction === "higher" ? <ArrowUp size={15} /> : <ArrowDown size={15} />}
          {metric.direction === "higher" ? "Higher is better" : "Lower is better"}
        </span>
      </div>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 12, left: -17, bottom: 4 }}>
            <CartesianGrid stroke="#e8ece8" strokeDasharray="3 5" vertical={false} />
            <XAxis
              dataKey="name"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#69736d", fontSize: 11 }}
              interval={0}
              height={48}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#8b938d", fontSize: 11 }}
              width={48}
            />
            <Tooltip
              cursor={{ fill: "#eff3ef" }}
              formatter={(value) => [formatMetric(value, metric.key), metric.label]}
              contentStyle={{
                borderRadius: 12,
                border: "1px solid #e1e7e1",
                boxShadow: "0 8px 24px rgba(26, 48, 34, .08)",
              }}
            />
            <Bar dataKey="value" name={metric.label} radius={[6, 6, 0, 0]} maxBarSize={52}>
              {data.map((entry) => (
                <Cell
                  key={entry.name}
                  fill={entry.type.toLowerCase().includes("quantum") ? "#789b87" : metric.color}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </article>
  );
}

function App() {
  const [results, setResults] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [apiStatus, setApiStatus] = useState("checking");
  const [loadingResults, setLoadingResults] = useState(true);
  const [form, setForm] = useState(EMPTY_FORM);
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");
  const [predictionError, setPredictionError] = useState("");
  const [dashboardRequestId, setDashboardRequestId] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const loadDashboard = async () => {
      try {
        const [health, modelData, resultData] = await Promise.all([
          getHealth(),
          getModels(),
          getResults(),
        ]);
        if (health.status !== "ok") {
          throw new Error("The API health check did not return an OK status.");
        }
        if (!Array.isArray(modelData.models) || !Array.isArray(resultData.models)) {
          throw new Error("The API returned model data in an unexpected format.");
        }
        if (cancelled) return;
        setAvailableModels(modelData.models);
        setResults(resultData);
        setApiStatus("online");
        setError("");
      } catch (loadError) {
        if (cancelled) return;
        console.error("Could not load dashboard data:", loadError);
        setApiStatus("offline");
        setError(loadError.message);
      } finally {
        if (!cancelled) setLoadingResults(false);
      }
    };
    loadDashboard();
    return () => {
      cancelled = true;
    };
  }, [dashboardRequestId]);

  const models = results?.models ?? EMPTY_MODELS;
  const chartModels = models.filter((model) =>
    METRICS.every((metric) => Number.isFinite(Number(model[metric.key]))),
  );

  const conclusion = (() => {
    if (!models.length) return "";
    const ranked = [...models].sort((first, second) => Number(second.r2) - Number(first.r2));
    const strongest = ranked[0];
    const classical = models.filter((model) => model.type.toLowerCase() === "classical");
    const quantum = models.filter((model) => model.type.toLowerCase().includes("quantum"));
    const allQuantumBelowClassical = quantum.every((quantumModel) => {
      const baseName = quantumModel.name.replace(" + Quantum", "");
      const baseline = classical.find((model) => model.name === baseName);
      return baseline && Number(quantumModel.r2) < Number(baseline.r2);
    });

    if (strongest.name === "Random Forest" && allQuantumBelowClassical) {
      return "Based on the evaluated test subset, Random Forest achieved the strongest predictive performance among the four evaluated approaches. The quantum-integrated variants did not outperform their classical counterparts in this experiment.";
    }
    return `Based on the evaluated test subset, ${strongest.name} achieved the highest R² among the four evaluated approaches. These results describe this experiment only and do not establish a quantum advantage.`;
  })();

  const retryDashboard = () => {
    setLoadingResults(true);
    setError("");
    setDashboardRequestId((requestId) => requestId + 1);
  };

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((previous) => ({ ...previous, [name]: value }));
    setPredictionError("");
  };

  const handlePredict = async (event) => {
    event.preventDefault();
    setPredicting(true);
    setPrediction(null);
    setPredictionError("");

    const input = Object.fromEntries(
      Object.entries(form).map(([key, value]) => [key, Number(value)]),
    );
    if (Object.values(input).some((value) => !Number.isFinite(value))) {
      setPredictionError("Enter a valid number for each of the five inputs.");
      setPredicting(false);
      return;
    }

    try {
      const response = await predict(input);
      const missingPredictions = MODEL_PREDICTIONS.filter(
        (model) => !Number.isFinite(Number(response.predictions?.[model.id])),
      );
      if (missingPredictions.length) {
        throw new Error("The API response did not include all four numeric model predictions.");
      }
      setPrediction(response);
      setApiStatus("online");
    } catch (predictError) {
      console.error("Prediction request failed:", predictError);
      setApiStatus("offline");
      setPredictionError(predictError.message);
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="app-shell">
      <nav className="navbar">
        <a className="brand" href="#top" aria-label="QuantumYield home">
          <span className="brand-mark"><Leaf size={21} /></span>
          <span className="brand-copy">
            <strong>QuantumYield</strong>
            <small>Crop analytics</small>
          </span>
        </a>
        <div className={`api-indicator ${apiStatus}`} aria-live="polite">
          <span className="status-dot" />
          {apiStatus === "checking" ? "Connecting to API" : apiStatus === "online" ? `API connected · ${availableModels.length} models` : "API unavailable"}
        </div>
      </nav>

      <main id="top">
        <section className="hero">
          <div className="hero-inner">
            <div className="hero-copy">
              <span className="hero-kicker"><Sprout size={15} /> CLASSICAL + QUANTUM MODEL STUDY</span>
              <h1>Understand the signals behind <em>crop yield.</em></h1>
              <p>Explore measured model performance and compare predictions from classical and quantum-integrated approaches.</p>
              <a className="hero-link" href="#playground">Try a prediction <ArrowDown size={15} /></a>
            </div>
            <div className="hero-art" aria-hidden="true">
              <div className="orbit orbit-one" />
              <div className="orbit orbit-two" />
              <div className="hero-plant"><Sprout size={96} strokeWidth={1.1} /></div>
              <span className="orbit-dot dot-one" />
              <span className="orbit-dot dot-two" />
              <span className="hero-art-caption">DATA-DRIVEN<br />CULTIVATION</span>
            </div>
          </div>
        </section>

        <section className="section overview-section">
          <div className="section-heading">
            <span className="section-icon"><Database size={19} /></span>
            <div>
              <span className="eyebrow">THE EXPERIMENT</span>
              <h2>Dataset overview</h2>
              <p>Evaluation context supplied by the prediction API.</p>
            </div>
          </div>

          {loadingResults && <LoadingPanel />}
          {error && <ApiError message={error} onRetry={retryDashboard} />}
          {results && (
            <>
              <div className="overview-grid">
                <article className="overview-card">
                  <span className="overview-icon"><Database size={18} /></span>
                  <div><span>Dataset samples</span><strong>{results.dataset.samples.toLocaleString()}</strong></div>
                </article>
                <article className="overview-card">
                  <span className="overview-icon"><Brain size={18} /></span>
                  <div><span>Input features</span><strong>{results.dataset.features.length}</strong></div>
                </article>
                <article className="overview-card">
                  <span className="overview-icon"><TrendingUp size={18} /></span>
                  <div><span>Prediction target</span><strong>{results.dataset.target}</strong></div>
                </article>
                <article className="overview-card">
                  <span className="overview-icon"><FlaskConical size={18} /></span>
                  <div><span>Compared models</span><strong>{models.length}</strong></div>
                </article>
              </div>
              <div className="feature-strip">
                <span>FEATURES</span>
                {results.dataset.features.map((feature) => <span className="feature-chip" key={feature}>{feature}</span>)}
                <span className="feature-target">Target · {results.dataset.target}</span>
              </div>
              <p className="evaluation-note">
                Metrics use the same {results.evaluation.comparison_samples}-row evaluation subset. MAPE is reported in {results.evaluation.metric_units.mape}.
              </p>
            </>
          )}
        </section>

        <section className="section comparison-section" id="comparison">
          <div className="section-heading">
            <span className="section-icon"><Activity size={19} /></span>
            <div>
              <span className="eyebrow">MEASURED PERFORMANCE</span>
              <h2>Model comparison</h2>
              <p>All displayed metrics are loaded directly from the verified backend results.</p>
            </div>
          </div>
          {loadingResults && !results && <LoadingPanel />}
          {results && (
            <div className="table-card">
              <div className="table-scroll">
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      <th scope="col">Approach</th>
                      <th scope="col">R² <ArrowUp size={12} /></th>
                      <th scope="col">MAE <ArrowDown size={12} /></th>
                      <th scope="col">MSE <ArrowDown size={12} /></th>
                      <th scope="col">RMSE <ArrowDown size={12} /></th>
                      <th scope="col">MAPE <ArrowDown size={12} /></th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((model) => {
                      const integrated = model.type.toLowerCase().includes("quantum");
                      return (
                        <tr key={model.name}>
                          <th scope="row">{model.name}</th>
                          <td><span className={`approach-tag ${integrated ? "quantum-tag" : ""}`}>{integrated ? "Quantum-integrated" : "Classical"}</span></td>
                          <td className="metric-primary">{formatMetric(model.r2, "r2")}</td>
                          <td>{formatMetric(model.mae, "mae")}</td>
                          <td>{formatMetric(model.mse, "mse")}</td>
                          <td>{formatMetric(model.rmse, "rmse")}</td>
                          <td>{formatMetric(model.mape, "mape")}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="table-footnote"><CheckCircle2 size={14} /> Backend-sourced values · no frontend metric estimates</div>
            </div>
          )}
        </section>

        <section className="section charts-section">
          <div className="section-heading">
            <span className="section-icon"><TrendingUp size={19} /></span>
            <div>
              <span className="eyebrow">VISUAL ANALYSIS</span>
              <h2>Performance at a glance</h2>
              <p>Each chart uses the same model results shown in the comparison table.</p>
            </div>
          </div>
          {loadingResults && !results && <LoadingPanel />}
          {results && chartModels.length === 0 && (
            <div className="error-banner" role="alert"><AlertCircle size={18} /><p>Metrics are unavailable for charting in the API response.</p></div>
          )}
          {results && chartModels.length > 0 && (
            <div className="charts-grid">
              {METRICS.map((metric) => <MetricChart key={metric.key} metric={metric} models={chartModels} />)}
            </div>
          )}
          <div className="chart-legend">
            <span><i className="legend-dot classical-dot" /> Classical</span>
            <span><i className="legend-dot quantum-dot" /> Quantum-integrated</span>
          </div>
        </section>

        <section className="section conclusion-section">
          <div className="conclusion-card">
            <span className="conclusion-icon"><CheckCircle2 size={21} /></span>
            <div>
              <span className="eyebrow">WHAT THE RESULTS SAY</span>
              <h2>Evidence before advantage</h2>
              {loadingResults ? <p>Loading measured results...</p> : <p>{conclusion || "Conclusion is unavailable until backend results load."}</p>}
            </div>
          </div>
        </section>

        <section className="methodology-section" id="methodology">
          <div className="section methodology-inner">
            <div className="section-heading">
              <span className="section-icon"><Cpu size={19} /></span>
              <div>
                <span className="eyebrow">HOW THE HYBRID MODELS WORK</span>
                <h2>Quantum-integrated methodology</h2>
                <p>Inference combines a classical baseline with trained quantum corrections.</p>
              </div>
            </div>
            <div className="pipeline">
              <article className="pipeline-step"><span>01</span><h3>Input data</h3><p>Five soil, fertilizer and weather features.</p></article>
              <span className="pipeline-arrow">→</span>
              <article className="pipeline-step"><span>02</span><h3>Preprocessing</h3><p>Apply the preprocessing saved with each model.</p></article>
              <span className="pipeline-arrow">→</span>
              <article className="pipeline-step"><span>03</span><h3>Classical model</h3><p>Generate the baseline yield estimate.</p></article>
              <span className="pipeline-arrow">→</span>
              <article className="pipeline-step quantum-step"><span>04</span><h3>QKRR + VQR</h3><p>Apply the saved quantum-integrated corrections.</p></article>
            </div>
          </div>
        </section>

        <section className="section playground-section" id="playground">
          <div className="section-heading">
            <span className="section-icon"><Zap size={19} /></span>
            <div>
              <span className="eyebrow">LIVE MODEL INFERENCE</span>
              <h2>Prediction playground</h2>
              <p>Submit one input set to compare all four trained model outputs.</p>
            </div>
          </div>

          <div className="playground-grid">
            <form className="input-card" onSubmit={handlePredict}>
              <div className="input-card-heading">
                <div><h3>Crop and soil inputs</h3><p>Enter numeric values for every feature.</p></div>
                <span className="input-heading-icon"><Leaf size={21} /></span>
              </div>
              <div className="input-grid">
                {FIELD_CONFIG.map((field) => (
                  <label className="input-group" htmlFor={`input-${field.key}`} key={field.key}>
                    <span>{field.label}</span>
                    <small>{field.detail}</small>
                    <input
                      id={`input-${field.key}`}
                      name={field.key}
                      type="number"
                      step="any"
                      value={form[field.key]}
                      onChange={handleChange}
                      placeholder="Enter value"
                      required
                    />
                  </label>
                ))}
              </div>
              <button className="predict-button" type="submit" disabled={predicting || loadingResults}>
                {predicting ? <><LoaderCircle className="spin" size={17} /> Predicting...</> : <><Play size={16} fill="currentColor" /> Compare model predictions</>}
              </button>
              {predictionError && <p className="form-error" role="alert"><AlertCircle size={15} />{predictionError}</p>}
              <p className="form-note">Predictions are model outputs from the connected FastAPI service.</p>
            </form>

            <div className="prediction-panel" aria-live="polite">
              <div className="prediction-panel-heading">
                <div><span className="eyebrow">INFERENCE OUTPUT</span><h3>Four-model comparison</h3></div>
                {prediction && <span className="output-status"><CheckCircle2 size={14} /> Complete</span>}
              </div>
              {predicting ? (
                <div className="prediction-state"><LoaderCircle className="spin" size={27} /><strong>Predicting...</strong><p>Waiting for the model outputs from the API.</p></div>
              ) : prediction ? (
                <>
                  <p className="output-caption">Returned predictions · model output values</p>
                  <div className="prediction-list">
                    {MODEL_PREDICTIONS.map((model) => (
                      <article className={`prediction-item ${model.tone}`} key={model.id}>
                        <span className="prediction-mark">{model.tone === "quantum" ? <Cpu size={17} /> : <FlaskConical size={17} />}</span>
                        <div className="prediction-label"><strong>{model.name}</strong><small>{model.group}</small></div>
                        <b>{Number(prediction.predictions[model.id]).toFixed(6)}</b>
                      </article>
                    ))}
                  </div>
                  <div className="output-input">
                    <span>Input used</span>
                    <p>{FIELD_CONFIG.map((field) => `${field.key === "temp" ? "Temp" : field.key} ${prediction.input[field.key]}`).join(" · ")}</p>
                  </div>
                </>
              ) : (
                <div className="prediction-state">
                  <div className="ready-mark"><Sprout size={26} /></div>
                  <strong>Ready when you are</strong>
                  <p>Complete the five inputs and run a prediction to see all four model outputs here.</p>
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="section final-note-section">
          <div className="final-note">
            <span><CheckCircle2 size={18} /></span>
            <p>Model metrics and predictions are served by the verified API. Results describe the evaluated dataset and should not be treated as a guarantee of future crop yield.</p>
          </div>
        </section>
      </main>

      <footer className="footer">
        <a className="brand footer-brand" href="#top">
          <span className="brand-mark"><Leaf size={18} /></span>
          <span className="brand-copy"><strong>QuantumYield</strong><small>Crop analytics</small></span>
        </a>
        <span>Classical and quantum-integrated crop yield research</span>
        <a href="#comparison">View results <ArrowUp size={14} /></a>
      </footer>
    </div>
  );
}

export default App;
