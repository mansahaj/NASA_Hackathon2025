import React from "react";
import * as tf from "@tensorflow/tfjs";

const API_BASE_URL = (process.env.REACT_APP_ML_API_BASE_URL || "").replace(
  /\/+$/,
  ""
);

function generateSyntheticLightCurve({ length = 200, transit = true }) {
  const time = Array.from({ length }, (_, i) => i / length);
  const baseline = time.map(() => 1 + (Math.random() - 0.5) * 0.01);
  if (transit) {
    const center = Math.floor(length * (0.4 + Math.random() * 0.2));
    const width = Math.floor(length * 0.05);
    for (let i = -width; i <= width; i++) {
      const idx = center + i;
      if (idx >= 0 && idx < length) {
        const depth = 0.01 + Math.random() * 0.01;
        const shape = Math.exp(-(i * i) / (2 * Math.pow(width / 2, 2)));
        baseline[idx] -= depth * shape;
      }
    }
  }
  return baseline;
}

function simpleHeuristic(lightCurve) {
  const window = 7;
  let minAvg = Infinity;
  for (let i = 0; i <= lightCurve.length - window; i++) {
    const avg =
      lightCurve.slice(i, i + window).reduce((a, b) => a + b, 0) / window;
    if (avg < minAvg) minAvg = avg;
  }
  const dip = 1 - minAvg;
  const score = Math.max(0, Math.min(1, dip * 50));
  return score;
}

export default function ExoplanetFinder() {
  const [lightCurve, setLightCurve] = React.useState([]);
  const [score, setScore] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [modelId, setModelId] = React.useState(null);
  const [modelName, setModelName] = React.useState("");
  const [modelUploading, setModelUploading] = React.useState(false);
  const [apiError, setApiError] = React.useState("");

  const handleGenerate = (withTransit) => {
    const lc = generateSyntheticLightCurve({ transit: withTransit });
    setLightCurve(lc);
    setScore(null);
  };

  const handleFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    const values = text
      .split(/[\n\r\t ,]+/)
      .map((v) => parseFloat(v))
      .filter((v) => Number.isFinite(v));
    if (values.length > 5) {
      setLightCurve(values.slice(0, 1024));
      setScore(null);
    }
  };

  const handleModelUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!API_BASE_URL) {
      setApiError("Backend URL not configured (REACT_APP_ML_API_BASE_URL)");
      return;
    }
    setApiError("");
    setModelUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const resp = await fetch(`${API_BASE_URL}/model`, {
        method: "POST",
        body: form,
      });
      if (!resp.ok) throw new Error(`Upload failed (${resp.status})`);
      const data = await resp.json();
      const newModelId = data.modelId || data.id || data.model_id || null;
      if (!newModelId) throw new Error("No modelId returned from API");
      setModelId(newModelId);
      setModelName(file.name);
    } catch (err) {
      console.error(err);
      setApiError(String(err.message || err));
      setModelId(null);
      setModelName("");
    } finally {
      setModelUploading(false);
    }
  };

  const clearModel = () => {
    setModelId(null);
    setModelName("");
    setApiError("");
  };

  const runDetection = async () => {
    setLoading(true);
    try {
      if (modelId && API_BASE_URL) {
        try {
          const body = {
            modelId,
            series: Array.from(lightCurve).slice(0, 4096),
          };
          const resp = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          if (!resp.ok) throw new Error(`Predict failed (${resp.status})`);
          const result = await resp.json();
          const raw =
            result.score ?? result.probability ?? result.pred ?? result[0];
          const val = typeof raw === "number" ? raw : parseFloat(raw);
          if (Number.isFinite(val)) {
            setScore(Math.max(0, Math.min(1, val)));
            return;
          }
        } catch (e) {
          console.warn("Falling back to heuristic due to API error", e);
        }
      }

      const lc = tf.tensor1d(lightCurve);
      const normalized = lc
        .sub(lc.mean())
        .div(lc.std().add(1e-6))
        .expandDims(0)
        .expandDims(-1);
      const kernel = tf.tensor3d([[-0.5], [1], [1], [1], [-0.5]], [5, 1, 1]);
      const conv = tf.conv1d(normalized, kernel, 1, "same");
      const smoothKernel = tf.tensor3d(
        new Array(7).fill(0).map(() => [[1 / 7]]),
        [7, 1, 1]
      );
      const smoothed = tf.conv1d(conv, smoothKernel, 1, "same");
      const stat = smoothed.min().dataSync()[0];
      const heuristicScore = simpleHeuristic(lightCurve);
      const combined = Math.max(
        0,
        Math.min(1, heuristicScore * 0.7 + Math.min(1, -stat / 5) * 0.3)
      );
      setScore(combined);
      lc.dispose();
      normalized.dispose();
      kernel.dispose();
      conv.dispose();
      smoothKernel.dispose();
      smoothed.dispose();
    } catch (e) {
      console.error(e);
      const heuristicScore = simpleHeuristic(lightCurve);
      setScore(heuristicScore);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page container">
      <h2>Exoplanet Finder (Demo)</h2>
      <p>
        Upload a light curve file (CSV or newline-separated brightness values)
        or generate a synthetic series. Then run the detector to estimate the
        likelihood of a transit.
      </p>

      <div className="card">
        <div className="controls">
          <input type="file" accept=".txt,.csv" onChange={handleFile} />
          <div
            style={{
              marginTop: 10,
              display: "flex",
              gap: 8,
              alignItems: "center",
              flexWrap: "wrap",
            }}
          >
            <input
              type="file"
              accept=".pkl,application/octet-stream"
              onChange={handleModelUpload}
              disabled={modelUploading || !API_BASE_URL}
            />
            {modelUploading && <span>Uploading model…</span>}
            {modelId && (
              <>
                <span>Active model: {modelName || modelId}</span>
                <button className="button secondary" onClick={clearModel}>
                  Remove Model
                </button>
              </>
            )}
            {!API_BASE_URL && (
              <span style={{ color: "#f88" }}>Backend not configured</span>
            )}
            {!!apiError && <span style={{ color: "#f88" }}>{apiError}</span>}
          </div>
          <div className="buttons-row">
            <button
              className="button secondary"
              onClick={() => handleGenerate(false)}
            >
              Generate No-Transit
            </button>
            <button className="button" onClick={() => handleGenerate(true)}>
              Generate With Transit
            </button>
            <button
              className="button primary"
              disabled={!lightCurve.length || loading}
              onClick={runDetection}
            >
              {loading ? "Analyzing…" : "Run Detector"}
            </button>
          </div>
        </div>

        <LightCurveChart data={lightCurve} />

        {score !== null && (
          <div className="result">
            <div className="scorebar">
              <div
                className="scorefill"
                style={{ width: `${Math.round(score * 100)}%` }}
              />
            </div>
            <p>
              <strong>Transit likelihood:</strong> {Math.round(score * 100)}%
            </p>
          </div>
        )}
      </div>

      <p className="fineprint">
        This is a simplified demonstration and not a scientific tool. Real
        missions (e.g., Kepler, TESS) use more rigorous pipelines and domain
        expertise.
      </p>
    </div>
  );
}

function LightCurveChart({ data }) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.clientWidth;
    const height = 220;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0b1021";
    ctx.fillRect(0, 0, width, height);
    // axes
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10);
    ctx.lineTo(40, height - 30);
    ctx.lineTo(width - 10, height - 30);
    ctx.stroke();
    // plot
    if (!data || data.length === 0) return;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const y = (v) => {
      if (max === min) return height / 2;
      return height - 40 - ((v - min) / (max - min)) * (height - 60) + 10;
    };
    ctx.strokeStyle = "#61dafb";
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const xx = 40 + (i / Math.max(1, data.length - 1)) * (width - 60);
      const yy = y(v);
      if (i === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
  }, [data]);

  return (
    <canvas
      ref={ref}
      style={{ width: "100%", height: 220, display: "block", borderRadius: 8 }}
    />
  );
}
