import { useMemo, useState } from "react";

const API_BASE = "http://localhost:8000";

function formatPercent(x) {
  if (typeof x !== "number" || Number.isNaN(x)) return "-";
  return `${(x * 100).toFixed(2)}%`;
}

function severityColor(label) {
  if (!label) return "var(--muted)";
  if (label === "No_DR") return "var(--good)";
  if (label === "Mild") return "var(--mild)";
  if (label === "Moderate") return "var(--moderate)";
  return "var(--bad)";
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // ✅ patient inputs
  const [name, setName] = useState("");   // 🆕
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [years, setYears] = useState("");
  const [blood, setBlood] = useState("");
  const [symptoms, setSymptoms] = useState("");

  const badgeStyle = useMemo(() => {
    const c = severityColor(result?.label);
    return { borderColor: c, color: c };
  }, [result?.label]);

  async function onPickFile(e) {
    setError("");
    setResult(null);

    const f = e.target.files?.[0] || null;
    setFile(f);

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(f ? URL.createObjectURL(f) : null);
  }

  async function onPredict() {
    setError("");
    setResult(null);

    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    if (!name || !age || !gender || !years || !blood) {
      setError("Please fill all patient details including name and blood group.");
      return;
    }

    setLoading(true);

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("name", name);   // 🆕
      form.append("age", age);
      form.append("gender", gender);
      form.append("diabetes_years", years);
      form.append("blood", blood);
      form.append("symptoms", symptoms);

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: form,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(JSON.stringify(data.detail));
      }

      setResult(data);
    } catch (e) {
      setError(e?.message || "API error.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1>AI Retinal Disease Detection System</h1>
        <p className="subtitle">
          Upload a retinal image and enter patient details for prediction.
        </p>
      </header>

      <main className="grid">
        {/* LEFT */}
        <section className="card">
          <h2>Upload & Patient Details</h2>

          <div className="uploadRow">
            <label className="fileButton">
              Choose Image
              <input type="file" onChange={onPickFile} />
            </label>

            <button className="primary" onClick={onPredict} disabled={loading}>
              {loading ? "Analyzing..." : "Predict"}
            </button>
          </div>

          {previewUrl && (
            <div className="preview">
              <img src={previewUrl} alt="Preview" />
            </div>
          )}

          {/* 🆕 NAME FIELD */}
          <input
            className="fullInput"
            type="text"
            placeholder="Patient Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />

          {/* FORM */}
          <div className="formRow">
            <input
              type="number"
              placeholder="Age"
              value={age}
              onChange={(e) => setAge(e.target.value)}
            />

            <select value={gender} onChange={(e) => setGender(e.target.value)}>
              <option value="">Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>

            <input
              type="number"
              placeholder="Diabetic Years"   // ✅ FIXED
              value={years}
              onChange={(e) => setYears(e.target.value)}
            />
          </div>

          <div className="formRow">
            <select value={blood} onChange={(e) => setBlood(e.target.value)}>
              <option value="">Blood Group</option>
              <option value="A+">A+</option>
              <option value="A-">A-</option>
              <option value="B+">B+</option>
              <option value="B-">B-</option>
              <option value="O+">O+</option>
              <option value="O-">O-</option>
              <option value="AB+">AB+</option>
              <option value="AB-">AB-</option>
            </select>

            <input
              type="text"
              placeholder="Symptoms"
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
            />
          </div>

          {error && <p className="alert">{error}</p>}
        </section>

        {/* RIGHT */}
        <section className="card">
          <h2>Result</h2>

          {!result && !loading && <p>No result yet</p>}

          {result && (
            <div className="resultCard">
              <h3 style={{ color: severityColor(result.label) }}>
                {result.label}
              </h3>

              <p>Confidence: {formatPercent(result.confidence)}</p>

              <div className="divider" />

              {/* 🆕 FULL PATIENT DETAILS */}
              <h4>Patient Details</h4>
              <p>Name: {result.patient.name}</p>
              <p>Age: {result.patient.age}</p>
              <p>Gender: {result.patient.gender}</p>
              <p>Blood Group: {result.patient.blood}</p>
              <p>Diabetic Years: {result.patient.diabetes_years}</p>
              <p>Symptoms: {result.patient.symptoms}</p>

              <div className="divider" />

              <p className="explainText">{result.explanation}</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}