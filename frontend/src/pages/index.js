// File: pages/index.js
import React, { useState } from 'react';
import { useRouter } from 'next/router';
import ModelComparison from '../components/ModelComparison';

export default function Home() {
  const [f, sF] = useState(null);  // file
  const [t, sT] = useState('');    // text
  const [r, sR] = useState(null);  // results from advanced pipeline
  const [m, sM] = useState('');    // message
  const ro = useRouter();

  // 1) handle file selection
  const hFS = (e) => {
    sF(e.target.files[0]);
  };

  // 2) load file into memory
  const hLF = () => {
    if (!f) {
      sM('No file selected.');
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      sT(ev.target.result);
      sM('File content loaded. Ready to upload to Flask.');
    };
    reader.readAsText(f);
  };

  // 3) upload text to Flask
  const hUT = async () => {
    if (!t.trim()) {
      sM('No text in memory. Please load file first.');
      return;
    }
    sM('Uploading text to Flask...');
    try {
      // Replace URL with your real Flask endpoint
      const resp = await fetch('http://127.0.0.1:5000', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: t })
      });
      const d = await resp.json();
      if (!resp.ok) {
        sM('Error: ' + (d.error || 'unknown'));
      } else {
        sM('Uploaded to Flask successfully!');
      }
    } catch (err) {
      sM('Error: ' + err.message);
    }
  };

  // 4) run advanced pipeline
  const hAdv = async () => {
    sM('Running advanced pipeline on Flask...');
    try {
      const resp = await fetch('http://127.0.0.1:5000', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const d = await resp.json();
      if (!resp.ok) {
        sM('Error: ' + (d.error || 'unknown'));
      } else {
        sR(d.results);
        sM(`Done in ${d.elapsed} seconds.`);
      }
    } catch (err) {
      sM('Error: ' + err.message);
    }
  };

  return (
    <div style={{ backgroundColor: '#000', color: '#fff', minHeight: '100vh', padding: 20, fontFamily: 'Arial' }}>
      <h1>Advanced Topic Modeling Tool</h1>
      <h3>Author: Kumari Ankita</h3>
      <p>
        1) Select your .txt file. <br/>
        2) "Load File" to read into memory. <br/>
        3) "Upload to Flask" to send text to the backend. <br/>
        4) "Run Advanced Analysis" to see model metrics + bubble chart. <br/>
        5) Also see "Preprocessed Data" and "Visualize Graph" links below.
      </p>

      <input type="file" accept=".txt" onChange={hFS} />
      <button onClick={hLF} style={{ marginLeft: 10 }}>Load File</button>
      <button onClick={hUT} style={{ marginLeft: 10 }}>Upload to Flask</button>

      <br/><br/>
      <button onClick={hAdv}>Run Advanced Analysis</button>
      <button onClick={() => ro.push('/preprocess')} style={{ marginLeft: 10 }}>Preprocessed Data</button>
      <button onClick={() => ro.push('/visualize')} style={{ marginLeft: 10 }}>Visualize Graph</button>

      <p style={{ color: 'yellow' }}>{m}</p>

      {r && <ModelComparison results={r} />}
    </div>
  );
}
