// File: pages/preprocess.js
import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';

export default function Preprocess() {
  const [freq, setFreq] = useState([]);
  const [msg, setMsg] = useState('');
  const rt = useRouter();

  useEffect(() => {
    async function fetchData() {
      setMsg('Loading...');
      try {
        const r = await fetch('http://127.0.0.1:5000');
        const d = await r.json();
        if (!r.ok) {
          setMsg('Error: ' + (d.error || 'unknown'));
        } else {
          setFreq(d.frequency);
          setMsg('');
        }
      } catch (err) {
        setMsg('Error: ' + err.message);
      }
    }
    fetchData();
  }, []);

  return (
    <div style={{ backgroundColor: '#000', color: '#fff', minHeight: '100vh', padding: 20, fontFamily: 'Arial' }}>
      <h1>Preprocessed Data</h1>
      <p style={{ color: 'yellow' }}>{msg}</p>
      {freq.length > 0 && (
        <table style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ border: '1px solid #777', padding: 5 }}>Word</th>
              <th style={{ border: '1px solid #777', padding: 5 }}>Count</th>
            </tr>
          </thead>
          <tbody>
            {freq.map((fItem, i) => (
              <tr key={i}>
                <td style={{ border: '1px solid #777', padding: 5 }}>{fItem.word}</td>
                <td style={{ border: '1px solid #777', padding: 5 }}>{fItem.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <br/>
      <button onClick={() => rt.push('/')}>Home</button>{' '}
      <button onClick={() => rt.push('/visualize')}>Visualize Graph</button>
    </div>
  );
}