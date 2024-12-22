// File: components/ModelComparison.js
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

export default function ModelComparison({ results }) {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!results?.models) return;

    const data = results.models;
    const w = 600, h = 400;
    const m = { top: 20, right: 20, bottom: 40, left: 50 };

    const svg = d3.select(chartRef.current)
      .attr('width', w)
      .attr('height', h)
      .style('background', '#333');

    svg.selectAll('*').remove();

    // x => coherence
    const xExt = d3.extent(data, d => d.coherence);
    const xScale = d3.scaleLinear()
      .domain([Math.min(xExt[0], 0.35), Math.max(xExt[1], 0.55)])
      .range([m.left, w - m.right]);

    // y => time
    const yExt = d3.extent(data, d => d.time_sec);
    const yScale = d3.scaleLinear()
      .domain([0, (yExt[1] || 100) * 1.2])
      .range([h - m.bottom, m.top]);

    // r => topic_diversity
    const rExt = d3.extent(data, d => d.topic_diversity);
    const rScale = d3.scaleSqrt()
      .domain(rExt)
      .range([5, 40]);

    // axes
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5);

    svg.append('g')
      .attr('transform', `translate(0,${h - m.bottom})`)
      .call(xAxis)
      .attr('color', 'white');

    svg.append('g')
      .attr('transform', `translate(${m.left},0)`)
      .call(yAxis)
      .attr('color', 'white');

    // labels
    svg.append('text')
      .attr('x', w/2)
      .attr('y', h - 5)
      .attr('fill', 'white')
      .attr('text-anchor', 'middle')
      .text('Coherence');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h/2)
      .attr('y', 15)
      .attr('fill', 'white')
      .attr('text-anchor', 'middle')
      .text('Time (sec)');

    // circles
    svg.selectAll('circle')
      .data(data)
      .enter().append('circle')
      .attr('cx', d => xScale(d.coherence))
      .attr('cy', d => yScale(d.time_sec))
      .attr('r', d => rScale(d.topic_diversity))
      .attr('fill', (d, i) => d3.schemeSet2[i % 8])
      .attr('fill-opacity', 0.8);

    // model labels
    svg.selectAll('.lbl')
      .data(data)
      .enter().append('text')
      .attr('x', d => xScale(d.coherence))
      .attr('y', d => yScale(d.time_sec))
      .attr('dy', '.35em')
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .attr('font-size', 10)
      .text(d => d.model);

  }, [results]);

  if (!results?.models) return null;

  return (
    <div>
      <h2>Model Performance (Bubble Chart)</h2>
      <p style={{ fontStyle: 'italic' }}>Bubble size = topic diversity</p>
      <svg ref={chartRef}></svg>

      <h3>Comparison Table</h3>
      <table style={{ borderCollapse: 'collapse', color: 'white' }}>
        <thead>
          <tr>
            <th>Model</th>
            <th>Coherence</th>
            <th>UMass</th>
            <th>NPMI</th>
            <th>UCI</th>
            <th>Topic Diversity</th>
            <th>Silhouette</th>
            <th>DBCV</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          {results.models.map((md, i) => (
            <tr key={i}>
              <td style={{ border: '1px solid #777', padding: 5 }}>{md.model}</td>
              <td style={{ border: '1px solid #777', padding: 5 }}>{md.coherence}</td>
              <td style={{ border: '1px solid #777', padding: 5 }}>{md.umass}</td>
              <td style={{ border: '1px solid #777', padding: 5 }}>{md.npmi}</td>
              <td style={{ border: '1px solid #777', padding: 5 }}>{md.uci}</td>
              <td style={{ border: '1px solid #777', padding: 5 }}>{md.topic_diversity}</td>
              <td style={{ border: '1px solid #777', padding: 5 }}>{md.silhouette}</td>
              <td style={{ border: '1px solid #777', padding: 5 }}>{md.dbcv}</td>
              <td style={{ border: '1px solid #777', padding: 5 }}>{md.time_sec}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Best Model</h3>
      <p style={{ color: 'lightgreen' }}>
        {results.best_model.model} with Coherence = {results.best_model.coherence},
        Time = {results.best_model.time_sec}s
      </p>

      <h3>Topics & Top Words</h3>
      <table style={{ borderCollapse: 'collapse', color: 'white' }}>
        <thead>
          <tr>
            <th>Topic</th>
            <th>Word 0</th>
            <th>Word 1</th>
            <th>Word 2</th>
            <th>Word 3</th>
            <th>Word 4</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(results.topics).map(([topic, words], idx) => (
            <tr key={idx}>
              <td style={{ border: '1px solid #777', padding: 5 }}>{topic}</td>
              {words.map((w, i2) => (
                <td key={i2} style={{ border: '1px solid #777', padding: 5 }}>{w}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Pipeline Diagram</h3>
      <div style={{ display: 'flex', alignItems: 'flex-start' }}>
        <div style={{ marginRight: 20, textAlign: 'center', color: 'white' }}>
          <div style={{ border: '2px solid #fff', padding: 10, marginBottom: 5 }}>BERT Embedding</div>
          <div style={{ marginBottom: 5 }}>↓</div>
          <div style={{ border: '2px solid #fff', padding: 10, marginBottom: 5 }}>UMAP</div>
          <div style={{ marginBottom: 5 }}>↓</div>
          <div style={{ border: '2px solid #fff', padding: 10, marginBottom: 5 }}>HDBSCAN</div>
          <div style={{ marginBottom: 5 }}>↓</div>
          <div style={{ border: '2px solid #fff', padding: 10 }}>c-TF-IDF</div>
        </div>
        <p style={{ color: 'white', maxWidth: 400 }}>
          This shows how we embed documents with BERT, reduce with UMAP, cluster with HDBSCAN, and extract class-specific words using c-TF-IDF.
        </p>
      </div>
    </div>
  );
}