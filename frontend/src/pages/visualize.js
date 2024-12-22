// File: pages/visualize.js
import React, { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/router';
import * as d3 from 'd3';

export default function Visualize() {
  const [gData, setGData] = useState(null);
  const [msg, setMsg] = useState('');
  const sRef = useRef(null);
  const rt = useRouter();

  useEffect(() => {
    async function fetchData() {
      setMsg('Loading graph data...');
      try {
        const r = await fetch('http://127.0.0.1:5000');
        const d = await r.json();
        if (!r.ok) {
          setMsg('Error: ' + (d.error || 'unknown'));
        } else {
          setGData(d.graph);
          setMsg('');
        }
      } catch (err) {
        setMsg('Error: ' + err.message);
      }
    }
    fetchData();
  }, []);

  useEffect(() => {
    if (!gData) return;

    const svg = d3.select(sRef.current);
    const width = 800, height = 600;
    svg.attr('width', width).attr('height', height).style('background', '#222');
    svg.selectAll('*').remove();

    const simulation = d3.forceSimulation(gData.nodes)
      .force('link', d3.forceLink(gData.links).id(d => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-30))
      .force('center', d3.forceCenter(width/2, height/2));

    const link = svg.append('g')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(gData.links)
      .enter().append('line')
      .attr('stroke-width', d => Math.sqrt(d.value));

    const node = svg.append('g')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .selectAll('circle')
      .data(gData.nodes)
      .enter().append('circle')
      .attr('r', 5)
      .attr('fill', '#69b3a2')
      .call(d3.drag()
        .on('start', dragStart)
        .on('drag', dragged)
        .on('end', dragEnd)
      );

    const label = svg.append('g')
      .selectAll('text')
      .data(gData.nodes)
      .enter().append('text')
      .text(d => d.id)
      .attr('fill', '#ccc')
      .attr('font-size', 10)
      .attr('dx', 8)
      .attr('dy', '.35em');

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);

      label
        .attr('x', d => d.x)
        .attr('y', d => d.y);
    });

    function dragStart(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    }
    function dragged(event, d) {
      d.fx = event.x; d.fy = event.y;
    }
    function dragEnd(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    }

  }, [gData]);

  return (
    <div style={{ backgroundColor: '#000', color: '#fff', minHeight: '100vh', padding: 20, fontFamily: 'Arial' }}>
      <h1>Co-occurrence Network</h1>
      <p style={{ color: 'yellow' }}>{msg}</p>
      <svg ref={sRef}></svg>
      <br/>
      <button onClick={() => rt.push('/preprocess')}>Back to Preprocessed</button>{' '}
      <button onClick={() => rt.push('/')}>Home</button>
    </div>
  );
}
