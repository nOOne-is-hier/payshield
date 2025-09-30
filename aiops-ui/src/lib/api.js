const BASE = '/api'; // vite proxy 사용

export async function fetchMetrics() {
  const r = await fetch(`${BASE}/metrics?cur_win=300`, { cache: 'no-store' });
  if (!r.ok) throw new Error('metrics error');
  return await r.json();
}

export async function fetchFeed(limit = 20) {
  const r = await fetch(`${BASE}/dashboard/summary?limit=${limit}`, { cache: 'no-store' });
  if (!r.ok) throw new Error('feed error');
  return await r.json();
}

export function connectSSE(onData, onError) {
  const es = new EventSource(`${BASE}/dashboard/stream`);
  es.onmessage = (ev) => {
    try {
      const item = JSON.parse(ev.data);
      onData(item);
    } catch (e) {}
  };
  es.addEventListener('ping', () => {});
  es.onerror = (e) => { onError?.(e); };
  return () => es.close();
}
