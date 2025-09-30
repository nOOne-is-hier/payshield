<template>
  <div class="max-w-6xl mx-auto p-6 space-y-6">
    <header class="flex items-center justify-between">
      <h1 class="text-2xl font-bold">Fraud Mini AIOps — Live</h1>
      <div class="flex items-center gap-2">
        <span class="text-sm text-slate-500">Auto-refresh: SSE {{ sseOk ? 'ON' : 'OFF' }} / fallback {{ intervalSec
        }}s</span>
        <button @click="togglePolling"
          class="px-3 py-1.5 rounded-lg bg-slate-900 text-white text-sm hover:bg-slate-800">
          {{ polling ? '폴링 정지' : '폴링 재개' }}
        </button>
      </div>
    </header>

    <!-- KPIs -->
    <section class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <KpiCard label="Threshold" :sub="`Window: ${metrics.window_size ?? 0}`">
        {{ fmt(metrics.threshold) }}
      </KpiCard>
      <KpiCard label="Anomaly Rate" :sub="`Model: ${metrics.model_version || '-'}`">
        {{ ((metrics.anomaly_rate || 0) * 100).toFixed(2) }}%
      </KpiCard>
      <KpiCard label="Latency P95" :sub="`Updated: ${metrics.updated_at || '-'}`">
        {{ Math.round(metrics.latency_p95_ms || 0) }} ms
      </KpiCard>
      <div class="card">
        <div class="kpi">Health</div>
        <div class="mt-2">
          <span :class="chipClass(psiLevel.amount)">amount: {{ psi.amount }}</span>
          <span class="mx-1"></span>
          <span :class="chipClass(psiLevel.score)">score: {{ psi.score }}</span>
        </div>
      </div>
    </section>

    <!-- PSI -->
    <section class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <PsiMeter title="Amount PSI" :psi="Number(psi.amount) || 0" />
      <PsiMeter title="Score PSI" :psi="Number(psi.score) || 0" />
    </section>

    <!-- Score facts -->
    <section class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <KpiCard label="score.p95" :sub="`Δp95: ${(metrics.score_stats?.delta_p95 ?? 0).toFixed(3)}`">
        {{ (metrics.score_stats?.p95 ?? 0).toFixed(3) }}
      </KpiCard>
      <KpiCard label="near_rate" :sub="`tail_rate: ${(metrics.score_stats?.tail_rate ?? 0).toFixed(3)}`">
        {{ (metrics.score_stats?.near_rate ?? 0).toFixed(3) }}
      </KpiCard>
      <div class="card">
        <div class="kpi">Near Band (~thr±0.02)</div>
        <div class="mt-2 meter">
          <div :style="{ width: nearPct + '%' }" class="bg-sky-400"></div>
        </div>
        <div class="text-xs text-slate-500 mt-1">{{ nearPct.toFixed(1) }}%</div>
      </div>
    </section>

    <!-- Summaries -->
    <section class="card">
      <div class="flex items-center justify-between mb-2">
        <div class="font-semibold">Agent Summaries</div>
        <button @click="refreshOnce" class="text-sm px-2 py-1 rounded bg-slate-100 hover:bg-slate-200">수동 새로고침</button>
      </div>
      <div v-if="feed.length === 0" class="text-slate-500 text-sm">아직 요약이 없습니다…</div>
      <ul v-else class="space-y-3">
        <SummaryItem v-for="(it, idx) in feed" :key="idx" :item="it" />
      </ul>
    </section>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import KpiCard from './components/KpiCard.vue'
import PsiMeter from './components/PsiMeter.vue'
import SummaryItem from './components/SummaryItem.vue'
import { fetchMetrics, fetchFeed, connectSSE } from './lib/api'

const metrics = ref({})
const feed = ref([])
const intervalSec = 2
let timer = null
const polling = ref(false)   // 기본은 SSE 우선, 폴백시 true

const psi = computed(() => {
  const drift = metrics.value?.drift || []
  const pick = (f) => {
    const d = drift.find(x => x.feature === f)
    return d ? Number(d.psi).toFixed(2) : '0.00'
  }
  return { amount: pick('amount'), score: pick('score') }
})
const psiLevel = computed(() => {
  const level = (v) => {
    const n = Number(v)
    if (n > 0.4) return 'high'
    if (n > 0.25) return 'warn'
    return 'normal'
  }
  return { amount: level(psi.value.amount), score: level(psi.value.score) }
})
const nearPct = computed(() => Math.min(100, Math.max(0, (metrics.value?.score_stats?.near_rate || 0) * 100)))

function chipClass(level) {
  return ['chip', level === 'high' ? 'chip-red' : (level === 'warn' ? 'chip-yellow' : 'chip-green')]
}
function fmt(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return v.toFixed(2);
  if (v == null) return '-';
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(2) : String(v);
}

async function pull() {
  const [m, f] = await Promise.all([fetchMetrics(), fetchFeed(20)])
  metrics.value = m
  feed.value = f.items || []
}

async function refreshOnce() {
  try { await pull() } catch (e) { console.error(e) }
}

function startPolling() {
  if (timer) return
  polling.value = true
  timer = setInterval(refreshOnce, intervalSec * 1000)
}
function stopPolling() {
  polling.value = false
  if (timer) clearInterval(timer)
  timer = null
}
function togglePolling() {
  polling.value ? stopPolling() : startPolling()
}

const sseOk = ref(false)
let closeSSE = null

onMounted(async () => {
  await refreshOnce()
  // SSE 우선 연결
  try {
    closeSSE = connectSSE((item) => {
      // 최신 1건 상단 삽입
      feed.value = [item, ...feed.value].slice(0, 20)
      // SSE 수신 시 metrics도 가끔 당겨오면 화면이 싱크
      if (!polling.value) { fetchMetrics().then(j => metrics.value = j).catch(() => { }) }
      sseOk.value = true
    }, () => {
      // 에러나면 폴링으로 폴백
      if (!polling.value) startPolling()
    })
  } catch {
    startPolling()
  }
})

onBeforeUnmount(() => {
  stopPolling()
  if (closeSSE) closeSSE()
})
</script>
