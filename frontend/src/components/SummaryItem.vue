<template>
    <li class="border border-slate-100 rounded-xl p-4 bg-slate-50">
        <div class="text-xs text-slate-400 mb-2">{{ tsText }}</div>
        <div class="whitespace-pre-wrap font-mono text-sm">{{ item.text }}</div>
        <div v-if="item.payload?.score_stats" class="mt-2 text-xs text-slate-500">
            score.p95={{ item.payload.score_stats.p95 }} /
            near={{ item.payload.score_stats.near_rate }} /
            tail={{ item.payload.score_stats.tail_rate }}
        </div>
    </li>
</template>

<script setup>
import { computed } from 'vue'
const props = defineProps({ item: Object })
const tsText = computed(() => props.item?.ts ? new Date(props.item.ts * 1000).toLocaleString() : '')
</script>
