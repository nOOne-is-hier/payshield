<template>
    <div class="card">
        <div class="flex items-center justify-between">
            <div class="font-medium">{{ title }}</div>
            <div :class="badgeClass">{{ level.toUpperCase() }}</div>
        </div>
        <div class="mt-3 meter">
            <div :style="{ width: widthPct + '%' }" :class="barClass"></div>
        </div>
        <div class="mt-2 text-sm text-slate-500">PSI: {{ psi.toFixed(2) }}</div>
    </div>
</template>

<script setup>
import { computed } from 'vue'
const props = defineProps({ title: String, psi: { type: Number, default: 0 } })

const level = computed(() => {
    const v = props.psi ?? 0
    if (v > 0.4) return 'high'
    if (v > 0.25) return 'warn'
    return 'normal'
})
const widthPct = computed(() => Math.max(0, Math.min(100, (props.psi ?? 0) * 100)))
const badgeClass = computed(() => ['chip', level.value === 'high' ? 'chip-red' : (level.value === 'warn' ? 'chip-yellow' : 'chip-green')])
const barClass = computed(() => {
    const v = props.psi ?? 0
    if (v > 0.4) return 'bg-red-500'
    if (v > 0.25) return 'bg-yellow-500'
    return 'bg-green-500'
})
</script>
