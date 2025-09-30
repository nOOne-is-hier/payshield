<template>
    <div class="card">
        <div class="kpi">{{ label }}</div>
        <div class="kpi-val">
            <slot />
        </div>
        <div v-if="sub" class="kpi mt-1">{{ sub }}</div>
    </div>
</template>

<script setup>
defineProps({ label: String, sub: String })
</script>
