<template>
  <v-container class="py-8">
    <v-card>
      <v-card-title>Details for {{ symbol }}</v-card-title>
      <v-card-text>
        <h3>Performance</h3>
        <LineChart v-if="priceChartData" :data="priceChartData" :chart-options="chartOptions" />

        <h3 class="mt-6">ADX & DMI</h3>
        <LineChart v-if="adxDmiChartData" :data="adxDmiChartData" :chart-options="chartOptions" />
      </v-card-text>
    </v-card>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useRoute } from 'vue-router';
import { Chart as ChartJS, registerables } from 'chart.js';
import { Line } from 'vue-chartjs';
import { stockApi } from '@/plugins';
import type {StockRead} from "@/generated";

definePage({
  name: '/StockDetails/:symbol'
})

ChartJS.register(...registerables);
const LineChart = Line;

const route = useRoute();
const symbol = route.params.symbol as string;

const stockData = ref<StockRead | null>(null);

const priceChartData = computed(() =>
  stockData.value
    ? {
      labels: stockData.value.charts?.map(p => p.date.substring(0, 10)),
      datasets: [{
        label: 'Price in $',
        data: stockData.value.charts?.map(p => p.close / 100),
        borderColor: 'blue',
        backgroundColor: 'lightblue'
      }]
    }
    : null
);

const adxDmiChartData = computed(() =>
  stockData.value
    ? {
      labels: stockData.value.charts?.map(p => p.date.substring(0, 10)),
      datasets: [
        { label: 'ADX 14', data: stockData.value.charts?.map(p => p.adx_14), borderColor: 'purple' },
        { label: '+DMI 14', data: stockData.value.charts?.map(p => p.dmi_positive_14), borderColor: 'green' },
        { label: '-DMI 14', data: stockData.value.charts?.map(p => p.dmi_negative_14), borderColor: 'red' }
      ]
    }
    : null
);

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: { y: { beginAtZero: false } },
};

onMounted(async () => {
  const { data } = await stockApi.stockGetStock(symbol, true);
  stockData.value = data;
});
</script>
