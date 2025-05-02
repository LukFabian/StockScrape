<template>
  <v-container class="py-8">
    <!-- Stock Counter -->
    <v-card class="mb-6" color="primary" dark>
      <v-card-title>Tracked Stocks</v-card-title>
      <v-card-text class="text-h3 text-center">
        {{ stockCount }}
      </v-card-text>
    </v-card>

    <v-row dense>
      <!-- Best Performer -->
      <v-col cols="12" md="6">
        <v-card v-if="bestChartData !== null && bestStock !== null">
          <v-card-title>📈 Best Performing (24h): {{ bestStock.name }} ({{ bestStock.symbol }})</v-card-title>
          <v-card-text>
            <LineChart :data="bestChartData" :chart-options="chartOptions"/>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Worst Performer -->
      <v-col cols="12" md="6">
        <v-card v-if="worstChartData !== null && worstStock !== null">
          <v-card-title>📉 Worst Performing (24h): {{ worstStock.name }} ({{ worstStock.symbol }})</v-card-title>
          <v-card-text>
            <LineChart :data="worstChartData" :chart-options="chartOptions"/>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { Chart as ChartJS, registerables } from 'chart.js';
import { Line } from 'vue-chartjs';
import {chartsApi, stocksApi} from '@/plugins';
import type {StockRead, ChartRead} from "@/generated";
import {DAY} from "@/utils.ts";

ChartJS.register(...registerables);
const LineChart = Line;

/* ------------ reactive state ------------ */
const stockCount   = ref(0);

const bestStock    = ref<StockRead | null>(null);
const worstStock   = ref<StockRead | null>(null);

const bestChart = ref<ChartRead[] | null>(null);
const worstChart = ref<ChartRead[] | null>(null);

/* graphs build themselves whenever the API data changes */
const bestChartData  = computed(() =>
  bestChart.value
    ? {
      labels: bestChart.value.map(p => p.date.substring(0, 10)), // extract date part of iso string
      datasets: [{ label: 'Price in $', data: bestChart.value.map(p => p.open / 100) }]
    }
    : null
);

const worstChartData = computed(() =>
  worstChart.value
    ? {
      labels: worstChart.value.map(p => p.date.substring(0, 10)), // extract date part of iso string
      datasets: [{ label: 'Price in $', data: worstChart.value.map(p => p.open / 100) }]
    }
    : null
);

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: { y: { beginAtZero: false } },
  plugins: { legend: { display: false } },
};

/* ------------ load data once on mount ------------ */
onMounted(async () => {
  /* parallel fetch */
  const [best, worst] = await Promise.all([
    stocksApi.stocksGetStocks('best'),
    stocksApi.stocksGetStocks('worst')
  ]);
  bestStock.value  = best.data;
  worstStock.value = worst.data;
  const now      = new Date();
  const weekAgo = new Date(now.getTime() - (DAY * 7));
  const [localChartBest, localChartWorst, stocksCount] = await Promise.all([
    chartsApi.chartsGetStockWithCharts(bestStock.value.symbol),
    chartsApi.chartsGetStockWithCharts(worstStock.value.symbol),
    stocksApi.stocksGetStocksCount()
  ]);
  bestChart.value = localChartBest.data.charts;
  worstChart.value = localChartWorst.data.charts;
  stockCount.value = stocksCount.data
});
</script>
