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
      <v-col cols="12">
        <v-card v-if="bestChartData !== null && bestStock !== null">
          <v-card-title>📈 Best Performing (24h): ({{ bestStock.symbol }})</v-card-title>
          <v-card-subtitle>Performance (7 days): {{ bestStock.performance.toFixed(2) + '%' }}</v-card-subtitle>
          <v-card-text>
            <LineChart :data="bestChartData" :chart-options="chartOptions" />
            <v-btn color="primary" class="mt-4" @click="goToStockDetails(bestStock.symbol)">
              View Full Details
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Worst Performer -->
      <v-col cols="12">
        <v-card v-if="worstChartData !== null && worstStock !== null">
          <v-card-title>📉 Worst Performing: ({{ worstStock.symbol }})</v-card-title>
          <v-card-subtitle>Performance (7 days): {{ worstStock.performance.toFixed(2) + '%' }}</v-card-subtitle>
          <v-card-text>
            <LineChart :data="worstChartData" :chart-options="chartOptions" />
            <v-btn color="primary" class="mt-4" @click="goToStockDetails(worstStock.symbol)">
              View Full Details
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { Chart as ChartJS, registerables } from 'chart.js';
import { Line } from 'vue-chartjs';
import { stocksApi } from '@/plugins';
import type { StockPerformanceRead } from "@/generated";

ChartJS.register(...registerables);
const LineChart = Line;

const router = useRouter();

const stockCount = ref(0);
const bestStock = ref<StockPerformanceRead | null>(null);
const worstStock = ref<StockPerformanceRead | null>(null);

const bestChartData = computed(() =>
  bestStock.value
    ? {
      labels: bestStock.value.charts?.map(p => p.date.substring(0, 10)),
      datasets: [{ label: 'Price in $', data: bestStock.value.charts?.map(p => p.close / 100), borderColor: 'green', backgroundColor: 'lightgreen' }]
    }
    : null
);

const worstChartData = computed(() =>
  worstStock.value
    ? {
      labels: worstStock.value.charts?.map(p => p.date.substring(0, 10)),
      datasets: [{ label: 'Price in $', data: worstStock.value.charts?.map(p => p.close / 100), borderColor: 'red', backgroundColor: 'pink' }]
    }
    : null
);

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: { y: { beginAtZero: false } },
  plugins: { legend: { display: false } },
};

const goToStockDetails = (symbol: string) => {
  router.push(`/StockDetails/${symbol}`);
};

onMounted(async () => {
  const [best, worst, stocksCount] = await Promise.all([
    stocksApi.stocksGetStocks('best'),
    stocksApi.stocksGetStocks('worst'),
    stocksApi.stocksGetStocksCount()
  ]);
  bestStock.value = best.data;
  worstStock.value = worst.data;
  stockCount.value = stocksCount.data;
});
</script>
