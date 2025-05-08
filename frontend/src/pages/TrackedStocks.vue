<template>
  <v-container class="py-8">
    <!-- Timespan Selector -->
    <v-select
      v-model="selectedTimeSpan"
      :items="timeSpans"
      item-title="label"
      item-value="value"
      label="Select Timespan"
      class="mb-6"
    />

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
          <v-card-title>
            📈 Best Performing ({{ selectedTimeSpanLabel }}): ({{ bestStock.symbol }})
          </v-card-title>
          <v-card-subtitle>
            Performance ({{ selectedTimeSpanLabel }}): {{ bestStock.performance.toFixed(2) + '%' }}
          </v-card-subtitle>
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
          <v-card-title>
            📉 Worst Performing ({{ selectedTimeSpanLabel }}): ({{ worstStock.symbol }})
          </v-card-title>
          <v-card-subtitle>
            Performance ({{ selectedTimeSpanLabel }}): {{ worstStock.performance.toFixed(2) + '%' }}
          </v-card-subtitle>
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
import { ref, computed, onMounted, watch } from 'vue';
import { useRouter } from 'vue-router';
import { Chart as ChartJS, registerables } from 'chart.js';
import { Line } from 'vue-chartjs';
import { stocksApi } from '@/plugins';
import type { StockPerformanceRead } from "@/generated";

ChartJS.register(...registerables);
const LineChart = Line;

const router = useRouter();

const timeSpans = [
  { label: '1 Day', value: '1d' },
  { label: '3 Days', value: '3d' },
  { label: '7 Days', value: '7d' },
  { label: '2 Weeks', value: '2w' },
  { label: '1 Month', value: '1m' },
  { label: '6 Months', value: '6m' },
  { label: '1 Year', value: '1y' },
  { label: '5 Years', value: '5y' },
  { label: 'Max', value: 'max' },
];

const selectedTimeSpan = ref('1y');
const stockCount = ref(0);
const bestStock = ref<StockPerformanceRead | null>(null);
const worstStock = ref<StockPerformanceRead | null>(null);

const selectedTimeSpanLabel = computed(() => {
  return timeSpans.find(t => t.value === selectedTimeSpan.value)?.label || '';
});

const getStartDate = (span: string) => {
  const now = new Date();
  const date = new Date(now);
  switch (span) {
    case '1d': date.setDate(now.getDate() - 1); break;
    case '3d': date.setDate(now.getDate() - 3); break;
    case '7d': date.setDate(now.getDate() - 7); break;
    case '2w': date.setDate(now.getDate() - 14); break;
    case '1m': date.setMonth(now.getMonth() - 1); break;
    case '6m': date.setMonth(now.getMonth() - 6); break;
    case '1y': date.setFullYear(now.getFullYear() - 1); break;
    case '5y': date.setFullYear(now.getFullYear() - 5); break;
    case 'max': return null;
    default: return null;
  }
  return date.toISOString();
};

const bestChartData = computed(() =>
  bestStock.value
    ? {
      labels: bestStock.value.charts
        ?.filter(p => filterByTimespan(p.date))
        .map(p => p.date.substring(0, 10)),
      datasets: [
        {
          label: 'Price in $',
          data: bestStock.value.charts
            ?.filter(p => filterByTimespan(p.date))
            .map(p => p.close / 100),
          borderColor: 'green',
          backgroundColor: 'lightgreen',
        },
      ],
    }
    : null
);

const worstChartData = computed(() =>
  worstStock.value
    ? {
      labels: worstStock.value.charts
        ?.filter(p => filterByTimespan(p.date))
        .map(p => p.date.substring(0, 10)),
      datasets: [
        {
          label: 'Price in $',
          data: worstStock.value.charts
            ?.filter(p => filterByTimespan(p.date))
            .map(p => p.close / 100),
          borderColor: 'red',
          backgroundColor: 'pink',
        },
      ],
    }
    : null
);

const filterByTimespan = (dateStr: string) => {
  const date = new Date(dateStr);
  const start = getStartDate(selectedTimeSpan.value);
  if (!start) return true; // max timeframe
  return new Date(start) <= date;
};

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: { y: { beginAtZero: false } },
  plugins: { legend: { display: false } },
};

const goToStockDetails = (symbol: string) => {
  router.push(`/StockDetails/${symbol}`);
};

const fetchData = async () => {
  const startDate = getStartDate(selectedTimeSpan.value);
  const [best, worst, stocksCount] = await Promise.all([
    stocksApi.stocksGetStocks('best', startDate),
    stocksApi.stocksGetStocks('worst', startDate),
    stocksApi.stocksGetStocksCount(),
  ]);
  bestStock.value = best.data;
  worstStock.value = worst.data;
  stockCount.value = stocksCount.data;
};

onMounted(() => {
  fetchData();
});

watch(selectedTimeSpan, () => {
  fetchData();
});
</script>
