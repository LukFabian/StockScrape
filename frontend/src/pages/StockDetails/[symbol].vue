<template>
  <v-container class="py-8">
    <v-card>
      <v-card-title>Details for {{ symbol }}</v-card-title>
      <v-card-text>
        <v-select
          v-model="selectedTimespan"
          :items="timespanOptions"
          label="Select Timespan"
          outlined
          dense
          class="mb-6"
        />

        <v-select
          v-model="selectedAnalysis"
          :items="analysisOptions"
          @update:modelValue="onAnalysisChange"
          label="Select Analysis Method"
          outlined
          dense
          class="mb-6"
        />

        <!-- SGP LSTM Classification Output -->
        <div v-if="classification" class="mt-6">
          <v-alert
            :type="classification.label === 1 ? 'success' : 'error'"
            border="start"
            elevation="2"
            colored-border
          >
            <div class="text-h6 font-weight-medium">
              Predicted Movement:
              <span v-if="classification.label === 1">📈 Likely Rise</span>
              <span v-else>📉 Likely Fall</span>
            </div>
            <div class="mt-2">
              Confidence: <strong>{{ (classification.probability * 100).toFixed(2) }}%</strong>
            </div>
          </v-alert>
        </div>

        <v-progress-circular
          v-if="loading"
          indeterminate
          color="primary"
          size="40"
          class="my-6 mx-auto d-block"
        />


        <h3>Performance</h3>
        <!-- Display either the priceChart or the priceChart with predictions -->
        <LineChart v-if="priceChartData && !priceChartWithAnalysisData" :data="priceChartData"
                   :chart-options="chartOptions"/>
        <LineChart v-if="priceChartWithAnalysisData" :data="priceChartWithAnalysisData" :chart-options="chartOptions"/>


        <h3 class="mt-6">ADX & DMI</h3>
        <LineChart v-if="adxDmiChartData" :data="adxDmiChartData" :chart-options="chartOptions"/>

        <h3 class="mt-6">RSI</h3>
        <LineChart v-if="rsiChartData" :data="rsiChartData" :chart-options="chartOptions"/>
      </v-card-text>
    </v-card>
  </v-container>
</template>

<script setup lang="ts">
import {ref, computed, onMounted} from 'vue';
import {useRoute} from 'vue-router';
import {Chart as ChartJS, registerables} from 'chart.js';
import {Line} from 'vue-chartjs';
import {analysisApi, stockApi} from '@/plugins';
import type {ClassificationPrediction, StockRead} from "@/generated";

definePage({
  name: '/StockDetails/:symbol'
})

ChartJS.register(...registerables);
const LineChart = Line;

const route = useRoute();
const symbol = route.params.symbol as string;

const stockData = ref<StockRead | null>(null);
const selectedTimespan = ref('1y');

const timespanOptions = [
  {title: '1 Day', value: '1d'},
  {title: '3 Days', value: '3d'},
  {title: '7 Days', value: '7d'},
  {title: '2 Weeks', value: '2w'},
  {title: '1 Month', value: '1m'},
  {title: '6 Months', value: '6m'},
  {title: '1 Year', value: '1y'},
  {title: '5 Years', value: '5y'},
  {title: 'Max', value: 'max'},
];

const selectedAnalysis = ref('None');
const priceChartWithAnalysisData = ref(null);
const analysisOptions = [
  'None',
  'Linear Regression',
  'SGP LSTM'
];
const classification = ref<ClassificationPrediction | null>(null)
const loading = ref<boolean>(false);

// Helper function to calculate start date based on selected timespan
function getStartDate(timespan: string): Date | null {
  const now = new Date();
  const start = new Date(now);
  switch (timespan) {
    case '1d':
      start.setDate(now.getDate() - 1);
      break;
    case '3d':
      start.setDate(now.getDate() - 3);
      break;
    case '7d':
      start.setDate(now.getDate() - 7);
      break;
    case '2w':
      start.setDate(now.getDate() - 14);
      break;
    case '1m':
      start.setMonth(now.getMonth() - 1);
      break;
    case '6m':
      start.setMonth(now.getMonth() - 6);
      break;
    case '1y':
      start.setFullYear(now.getFullYear() - 1);
      break;
    case '5y':
      start.setFullYear(now.getFullYear() - 5);
      break;
    case 'max':
      return null;
    default:
      return null;
  }
  return start;
}

async function onAnalysisChange(): Promise<void> {
  loading.value = true;
  try {
    const baseChart = {
      labels: filteredCharts.value.map(p => p.date.substring(0, 10)),
      datasets: [
        {
          label: 'Price in $',
          data: filteredCharts.value.map(p => p.close / 100),
          borderColor: 'blue',
          backgroundColor: 'lightblue'
        }
      ]
    };

    if (selectedAnalysis.value === 'None') {
      priceChartWithAnalysisData.value = baseChart;
      return;
    }

    if (selectedAnalysis.value === 'Linear Regression') {
      const startDate = getStartDate(selectedTimespan.value);
      const {data} = await analysisApi.analysisPutStockAnalysisRegression(
        symbol,
        selectedAnalysis.value.toUpperCase(),
        startDate?.toISOString()
      );

      if (data.charts) {
        const analysisData = data.charts.map(p => p.close / 100);
        baseChart.datasets.push({
          label: selectedAnalysis.value,
          data: analysisData,
          borderColor: 'red',
          backgroundColor: 'blue'
        });
        baseChart.labels = data.charts.map(p => p.date.substring(0, 10));
      }

      priceChartWithAnalysisData.value = baseChart;
    } else if (selectedAnalysis.value === 'SGP LSTM') {
      const {data} = await analysisApi.analysisPutStockAnalysisClassification(
        symbol,
        selectedAnalysis.value.toUpperCase()
      );
      classification.value = data;
    }
  } catch (error) {
    console.error('Failed to fetch analysis:', error);
  } finally {
    loading.value = false;
  }
}

// Filtered chart data based on selected timespan
const filteredCharts = computed(() => {
  if (!stockData.value || !stockData.value?.charts) return [];
  const startDate = getStartDate(selectedTimespan.value);
  if (!startDate) return stockData.value.charts;
  return stockData.value?.charts.filter(chart => new Date(chart.date) >= startDate);
});

const priceChartData = computed(() =>
  filteredCharts.value?.length
    ? {
      labels: filteredCharts.value.map(p => p.date.substring(0, 10)),
      datasets: [
        {
          label: 'Price in $',
          data: filteredCharts.value.map(p => p.close / 100),
          borderColor: 'blue',
          backgroundColor: 'lightblue'
        }
      ]
    }
    : null
);

const adxDmiChartData = computed(() =>
  filteredCharts.value?.length
    ? {
      labels: filteredCharts.value.map(p => p.date.substring(0, 10)),
      datasets: [
        {label: 'ADX 14', data: filteredCharts.value.map(p => p.adx_14), borderColor: 'purple'},
        {label: '+DMI 14', data: filteredCharts.value.map(p => p.dmi_positive_14), borderColor: 'green'},
        {label: '-DMI 14', data: filteredCharts.value.map(p => p.dmi_negative_14), borderColor: 'red'},
        {label: 'ADX 120', data: filteredCharts.value.map(p => p.adx_120), borderColor: 'blue'},
        {label: '+DMI 120', data: filteredCharts.value.map(p => p.dmi_positive_120), borderColor: 'yellow'},
        {label: '-DMI 120', data: filteredCharts.value.map(p => p.dmi_negative_120), borderColor: 'orange'}
      ].filter((obj) => obj.data.some(p => p !== null))
    }
    : null
);

const rsiChartData = computed(() =>
  filteredCharts.value?.length
    ? {
      labels: filteredCharts.value.map(p => p.date.substring(0, 10)),
      datasets: [
        {label: 'RSI 14', data: filteredCharts.value.map(p => p.rsi_14), borderColor: 'purple'},
        {label: 'RSI 120', data: filteredCharts.value.map(p => p.rsi_120), borderColor: 'green'},
      ].filter((obj) => obj.data.some(p => p !== null))   // checks if any element is not null
    }
    : null
);

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {y: {beginAtZero: false}},
};

onMounted(async () => {
  const {data} = await stockApi.stockGetStock(symbol, true);
  stockData.value = data;
});
</script>
