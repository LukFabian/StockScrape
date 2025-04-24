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
        <v-card v-if="bestChartData !== null">
          <v-card-title>📈 Best Performing (24h): {{ bestStock.name }} ({{ bestStock.ticker }})</v-card-title>
          <v-card-text>
            <LineChart :data="bestChartData" :chart-options="chartOptions"/>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Worst Performer -->
      <v-col cols="12" md="6">
        <v-card v-if="worstChartData !== null">
          <v-card-title>📉 Worst Performing (24h): {{ worstStock.name }} ({{ worstStock.ticker }})</v-card-title>
          <v-card-text>
            <LineChart :data="worstChartData" :chart-options="chartOptions"/>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup lang="ts">
import {ref, computed} from 'vue'
import {Chart as ChartJS, registerables} from 'chart.js'
import {Line} from 'vue-chartjs'
import {stocksApi} from "@/plugins";

ChartJS.register(...registerables)

// Register LineChart as a wrapper component
const LineChart = Line

// Data: Counter
const stockCount = ref(1245)

let bestStock: Stock | null = null;
let worstStock: Stock | null = null;
let bestChartData = null;
let worstChartData = null;

onMounted(() => {
  stocksApi.stocksGetStocks("best").then((stock) => {
    bestStock = stock;
    bestChartData = computed(() => ({
      labels: bestStock.value.data.map((point) => point.time),
      datasets: [
        {
          label: 'Price in €',
          data: bestStock.value.data.map((point) => point.price),
        },
      ],
      chartOptions: {
        responsive: true
      }
    }))
  });
  stocksApi.stocksGetStocks("worst").then((stock) => {
    worstStock = stock;
    worstChartData = computed(() => ({
      labels: worstStock.value.data.map((point) => point.time),
      datasets: [
        {
          label: 'Price in €',
          data: worstStock.value.data.map((point) => point.price),
        },
      ],
      chartOptions: {
        responsive: true
      }
    }))
  });
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    y: {
      beginAtZero: false,
    },
  },
  plugins: {
    legend: {
      display: false,
    },
  },
}
</script>

<style scoped>
.v-card-text {
  height: 300px;
}
</style>
