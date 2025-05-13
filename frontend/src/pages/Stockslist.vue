<template>
  <v-card>
    <v-card-title>Tracked Stocks</v-card-title>
    <v-divider></v-divider>

    <v-list v-if="stocks.length > 0">
      <v-list-item
        v-for="stock in stocks"
        :key="stock.symbol"
        :to="`/StockDetails/${stock.symbol}`"
        link
      >
        <v-list-item-title>{{ stock.symbol }}</v-list-item-title>
        <v-list-item-subtitle v-if="stock.charts !== undefined">
          Performance (14 days): {{ stock.performance.toFixed(2) }}% | Last Close: ${{ (stock.charts[stock.charts.length - 1].close / 100).toFixed(2) }}
        </v-list-item-subtitle>
      </v-list-item>
    </v-list>
  </v-card>
</template>

<script setup lang="ts">
import {stocksApi} from '@/plugins';
import {onMounted, ref} from "vue";
import type {StockPerformanceRead} from "@/generated/index.js";

const stocks = ref<StockPerformanceRead[]>([]);

onMounted(async () => {
  try {
    const res = await stocksApi.stocksGetAllStocks();
    stocks.value = res.data
    console.log(stocks.value)
  } catch (error) {
    console.error('Failed to fetch stocks:', error);
  }
});
</script>
