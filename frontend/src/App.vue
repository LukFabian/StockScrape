<template>
  <v-app>
    <!-- Persistent Header -->
    <v-app-bar app color="primary" dark>
      <v-toolbar-title>My Stock Tracker</v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn text to="/">Home</v-btn>
      <v-btn text to="/Stockslist">Stockslist</v-btn>
    </v-app-bar>
    <router-view />
  </v-app>
</template>

<script lang="ts" setup>
  //
</script>
