/**
 * plugins/index.ts
 *
 * Automatically included in `./src/main.ts`
 */

// Plugins
import pinia from '../stores'
import router from '../router'

// Vuetify styles
import 'vuetify/styles'
import '@mdi/font/css/materialdesignicons.css'
// Types
import type { App } from 'vue'
import {createVuetify} from "vuetify";
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import { Configuration } from '@/generated/configuration.ts';
import { StocksApi } from '@/generated/api.ts';
import { http } from '@/lib/http';


const vuetify = createVuetify({
  theme: {
    defaultTheme: 'dark',
  },
  components,
  directives,
})

export function registerPlugins (app: App) {
  app
    .use(vuetify)
    .use(router)
    .use(pinia)
}

const config = new Configuration({
  basePath: '',                 // leave empty – we hand the baseURL via axios
});

export const stocksApi = new StocksApi(config, undefined, http);
