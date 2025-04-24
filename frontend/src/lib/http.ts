import axios from 'axios';

export const http = axios.create({
  baseURL: import.meta.env.VITE_API_URL,   // ← env‑specific
  timeout: 10_000,
});

http.interceptors.request.use(cfg => {
  const token = localStorage.getItem('token');
  if (token) cfg.headers!.Authorization = `Bearer ${token}`;
  return cfg;
});
