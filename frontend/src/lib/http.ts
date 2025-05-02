import axios from 'axios';

export const http = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 10_000,
});

http.interceptors.request.use(cfg => {
  const token = localStorage.getItem('token');
  if (token) cfg.headers!.Authorization = `Bearer ${token}`;
  return cfg;
});
