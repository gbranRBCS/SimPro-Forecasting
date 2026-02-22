/// <reference types="vite/client" />
import axios from "axios";

function getBaseUrl() {
  const envUrl = import.meta.env.VITE_API_URL;
  return envUrl;
}

const api = axios.create({
  baseURL: getBaseUrl(),
});

// Token is read from local storage for authenticated requests
function getStoredToken() {
  return localStorage.getItem("token");
}

// request config gets the bearer token only when one exists
api.interceptors.request.use((config) => {
  const token = getStoredToken();

  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  return config;
});

export default api;