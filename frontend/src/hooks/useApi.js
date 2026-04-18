import axios from 'axios'

const API_KEY = localStorage.getItem('gr_api_key') || ''

export const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'X-API-Key': API_KEY },
})

export function setApiKey(key) {
  localStorage.setItem('gr_api_key', key)
  api.defaults.headers['X-API-Key'] = key
}
