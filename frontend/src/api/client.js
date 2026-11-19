/**
 * API client with automatic fallback for standalone operation.
 */

import {
  FALLBACK_OVERVIEW,
  FALLBACK_CONTROLLERS_COMPARISON,
  FALLBACK_FORECAST_METRICS,
  FALLBACK_RESEARCH_RESULTS,
  FALLBACK_RISK_RESULTS,
  FALLBACK_ADAPTIVE_RESULTS,
  generateSampleAdaptiveTrajectory,
  generateSampleTimeseries
} from './fallbackData';

const BASE_URL = '/api';

async function fetchJson(endpoint, fallback) {
  try {
    const res = await fetch(`${BASE_URL}${endpoint}`);
    if (!res.ok) {
      console.warn(`API ${endpoint} returned ${res.status}; using fallback data.`);
      return fallback && !Array.isArray(fallback) ? { ...fallback, _data_source: 'saved_or_illustrative' } : fallback;
    }
    const result = await res.json();
    return result && !Array.isArray(result) ? { ...result, _data_source: 'research_api' } : result;
  } catch (err) {
    console.info(`API ${endpoint} unavailable (${err.message}); using fallback data.`);
    return fallback && !Array.isArray(fallback) ? { ...fallback, _data_source: 'saved_or_illustrative' } : fallback;
  }
}

export const api = {
  async getHealth() {
    return fetchJson('/health', { status: 'offline', mode: 'standalone_fallback' });
  },

  async getOverview() {
    return fetchJson('/overview', FALLBACK_OVERVIEW);
  },

  async getTimeseries({ controllerId = 'mpc_gnn_transformer', partition = 'test', downsample = 1, limit = 1313 } = {}) {
    const query = new URLSearchParams({
      controller_id: controllerId,
      partition,
      downsample: String(downsample),
      limit: String(limit)
    });
    const fallback = {
      controller_id: controllerId,
      partition,
      total_points: 72,
      original_length: 1313,
      downsample_factor: downsample,
      data: generateSampleTimeseries(72),
    };
    return fetchJson(`/energy/timeseries?${query.toString()}`, fallback);
  },

  async getForecastMetrics() {
    return fetchJson('/forecast/metrics', FALLBACK_FORECAST_METRICS);
  },

  async getBatteryDispatch({ controllerId = 'mpc_gnn_transformer', partition = 'test' } = {}) {
    const query = new URLSearchParams({ controller_id: controllerId, partition });
    const fallback = {
      controller_id: controllerId,
      specs: {
        nominal_capacity_kwh: 500.0,
        max_power_kw: 150.0,
        charging_efficiency: 0.95,
        discharging_efficiency: 0.95,
        round_trip_efficiency: 0.9025,
        min_soc: 0.10,
        max_soc: 0.90,
        degradation_rate_usd_per_kwh: 0.02,
      },
      metrics: {
        initial_soc: 0.50,
        terminal_soc: 0.67,
        throughput_kwh_ac: 31540.8,
        full_cycle_equivalents: 31.54,
        degradation_cost_usd: 630.82,
        charge_energy_kwh_ac: 16621.1,
        discharge_energy_kwh_ac: 14919.8,
        terminal_inventory_credit_usd: 12.53,
      },
      soc_trajectory_sample: generateSampleTimeseries(48).map((p, idx) => ({
        step: idx * 4,
        timestamp: p.timestamp,
        soc: p.battery_soc,
        power_kw: p.battery_power_kw,
      })),
    };
    return fetchJson(`/battery/dispatch?${query.toString()}`, fallback);
  },

  async getControllers() {
    const comparison = await this.getControllersComparison();
    return {
      controllers: comparison.controllers.map(c => ({
        id: c.controller_id,
        display_name: c.display_name,
        category: c.category,
        causality: c.causality,
        is_deployable: c.is_deployable,
      }))
    };
  },

  async getControllersComparison(partition = 'test') {
    return fetchJson(`/controllers/comparison?partition=${partition}`, FALLBACK_CONTROLLERS_COMPARISON);
  },

  async getResearchResults() {
    return fetchJson('/research/results', FALLBACK_RESEARCH_RESULTS);
  },

  async getRiskResults() {
    return fetchJson('/risk/results', FALLBACK_RISK_RESULTS);
  },

  async getAdaptiveResults() {
    return fetchJson('/adaptive/results', FALLBACK_ADAPTIVE_RESULTS);
  },

  async getAdaptiveTrajectory() {
    return fetchJson('/adaptive/trajectory', generateSampleAdaptiveTrajectory(71));
  },

  async getAssistantStatus() {
    const response = await fetch(`${BASE_URL}/assistant/status`);
    if (!response.ok) throw new Error('The assistant service is unavailable.');
    return response.json();
  },

  async sendAssistantMessage(message, history = [], options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120000);
    try {
      const { accessKey, ...requestOptions } = options;
      const headers = { 'Content-Type': 'application/json' };
      if (accessKey) headers.Authorization = `Bearer ${accessKey}`;
      const response = await fetch(`${BASE_URL}/assistant/chat`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ message, history, ...requestOptions }),
        signal: controller.signal,
      });
      const result = await response.json();
      if (!response.ok || result.status === 'error' || result.status === 'invalid_input') throw new Error(result.error || result.answer || 'The selected assistant could not answer.');
      return result;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Gemini took too long to answer. Try again with a shorter question.');
      throw error;
    } finally { clearTimeout(timeout); }
  },

  getExportUrl(controllerId, partition = 'test', format = 'csv') {
    return `${BASE_URL}/simulations/export?controller_id=${controllerId}&partition=${partition}&format=${format}`;
  }
};
