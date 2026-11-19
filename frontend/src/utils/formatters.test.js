import test from 'node:test';
import assert from 'node:assert/strict';
import { formatCurrency, formatNumber, formatEnergy, formatPercent, formatDate } from './formatters.js';

test('formatCurrency formats USD correctly', () => {
  assert.equal(formatCurrency(36532.73), '$36,532.73');
  assert.equal(formatCurrency(0), '$0.00');
  assert.equal(formatCurrency(null), '$0.00');
  assert.equal(formatCurrency(-15.87), '-$15.87');
});

test('formatNumber formats decimals properly', () => {
  assert.equal(formatNumber(1234.56, 1), '1,234.6');
  assert.equal(formatNumber(45.502, 2), '45.50');
  assert.equal(formatNumber(0, 0), '0');
});

test('formatEnergy converts kWh to MWh above 1000', () => {
  assert.equal(formatEnergy(500), '500.0 kWh');
  assert.equal(formatEnergy(76340.4), '76.3 MWh');
  assert.equal(formatEnergy(364407.7), '364.4 MWh');
});

test('formatPercent prefixes positive sign', () => {
  assert.equal(formatPercent(0.213), '+0.21%');
  assert.equal(formatPercent(-0.043), '-0.04%');
});

test('formatDate formats ISO string with UTC suffix', () => {
  const formatted = formatDate('2026-11-28T03:00:00Z');
  assert.ok(formatted.includes('Nov 28'));
  assert.ok(formatted.includes('03:00'));
  assert.ok(formatted.includes('UTC'));
});
