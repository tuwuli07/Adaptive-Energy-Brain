import test from 'node:test';
import assert from 'node:assert/strict';
import { integratePower } from './energyAnalysis.js';

test('energy follows irregular timestamps, never row counts or assumed final hours', () => {
  assert.deepEqual(integratePower([
    { timestamp: '2026-01-01T00:00:00Z', power: 10 },
    { timestamp: '2026-01-01T00:30:00Z', power: 20 },
    { timestamp: '2026-01-01T02:00:00Z', power: 100 },
  ], 'power'), { energy: 35, hours: 2 });
});
test('missing measurements remain unavailable rather than fabricated zero', () => {
  assert.deepEqual(integratePower([{ timestamp: '2026-01-01T00:00:00Z' }, { timestamp: '2026-01-01T01:00:00Z' }], 'power'), { energy: null, hours: 0 });
  assert.deepEqual(integratePower([], 'power'), { energy: null, hours: 0 });
});
