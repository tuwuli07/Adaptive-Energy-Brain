import test from 'node:test';
import assert from 'node:assert/strict';
import { sourcePolicy, validateUploadFile } from './dataPolicy.js';

test('uploaded sources use their own analysis and reference pages stay separate', () => {
  for (const route of ['/', '/energy', '/battery', '/explorer']) assert.equal(sourcePolicy('upload-id', route), 'analysis');
  for (const route of ['/forecasting', '/adaptive', '/assistant']) assert.equal(sourcePolicy('upload-id', route), 'unsupported');
  for (const route of ['/research', '/comparison', '/risk']) assert.equal(sourcePolicy('upload-id', route), 'standard');
  assert.equal(sourcePolicy('none', '/'), 'empty');
  assert.equal(sourcePolicy('campus', '/forecasting'), 'empty');
  assert.equal(sourcePolicy('upload-id', '/data'), 'standard');
});
test('file selection enforces real CSV and size restrictions', () => {
  assert.equal(validateUploadFile({ name: 'power.CSV', size: 50 }), null);
  assert.match(validateUploadFile({ name: 'power.xlsx', size: 50 }), /CSV/);
  assert.match(validateUploadFile({ name: 'power.csv', size: 0 }), /empty/);
  assert.match(validateUploadFile({ name: 'power.csv', size: 10485761 }), /10 MiB/);
});
