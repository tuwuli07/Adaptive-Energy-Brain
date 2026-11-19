import test from 'node:test';
import assert from 'node:assert/strict';
import { datasetRequest, uploadDataset } from './datasets.js';

test('dataset switching requests the chosen dataset, without a campus fallback', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async path => { assert.equal(path, '/api/datasets/my-id/analysis'); return { ok: true, json: async () => ({ records: [{ demand_kw: 23 }] }) }; };
    assert.equal((await datasetRequest('/my-id/analysis')).records[0].demand_kw, 23);
    globalThis.fetch = async () => { throw new Error('offline'); };
    await assert.rejects(datasetRequest('/my-id/analysis'), /Cannot reach/);
    globalThis.fetch = async () => ({ ok: false, json: async () => ({ error: 'Duplicate timestamps', validation: { duplicate_timestamps: 2 } }) });
    await assert.rejects(datasetRequest('/my-id/mapping'), e => e.validation.duplicate_timestamps === 2);
    globalThis.fetch = async () => ({ json: async () => { throw new Error('HTML response'); } });
    await assert.rejects(datasetRequest(), /unavailable/);
  } finally { globalThis.fetch = original; }
});
test('multipart upload reports actual progress and propagates validation errors', async () => {
  const original = globalThis.XMLHttpRequest;
  const events = [];
  try {
    let failure = false;
    globalThis.XMLHttpRequest = class {
      upload = {};
      open(method, path) { assert.equal(method, 'POST'); assert.equal(path, '/api/datasets/upload'); }
      send(body) {
        assert.ok(body.get('file'));
        this.upload.onprogress({ lengthComputable: true, loaded: 5, total: 10 });
        this.status = failure ? 400 : 201;
        this.responseText = JSON.stringify(failure ? { error: 'Malformed CSV' } : { id: 'stored-dataset' });
        this.onload();
      }
    };
    assert.equal((await uploadDataset(new Blob(['a,b\n1,2']), n => events.push(n))).id, 'stored-dataset');
    assert.deepEqual(events, [50]);
    failure = true;
    await assert.rejects(uploadDataset(new Blob(['bad']), () => {}), /Malformed CSV/);
  } finally { globalThis.XMLHttpRequest = original; }
});
