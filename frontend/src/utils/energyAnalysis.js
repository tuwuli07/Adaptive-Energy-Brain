// Samples describe average power from their timestamp to the next timestamp.
// No last-row duration or missing measurement is invented.
export function integratePower(records, field) {
  let energy = 0, hours = 0;
  for (let i = 0; i < records.length - 1; i++) {
    const value = records[i][field];
    const dt = (Date.parse(records[i + 1].timestamp) - Date.parse(records[i].timestamp)) / 3600000;
    if (Number.isFinite(value) && Number.isFinite(dt) && dt > 0) { energy += value * dt; hours += dt; }
  }
  return { energy: hours ? energy : null, hours };
}
