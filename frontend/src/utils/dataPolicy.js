export const UPLOAD_ROUTES = new Set(['/', '/energy', '/battery', '/explorer']);
export function sourcePolicy(source, pathname) {
  if (['/data', '/about', '/research', '/comparison', '/risk'].includes(pathname)) return 'standard';
  if (!source || source === 'none' || source === 'campus') return 'empty';
  return UPLOAD_ROUTES.has(pathname) ? 'analysis' : 'unsupported';
}
export function validateUploadFile(file) {
  if (!file?.name?.toLowerCase().endsWith('.csv')) return 'Choose a CSV file.';
  if (!file.size) return 'The CSV is empty.';
  if (file.size > 10 * 1024 * 1024) return 'The file exceeds the 10 MiB limit.';
  return null;
}
