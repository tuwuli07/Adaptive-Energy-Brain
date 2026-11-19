// Upload data has no campus fallback: failures must remain visible.
export async function datasetRequest(path = '', options = {}) {
  let response;
  try { response = await fetch(`/api/datasets${path}`, options); }
  catch { throw new Error('Cannot reach data storage. Check your connection and try again.'); }
  let result;
  try { result = await response.json(); }
  catch { throw new Error('Data storage is unavailable. Start the project server and try again.'); }
  if (!response.ok) {
    const error = new Error(result.error || 'The dataset request failed.');
    error.validation = result.validation;
    throw error;
  }
  return result;
}

export function uploadDataset(file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', '/api/datasets/upload');
    request.timeout = 60000;
    request.upload.onprogress = event => {
      if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100));
    };
    request.onerror = () => reject(new Error('Upload could not reach the server. Please try again.'));
    request.ontimeout = () => reject(new Error('Upload timed out. Please try again.'));
    request.onload = () => {
      let result;
      try { result = JSON.parse(request.responseText); }
      catch { reject(new Error('Upload service is unavailable. Start the project server and try again.')); return; }
      if (request.status < 200 || request.status >= 300) {
        const error = new Error(result.error || 'Upload failed.');
        error.validation = result.validation;
        reject(error);
      } else resolve(result);
    };
    const body = new FormData();
    body.append('file', file);
    request.send(body);
  });
}
