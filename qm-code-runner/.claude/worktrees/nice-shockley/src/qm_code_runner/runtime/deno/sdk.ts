/**
 * qm-code-runner SDK for Deno runtime.
 */

const METADATA_FILE = "/metadata/.qm_metadata.json";

/**
 * Set the result metadata to be returned to the backend.
 *
 * This separates structured results from stdout/stderr logs.
 *
 * @param data - Any JSON-serializable data (object, array, string, number, etc.)
 *
 * @example
 * import { setMetadata } from './sdk.ts';
 *
 * const result = { status: 'success', count: 42 };
 * setMetadata(result);
 */
export function setMetadata(data: unknown): void {
  Deno.writeTextFileSync(METADATA_FILE, JSON.stringify(data));
}

/**
 * Get previously set metadata (useful for reading/modifying).
 *
 * @returns The previously set metadata, or null if not set.
 */
export function getMetadata(): unknown {
  try {
    const content = Deno.readTextFileSync(METADATA_FILE);
    return JSON.parse(content);
  } catch {
    return null;
  }
}

/**
 * Load a file from the flow's environment.
 * Only available during flow execution, not test runs.
 *
 * @param path - Path to the file within the environment.
 * @returns The file content as a string.
 */
export async function loadFile(path: string): Promise<string> {
  const webdavUrl = Deno.env.get("QM_WEBDAV_URL");
  if (!webdavUrl) {
    throw new Error(
      "loadFile() is only available during flow execution. " +
        "For test runs, use mounted environments instead.",
    );
  }
  const url = webdavUrl.replace(/\/?$/, "/") + path.replace(/^\//, "");
  const resp = await fetch(url, { method: "GET" });
  if (resp.status === 404) {
    throw new Error(`File not found: ${path}`);
  }
  if (!resp.ok) {
    throw new Error(`Failed to load file: HTTP ${resp.status}`);
  }
  return await resp.text();
}

/**
 * Save a file to the flow's environment.
 * Only available during flow execution, not test runs.
 *
 * @param path - Path to the file within the environment.
 * @param content - The file content to save.
 */
export async function saveFile(path: string, content: string): Promise<void> {
  const webdavUrl = Deno.env.get("QM_WEBDAV_URL");
  if (!webdavUrl) {
    throw new Error(
      "saveFile() is only available during flow execution. " +
        "For test runs, use mounted environments instead.",
    );
  }
  const url = webdavUrl.replace(/\/?$/, "/") + path.replace(/^\//, "");
  const resp = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/octet-stream" },
    body: content,
  });
  if (!resp.ok) {
    throw new Error(`Failed to save file: HTTP ${resp.status}`);
  }
}
