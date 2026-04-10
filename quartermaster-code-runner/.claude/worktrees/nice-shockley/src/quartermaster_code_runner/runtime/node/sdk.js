/**
 * quartermaster-code-runner SDK for Node.js runtime.
 */

const fs = require('fs');

const METADATA_FILE = '/metadata/.quartermaster_metadata.json';

/**
 * Set the result metadata to be returned to the backend.
 * 
 * This separates structured results from stdout/stderr logs.
 * 
 * @param {any} data - Any JSON-serializable data (object, array, string, number, etc.)
 * 
 * @example
 * const { setMetadata } = require('./sdk');
 * 
 * const result = { status: 'success', count: 42 };
 * setMetadata(result);
 */
function setMetadata(data) {
    fs.writeFileSync(METADATA_FILE, JSON.stringify(data));
}

/**
 * Get previously set metadata (useful for reading/modifying).
 * 
 * @returns {any} The previously set metadata, or null if not set.
 */
function getMetadata() {
    if (!fs.existsSync(METADATA_FILE)) {
        return null;
    }
    return JSON.parse(fs.readFileSync(METADATA_FILE, 'utf8'));
}

/**
 * Load a file from the flow's environment.
 * Only available during flow execution, not test runs.
 *
 * @param {string} filePath - Path to the file within the environment.
 * @returns {Promise<string>} The file content as a string.
 */
function loadFile(filePath) {
    return new Promise((resolve, reject) => {
        const webdavUrl = process.env.QM_WEBDAV_URL;
        if (!webdavUrl) {
            return reject(new Error(
                'loadFile() is only available during flow execution. ' +
                'For test runs, use mounted environments instead.'
            ));
        }
        const url = new URL(filePath.replace(/^\//, ''), webdavUrl.replace(/\/?$/, '/'));
        const mod = url.protocol === 'https:' ? require('https') : require('http');
        mod.get(url.href, (res) => {
            if (res.statusCode === 404) {
                return reject(new Error(`File not found: ${filePath}`));
            }
            if (res.statusCode < 200 || res.statusCode >= 300) {
                return reject(new Error(`Failed to load file: HTTP ${res.statusCode}`));
            }
            const chunks = [];
            res.on('data', (chunk) => chunks.push(chunk));
            res.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
            res.on('error', reject);
        }).on('error', reject);
    });
}

/**
 * Save a file to the flow's environment.
 * Only available during flow execution, not test runs.
 *
 * @param {string} filePath - Path to the file within the environment.
 * @param {string} content - The file content to save.
 * @returns {Promise<void>}
 */
function saveFile(filePath, content) {
    return new Promise((resolve, reject) => {
        const webdavUrl = process.env.QM_WEBDAV_URL;
        if (!webdavUrl) {
            return reject(new Error(
                'saveFile() is only available during flow execution. ' +
                'For test runs, use mounted environments instead.'
            ));
        }
        const url = new URL(filePath.replace(/^\//, ''), webdavUrl.replace(/\/?$/, '/'));
        const mod = url.protocol === 'https:' ? require('https') : require('http');
        const data = Buffer.from(content, 'utf8');
        const options = {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/octet-stream',
                'Content-Length': data.length,
            },
        };
        const req = mod.request(url.href, options, (res) => {
            if (res.statusCode < 200 || res.statusCode >= 300) {
                return reject(new Error(`Failed to save file: HTTP ${res.statusCode}`));
            }
            resolve();
        });
        req.on('error', reject);
        req.end(data);
    });
}

module.exports = { setMetadata, getMetadata, loadFile, saveFile };
