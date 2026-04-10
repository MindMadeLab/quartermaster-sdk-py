// Package sdk provides quartermaster-code-runner SDK for Go runtime.
package sdk

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

const metadataFile = "/metadata/.quartermaster_metadata.json"

// SetMetadata sets the result metadata to be returned to the backend.
//
// This separates structured results from stdout/stderr logs.
//
// Example:
//
//	import "sdk"
//
//	result := map[string]interface{}{"status": "success", "count": 42}
//	sdk.SetMetadata(result)
func SetMetadata(data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return os.WriteFile(metadataFile, jsonData, 0644)
}

// GetMetadata gets previously set metadata (useful for reading/modifying).
//
// Returns nil if no metadata has been set.
func GetMetadata(dest interface{}) error {
	data, err := os.ReadFile(metadataFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	return json.Unmarshal(data, dest)
}

var errNoWebDAV = errors.New(
	"LoadFile/SaveFile is only available during flow execution. " +
		"For test runs, use mounted environments instead",
)

// LoadFile loads a file from the flow's environment.
// Only available during flow execution, not test runs.
func LoadFile(path string) (string, error) {
	webdavURL := os.Getenv("QM_WEBDAV_URL")
	if webdavURL == "" {
		return "", errNoWebDAV
	}
	url := strings.TrimRight(webdavURL, "/") + "/" + strings.TrimLeft(path, "/")
	resp, err := http.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to load file: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == 404 {
		return "", fmt.Errorf("file not found: %s", path)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("failed to load file: HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}
	return string(body), nil
}

// SaveFile saves a file to the flow's environment.
// Only available during flow execution, not test runs.
func SaveFile(path, content string) error {
	webdavURL := os.Getenv("QM_WEBDAV_URL")
	if webdavURL == "" {
		return errNoWebDAV
	}
	url := strings.TrimRight(webdavURL, "/") + "/" + strings.TrimLeft(path, "/")
	req, err := http.NewRequest("PUT", url, strings.NewReader(content))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to save file: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("failed to save file: HTTP %d", resp.StatusCode)
	}
	return nil
}
