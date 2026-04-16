// Minimal JSON merge helper for zarr.json attributes.
#pragma once

#include <stddef.h>

// Skip a JSON value starting at s[pos]. Returns the index just past the
// parsed value, or -1 on malformed input. Skips leading whitespace.
ptrdiff_t
json_skip_value(const char* s, size_t len, size_t pos);

// Check that the entire string (modulo trailing whitespace) is a single
// well-formed JSON value. Returns 0 on success, non-zero on error.
int
json_validate(const char* s, size_t len);

// Merge attr_key -> json_value into the "attributes" object of the given
// zarr.json text. Outputs a freshly malloc'd buffer and its length.
// Caller frees *out_buf.
// Returns 0 on success, non-zero on error (malformed, missing attributes).
int
json_merge_attribute(const char* zjson,
                     size_t zjson_len,
                     const char* attr_key,
                     const char* json_value,
                     char** out_buf,
                     size_t* out_len);
