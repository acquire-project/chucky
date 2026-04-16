#include "zarr/s3_client.h"
#include "util/prelude.h"

#include <aws/auth/credentials.h>
#include <aws/common/byte_buf.h>
#include <aws/common/condition_variable.h>
#include <aws/common/mutex.h>
#include <aws/common/string.h>
#include <aws/http/request_response.h>
#include <aws/io/channel_bootstrap.h>
#include <aws/io/event_loop.h>
#include <aws/io/host_resolver.h>
#include <aws/io/retry_strategy.h>
#include <aws/io/stream.h>
#include <aws/io/tls_channel_handler.h>
#include <aws/io/uri.h>
#include <aws/s3/s3.h>
#include <aws/s3/s3_client.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- s3_client ---

struct s3_client
{
  struct aws_allocator* alloc;
  struct aws_event_loop_group* el_group;
  struct aws_host_resolver* resolver;
  struct aws_client_bootstrap* bootstrap;
  struct aws_tls_ctx* tls_ctx;
  struct aws_credentials_provider* cred_provider;
  struct aws_retry_strategy* retry_strategy;
  struct aws_s3_client* client;
  struct aws_signing_config_aws signing_config;

  // Parsed endpoint (when custom endpoint is set)
  struct aws_uri endpoint_uri;
  int has_endpoint;

  // Region string (kept alive for signing config)
  struct aws_string* region;

  // Timeout for blocking waits (INT64_MAX = no timeout)
  uint64_t timeout_ns;
};

struct s3_client*
s3_client_create(const struct s3_client_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->region);
  CHECK(Fail, cfg->endpoint);
  struct aws_allocator* alloc = aws_default_allocator();
  aws_s3_library_init(alloc);

  struct s3_client* c = (struct s3_client*)calloc(1, sizeof(struct s3_client));
  CHECK(Fail, c);
  c->alloc = alloc;

  // Event loop group
  c->el_group = aws_event_loop_group_new_default(alloc, 0, NULL);
  CHECK(Fail_alloc, c->el_group);

  // Host resolver
  struct aws_host_resolver_default_options resolver_opts = {
    .el_group = c->el_group,
    .max_entries = 8,
  };
  c->resolver = aws_host_resolver_new_default(alloc, &resolver_opts);
  CHECK(Fail_el, c->resolver);

  // Client bootstrap
  struct aws_client_bootstrap_options bootstrap_opts = {
    .event_loop_group = c->el_group,
    .host_resolver = c->resolver,
  };
  c->bootstrap = aws_client_bootstrap_new(alloc, &bootstrap_opts);
  CHECK(Fail_resolver, c->bootstrap);

  // Parse endpoint to determine TLS mode
  int use_tls = 1;
  {
    struct aws_byte_cursor ep_cur = aws_byte_cursor_from_c_str(cfg->endpoint);
    CHECK(Fail_bootstrap,
          aws_uri_init_parse(&c->endpoint_uri, alloc, &ep_cur) ==
            AWS_OP_SUCCESS);
    c->has_endpoint = 1;
    const struct aws_byte_cursor* scheme = aws_uri_scheme(&c->endpoint_uri);
    struct aws_byte_cursor http_scheme = aws_byte_cursor_from_c_str("http");
    if (aws_byte_cursor_eq_ignore_case(scheme, &http_scheme))
      use_tls = 0;
  }

  if (use_tls) {
    struct aws_tls_ctx_options tls_opts;
    aws_tls_ctx_options_init_default_client(&tls_opts, alloc);
    c->tls_ctx = aws_tls_client_ctx_new(alloc, &tls_opts);
    aws_tls_ctx_options_clean_up(&tls_opts);
    CHECK(Fail_bootstrap, c->tls_ctx);
  }

  // Credentials provider (default chain)
  struct aws_credentials_provider_chain_default_options cred_opts = {
    .bootstrap = c->bootstrap,
  };
  c->cred_provider =
    aws_credentials_provider_new_chain_default(alloc, &cred_opts);
  CHECK(Fail_tls, c->cred_provider);

  // Region
  c->region = aws_string_new_from_c_str(alloc, cfg->region);
  CHECK(Fail_cred, c->region);

  // Signing config
  c->signing_config.algorithm = AWS_SIGNING_ALGORITHM_V4;
  c->signing_config.config_type = AWS_SIGNING_CONFIG_AWS;
  c->signing_config.credentials_provider = c->cred_provider;
  c->signing_config.region = aws_byte_cursor_from_string(c->region);
  c->signing_config.service = aws_byte_cursor_from_c_str("s3");
  c->signing_config.signed_body_header = AWS_SBHT_X_AMZ_CONTENT_SHA256;

  // Retry strategy (only if caller requested non-default settings)
  if (cfg->max_retries || cfg->backoff_scale_ms || cfg->max_backoff_secs) {
    struct aws_standard_retry_options retry_opts = {
      .backoff_retry_options = {
        .el_group = c->el_group,
        .max_retries = cfg->max_retries,
        .backoff_scale_factor_ms = cfg->backoff_scale_ms,
        .max_backoff_secs = cfg->max_backoff_secs,
      },
    };
    c->retry_strategy = aws_retry_strategy_new_standard(alloc, &retry_opts);
    CHECK(Fail_region, c->retry_strategy);
  }

  // Timeout
  c->timeout_ns = cfg->timeout_ns ? cfg->timeout_ns : (uint64_t)INT64_MAX;

  // S3 client config
  struct aws_s3_client_config s3cfg = {
    .region = aws_byte_cursor_from_string(c->region),
    .client_bootstrap = c->bootstrap,
    .signing_config = &c->signing_config,
    .part_size = cfg->part_size,
    .throughput_target_gbps = cfg->throughput_gbps,
    .tls_mode = use_tls ? AWS_MR_TLS_ENABLED : AWS_MR_TLS_DISABLED,
    .retry_strategy = c->retry_strategy,
  };

  if (use_tls && c->tls_ctx) {
    struct aws_tls_connection_options* tls_conn_opts =
      (struct aws_tls_connection_options*)calloc(
        1, sizeof(struct aws_tls_connection_options));
    CHECK(Fail_region, tls_conn_opts);
    aws_tls_connection_options_init_from_ctx(tls_conn_opts, c->tls_ctx);
    s3cfg.tls_connection_options = tls_conn_opts;
  }

  c->client = aws_s3_client_new(alloc, &s3cfg);

  // Clean up tls_connection_options (client copies what it needs)
  if (s3cfg.tls_connection_options) {
    aws_tls_connection_options_clean_up(
      (struct aws_tls_connection_options*)s3cfg.tls_connection_options);
    free((void*)s3cfg.tls_connection_options);
  }

  CHECK(Fail_region, c->client);

  return c;

Fail_region:
  if (c->retry_strategy)
    aws_retry_strategy_release(c->retry_strategy);
  aws_string_destroy(c->region);
Fail_cred:
  aws_credentials_provider_release(c->cred_provider);
Fail_tls:
  if (c->tls_ctx)
    aws_tls_ctx_release(c->tls_ctx);
Fail_bootstrap:
  aws_client_bootstrap_release(c->bootstrap);
Fail_resolver:
  aws_host_resolver_release(c->resolver);
Fail_el:
  aws_event_loop_group_release(c->el_group);
Fail_alloc:
  if (c->has_endpoint)
    aws_uri_clean_up(&c->endpoint_uri);
  free(c);
Fail:
  return NULL;
}

void
s3_client_destroy(struct s3_client* c)
{
  if (!c)
    return;

  aws_s3_client_release(c->client);
  if (c->retry_strategy)
    aws_retry_strategy_release(c->retry_strategy);
  aws_credentials_provider_release(c->cred_provider);
  if (c->tls_ctx)
    aws_tls_ctx_release(c->tls_ctx);
  aws_client_bootstrap_release(c->bootstrap);
  aws_host_resolver_release(c->resolver);
  aws_event_loop_group_release(c->el_group);
  aws_string_destroy(c->region);
  if (c->has_endpoint)
    aws_uri_clean_up(&c->endpoint_uri);
  free(c);

  aws_s3_library_clean_up();
}

// --- Blocking meta request helpers ---

struct blocking_meta_request
{
  struct aws_mutex mutex;
  struct aws_condition_variable cv;
  int done;
  int error_code;
  int response_status;
};

static void
on_meta_request_finish(struct aws_s3_meta_request* meta_request,
                       const struct aws_s3_meta_request_result* result,
                       void* user_data)
{
  (void)meta_request;
  struct blocking_meta_request* ctx = (struct blocking_meta_request*)user_data;
  aws_mutex_lock(&ctx->mutex);
  ctx->error_code = result->error_code;
  ctx->response_status = result->response_status;
  ctx->done = 1;
  aws_condition_variable_notify_one(&ctx->cv);
  aws_mutex_unlock(&ctx->mutex);
}

static bool
is_meta_request_done(void* user_data)
{
  struct blocking_meta_request* ctx = (struct blocking_meta_request*)user_data;
  return ctx->done != 0;
}

// Build an HTTP PUT request message for the given bucket/key.
static struct aws_http_message*
make_put_message(struct aws_allocator* alloc,
                 struct s3_client* c,
                 const char* bucket,
                 const char* key,
                 size_t content_length)
{
  struct aws_http_message* msg = aws_http_message_new_request(alloc);
  if (!msg)
    return NULL;

  aws_http_message_set_request_method(msg, aws_byte_cursor_from_c_str("PUT"));

  // Build path and Host header
  char path[4096];
  if (c->has_endpoint) {
    // Path-style: /{bucket}/{key}
    snprintf(path, sizeof(path), "/%s/%s", bucket, key);
    aws_http_message_set_request_path(msg, aws_byte_cursor_from_c_str(path));

    const struct aws_byte_cursor* authority =
      aws_uri_authority(&c->endpoint_uri);
    aws_http_message_add_header(
      msg,
      (struct aws_http_header){ .name = aws_byte_cursor_from_c_str("Host"),
                                .value = *authority });
  } else {
    // Virtual-hosted style: /{key}, Host: {bucket}.s3.{region}.amazonaws.com
    snprintf(path, sizeof(path), "/%s", key);
    aws_http_message_set_request_path(msg, aws_byte_cursor_from_c_str(path));

    char host[512];
    snprintf(host,
             sizeof(host),
             "%s.s3.%s.amazonaws.com",
             bucket,
             aws_string_c_str(c->region));
    aws_http_message_add_header(
      msg,
      (struct aws_http_header){ .name = aws_byte_cursor_from_c_str("Host"),
                                .value = aws_byte_cursor_from_c_str(host) });
  }

  // Content-Length (only for small puts where we know the size)
  if (content_length > 0) {
    char cl[32];
    snprintf(cl, sizeof(cl), "%zu", content_length);
    aws_http_message_add_header(
      msg,
      (struct aws_http_header){ .name =
                                  aws_byte_cursor_from_c_str("Content-Length"),
                                .value = aws_byte_cursor_from_c_str(cl) });
  }

  return msg;
}

// --- s3_client_put (small blocking PUT) ---

int
s3_client_put(struct s3_client* c,
              const char* bucket,
              const char* key,
              const void* data,
              size_t len)
{
  struct aws_http_message* msg =
    make_put_message(c->alloc, c, bucket, key, len);
  CHECK(Fail, msg);

  // Body stream
  struct aws_byte_cursor body_cur = aws_byte_cursor_from_array(data, len);
  struct aws_input_stream* body_stream =
    aws_input_stream_new_from_cursor(c->alloc, &body_cur);
  CHECK(Fail_msg, body_stream);
  aws_http_message_set_body_stream(msg, body_stream);

  struct blocking_meta_request ctx = {
    .mutex = AWS_MUTEX_INIT,
    .cv = AWS_CONDITION_VARIABLE_INIT,
  };

  struct aws_s3_meta_request_options opts = {
    .type = AWS_S3_META_REQUEST_TYPE_PUT_OBJECT,
    .message = msg,
    .finish_callback = on_meta_request_finish,
    .user_data = &ctx,
  };

  // Set endpoint override for custom endpoints
  if (c->has_endpoint)
    opts.endpoint = &c->endpoint_uri;

  struct aws_s3_meta_request* meta_req =
    aws_s3_client_make_meta_request(c->client, &opts);
  CHECK(Fail_stream, meta_req);

  // Wait for completion (with timeout)
  aws_mutex_lock(&ctx.mutex);
  aws_condition_variable_wait_for_pred(
    &ctx.cv, &ctx.mutex, (int64_t)c->timeout_ns, is_meta_request_done, &ctx);
  int timed_out = !ctx.done;
  aws_mutex_unlock(&ctx.mutex);

  if (timed_out) {
    log_error("s3_client_put(%s/%s): timed out", bucket, key);
    aws_s3_meta_request_cancel(meta_req);
    // Wait for CRT to acknowledge cancellation
    aws_mutex_lock(&ctx.mutex);
    aws_condition_variable_wait_pred(
      &ctx.cv, &ctx.mutex, is_meta_request_done, &ctx);
    aws_mutex_unlock(&ctx.mutex);
  }

  aws_s3_meta_request_release(meta_req);
  aws_input_stream_release(body_stream);
  aws_http_message_release(msg);

  if (timed_out)
    return 1;

  if (ctx.error_code != 0) {
    log_error("s3_client_put(%s/%s): error %d (HTTP %d)",
              bucket,
              key,
              ctx.error_code,
              ctx.response_status);
    return 1;
  }
  return 0;

Fail_stream:
  aws_input_stream_release(body_stream);
Fail_msg:
  aws_http_message_release(msg);
Fail:
  return 1;
}

// --- s3_upload (streaming PUT with async writes) ---

struct s3_upload
{
  struct s3_client* client;
  struct aws_s3_meta_request* meta_request;
  struct aws_http_message* message;

  // Completion tracking
  struct aws_mutex mutex;
  struct aws_condition_variable cv;
  int finished;
  int error_code;
  int response_status;
};

static void
on_upload_finish(struct aws_s3_meta_request* meta_request,
                 const struct aws_s3_meta_request_result* result,
                 void* user_data)
{
  (void)meta_request;
  struct s3_upload* u = (struct s3_upload*)user_data;
  aws_mutex_lock(&u->mutex);
  u->error_code = result->error_code;
  u->response_status = result->response_status;
  u->finished = 1;
  aws_condition_variable_notify_one(&u->cv);
  aws_mutex_unlock(&u->mutex);
}

static bool
is_upload_finished(void* user_data)
{
  struct s3_upload* u = (struct s3_upload*)user_data;
  return u->finished != 0;
}

struct s3_upload*
s3_upload_begin(struct s3_client* c, const char* bucket, const char* key)
{
  struct s3_upload* u = (struct s3_upload*)calloc(1, sizeof(struct s3_upload));
  CHECK(Fail, u);

  u->client = c;
  u->mutex = (struct aws_mutex)AWS_MUTEX_INIT;
  u->cv = (struct aws_condition_variable)AWS_CONDITION_VARIABLE_INIT;

  // Build PUT message (no Content-Length for streaming upload)
  u->message = make_put_message(c->alloc, c, bucket, key, 0);
  CHECK(Fail_alloc, u->message);

  struct aws_s3_meta_request_options opts = {
    .type = AWS_S3_META_REQUEST_TYPE_PUT_OBJECT,
    .message = u->message,
    .send_using_async_writes = true,
    .finish_callback = on_upload_finish,
    .user_data = u,
  };

  if (c->has_endpoint)
    opts.endpoint = &c->endpoint_uri;

  u->meta_request = aws_s3_client_make_meta_request(c->client, &opts);
  CHECK(Fail_msg, u->meta_request);

  return u;

Fail_msg:
  aws_http_message_release(u->message);
Fail_alloc:
  free(u);
Fail:
  return NULL;
}

int
s3_upload_write(struct s3_upload* u, const void* data, size_t len)
{
  if (len == 0)
    return 0;

  struct aws_byte_cursor cur = aws_byte_cursor_from_array(data, len);
  struct aws_future_void* future =
    aws_s3_meta_request_write(u->meta_request, cur, false);
  aws_future_void_wait(future, u->client->timeout_ns);
  int err = aws_future_void_get_error(future);
  aws_future_void_release(future);

  if (err) {
    log_error("s3_upload_write: error %d", err);
    return 1;
  }
  return 0;
}

int
s3_upload_finish_async(struct s3_upload* u)
{
  struct aws_byte_cursor empty = { .ptr = NULL, .len = 0 };
  struct aws_future_void* future =
    aws_s3_meta_request_write(u->meta_request, empty, true);
  aws_future_void_wait(future, u->client->timeout_ns);
  int err = aws_future_void_get_error(future);
  aws_future_void_release(future);

  if (err)
    log_error("s3_upload_finish_async: EOF write error %d", err);
  return err ? 1 : 0;
}

int
s3_upload_wait(struct s3_upload* u)
{
  aws_mutex_lock(&u->mutex);
  aws_condition_variable_wait_for_pred(
    &u->cv, &u->mutex, (int64_t)u->client->timeout_ns, is_upload_finished, u);
  int timed_out = !u->finished;
  aws_mutex_unlock(&u->mutex);

  if (timed_out) {
    log_error("s3_upload_wait: timed out");
    aws_s3_meta_request_cancel(u->meta_request);
    // Wait for CRT to acknowledge cancellation
    aws_mutex_lock(&u->mutex);
    aws_condition_variable_wait_pred(&u->cv, &u->mutex, is_upload_finished, u);
    aws_mutex_unlock(&u->mutex);
    return 1;
  }

  if (u->error_code)
    log_error(
      "s3_upload_wait: error %d (HTTP %d)", u->error_code, u->response_status);
  return u->error_code ? 1 : 0;
}

void
s3_upload_destroy(struct s3_upload* u)
{
  if (!u)
    return;
  if (u->meta_request)
    aws_s3_meta_request_release(u->meta_request);
  if (u->message)
    aws_http_message_release(u->message);
  free(u);
}

void
s3_upload_abort(struct s3_upload* u)
{
  if (!u)
    return;
  if (u->meta_request)
    aws_s3_meta_request_cancel(u->meta_request);

  // Wait for CRT to acknowledge cancellation
  aws_mutex_lock(&u->mutex);
  aws_condition_variable_wait_pred(&u->cv, &u->mutex, is_upload_finished, u);
  aws_mutex_unlock(&u->mutex);

  s3_upload_destroy(u);
}
