#pragma once

#include "ggml.h"

#include <string>
#include <vector>

// opaque public type, implementation lives in this header (internal use only)
struct llama_kv_cache_policy {
    // per-layer cache types, indexed by model layer id
    // empty vector = uniform cache type
    std::vector<ggml_type> type_k;
    std::vector<ggml_type> type_v;

    // canonical hash of the policy file (empty if the file has no hash)
    std::string sha256;

    // deterministic one-line summary of the policy
    std::string summary;
};

// parse a policy from the given JSON string, returns NULL on error
struct llama_kv_cache_policy * llama_kv_cache_policy_parse(const char * json_str);

// free a policy handle
void llama_kv_cache_policy_free(struct llama_kv_cache_policy * policy);
