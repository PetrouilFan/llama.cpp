#include "llama.h"

#include "llama-kv-cache-policy.h"

#include "llama-impl.h"

#include "nlohmann/json.hpp"

extern "C" {
#include "sha256/sha256.h"
}

#define SHA256_DIGEST_SIZE 32

#include <cstdio>
#include <cstring>

using json = nlohmann::ordered_json;

static bool ggml_type_from_str(const char * name, ggml_type & type) {
    for (int t = GGML_TYPE_COUNT - 1; t >= 0; --t) {
        if (strcmp(name, ggml_type_name((ggml_type) t)) == 0) {
            type = (ggml_type) t;
            return true;
        }
    }
    return false;
}

static bool parse_type_list(const json & j, const char * key, std::vector<ggml_type> & out) {
    if (!j.contains(key)) {
        return false;
    }

    const json & arr = j[key];
    if (!arr.is_array()) {
        return false;
    }

    out.clear();
    for (const auto & item : arr) {
        if (!item.is_string()) {
            return false;
        }
        ggml_type type;
        if (!ggml_type_from_str(item.get<std::string>().c_str(), type)) {
            return false;
        }
        out.push_back(type);
    }

    return true;
}

// canonical SHA-256 of the policy: json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=True)
// with policy_sha256/created_at/git_commit excluded (paper spec, see create_policy.py)
static std::string compute_policy_sha256(const json & j) {
    // std::map-based json sorts the keys, matching python sort_keys=True
    nlohmann::json sorted = nlohmann::json::object();
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (strcmp(it.key().c_str(), "policy_sha256") == 0 ||
            strcmp(it.key().c_str(), "created_at")    == 0 ||
            strcmp(it.key().c_str(), "git_commit")    == 0) {
            continue;
        }
        sorted[it.key()] = it.value();
    }

    const std::string canonical = sorted.dump(2, ' ', true);
    unsigned char hash[SHA256_DIGEST_SIZE];
    sha256_hash(hash, (const unsigned char *) canonical.c_str(), canonical.size());

    char buf[SHA256_DIGEST_SIZE * 2 + 1];
    for (size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        snprintf(buf + 2 * i, 3, "%02x", hash[i]);
    }

    return std::string(buf);
}

static std::string build_summary(const std::vector<ggml_type> & type_k, const std::vector<ggml_type> & type_v) {
    // run-length encoded, e.g. "k=q4_0 v=q3_0_64x24,q2_nux12"
    const auto encode = [](const std::vector<ggml_type> & types) {
        std::string res;
        for (size_t i = 0; i < types.size();) {
            size_t j = i + 1;
            while (j < types.size() && types[j] == types[i]) {
                ++j;
            }
            if (!res.empty()) {
                res += ",";
            }
            res += ggml_type_name(types[i]);
            if (j - i > 1) {
                res += "x" + std::to_string(j - i);
            }
            i = j;
        }
        return res;
    };

    return "k=" + encode(type_k) + " v=" + encode(type_v);
}

struct llama_kv_cache_policy * llama_kv_cache_policy_parse(const char * json_str) {
    if (json_str == nullptr) {
        return nullptr;
    }

    json j;
    try {
        j = json::parse(json_str);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: invalid policy JSON: %s\n", __func__, e.what());
        return nullptr;
    }

    if (!j.is_object()) {
        LLAMA_LOG_ERROR("%s: policy must be a JSON object\n", __func__);
        return nullptr;
    }

    if (!j.contains("schema_version") || j["schema_version"] != 1) {
        LLAMA_LOG_ERROR("%s: unsupported policy schema_version (expected 1)\n", __func__);
        return nullptr;
    }

    if (!j.contains("k_type") || !j["k_type"].is_string()) {
        LLAMA_LOG_ERROR("%s: policy is missing k_type\n", __func__);
        return nullptr;
    }

    ggml_type type_k;
    if (!ggml_type_from_str(j["k_type"].get<std::string>().c_str(), type_k)) {
        LLAMA_LOG_ERROR("%s: unknown k_type: %s\n", __func__, j["k_type"].get<std::string>().c_str());
        return nullptr;
    }

    std::vector<ggml_type> type_v;
    if (j.contains("v_layer_types")) {
        if (!parse_type_list(j, "v_layer_types", type_v)) {
            LLAMA_LOG_ERROR("%s: invalid v_layer_types\n", __func__);
            return nullptr;
        }
    } else {
        LLAMA_LOG_ERROR("%s: policy is missing v_layer_types\n", __func__);
        return nullptr;
    }

    if (type_v.empty()) {
        LLAMA_LOG_ERROR("%s: v_layer_types is empty\n", __func__);
        return nullptr;
    }

    if (j.contains("n_layers")) {
        const uint32_t n_layers = j["n_layers"].get<uint32_t>();
        if (n_layers != type_v.size()) {
            LLAMA_LOG_ERROR("%s: n_layers (%u) does not match v_layer_types size (%zu)\n", __func__, n_layers, type_v.size());
            return nullptr;
        }
    }

    if (j.contains("n_kv_heads") && j["n_kv_heads"].get<uint32_t>() == 0) {
        LLAMA_LOG_ERROR("%s: invalid n_kv_heads\n", __func__);
        return nullptr;
    }

    if (j.contains("head_dim") && j["head_dim"].get<uint32_t>() == 0) {
        LLAMA_LOG_ERROR("%s: invalid head_dim\n", __func__);
        return nullptr;
    }

    const std::string sha256 = compute_policy_sha256(j);

    if (j.contains("policy_sha256")) {
        if (!j["policy_sha256"].is_string()) {
            LLAMA_LOG_ERROR("%s: invalid policy_sha256\n", __func__);
            return nullptr;
        }
        const std::string expected = j["policy_sha256"].get<std::string>();
        if (expected != sha256) {
            LLAMA_LOG_ERROR("%s: policy hash mismatch: file has %s, computed %s\n", __func__, expected.c_str(), sha256.c_str());
            return nullptr;
        }
    }

    auto * policy = new llama_kv_cache_policy();
    policy->type_k.assign(type_v.size(), type_k);
    policy->type_v = std::move(type_v);
    policy->sha256 = sha256;
    policy->summary = build_summary(policy->type_k, policy->type_v);

    return policy;
}

void llama_kv_cache_policy_free(struct llama_kv_cache_policy * policy) {
    delete policy;
}
