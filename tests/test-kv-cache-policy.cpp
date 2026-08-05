#include "llama.h"
#include "../src/llama-kv-cache-policy.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// expected policy hashes (canonical JSON sha256, see paper/scripts/create_policy.py)
// verified against the legacy repo manifest and the files' own policy_sha256 fields
static const struct {
    const char * file;
    const char * sha256;
    const char * summary;
} EXPECTED[] = {
    { "q4xq4.json",
      "873f809c14179f6f85a76a6029e6e73ab4d79ffeb90e039633140fdceb241905",
      "k=q4_0x36 v=q4_0x36" },
    { "q4xq3_64.json",
      "4e9a23f2db8e155583496c2d2f6fc02a5b02bae6213760ba1510fadf0bbaaeb0",
      "k=q4_0x36 v=q3_0_64x36" },
    { "c2.json",
      "c258819b5134e353298aac7ce9f8130dc70d917d20c9b90cca39113a1519a1a3",
      "k=q4_0x36 v=q4_0,q2_0x6,q4_0,q2_0,q3_0_64,q2_0x4,q3_0_64,q2_0,q3_0_64,q2_0x3,q4_0,q2_0x2,q4_0,q2_0x10,q3_0_64,q2_0" },
    { "28q2_8q4.json",
      "1e7b1fed28462f9b608dd0f6e32dbf2e0025a6c23bb3afdcbf6826b90cd32779",
      "k=q4_0x36 v=q4_0,q2_0x6,q4_0,q2_0,q4_0,q2_0x4,q4_0,q2_0,q4_0,q2_0x3,q4_0,q2_0x2,q4_0,q2_0x10,q4_0,q2_0" },
};

static const char * POLICY_JSON =
    "{\n"
    "  \"schema_version\": 1,\n"
    "  \"policy_id\": \"test\",\n"
    "  \"k_type\": \"q4_0\",\n"
    "  \"v_layer_types\": [\"q3_0_64\", \"q3_0_64\", \"q2_nu\"],\n"
    "  \"n_layers\": 3\n"
    "}\n";

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    out = ss.str();
    return true;
}

static std::string tamper_hash(const std::string & json) {
    const std::string needle = "\"policy_sha256\": \"";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return json;
    }
    std::string res = json;
    const size_t hpos = pos + needle.size();
    res[hpos] = res[hpos] == '0' ? '1' : '0';
    return res;
}

static std::string set_n_layers(const std::string & json, const char * value) {
    const std::string needle = "\"n_layers\": 36";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return json;
    }
    std::string res = json;
    res.replace(pos, needle.size(), std::string("\"n_layers\": ") + value);
    return res;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <policies-dir>\n", argv[0]);
        return 1;
    }

    const std::string dir = argv[1];

    int num_failed = 0;

    for (const auto & exp : EXPECTED) {
        std::string json;
        if (!read_file(dir + "/" + exp.file, json)) {
            fprintf(stderr, "cannot open %s\n", exp.file);
            return 1;
        }

        llama_kv_cache_policy * policy = llama_kv_cache_policy_parse(json.c_str());
        if (policy == nullptr) {
            fprintf(stderr, "%s: parse failed\n", exp.file);
            num_failed++;
            continue;
        }

        if (policy->sha256 != exp.sha256) {
            fprintf(stderr, "%s: hash mismatch: %s\n", exp.file, policy->sha256.c_str());
            num_failed++;
        }

        if (policy->type_k.size() != 36 || policy->type_v.size() != 36) {
            fprintf(stderr, "%s: wrong layer count (%zu/%zu)\n", exp.file, policy->type_k.size(), policy->type_v.size());
            num_failed++;
        }

        for (size_t i = 0; i < policy->type_k.size(); ++i) {
            if (policy->type_k[i] != GGML_TYPE_Q4_0) {
                fprintf(stderr, "%s: k layer %zu is not q4_0\n", exp.file, i);
                num_failed++;
                break;
            }
        }

        if (policy->summary != exp.summary) {
            fprintf(stderr, "%s: summary mismatch:\n  got: %s\n  exp: %s\n", exp.file, policy->summary.c_str(), exp.summary);
            num_failed++;
        }

        llama_kv_cache_policy_free(policy);

        printf("%s: PASSED (%s)\n", exp.file, exp.summary);
    }

    // positive case without policy_sha256
    {
        llama_kv_cache_policy * policy = llama_kv_cache_policy_parse(POLICY_JSON);
        if (policy == nullptr) {
            fprintf(stderr, "inline policy: parse failed\n");
            num_failed++;
        } else {
            if (policy->type_v != std::vector<ggml_type>({GGML_TYPE_Q3_0_64, GGML_TYPE_Q3_0_64, GGML_TYPE_Q2_NU})) {
                fprintf(stderr, "inline policy: wrong v types\n");
                num_failed++;
            }
            if (policy->sha256.empty()) {
                fprintf(stderr, "inline policy: empty hash\n");
                num_failed++;
            }
            llama_kv_cache_policy_free(policy);
            printf("inline policy: PASSED\n");
        }
    }

    // negative cases - all must return NULL
    {
        std::string json;
        if (!read_file(dir + "/q4xq4.json", json)) {
            fprintf(stderr, "cannot open q4xq4.json\n");
            return 1;
        }

        const char * bad_inputs[] = {
            nullptr,                       // null input
            "",                            // empty string
            "{",                           // truncated JSON
            "[]",                          // not an object
            "{}",                          // missing fields
            "{\"schema_version\": 2}",     // wrong schema version
            "{\"schema_version\": 1, \"k_type\": \"q4_0\", \"v_layer_types\": []}", // empty v_layer_types
            "{\"schema_version\": 1, \"k_type\": \"q9_9\", \"v_layer_types\": [\"q4_0\"]}", // unknown k_type
            "{\"schema_version\": 1, \"k_type\": \"q4_0\", \"v_layer_types\": [\"q9_9\"]}", // unknown v type
            "{\"schema_version\": 1, \"k_type\": \"q4_0\", \"v_layer_types\": [\"q4_0\"], \"n_layers\": 2}", // n_layers mismatch
        };

        for (size_t i = 0; i < sizeof(bad_inputs) / sizeof(bad_inputs[0]); ++i) {
            llama_kv_cache_policy * policy = llama_kv_cache_policy_parse(bad_inputs[i]);
            if (policy != nullptr) {
                fprintf(stderr, "negative case %zu: expected NULL\n", i);
                num_failed++;
                llama_kv_cache_policy_free(policy);
            }
        }

        // tampered policy_sha256 must fail
        const std::string tampered = tamper_hash(json);
        llama_kv_cache_policy * policy = llama_kv_cache_policy_parse(tampered.c_str());
        if (policy != nullptr) {
            fprintf(stderr, "tampered hash: expected NULL\n");
            num_failed++;
            llama_kv_cache_policy_free(policy);
        }

        // n_layers mismatch must fail
        const std::string wrong_n = set_n_layers(json, "35");
        policy = llama_kv_cache_policy_parse(wrong_n.c_str());
        if (policy != nullptr) {
            fprintf(stderr, "n_layers mismatch: expected NULL\n");
            num_failed++;
            llama_kv_cache_policy_free(policy);
        }

        printf("negative cases: PASSED\n");
    }

    if (num_failed != 0) {
        fprintf(stderr, "%d checks failed\n", num_failed);
        return 1;
    }

    printf("OK\n");
    return 0;
}
