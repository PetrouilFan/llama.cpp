#include "ggml.h"
#include "ggml-cpu.h"

#define GGML_COMMON_DECL_C
#include "../ggml/src/ggml-common.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// block sizes must match the paper formats exactly
static_assert(sizeof(block_q2_nu) == 18, "q2_nu block must be 18 bytes");
static_assert(sizeof(block_q3_0)  == 14, "q3_0 block must be 14 bytes");
static_assert(sizeof(block_q3_0_64) == 26, "q3_0_64 block must be 26 bytes");

struct test_record {
    std::string type;
    std::string ccase;
    int sub;
    std::vector<float> input;
    std::vector<float> dequant;
    std::vector<uint8_t> packed;
};

static bool parse_records(const json & j, const char * type_name, std::vector<test_record> & out) {
    for (const auto & r : j["records"]) {
        test_record rec;
        rec.type     = type_name;
        rec.ccase    = r["case"];
        rec.sub      = r["sub"];

        for (const auto & v : r["input"]) {
            uint32_t bits;
            sscanf(((const std::string &) v[0]).c_str(), "%" SCNx32, &bits);
            float f = strtof(((const std::string &) v[1]).c_str(), nullptr);
            if (memcmp(&bits, &f, sizeof(float)) != 0) {
                fprintf(stderr, "%s: input hex/f32 mismatch (case %s sub %d)\n", type_name, rec.ccase.c_str(), rec.sub);
                return false;
            }
            rec.input.push_back(f);
        }

        for (const auto & v : r["dequant"]) {
            rec.dequant.push_back(strtof(((const std::string &) v[1]).c_str(), nullptr));
        }

        const std::string hex = r["packed"];
        if (hex.size() % 2 != 0) {
            fprintf(stderr, "%s: odd packed hex length\n", type_name);
            return false;
        }

        // golden files store the fp16 scale separately from the packed codes
        uint32_t scale_bits;
        sscanf(((const std::string &) r["scale"]).c_str(), "%" SCNx32, &scale_bits);
        const uint16_t scale = (uint16_t) scale_bits;
        rec.packed.push_back((uint8_t) (scale & 0xff));
        rec.packed.push_back((uint8_t) (scale >> 8));

        for (size_t i = 0; i < hex.size(); i += 2) {
            unsigned int byte;
            sscanf(hex.c_str() + i, "%2x", &byte);
            rec.packed.push_back((uint8_t) byte);
        }

        out.push_back(std::move(rec));
    }
    return true;
}

static bool check_quantize(ggml_type type, const test_record & rec, size_t blck_size, size_t type_size) {
    if (rec.input.size() != rec.packed.size() / type_size * blck_size) {
        fprintf(stderr, "%s: record size mismatch (%zu input, %zu packed)\n", rec.type.c_str(), rec.input.size(), rec.packed.size());
        return false;
    }

    std::vector<uint8_t> qbuf(rec.packed.size());
    const auto * traits = ggml_get_type_traits(type);
    traits->from_float_ref(rec.input.data(), qbuf.data(), (int64_t) rec.input.size());

    if (memcmp(qbuf.data(), rec.packed.data(), rec.packed.size()) != 0) {
        fprintf(stderr, "%s: packed bytes differ (case %s sub %d)\n", rec.type.c_str(), rec.ccase.c_str(), rec.sub);
        return false;
    }
    return true;
}

static bool check_dequant(ggml_type type, const test_record & rec, float tol) {
    std::vector<float> out(rec.input.size());
    const auto * traits = ggml_get_type_traits(type);
    traits->to_float(rec.packed.data(), out.data(), (int64_t) rec.input.size());

    for (size_t i = 0; i < rec.input.size(); ++i) {
        if (std::isnan(out[i]) || std::isinf(out[i])) {
            fprintf(stderr, "%s: non-finite dequant (case %s sub %d idx %zu)\n", rec.type.c_str(), rec.ccase.c_str(), rec.sub, i);
            return false;
        }
        if (fabsf(out[i] - rec.dequant[i]) > tol) {
            fprintf(stderr, "%s: dequant mismatch (case %s sub %d idx %zu: %f vs %f)\n", rec.type.c_str(), rec.ccase.c_str(), rec.sub, i, out[i], rec.dequant[i]);
            return false;
        }
    }
    return true;
}

static bool check_traits(ggml_type type, const char * name, size_t blck_size, size_t type_size) {
    const auto * traits = ggml_get_type_traits(type);
    if (traits->blck_size != blck_size) {
        fprintf(stderr, "%s: wrong blck_size %zu\n", name, traits->blck_size);
        return false;
    }
    if (traits->type_size != type_size) {
        fprintf(stderr, "%s: wrong type_size %zu\n", name, traits->type_size);
        return false;
    }
    if (!traits->is_quantized || !traits->to_float || !traits->from_float_ref) {
        fprintf(stderr, "%s: missing traits\n", name);
        return false;
    }
    if (strcmp(ggml_type_name(type), name) != 0) {
        fprintf(stderr, "%s: wrong type name\n", name);
        return false;
    }
    const auto * cpu_traits = ggml_get_type_traits_cpu(type);
    if (!cpu_traits->from_float || !cpu_traits->vec_dot || cpu_traits->vec_dot_type != GGML_TYPE_Q8_0) {
        fprintf(stderr, "%s: wrong CPU traits\n", name);
        return false;
    }
    return true;
}

static bool check_vec_dot(ggml_type type, const test_record & rec) {
    std::vector<uint8_t> qbuf(rec.packed.size());
    std::vector<float> dequant(rec.input.size());
    std::vector<uint8_t> qy(rec.input.size() * sizeof(block_q8_0) / 32);

    const auto * traits = ggml_get_type_traits(type);
    const auto * cpu_traits = ggml_get_type_traits_cpu(type);
    const auto * y_traits = ggml_get_type_traits_cpu(cpu_traits->vec_dot_type);

    cpu_traits->from_float(rec.input.data(), qbuf.data(), (int64_t) rec.input.size());
    y_traits->from_float(rec.dequant.data(), qy.data(), (int64_t) rec.dequant.size());
    traits->to_float(qbuf.data(), dequant.data(), (int64_t) rec.input.size());

    float vd = 0.0f;
    cpu_traits->vec_dot((int) rec.input.size(), &vd, 0, qbuf.data(), 0, qy.data(), 0, 1);

    std::vector<float> dequant_y(rec.dequant.size());
    ggml_get_type_traits(cpu_traits->vec_dot_type)->to_float(qy.data(), dequant_y.data(), (int64_t) rec.dequant.size());

    double ref = 0.0;
    for (size_t i = 0; i < rec.input.size(); ++i) {
        ref += dequant[i] * dequant_y[i];
    }

    // relative tolerance covers fp16 scale rounding in the vec_dot path
    const double tol = 1e-3 * (fabs(ref) + 1.0);
    if (fabs((double) vd - ref) > tol) {
        fprintf(stderr, "%s: vec_dot mismatch (case %s sub %d: %f vs %f)\n", rec.type.c_str(), rec.ccase.c_str(), rec.sub, vd, ref);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <reference-dir>\n", argv[0]);
        return 1;
    }

    ggml_backend_cpu_init();
    ggml_quantize_init(GGML_TYPE_COUNT);

    const std::string dir = argv[1];

    struct type_spec {
        const char * file;
        const char * name;
        ggml_type    type;
        size_t       blck_size;
        size_t       type_size;
    };

    const type_spec specs[] = {
        { "q2_uniform_vectors.json", "q2_0",   GGML_TYPE_Q2_0,    QK2_0,   sizeof(block_q2_0) },
        { "q2_nu_vectors.json",      "q2_nu",  GGML_TYPE_Q2_NU,   QK2_NU,  sizeof(block_q2_nu) },
        { "q3_0_vectors.json",       "q3_0",   GGML_TYPE_Q3_0,    QK3_0,   sizeof(block_q3_0) },
        { "q3_0_64_vectors.json",    "q3_0_64", GGML_TYPE_Q3_0_64, QK3_0_64, sizeof(block_q3_0_64) },
    };

    int num_failed = 0;

    for (const auto & spec : specs) {
        std::ifstream file(dir + "/" + spec.file);
        if (!file.is_open()) {
            fprintf(stderr, "cannot open %s\n", spec.file);
            return 1;
        }
        std::stringstream ss;
        ss << file.rdbuf();

        json j = json::parse(ss.str());
        if (j["format_version"] != 1 || j["type"] != spec.name) {
            fprintf(stderr, "%s: bad golden header\n", spec.file);
            return 1;
        }

        std::vector<test_record> records;
        if (!parse_records(j, spec.name, records)) {
            return 1;
        }

        num_failed += !check_traits(spec.type, spec.name, spec.blck_size, spec.type_size);

        for (const auto & rec : records) {
            num_failed += !check_quantize(spec.type, rec, spec.blck_size, spec.type_size);
            num_failed += !check_dequant(spec.type, rec, 1e-6f);
            num_failed += !check_vec_dot(spec.type, rec);
        }

        printf("%s: %zu records, %s\n", spec.name, records.size(), num_failed == 0 ? "PASSED" : "FAILED");
    }

    if (num_failed != 0) {
        fprintf(stderr, "%d checks failed\n", num_failed);
        return 1;
    }

    printf("OK\n");
    return 0;
}
