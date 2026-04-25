#pragma once
#include <cstddef>
#include <cstdint>

size_t get_rss_bytes();

struct CpuSnapshot { uint64_t user, sys, idle; };
CpuSnapshot cpu_snapshot();
double cpu_util_percent(CpuSnapshot before, CpuSnapshot after);
