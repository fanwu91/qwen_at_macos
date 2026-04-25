#include "metrics.h"
#include <mach/mach.h>
#include <mach/mach_host.h>

size_t get_rss_bytes() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0;
    return info.phys_footprint;
}

CpuSnapshot cpu_snapshot() {
    host_cpu_load_info_data_t info;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t)&info, &count) != KERN_SUCCESS)
        return {0, 0, 0};
    return {
        info.cpu_ticks[CPU_STATE_USER],
        info.cpu_ticks[CPU_STATE_SYSTEM],
        info.cpu_ticks[CPU_STATE_IDLE]
    };
}

double cpu_util_percent(CpuSnapshot before, CpuSnapshot after) {
    uint64_t used = (after.user - before.user) + (after.sys - before.sys);
    uint64_t total = used + (after.idle - before.idle);
    if (total == 0) return 0.0;
    return 100.0 * used / total;
}
