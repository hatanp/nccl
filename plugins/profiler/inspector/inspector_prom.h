/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef INSPECTOR_INSPECTOR_PROM_H_
#define INSPECTOR_INSPECTOR_PROM_H_

#include <stdio.h>
#include "inspector.h"
#include "inspector_proxy_stats.h"

#ifdef __cplusplus
extern "C" {
#endif

int ncclInspectorStepBegin(int64_t step, uint64_t wallTimeNs);
int ncclInspectorStepEnd(int64_t step, uint64_t wallTimeNs);

#ifdef __cplusplus
}
#endif

// Forward declarations
struct inspectorCommInfoList;
struct inspectorDumpThread;

bool inspectorPromStreamingEnabled();
inspectorResult_t inspectorPromRecordCompleted(struct inspectorCommInfo* commInfo,
                                               const struct inspectorCompletedOpInfo* op);
int64_t inspectorPromCurrentStep();
inspectorResult_t inspectorPromRecordProxyOp(
  const struct inspectorProxyOpInfo* op);

// Prometheus-related function declarations
inspectorResult_t inspectorPromCommInfoListDump(struct inspectorCommInfoList* commList,
                                                const char* output_root,
                                                struct inspectorDumpThread* dumpThread);
inspectorResult_t inspectorPromCommInfoListsDump(struct inspectorCommInfoList* first,
                                                 struct inspectorCommInfoList* second,
                                                 const char* output_root,
                                                 struct inspectorDumpThread* dumpThread);

// Prometheus-specific configuration
int64_t inspectorPromValidateInterval(int64_t interval);

#endif  // INSPECTOR_INSPECTOR_PROM_H_
