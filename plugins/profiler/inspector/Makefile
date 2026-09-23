# Standalone Makefile for NCCL Inspector (independent from main NCCL)
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

# Compiler settings
CXX ?= g++
NVCC ?= nvcc

# Build options
DEBUG ?= 0
ASAN ?= 0
UBSAN ?= 0
NCCL_INSPECTOR_ENABLE_WARN ?= 0

# CUDA paths - default to /usr/local/cuda or require explicit CUDA_HOME
CUDA_HOME ?= $(shell \
  if [ -d /usr/local/cuda ]; then echo /usr/local/cuda; \
  else echo "ERROR: CUDA installation not found. Please set CUDA_HOME." >&2; exit 1; fi)
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_LIB ?= $(CUDA_HOME)/lib64

# Detect CUDA version
NVCC := $(CUDA_HOME)/bin/nvcc
CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)

# CUDA 13.0 requires c++17
ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 13; echo $$?),0)
  CXXSTD ?= -std=c++17
else
  CXXSTD ?= -std=c++14
endif

# Build directories
OBJDIR := obj
SRCDIR := .

# Compiler flags
CXXFLAGS := $(CXXSTD) -fPIC -Wall -Wextra -O2 -DNCCL_INSPECTOR_ENABLE_WARN=$(NCCL_INSPECTOR_ENABLE_WARN)
INCLUDES := -I$(CUDA_INC) -I$(SRCDIR) -I$(SRCDIR)/nccl
LDFLAGS := -shared -L$(CUDA_LIB)
LIBS := -lcudart -ldl -lpthread

# Debug flags
ifneq ($(DEBUG), 0)
CXXFLAGS += -g -DDEBUG
endif

# Sanitizer flags
ifneq ($(ASAN), 0)
CXXFLAGS += -fsanitize=address
LDFLAGS += -fsanitize=address -static-libasan
endif

ifneq ($(UBSAN), 0)
CXXFLAGS += -fsanitize=undefined
LDFLAGS += -fsanitize=undefined -static-libubsan
endif

# Source files
SRCS := inspector.cc \
	inspector_json.cc \
	inspector_prom.cc \
	inspector_plugin.cc \
	inspector_cudawrap.cc \
	inspector_ring.cc \
	inspector_event_pool.cc \
	inspector_proxy_pool.cc \
	inspector_parent_identity.cc \
	json.cc \
	version.cc

OBJS := $(SRCS:%.cc=$(OBJDIR)/%.o)

# Target library
TARGET := libnccl-profiler-inspector.so

# Build rules
.PHONY: all clean help check-cuda test-ring test-prom-stats test-proxy-stats test-proxy-pool test-proxy-lifecycle test-parent-identity

all: check-cuda $(TARGET)

test-ring:
	$(CXX) $(CXXSTD) -Wall -Wextra $(INCLUDES) \
		tests/inspector_ring_test.cc inspector_ring.cc -o inspector_ring_test
	./inspector_ring_test
	rm -f inspector_ring_test

test-prom-stats:
	$(CXX) $(CXXSTD) -Wall -Wextra $(INCLUDES) \
		tests/inspector_prom_stats_test.cc -o inspector_prom_stats_test
	./inspector_prom_stats_test
	rm -f inspector_prom_stats_test

test-proxy-stats:
	$(CXX) $(CXXSTD) -Wall -Wextra $(INCLUDES) \
		tests/inspector_proxy_stats_test.cc -o inspector_proxy_stats_test
	./inspector_proxy_stats_test
	rm -f inspector_proxy_stats_test

test-proxy-pool:
	$(CXX) $(CXXSTD) -Wall -Wextra $(INCLUDES) \
		tests/inspector_proxy_pool_test.cc inspector_proxy_pool.cc -o inspector_proxy_pool_test
	./inspector_proxy_pool_test
	rm -f inspector_proxy_pool_test

test-parent-identity:
	$(CXX) $(CXXSTD) -Wall -Wextra -pthread $(INCLUDES) \
		tests/inspector_parent_identity_test.cc inspector_parent_identity.cc -o inspector_parent_identity_test
	./inspector_parent_identity_test
	rm -f inspector_parent_identity_test

test-proxy-lifecycle: all
	$(CXX) $(CXXSTD) -Wall -Wextra $(INCLUDES) \
		tests/inspector_proxy_lifecycle_test.cc -L. \
		-Wl,-rpath,'$$ORIGIN' -lnccl-profiler-inspector -o inspector_proxy_lifecycle_test
	./inspector_proxy_lifecycle_test
	./inspector_proxy_lifecycle_test cap1
	./inspector_proxy_lifecycle_test cap0
	./inspector_proxy_lifecycle_test off
	./inspector_proxy_lifecycle_test missing
	rm -f inspector_proxy_lifecycle_test

# Check that CUDA installation exists
check-cuda:
	@if [ ! -d "$(CUDA_HOME)" ]; then \
		echo "ERROR: CUDA installation not found at $(CUDA_HOME)"; \
		echo "Please install CUDA or set CUDA_HOME to the correct path"; \
		echo "Example: make CUDA_HOME=/path/to/cuda"; \
		exit 1; \
	fi
	@if [ ! -f "$(CUDA_INC)/cuda.h" ]; then \
		echo "ERROR: cuda.h not found at $(CUDA_INC)/cuda.h"; \
		echo "Please check your CUDA installation"; \
		exit 1; \
	fi
	@if [ ! -d "$(CUDA_LIB)" ]; then \
		echo "ERROR: CUDA library directory not found at $(CUDA_LIB)"; \
		echo "Please check your CUDA installation"; \
		exit 1; \
	fi
	@if ! ls $(CUDA_LIB)/libcudart.so* >/dev/null 2>&1; then \
		echo "ERROR: libcudart.so not found in $(CUDA_LIB)"; \
		echo "Please check your CUDA installation"; \
		exit 1; \
	fi
	@echo "Using CUDA installation at: $(CUDA_HOME)"

$(TARGET): $(OBJDIR) $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)
	@echo "Built standalone NCCL Inspector: $@"

$(OBJDIR)/%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/version.o: version.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

version.cc:
	@echo "// Auto-generated version info" > version.cc
	@echo "#ifdef __cplusplus" >> version.cc
	@echo "extern \"C\" {" >> version.cc
	@echo "#endif" >> version.cc
	@echo "const char* get_git_version_info() {" >> version.cc
	@if command -v git >/dev/null 2>&1; then \
		echo "  return \"$(shell git describe --always --dirty 2>/dev/null || echo 'unknown')\"; " >> version.cc; \
	else \
		echo "  return \"standalone-build\"; " >> version.cc; \
	fi
	@echo "}" >> version.cc
	@echo "#ifdef __cplusplus" >> version.cc
	@echo "}" >> version.cc
	@echo "#endif" >> version.cc

$(OBJDIR):
	mkdir -p $(OBJDIR)


clean:
	rm -rf $(OBJDIR) $(TARGET) version.cc

# Help target
help:
	@echo "Standalone NCCL Inspector Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build the inspector library (default)"
	@echo "  clean   - Clean build artifacts"
	@echo "  help    - Show this help"
	@echo "  test-ring - Run the host-only completed-ring unit test"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1                      - Enable debug build"
	@echo "  ASAN=1                       - Enable address sanitizer"
	@echo "  UBSAN=1                      - Enable undefined behavior sanitizer"
	@echo "  NCCL_INSPECTOR_ENABLE_WARN=1 - Emit inspector messages at WARN level (default: INFO)"
	@echo ""
	@echo "Environment Variables:"
	@echo "  CUDA_HOME - Path to CUDA installation (default: /usr/local/cuda)"
	@echo "  CXX       - C++ compiler (default: g++)"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make DEBUG=1"
	@echo "  make CUDA_HOME=/path/to/cuda"
