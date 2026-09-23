#include "inspector_parent_identity.h"

#include <stdio.h>

const char* inspectorParentProcessInstance() {
  // Random file-scope namespace, generated once per plugin load. Independently
  // saved outputs need no address/PID/timestamp matching. This is not a global
  // application ID. Entropy failure leaves all parent metadata unavailable.
  struct Instance {
    char hex[33] = {};
    Instance() {
      unsigned char bytes[16];
      FILE* source = fopen("/dev/urandom", "rb");
      if (source == nullptr) return;
      size_t count = fread(bytes, 1, sizeof(bytes), source);
      fclose(source);
      if (count != sizeof(bytes)) return;
      const char* digits = "0123456789abcdef";
      for (size_t i = 0; i < sizeof(bytes); i++) {
        hex[2 * i] = digits[bytes[i] >> 4];
        hex[2 * i + 1] = digits[bytes[i] & 15];
      }
    }
  };
  static const Instance instance;
  return instance.hex;
}

uint64_t inspectorParentNextId() {
  static std::atomic<uint64_t> counter{0};
  if (inspectorParentProcessInstance()[0] == '\0') return 0;
  return inspectorParentAllocateId(counter);
}
