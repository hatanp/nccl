#include "inspector_proxy_stats.h"

#include <assert.h>
#include <string>

int main() {
  inspectorProxyStepTimeline send;
  send.isSend = true;
  send.startUsecs = 90;
  send.states[0] = 100;
  send.states[1] = 130;
  send.states[2] = 180;
  send.stopUsecs = 260;
  inspectorProxyWaitDurations sendDurations =
    inspectorProxyComputeWaitDurations(send);
  assert(sendDurations.usecs[inspectorProxySendGpuWait] == 30);
  assert(sendDurations.usecs[inspectorProxySendPeerWait] == 50);
  assert(sendDurations.usecs[inspectorProxySendWait] == 80);
  assert(sendDurations.validMask == 0x7);

  inspectorProxyStepTimeline recv;
  recv.isSend = false;
  recv.states[0] = 200;
  recv.states[1] = 250;
  recv.states[2] = 270;
  recv.stopUsecs = 300;
  inspectorProxyWaitDurations recvDurations =
    inspectorProxyComputeWaitDurations(recv);
  assert(recvDurations.usecs[inspectorProxyRecvWait] == 50);
  assert(recvDurations.usecs[inspectorProxyRecvFlushWait] == 20);
  assert(recvDurations.usecs[inspectorProxyRecvGpuWait] == 30);
  assert(recvDurations.validMask == (0x7u << 3));

  inspectorProxyStepTimeline partial;
  partial.isSend = true;
  partial.states[0] = 100;
  partial.states[1] = 0;
  partial.states[2] = 150;
  partial.stopUsecs = 140;
  inspectorProxyWaitDurations partialDurations =
    inspectorProxyComputeWaitDurations(partial);
  assert(partialDurations.validMask == 0);

  assert(std::string(inspectorProxyWaitPhaseName(inspectorProxyRecvFlushWait))
         == "recv_flush_wait");
  return 0;
}
