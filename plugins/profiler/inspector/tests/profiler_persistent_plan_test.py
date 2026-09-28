"""Portable regression for core profiler plan-state propagation (no CUDA).

Compile exact source helper bodies against small host fixtures. Also compile a
negative control restoring the old mutable-planner predicate, which must fail.
Generated sources/binaries belong in the explicit --build-dir, not the repo.
"""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def function(path, signature):
    source = path.read_text()
    start = source.index(signature)
    brace = source.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


PREAMBLE = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { if (!(x)) {std::fprintf(stderr,"check failed line %d: %s\n",__LINE__,#x);std::exit(1);} } while(0)
using ncclResult_t=int;
constexpr int ncclSuccess=0,ncclFuncRecv=1,MAX_PROFILER_EVENTS_PER_CHANNEL=64;
constexpr int ncclProxyOpReady=1,ncclProxyOpProgress=2,ncclProxyOpNone=3;
#define NCCLCHECK(expr) do {auto result=(expr);if(result!=ncclSuccess)return result;} while(0)
struct ncclProxyConnector {};
struct ncclDevProfiler {struct Entry {uint64_t timestamp=0,counter=0;} data[64];};
struct ncclProxyOp {bool incWorkCounter=true;int channelId=0,coll=0;uint64_t workCounter=0;uint8_t *sendbuff=nullptr,*recvbuff=nullptr;};
struct ncclComm {struct {bool persistent=false;} planner;struct {uint64_t workCounter[1]={0};ncclProxyConnector recvProxyConn[1],sendProxyConn[1];ncclDevProfiler *workStarted,*workCompleted;} profiler;};
static std::vector<ncclProxyOp> queued;
static ncclResult_t ncclLocalOpAppend(ncclComm*,ncclProxyConnector*,ncclProxyOp* op) {queued.push_back(*op);return 0;}
struct ncclProxySubArgs {uint64_t base=0,workCounter=0,posted=0,transmitted=0,nsteps=1;int channelId=0;void *sendbuff=nullptr,*recvbuff=nullptr;};
struct ncclProxyArgs {int state=ncclProxyOpReady,nsubs=1,done=0;ncclProxySubArgs subs[1];};
struct ncclProxyState {};
static uint64_t observedStart,observedStop;
static ncclResult_t ncclProfilerStartKernelChEvent(ncclProxyArgs*,int,uint64_t value) {observedStart=value;return 0;}
static ncclResult_t ncclProfilerStopKernelChEvent(ncclProxyArgs*,int,uint64_t value) {observedStop=value;return 0;}
static bool enabled=true;
static bool ncclProfilerNeedsProxy(ncclComm*,ncclProxyOp*) {return enabled;}
'''
TESTS = r'''
struct Plan {bool persistent;ncclProxyOp op;};
struct Fixture {
 ncclComm comm;ncclDevProfiler started[1],stopped[1];uint64_t gpuCounter=0;
 Fixture(){comm.profiler.workStarted=started;comm.profiler.workCompleted=stopped;}
 Plan schedule(bool persistent){
  comm.planner.persistent=persistent;Plan plan{persistent,{}};bool needed=false;
  CHECK(dispatch(&comm,&plan.op,plan.persistent,&needed)==0 && needed);return plan;
 }
 void execute(Plan& plan){
  const bool before=comm.planner.persistent;
  CHECK(dispatch(&comm,&plan.op,plan.persistent,nullptr)==0);
  CHECK(comm.planner.persistent==before); // no shared-planner mutation
  const auto uploaded=queued.back();++gpuCounter;auto i=gpuCounter%64;
  started[0].data[i].counter=stopped[0].data[i].counter=gpuCounter;
  started[0].data[i].timestamp=gpuCounter*1000;
  stopped[0].data[i].timestamp=gpuCounter*1000+100;
  ncclProxyArgs args;args.subs[0].workCounter=uploaded.workCounter;
  args.subs[0].sendbuff=uploaded.sendbuff;args.subs[0].recvbuff=uploaded.recvbuff;
  observedStart=observedStop=0;
  CHECK(profilerProxyProgress(nullptr,&args)==0);
  CHECK(profilerProxyProgress(nullptr,&args)==0);
  CHECK(args.state==ncclProxyOpNone);
  CHECK(uploaded.workCounter==gpuCounter);
  CHECK(observedStart==gpuCounter*1000 && observedStop==gpuCounter*1000+100);
 }
};
int main(){
 Fixture graph;auto captured=graph.schedule(true);
 CHECK(graph.comm.profiler.workCounter[0]==0);
 for(int i=0;i<130;i++)graph.execute(captured);
 Fixture eager;
 for(int i=0;i<130;i++){auto plan=eager.schedule(false);eager.execute(plan);}
 Fixture mixed;auto replay=mixed.schedule(true);mixed.execute(replay);
 auto once=mixed.schedule(false);mixed.execute(once);
 for(int i=0;i<130;i++)mixed.execute(replay); // multiple64-slot wraps, fresh every time
 Fixture separate;auto plan=separate.schedule(true);separate.execute(plan);
 Fixture other;auto unrelated=other.schedule(false);other.execute(unrelated);
 separate.execute(plan);
 // Eager plan prepared before capture must not double-increment at late upload.
 Fixture late;auto pending=late.schedule(false);(void)late.schedule(true);late.execute(pending);
 // Non-incrementing companions copy the current counter without advancing it.
 ncclProxyOp companion;companion.incWorkCounter=false;
 auto counter=late.comm.profiler.workCounter[0];
 CHECK(dispatch(&late.comm,&companion,true,nullptr)==0);
 CHECK(companion.workCounter==counter && late.comm.profiler.workCounter[0]==counter);
 // Keep the no-kernel-profiler fallback dispatch unchanged.
 enabled=false;ncclProxyOp disabled;bool needed=false;
 CHECK(dispatch(&late.comm,&disabled,true,&needed)==0);
 CHECK(late.comm.profiler.workCounter[0]==counter+1);
 std::puts("PASS graph/eager/mixed-same-comm/different-comm/late-eager/companion/disabled paths");
}
'''


def run(build_dir, compiler):
    build_dir.mkdir(parents=True, exist_ok=True)
    enqueue = ROOT / 'src/enqueue.cc'
    for name, expected in (
        ('ncclResult_t ncclAddProxyOpIfNeeded(', 'ncclProxySaveOp(comm, op, plan->persistent, &needed)'),
        ('static ncclResult_t uploadProxyOps(', 'ncclProxySaveOp(comm, op, plan->persistent, nullptr)'),
    ):
        assert expected in function(enqueue, name), 'caller lost immutable plan state'
    declaration = (ROOT / 'src/include/proxy.h').read_text()
    assert 'struct ncclProxyOp* proxyOp, bool persistent, bool* justInquire);' in declaration
    proxy = ROOT / 'src/proxy.cc'
    public = function(proxy, 'ncclResult_t ncclProxySaveOp(')
    assert 'bool persistent' in public.split('{', 1)[0]
    case = public.split('case ncclPatternProfiler:', 1)[1].split('\n    break;', 1)[0].strip()
    assert 'SaveProxyProfiler(comm, op, persistent, justInquire)' in case
    helpers = function(proxy, 'static void incWorkCounter(') + '\n' + function(proxy, 'static ncclResult_t SaveProxyProfiler(')
    assert 'comm->planner.persistent' not in helpers
    dispatch = '\nstatic ncclResult_t dispatch(ncclComm* comm,ncclProxyOp* op,bool persistent,bool* justInquire) {\n' + case + '\nreturn ncclSuccess;\n}\n'
    progress = function(ROOT / 'src/transport/profiler.cc', 'static ncclResult_t profilerProxyProgress(')
    for negative in (False, True):
        body = helpers
        if negative:
            body = body.replace('if (!persistent)', 'if (!comm->planner.persistent)').replace('if (persistent)', 'if (comm->planner.persistent)')
        stem = 'profiler_plan_mutable_negative' if negative else 'profiler_plan_persistent'
        source = build_dir / (stem + '.cc')
        source.write_text(PREAMBLE + body + dispatch + progress + TESTS)
        binary = build_dir / stem
        # Exact production helpers contain intentionally unused fixture parameters.
        subprocess.run([compiler, '-std=c++14', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter', str(source), '-o', str(binary)], check=True)
        result = subprocess.run([str(binary)], capture_output=True, text=True)
        if negative:
            assert result.returncode == 1 and 'uploaded.workCounter==gpuCounter' in result.stderr, result.stderr
            print('PASS negative control: old mutable-planner predicate fails mixed replay freshness')
        else:
            assert result.returncode == 0, result.stderr
            print(result.stdout.strip())
    print('PASS caller/declaration/dispatch plan-state propagation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--cxx', default='c++')
    args = parser.parse_args()
    run(args.build_dir, args.cxx)
