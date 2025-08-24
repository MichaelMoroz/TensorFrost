import numpy as np
import TensorFrost as tf

ctx = tf.VulkanContext()

N = 1024
aBuf = tf.createBuffer(ctx, N, 4, True)
bBuf = tf.createBuffer(ctx, N, 4, True)
outBuf = tf.createBuffer(ctx, N, 4, False)

# write inputs via mapped views
a_map = ctx.device.mapMemory(aBuf.memory, 0, aBuf.size)
b_map = ctx.device.mapMemory(bBuf.memory, 0, bBuf.size)
np.frombuffer(a_map, dtype=np.float32)[:] = np.arange(N, dtype=np.float32)
np.frombuffer(b_map, dtype=np.float32)[:] = 2 * np.arange(N, dtype=np.float32)
ctx.device.unmapMemory(aBuf.memory)
ctx.device.unmapMemory(bBuf.memory)

code = r'''
[[vk::binding(2,0)]] RWStructuredBuffer<float> C;
[[vk::binding(0,0)]] StructuredBuffer<float>   A;
[[vk::binding(1,0)]] StructuredBuffer<float>   B;

[shader("compute")] [numthreads(64,1,1)]
void computeMain(uint3 tid: SV_DispatchThreadID) {
    C[tid.x] = 2.0f*A[tid.x] + B[tid.x];
}
'''

prog = tf.createComputeProgramFromSlang(
    ctx, "vecadd", code, "computeMain",
    [aBuf, bBuf], [outBuf]
)

tf.runProgram(ctx, prog, N)

# read back
out_map = ctx.device.mapMemory(outBuf.memory, 0, outBuf.size)
out = np.frombuffer(out_map, dtype=np.float32).copy()
ctx.device.unmapMemory(outBuf.memory)

ok = np.allclose(out, 4 * np.arange(N, dtype=np.float32))
print("Compute result is", "correct" if ok else "incorrect")

tf.destroyComputeProgram(ctx, prog)
tf.destroyBuffer(ctx, aBuf)
tf.destroyBuffer(ctx, bBuf)
tf.destroyBuffer(ctx, outBuf)