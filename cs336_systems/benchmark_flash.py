import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from cs336_systems.flash_attention import FlashAttentionTritonImpl

def benchmark(implementation, seq_len, d_head, precision, is_causal):
    # 1. 准备输入数据
    # 注意：需要4维输入 (batch, head, seq_len, dim)
    # 我们的实现是3维的，所以暂时只用一个head
    batch_size = 1
    num_heads = 1
    
    q = torch.randn((batch_size, num_heads, seq_len, d_head), device='cuda', dtype=precision, requires_grad=True)
    k = torch.randn((batch_size, num_heads, seq_len, d_head), device='cuda', dtype=precision, requires_grad=True)
    v = torch.randn((batch_size, num_heads, seq_len, d_head), device='cuda', dtype=precision, requires_grad=True)
    grad_o = torch.randn_like(q)

    # 我们的实现是3维的，先去掉head维度
    q_3d, k_3d, v_3d = q.squeeze(1), k.squeeze(1), v.squeeze(1)
    grad_o_3d = grad_o.squeeze(1)

    # 2. 定义要测试的函数
    if implementation == 'triton':
        # Triton版本的前向和反向
        def fwd_bwd():
            o = FlashAttentionTritonImpl.apply(q_3d, k_3d, v_3d, is_causal)
            o.backward(grad_o_3d, retain_graph=True)
        
        def fwd():
            FlashAttentionTritonImpl.apply(q_3d, k_3d, v_3d, is_causal)

    elif implementation == 'pytorch':
        # PyTorch官方版本的前向和反向
        def fwd_bwd():
            # 使用官方内置的、高度优化的注意力函数
            o = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            o.backward(grad_o, retain_graph=True)
        
        def fwd():
            F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    else:
        raise ValueError("Unknown implementation")

    # 3. 使用 triton.testing.do_bench 进行测量
    # quantiles可用于更稳定的测量，例如取中位数
    quantiles = [0.5, 0.2, 0.8]
    
    # 测量端到端（前向+反向）
    fwd_bwd_latency = triton.testing.do_bench(fwd_bwd, quantiles=quantiles)[0]
    
    # 测量纯前向
    fwd_latency = triton.testing.do_bench(fwd, quantiles=quantiles)[0]
    
    # 反向时间 = 端到端时间 - 前向时间
    bwd_latency = fwd_bwd_latency - fwd_latency
    
    return fwd_latency, bwd_latency, fwd_bwd_latency

# --- 主循环 ---
if __name__ == "__main__":
    results = []
    
    # 遍历所有实验配置
    for seq_len in [128, 256, 512, 1024, 2048, 4096, 8192]:
        for d_head in [16, 32, 64, 128]:
            for precision in [torch.float32, torch.bfloat16]:
                for impl in ['triton', 'pytorch']:
                    # 对于大尺寸配置，使用 try-except 捕获OOM错误
                    try:
                        fwd, bwd, fwd_bwd = benchmark(impl, seq_len, d_head, precision, is_causal=True)
                        print(f"Impl: {impl:8s}, SeqLen: {seq_len:5d}, D_Head: {d_head:3d}, Prec: {str(precision)[6:]:8s}, Fwd: {fwd:.4f}ms, Bwd: {bwd:.4f}ms, Total: {fwd_bwd:.4f}ms")
                        results.append([impl, seq_len, d_head, str(precision), fwd, bwd, fwd_bwd])
                    except Exception as e:
                        print(f"Impl: {impl:8s}, SeqLen: {seq_len:5d}, D_Head: {d_head:3d}, Prec: {str(precision)[6:]:8s}, Error: OOM or other")