import torch
import math

import triton
import triton.language as tl

class FlashAttentionPytorchImpl(torch.autograd.Function):
    """
    FlashAttention-2 前向传播的纯PyTorch参考实现。
    """
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, N_q, d_head = Q.shape
        _, N_k, _ = K.shape
        B_q, B_k = 64, 64
        T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)
        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, N_q), device=Q.device)
        for b in range(batch_size):
            Q_b, K_b, V_b = Q[b], K[b], V[b]
            for i in range(T_q):
                q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
                current_B_q = q_end - q_start
                Q_i = Q_b[q_start:q_end, :]
                O_i = torch.zeros((current_B_q, d_head), device=Q.device, dtype=torch.float32)
                l_i = torch.zeros(current_B_q, device=Q.device, dtype=torch.float32)
                m_i = torch.full((current_B_q,), -float('inf'), device=Q.device, dtype=torch.float32)
                for j in range(T_k):
                    k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)
                    K_j, V_j = K_b[k_start:k_end, :], V_b[k_start:k_end, :]
                    scale = d_head**-0.5
                    S_ij = (Q_i @ K_j.T) * scale
                    if is_causal:
                        q_indices = torch.arange(q_start, q_end, device=Q.device)
                        k_indices = torch.arange(k_start, k_end, device=K.device)
                        causal_mask = q_indices[:, None] >= k_indices[None, :]
                        S_ij = torch.where(causal_mask, S_ij, -float('inf'))
                    m_i_new = torch.maximum(m_i, torch.max(S_ij, dim=1).values)
                    P_ij_tilde = torch.exp(S_ij - m_i_new.unsqueeze(1))
                    exp_m_diff = torch.exp(m_i - m_i_new)
                    l_i = exp_m_diff * l_i + torch.sum(P_ij_tilde, dim=1)
                    O_i = torch.diag(exp_m_diff) @ O_i + P_ij_tilde @ V_j
                    m_i = m_i_new
                O_i_normalized = torch.diag(1.0 / l_i) @ O_i
                O[b, q_start:q_end, :] = O_i_normalized.to(Q.dtype)
                L[b, q_start:q_end] = m_i + torch.log(l_i)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_O):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = torch.zeros_like(Q), torch.zeros_like(K), torch.zeros_like(V)
        batch_size, N_q, d_head = Q.shape
        _, N_k, _ = K.shape
        scale = d_head**-0.5
        D = torch.sum(grad_O * O, dim=-1)
        B_q, B_k = 64, 64
        T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)
        for b in range(batch_size):
            for i in range(T_q):
                q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
                Q_i, dO_i, L_i, D_i = Q[b, q_start:q_end, :], grad_O[b, q_start:q_end, :], L[b, q_start:q_end], D[b, q_start:q_end]
                for j in range(T_k):
                    k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)
                    K_j, V_j = K[b, k_start:k_end, :], V[b, k_start:k_end, :]
                    S_ij = (Q_i @ K_j.T) * scale
                    if is_causal:
                        q_indices = torch.arange(q_start, q_end, device=Q.device)
                        k_indices = torch.arange(k_start, k_end, device=K.device)
                        causal_mask = q_indices[:, None] >= k_indices[None, :]
                        S_ij = torch.where(causal_mask, S_ij, -float('inf'))
                    P_ij = torch.exp(S_ij - L_i.unsqueeze(1))
                    dV[b, k_start:k_end, :] += P_ij.T @ dO_i
                    dP_ij = dO_i @ V_j.T
                    dS_ij = P_ij * (dP_ij - D_i.unsqueeze(1))
                    dQ[b, q_start:q_end, :] += (dS_ij * scale) @ K_j
                    dK[b, k_start:k_end, :] += (dS_ij * scale).T @ Q_i
        return dQ, dK, dV, None

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_q, N_k, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    q_offset = batch_index * stride_qb + query_tile_index * Q_TILE_SIZE * stride_qq
    k_offset = batch_index * stride_kb
    v_offset = batch_index * stride_vb
    o_offset = batch_index * stride_ob + query_tile_index * Q_TILE_SIZE * stride_oq
    l_offset = batch_index * stride_lb + query_tile_index * Q_TILE_SIZE * stride_lq
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + q_offset, shape=(N_q, D), strides=(stride_qq, stride_qd),
        offsets=(0, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
    )
    K_base_ptr = K_ptr + k_offset
    V_base_ptr = V_ptr + v_offset
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + o_offset, shape=(N_q, D), strides=(stride_oq, stride_od),
        offsets=(0, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + l_offset, shape=(N_q,), strides=(stride_lq,),
        offsets=(0,), block_shape=(Q_TILE_SIZE,), order=(0,)
    )
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))
    T_k = tl.cdiv(N_k, K_TILE_SIZE)
    for j in range(T_k):
        k_tile_start = j * K_TILE_SIZE
        K_block_ptr = tl.make_block_ptr(
            base=K_base_ptr, shape=(D, N_k), strides=(stride_kd, stride_kk),
            offsets=(0, k_tile_start), block_shape=(D, K_TILE_SIZE), order=(0, 1)
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_base_ptr, shape=(N_k, D), strides=(stride_vk, stride_kd),
            offsets=(k_tile_start, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0)
        )
        K_j = tl.load(K_block_ptr, boundary_check=(1, 0))
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1))
        S_ij = tl.dot(Q_i.to(tl.float32), K_j.to(tl.float32)) * scale
        if is_causal:
            q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))
        m_i_new = tl.maximum(m_i, tl.max(S_ij, 1))
        exp_m_diff = tl.exp(m_i - m_i_new)
        P_ij_tilde = tl.exp(S_ij - m_i_new[:, None])
        l_i = l_i * exp_m_diff + tl.sum(P_ij_tilde, 1)
        o_i = o_i * exp_m_diff[:, None]
        o_i += tl.dot(P_ij_tilde.to(V_j.dtype), V_j)
        m_i = m_i_new
    O_i_normalized = o_i / l_i[:, None]
    tl.store(O_block_ptr, O_i_normalized.to(O_ptr.type.element_ty), boundary_check=(0, 1))
    L_i = m_i + tl.log(l_i)
    tl.store(L_block_ptr, L_i, boundary_check=(0,))

@torch.compile(fullgraph=True)
def _flash_attention_backward_compiled(
    Q, K, V, O, L, grad_O, is_causal,
    batch_size, N_q, N_k, d_head,
    B_q, B_k
):
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    scale = d_head**-0.5
    D = torch.sum(grad_O * O, dim=-1)
    T_q = math.ceil(N_q / B_q)
    T_k = math.ceil(N_k / B_k)
    for b in range(batch_size):
        for i in range(T_q):
            q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
            Q_i, dO_i, L_i, D_i = Q[b, q_start:q_end, :], grad_O[b, q_start:q_end, :], L[b, q_start:q_end], D[b, q_start:q_end]
            for j in range(T_k):
                k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)
                K_j, V_j = K[b, k_start:k_end, :], V[b, k_start:k_end, :]
                S_ij = (Q_i @ K_j.T) * scale
                if is_causal:
                    q_indices = torch.arange(q_start, q_end, device=Q.device)
                    k_indices = torch.arange(k_start, k_end, device=K.device)
                    causal_mask = q_indices[:, None] >= k_indices[None, :]
                    S_ij = torch.where(causal_mask, S_ij, -float('inf'))
                P_ij = torch.exp(S_ij - L_i.unsqueeze(1))
                dV[b, k_start:k_end, :] += P_ij.T @ dO_i
                dP_ij = dO_i @ V_j.T
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(1))
                dQ[b, q_start:q_end, :] += (dS_ij * scale) @ K_j
                dK[b, k_start:k_end, :] += (dS_ij * scale).T @ Q_i
    return dQ, dK, dV

class FlashAttentionTritonImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, N_q, d_head = Q.shape
        _, N_k, _ = K.shape
        scale = d_head**-0.5
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, N_q), device=Q.device, dtype=torch.float32)
        Q_TILE_SIZE, K_TILE_SIZE = 64, 64
        T_q = math.ceil(N_q / Q_TILE_SIZE)
        grid = (T_q, batch_size)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2), K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2), O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k, scale,
            D=d_head, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
        
    @staticmethod
    def backward(ctx, grad_O):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        batch_size, N_q, d_head = Q.shape
        _, N_k, _ = K.shape
        
        B_q, B_k = 64, 64
        
        dQ, dK, dV = _flash_attention_backward_compiled(
            Q, K, V, O, L, grad_O, is_causal,
            batch_size, N_q, N_k, d_head,
            B_q, B_k
        )
        
        return dQ, dK, dV, None