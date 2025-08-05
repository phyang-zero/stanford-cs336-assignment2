import torch
import torch.nn as nn
import torch.distributed as dist

class DDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module):
        """
        DDP封装类的构造函数。
        """
        super().__init__()
        
        # 1. 保存模型和获取分布式信息
        self.module = module
        # 检查分布式环境是否已初始化
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

            # 2. 同步初始模型权重
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
        else:
            # 在非分布式环境（例如，简单的单元测试）中运行时，提供默认值
            self.rank = 0
            self.world_size = 1
            
        # 3. 为每个参数注册一个“钩子”（hook）
        self.handles = []
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param):
        """
        这是被自动调用的钩子函数。
        它的核心任务是发起一个“异步”的梯度同步操作。
        """
        if param.grad is None or self.world_size <= 1:
            return

        # a. 对梯度进行原地（in-place）的 all-reduce 求和操作
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)

    def forward(self, *args, **kwargs):
        """
        前向传播很简单，直接调用被封装的模型的forward方法。
        """
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        这个方法在 loss.backward() 之后，optimizer.step() 之前调用。
        它的作用是等待所有异步的梯度同步操作全部完成。
        """
        if self.world_size <= 1:
            return

        for handle in self.handles:
            handle.wait()
        
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= self.world_size

        self.handles.clear()

class DDPBucketed(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        if not dist.is_initialized():
            self.rank = 0
            self.world_size = 1
        else:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # 同步初始权重
        for param in self.module.parameters():
            if dist.is_initialized():
                dist.broadcast(param.data, src=0)

        # --- 核心的分桶逻辑 ---
        self.buckets = []
        self.param_to_bucket_index = {}
        
        # 使用id来处理权重绑定的情况，确保每个参数只被分桶一次
        seen_param_ids = set()
        
        # 反向遍历参数进行分桶
        params_in_reverse = list(self.module.parameters())[::-1]
        
        current_bucket = []
        current_bucket_size = 0
        bucket_size_bytes = bucket_size_mb * 1024 * 1024

        for param in params_in_reverse:
            if param.requires_grad and id(param) not in seen_param_ids:
                param_size = param.numel() * param.element_size()
                if current_bucket_size + param_size > bucket_size_bytes and current_bucket:
                    self.buckets.append(current_bucket)
                    current_bucket = []
                    current_bucket_size = 0
                
                current_bucket.append(param)
                current_bucket_size += param_size
                seen_param_ids.add(id(param))
        
        if current_bucket:
            self.buckets.append(current_bucket)

        # 建立从参数ID到桶索引的映射
        for i, bucket in enumerate(self.buckets):
            for param in bucket:
                self.param_to_bucket_index[id(param)] = i

        # 为每个桶准备状态跟踪
        self.bucket_grad_ready_counts = [0] * len(self.buckets)
        self.handles = []

        # 为每个参数注册钩子
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param):
        if param.grad is None or self.world_size <= 1:
            return

        # 使用参数ID来查找桶
        bucket_index = self.param_to_bucket_index[id(param)]
        bucket = self.buckets[bucket_index]
        
        self.bucket_grad_ready_counts[bucket_index] += 1

        if self.bucket_grad_ready_counts[bucket_index] == len(bucket):
            grads_to_reduce = [p.grad for p in bucket]
            flat_grads = torch._utils._flatten_dense_tensors(grads_to_reduce)
            
            handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, flat_grads, bucket))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if self.world_size <= 1:
            return

        for handle, flat_grads, bucket in self.handles:
            handle.wait()
            flat_grads /= self.world_size
            
            unflattened_grads = torch._utils._unflatten_dense_tensors(flat_grads, bucket)
            for param, grad in zip(bucket, unflattened_grads):
                param.grad = grad

        self.handles.clear()
        self.bucket_grad_ready_counts = [0] * len(self.buckets)