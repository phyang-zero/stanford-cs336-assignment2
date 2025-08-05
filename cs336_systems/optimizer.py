import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Type, Any

class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.all_params = list(params)
        
        # 创建一个从参数ID到其全局索引的映射
        self.param_to_global_idx = {id(p): i for i, p in enumerate(self.all_params)}
        
        # 获取分布式信息
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        # 每个rank只负责一部分参数
        self.param_shard = self.all_params[self.rank::self.world_size]
        
        # 本地优化器只管理分片后的参数
        self.local_optimizer = optimizer_cls(self.param_shard, **kwargs)
        
        # 使用本地优化器的param_groups来初始化父类，以兼容PyTorch的接口
        super().__init__(self.local_optimizer.param_groups, self.local_optimizer.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # 1. 本地优化器执行一步，只更新本地分片内的参数
        loss = self.local_optimizer.step(closure)

        if self.world_size > 1:
            # 2. 将更新后的参数同步到所有进程
            for param in self.all_params:
                param_idx = self.param_to_global_idx[id(param)]
                owner_rank = param_idx % self.world_size
                dist.broadcast(param.data, src=owner_rank)
        
        return loss