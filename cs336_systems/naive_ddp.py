import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    """初始化每个进程的分布式环境"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # 在单GPU上，rank 0和1都会被分配到device 0
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def main_worker(rank, world_size):
    """每个GPU进程将要执行的主函数"""
    print(f"进程 {rank} 正在初始化...")
    setup(rank, world_size)
    
    device_id = rank % torch.cuda.device_count()

    # 1. 在每个GPU上创建模型和优化器
    model = nn.Linear(10, 10).to(device_id)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 2. 同步初始模型权重
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # 3. 准备数据
    full_batch = torch.randn(16, 10).to(device_id)
    local_batch_size = full_batch.shape[0] // world_size
    local_batch = full_batch[rank * local_batch_size : (rank + 1) * local_batch_size]

    # 4. 训练循环
    for step in range(5):
        optimizer.zero_grad()
        output = model(local_batch)
        loss = output.sum()
        loss.backward()

        # 核心：朴素DDP的梯度同步
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        
        optimizer.step()
        
        if rank == 0:
            print(f"步骤 {step}: 完成梯度同步和权重更新。")

    # 验证所有模型的权重是否仍然一致
    for param in model.parameters():
        tensor_list = [torch.empty_like(param.data) for _ in range(world_size)]
        dist.all_gather(tensor_list, param.data)
        if rank == 0:
            for i in range(1, world_size):
                assert torch.allclose(tensor_list[0], tensor_list[i]), "模型权重在各进程间不一致！"
    
    if rank == 0:
        print("\n所有训练步骤完成，且所有进程的模型权重保持一致！")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(main_worker,
             args=(world_size,),
             nprocs=world_size,
             join=True)