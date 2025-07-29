import torch
import timeit
import numpy as np
import argparse
from cs336_basics.model import BasicsTransformerLM

def run_benchmark(model_size, context_length, num_warmup, num_measure, measure_backward):
    """
    运行基准测试的主函数
    """
    # 1. 设置设备为GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("警告：没有检测到GPU，将在CPU上运行。计时结果可能不准确。")

    # 2. 根据模型尺寸，从Table 1中获取超参数
    configs = {
        "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
        "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }
    model_params = configs[model_size]
    
    # 3. 初始化模型
    #    词汇表大小固定为10000
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=model_params['d_model'],
        num_layers=model_params['num_layers'],
        num_heads=model_params['num_heads'],
        d_ff=model_params['d_ff'],
        rope_theta=10000.0
    ).to(device)
    model.eval()

    # 4. 生成随机数据
    #    批处理大小固定为4
    batch_size = 4
    inputs = torch.randint(0, model.vocab_size, (batch_size, context_length), device=device)

    # 5. 热身阶段
    print(f"正在进行 {num_warmup} 次热身...")
    for _ in range(num_warmup):
        if measure_backward:
            logits = model(inputs)
            targets = torch.randint(0, model.vocab_size, (batch_size, context_length), device=device)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            model.zero_grad()
        else:
            with torch.no_grad():
                model(inputs)

    torch.cuda.synchronize(device)

    # 6. 测量阶段
    timings = []
    print(f"正在进行 {num_measure} 次测量...")
    for _ in range(num_measure):
        torch.cuda.synchronize(device)
        start_time = timeit.default_timer()

        if measure_backward:
            logits = model(inputs)
            targets = torch.randint(0, model.vocab_size, (batch_size, context_length), device=device)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            model.zero_grad()
        else:
            with torch.no_grad():
                model(inputs)

        torch.cuda.synchronize(device)
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)

    # 7. 报告结果
    avg_time = np.mean(timings)
    std_dev = np.std(timings)
    print("-" * 50)
    print(f"模型尺寸: {model_size}, 上下文长度: {context_length}")
    print(f"测量模式: {'Forward+Backward' if measure_backward else 'Forward Only'}")
    print(f"平均时间: {avg_time * 1000:.3f} ms")
    print(f"标准差: {std_dev * 1000:.3f} ms")
    print("-" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer Model Benchmark")
    parser.add_argument('--model_size', type=str, default='xl', 
                        choices=['small', 'medium', 'large', 'xl', '2.7B'], 
                        help='Size of the model to benchmark')
    parser.add_argument('--context_length', type=int, default=512, 
                        help='Context length for the input sequence')
    parser.add_argument('--num_warmup', type=int, default=5, 
                        help='Number of warmup steps')
    parser.add_argument('--num_measure', type=int, default=10, 
                        help='Number of measurement steps')
    parser.add_argument('--mode', type=str, default='all', choices=['forward', 'all'],
                        help='Benchmark mode: "forward" for forward pass only, "all" for forward+backward')
    
    args = parser.parse_args()
    
    run_benchmark(
        model_size=args.model_size, 
        context_length=args.context_length,
        num_warmup=args.num_warmup, 
        num_measure=args.num_measure, 
        measure_backward=(args.mode == 'all')
    )