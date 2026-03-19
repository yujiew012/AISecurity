"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
#根据globals()获取当前定义的所有变量，是这几种类型int, float, bool, str并且不能以-开头
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
#执行configurator文件，并解析且更新全局变量
exec(open('configurator.py').read()) # overrides from command line or config file
#根据globals取出更新值，形成config字典
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
#DDP分布式数据并行初始化
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])#当前进程的全程rank
    ddp_local_rank = int(os.environ['LOCAL_RANK'])#当前进程在本机上的GPU编号
    ddp_world_size = int(os.environ['WORLD_SIZE'])#总进程数
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)#将当前进程绑定到指定的 GPU
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    #梯度累积步数必须能被 GPU 总数（DDP总进程数）整除，否则程序直接报错终止
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
#：计算每次迭代（即一次参数更新）实际处理的 token 总数，用于日志和效率监控。
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
#仅主进程创建目录
if master_process:
    os.makedirs(out_dir, exist_ok=True)
#设置随机种子
torch.manual_seed(1337 + seed_offset)
#开启TF32，精度较好且速度快
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
#映射数据类型
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
#创建混合精度上下文（cpu不支持混合精度）
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 简易版数据加载器
#从巨大的二进制文件（train.bin/val.bin）中随机抽取一小批连续的数据，作为模型的输入（x）和预测目标（y）
data_dir = os.path.join('data', dataset)
def get_batch(split):
    #np.memmap是numpy的内存映射文件，不把整个文件加载到内存，而是只映射文件地址，读取时才加载对应部分，适合超大数据集
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    #在0~len(data) - block_size长度范围内取batch_size个
    ix = torch.randint(len(data) - block_size, (batch_size,))
    #unit16转为int16转为pytorch参数类型
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        #pin_memory:普通CPU内存是可分页的（系统可能把内存换到硬盘），锁定后内存不会被换出，GPU 可以直接 DMA（直接内存访问）读取，无需CPU中转，大幅提升数据传输速度
        #non_blocking=True：数据传输和GPU计算并行（比如GPU在做前一轮的前向传播时,CPU异步把下一轮的数据传到 GPU），隐藏数据传输的耗时，提升训练吞吐量
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
#先在这里初始化这些变量，如果是从检查点恢复训练（init_from='resume'），可以覆盖这些值
iter_num = 0#初始化迭代数
best_val_loss = 1e9#初始化最佳验证损失

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    #读取词汇表大小
    #若文件不存在，None，后续模型初始化时，会采用GPT-2的默认值
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)#解包model_args创建配置对象
    model = GPT(gptconf)#用配置创建全新对象
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    #断点恢复训练
    #拼接断点路径
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    #取出断点保存的模型配置
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # 创建和原来模型配置一样的模型
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    #去除断点模型权重字典
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'#ddp训练时加的模型权重前缀
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            #删除原来的并且更新
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    #恢复最佳迭代数和最佳验证损失
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    #从GPT-2预训练权重初始化
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # 定义要覆盖的参数
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# 若上下文长度小则裁剪并且更新（可减小不可增大）
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

#初始化混合精度梯度缩放器（仅float16时进行，避免开销大）
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 初始化优化器，weight_decay:权重衰减，AdamW优化器动量参数（b1,b2）
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
#恢复优化器状态
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# 编译模型
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model#保存原始未编译模型
    model = torch.compile(model) # requires PyTorch 2.0

# 多卡训练时封装模型，实现数据并行和梯度同步，充分利用多 GPU 算力。
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches

@torch.no_grad()#禁用梯度计算（评估损失仅前向传播，不用反向传播节省内存 ）
def estimate_loss():
    out = {}
    model.eval()#切换模式到评估状态
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)#评估eval_iters个批次，创建长度为eval_iter的一维全0张量
        for k in range(eval_iters):
            X, Y = get_batch(split)#获取该数据集的一个批次
            with ctx:#混合精度上下文
                logits, loss = model(X, Y)#计算损失
            losses[k] = loss.item()#取出
        out[split] = losses.mean()#平均
    model.train()#切换状态
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):#学习率调度器
    # 1) 线性热身（当前迭代数<热身迭代数）
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)#+1避免it等于0时学习率为0
    # 2) 衰减结束
    if it > lr_decay_iters:
        return min_lr
    # 3) 余弦衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# 仅主进程时初始化日志
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
#获取未经分布式包装的原始模型对象
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0#初始化MFU滑动平均值
while True:#无限循环

    # decay_lr不等于0就在衰减期，衰减器就调用get_lr,否则采用初始Learning_rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr#保证所有参数使用相同学习率

    # 评估损失+保存检查点
    #每eval_interval评估一次
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        #保存最优模型/强制保存
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:#除去0，未训练时
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:#为评估模式时
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    #梯度累积，将一个「逻辑大批次」拆分为N个「物理微批次」（micro step），每个微批次计算梯度但不更新权重，累积N个微批次的梯度后再统一更新权重
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            #仅在N个时，反向传播时同步梯度
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:#混合精度上下文
            logits, loss = model(X, Y)#前向传播
            loss = loss / gradient_accumulation_steps #缩放损失
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        #单步训练耗时从「GPU 耗时 + CPU 耗时」→ 「max (GPU 耗时，CPU 耗时)
        X, Y = get_batch('train')#异步预取下一个批次
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # 梯度裁剪
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)#反缩放梯度（将参数梯度控制在grad_clip内）
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)#更新优化器
    scaler.update()#更新缩放因子
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)#清空梯度

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    #仅主进程并且到达间隔数
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # 去除前五轮的，避免不稳定，计算模型浮点利用率
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()#销毁进程组
