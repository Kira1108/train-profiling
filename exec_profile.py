import logging
logging.basicConfig(level = logging.INFO)
import torch
from dataclasses import dataclass
from torch.amp import autocast
from torch.utils import benchmark
import typer
import time
import psutil
import cpuinfo


app = typer.Typer()

@app.command()
def showcpu():
    # Get CPU info using cpuinfo
    cpu_info = cpuinfo.get_cpu_info()
    cpu_name = cpu_info['brand_raw']
    
    # Get the number of physical cores
    physical_cores = psutil.cpu_count(logical=False)
    
    # Get the maximum frequency
    max_freq = psutil.cpu_freq().max

    logging.info(f"CPU: {cpu_name}")
    logging.info(f"CPU Cores: {physical_cores}")
    logging.info(f"CPU Max Frequency: {max_freq} MHz")

@dataclass
class TensorSize:

    t:torch.Tensor

    @property
    def nbytes(self):
        return self.t.element_size() * self.t.nelement()

    @property
    def ngb(self):
        return self.nbytes / 1e9

    def __repr__(self):
        return f"<TensorSize: {self.nbytes:.2f} Bytes, {self.ngb:.2f} GB>"

    def __str__(self):
        return f"<TensorSize: {self.nbytes:.2f} Bytes, {self.ngb:.2f} GB>"

@app.command()
def tflops(
        dimension:int = 8192,
        fmt:str = 'fp16',
        warmup_steps:int = 10,
        profile_steps:int = 1000):

    """
    Test peak TFLOPS performance of a GPU with a simple matrix multiplication operation.\n
    
    \n\nArguments:\n
    dimension: int - The dimension of the square matrix.\n
    fmt: str - The data type to use. One of ['fp16', 'fp32']\n
    warmup_steps: int - The number of warmup steps, default is 10.\n
    profile_steps: int - The number of profile steps, default is 100.\n
    """
    
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine.")
    
    print("Cuda is available, start testing.")
    N = dimension

    dtype_config = {
        "fp16":torch.float16,
        "fp32":torch.float32
    }

    if not fmt in dtype_config:
        raise ValueError(f"fmt should be one of {list(dtype_config.keys())}")

    dtype = dtype_config[fmt]

    logging.info(f"Creating tesnors of shape {str(N)} x {str(N)}")
    device = torch.device("cuda")
    A = torch.randn(N,N,dtype = dtype).to(device)
    B = torch.randn(N,N,dtype = dtype).to(device)

    logging.info(f"Tensor size- {TensorSize(A)}")
    logging.info("Start Warmup steps")
    for _ in range(warmup_steps):
        with autocast(device_type = 'cuda'):
            torch.matmul(A, B)

    logging.info("Start benchmarking")
    t = benchmark.Timer(
      stmt='A @ B',
      globals={'A': A, 'B': B})

    x = t.timeit(profile_steps)
    elapsed_time = x.median

    num_operations = 2 * N**3
    tflops = num_operations / elapsed_time / 1e12

    logging.info(f"Result: TFLOPS: {tflops:.2f}")
    
    return tflops

@app.command()
def c2g():
    "Test transfer speed from CPU to GPU"
    
    # Ensure CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")  # CUDA device object
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        raise ValueError("CUDA is not available. Please check your GPU support and CUDA drivers.")

    for i in range(10):
        # Create a large tensor to test transfer speed
        # Note: Adjust size to fit your GPU memory
        size = 1024 * 1024 * 100  # 100MB
        tensor = torch.empty(size, dtype=torch.float32).cpu()  # Initially on CPU

        # Fill with some data (optional, but helps avoid certain optimizations)
        torch.randn(size, dtype=torch.float32, out=tensor)

        # Measure CPU to GPU transfer time
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        tensor = tensor.to(device)  # Transfer to GPU
        torch.cuda.synchronize()  # Wait for GPU to finish all previous commands
        end_time.record()

        # Wait for events to complete to measure time
        torch.cuda.synchronize()
        elapsed_time_ms = start_time.elapsed_time(end_time)

        logging.info(f"CPU to GPU transfer of {size / (1024**2):.2f} MB took {elapsed_time_ms:.2f} ms")
        logging.info(f"Transfer rate: {size / (1024**2) / (elapsed_time_ms / 1000):.2f} MB/s")  


@app.command()        
def g2g():
    """Test gpu to gpu transfer speed"""
    gpu0_id=0
    gpu1_id=1
    num_iterations=100
    buffer_size=1024*1024*1024
    
    tensor_gpu0 = torch.rand(buffer_size // 4, device=gpu0_id)  # 4 bytes per float32 element
    tensor_gpu1 = torch.empty_like(tensor_gpu0, device=gpu1_id)

    # Warm-up phase: perform 10 initial transfers
    for _ in range(10):
        tensor_gpu1.copy_(tensor_gpu0)

    # Synchronize GPUs
    torch.cuda.synchronize(gpu0_id)
    torch.cuda.synchronize(gpu1_id)

    # Measure transfer time
    start_time = time.time()

    # Perform the transfer num_iterations times
    for _ in range(num_iterations):
        tensor_gpu1.copy_(tensor_gpu0)

    # Record end time
    end_time = time.time()

    # Synchronize GPUs again
    torch.cuda.synchronize(gpu0_id)
    torch.cuda.synchronize(gpu1_id)

    # Calculate average transfer time and throughput
    average_time = (end_time - start_time) / num_iterations
    throughput = buffer_size / average_time / (1024 * 1024 * 1024)  # GB/s

    logging.info(f"GPU {gpu0_id} to GPU {gpu1_id} transfer speed: {throughput:.2f} GB/s")
    logging.info(f"Average transfer time per iteration: {average_time * 1000:.2f} ms")
        

@app.command()
def hello():
    "Simple health check hello command."
    logging.info("Hi there")
    
if __name__ == "__main__":
    app()