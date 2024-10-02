import os
import subprocess
import time

def get_gpu_processes():
    """
    使用 nvidia-smi 获取当前 GPU 使用的进程信息，返回包含 GPU 占用信息的字典。
    """
    try:
        # 使用 nvidia-smi 查看GPU的使用情况
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, text=True)
        # 将输出结果解析为字典
        gpu_processes = {}
        for line in result.stdout.strip().split("\n"):
            if line:
                pid, gpu_id = line.split(', ')
                gpu_processes[int(pid)] = gpu_id
        return gpu_processes
    except Exception as e:
        print(f"获取GPU使用情况失败: {e}")
        return {}

def is_process_running(pid):
    """
    使用 ps 命令检测指定 PID 是否仍在运行。
    """
    try:
        # 使用 ps 检查进程状态
        result = subprocess.run(['ps', '-p', str(pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 如果返回码为 0，表示 ps 成功找到了该进程
        if result.returncode == 0:
            return True
        else:
            return False
    except Exception as e:
        print(f"检测进程时出错: {e}")
        return False

def wait_for_gpu_release(target_pid):
    """
    检测指定 PID 的进程是否还在占用 GPU，若该进程结束，则开始执行任务。
    """
    print(f"开始监控 PID {target_pid} 的 GPU 占用情况...")
    
    while True:
        gpu_processes = get_gpu_processes()
        
        # 检查目标进程是否在GPU使用列表中
        if target_pid not in gpu_processes:
            print(f"PID {target_pid} 已经释放 GPU 资源，开始执行任务...")
            break
        
        # 如果进程仍然在运行，等待一段时间
        if is_process_running(target_pid):
            print(f"PID {target_pid} 仍在运行，等待进程结束...")
            time.sleep(10)  # 等待10秒后再次检查
        else:
            print(f"PID {target_pid} 进程已结束，GPU 资源可能已经释放，准备开始任务...")
            break

    # 在 GPU 资源释放后，执行你的代码
    run_your_task()

def run_your_task():
    """
    在 GPU 资源释放后，执行你的主要任务。
    """
    print("开始执行你的任务...")
    # 这里插入你的主要代码
    # Example:
    # model = train_model()  # 替换为你的训练模型或其他任务
    time.sleep(5)  # 模拟任务运行
    print("任务完成！")

# 例子: 监控 PID 12345，任务退出后开始执行
if __name__ == "__main__":
    target_pid = 2536674  # 替换为你要监控的 PID
    wait_for_gpu_release(target_pid)