import os
import signal
import concurrent.futures
import multiprocessing


# 任务函数
def my_task(task_id):
    try:
        # 模拟一些工作，可能会出错
        if task_id % 2 == 0:  # 模拟偶数任务出错
            raise ValueError(f"Task {task_id} encountered an error.")
        return f"Task {task_id} completed successfully."
    except Exception as e:
        return f"Task {task_id} failed: {str(e)}"


# 主函数
def main():
    batch_size = 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(my_task, i): i for i in range(10)}  # 提交任务

        for future in concurrent.futures.as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()  # 获取结果
                print(result)
            except Exception as exc:
                print(f"Task {task_id} generated an exception: {exc}")
                # 在这里可以添加你处理进程的逻辑，比如杀死进程：
                pid = os.getpid()  # 获取进程 ID
                os.kill(pid, signal.SIGKILL)  # 杀死进程
            finally:
                # 可根据需要执行进一步的清理工作
                pass


if __name__ == "__main__":
    main()
