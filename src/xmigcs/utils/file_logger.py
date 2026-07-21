# Copyright (c) 2025 Xuxin @ 747302550. 保留所有权利. 未经许可，禁止复制、修改或分发
import datetime
import csv
import os
import os.path as osp
import time

class FileLogger:
    def __init__(self, dt, save_dir, variable_name="observation", flush_every_n=None):
        """
        flush_every_n: 每写入 n 条记录后强制刷新磁盘缓冲区, None自动flush
        """
        self.dt = dt
        if flush_every_n:
            self.flush_every_n = max(1, int(flush_every_n))
        else:
            self.flush_every_n = None

        # file
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_name = f"{current_time}_{variable_name}.csv"
        os.makedirs(save_dir, exist_ok=True)
        log_path = osp.join(save_dir, log_name)
        self.file = open(log_path, "w", newline="")
        self.csv_writer = csv.writer(self.file)

        self.count = 0
        self._start_time = time.perf_counter()

    def data_log(self, observation_to_log):
        if self.file.closed:
            raise RuntimeError("FileLogger 已关闭")

        rel_time = round(time.perf_counter() - self._start_time, 3)
        formatted_data = []
        formatted_data.append(rel_time)
        for x in observation_to_log:
            if isinstance(x, float):
                formatted_data.append(round(x, 3))
            else:
                formatted_data.append(x)
        self.csv_writer.writerow(formatted_data)
        self.count += 1

        if self.flush_every_n:
            if self.count % self.flush_every_n == 0:
                self.file.flush()

    def data_log_no_time(self, observation_to_log):
        "无时间戳,所有数据写在一行上"
        if self.file.closed:
            raise RuntimeError("FileLogger file is closed")

        data1 = list(observation_to_log)
        data_str = ",".join(map(str, data1))  # 将数据转换为字符串并用逗号分隔
        self.file.write(data_str + ",")  # 写入文件（追加模式，不换行）
        if (self.count % self.flush_every_n) == 0:
            self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
