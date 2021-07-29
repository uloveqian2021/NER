# -*- coding:utf-8 -*-
"""
@author : 
@time   :20-7-21 上午11:18 
@IDE    :PyCharm
@document   :1.py
"""

import paramiko  # 用于调用scp命令
from scp import SCPClient
import datetime
import os
# 将指定目录的图片文件上传到服务器指定目录
# remote_path远程服务器目录
# file_path本地文件夹路径
# img_name是file_path本地文件夹路径下面的文件名称


def upload_file(remote_path="/wang/save_model/ccks2020ee/", file_path="./README.md", is_file=False):
    # img_name示例：07670ff76fc14ab496b0dd411a33ac95-6.webp

    host = ""  # 服务器ip地址
    port = 22  # 端口号
    username = ""  # ssh 用户名
    password = ""  # 密码

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(host, port, username, password)
    scpclient = SCPClient(ssh_client.get_transport(), socket_timeout=15.0)
    if not is_file:
        local_path = file_path
        try:
            scpclient.put(local_path, remote_path)
        except FileNotFoundError as e:
            print(e)
            print("can not find file" + local_path)
        else:
            print("sucessful!")
    else:
        local_path = file_path
        file_list = os.listdir(local_path)
        for files in file_list:
            fpath = local_path + files
            try:
                scpclient.put(fpath, remote_path)
            except FileNotFoundError as e:
                print(e)
                print("can not find file" + local_path)
            else:
                print("sucessful!")
    ssh_client.close()


def upload_file2(remote_path="/mnt/bd1/pubuser/wbq_data/", file_path="./"):
    # img_name示例：07670ff76fc14ab496b0dd411a33ac95-6.webp

    host = ""  # 服务器ip地址
    port = 22  # 端口号
    username = ""  # ssh 用户名
    password = ""  # 密码
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(host, port, username, password)
    # scpclient = SCPClient(ssh_client.get_transport(),socket_timeout=15.0)
    scpclient = SCPClient(ssh_client.get_transport(), socket_timeout=15.0)
    # local_path = file_path + "\\" + img_name
    local_path = file_path
    try:
        # scpclient.put(local_path, remote_path)
        scpclient.get(local_path, remote_path)
    except FileNotFoundError as e:
        print(e)
        print("can not find file" + local_path)
        pass
    else:
        print("sucessful!")
    ssh_client.close()


# upload_file(remote_path="/mnt/bd1/pubuser/wbq_data/", file_path='../data/', is_file=True)
