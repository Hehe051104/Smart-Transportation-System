import socket

# 创建UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 设置服务器地址和端口（与接收器保持一致）
server_address = ('localhost', 12345)

print(f'UDP发送方启动')
print(f'目标地址: {server_address[0]}:{server_address[1]}')

try:
    while True:
        # 获取用户输入
        message = input("请输入要发送的消息 (输入'quit'退出): ")
        
        if message.lower() == 'quit':
            break
            
        # 发送数据
        sock.sendto(message.encode('utf-8'), server_address)
        print(f'已发送: {message}')
        
except KeyboardInterrupt:
    print('\n程序已停止')
    
finally:
    sock.close()