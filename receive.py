import socket

# 创建UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 绑定IP地址和端口（更改端口号以避免冲突）
server_address = ('localhost', 12345)
print(f'UDP接收方启动，监听 {server_address[0]}:{server_address[1]}')
sock.bind(server_address)

try:
    while True:
        # 接收数据
        data, address = sock.recvfrom(4096)
        print(f'来自 {address} 的消息: {data.decode("utf-8")}')
        
except KeyboardInterrupt:
    print('\n程序已停止')

finally:
    sock.close()