
随着[LLM](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=LLM&zhida_source=entity)的普及，对现在[kernel](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=kernel&zhida_source=entity)开发人员的要求已经不再是能写一张卡上的代码，而是需要能写一个集群内的kernel，这也是我们做[Triton-distributed](https://link.zhihu.com/?target=https%3A//github.com/ByteDance-Seed/Triton-distributed/)的初衷。为此，需要掌握一些[分布式通信](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E5%88%86%E5%B8%83%E5%BC%8F%E9%80%9A%E4%BF%A1&zhida_source=entity)基础知识，原本学习这类知识需要更长的时间，只有专业的硕士/博士生深耕多年才能对相关优化得心应手，但借助现在旗舰的LLM产品，可以快速获取相关知识，本文是通过让[gemini](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=gemini&zhida_source=entity)解读NV的174页RDMA文档[RDMA Aware Networks Programming User Manual](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/networking/display/rdmaawareprogrammingv17)做的笔记（如果LLM在某些知识上骗了我，请路过的专家们指出）

## RDMA, InfiniBand, [RoCE](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=RoCE&zhida_source=entity)基本概念

RDMA (Remote Direct Memory Access)技术本质上是允许一个设备直接访问另一个远端设备的内存，这个过程无需**目标机器**的[CPU](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=CPU&zhida_source=entity)/操作系统/host 内存参与。注意，发起通信的源头机器，可能是需要CPU/操作系统/host 内存的。RDMA的核心优势在于[zero-copy](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=zero-copy&zhida_source=entity)，cpu offload, [kernel bypass](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=kernel+bypass&zhida_source=entity)。

RDMA是一种技术理念，InfiniBand和RoCE是理念的具体实现，它们俩是[网络协议](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E7%BD%91%E7%BB%9C%E5%8D%8F%E8%AE%AE&zhida_source=entity)和架构。RoCE是基于[以太网](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E4%BB%A5%E5%A4%AA%E7%BD%91&zhida_source=entity)的网线/交换机做通信介质、基于[以太网协议](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E4%BB%A5%E5%A4%AA%E7%BD%91%E5%8D%8F%E8%AE%AE&zhida_source=entity)构建的。InfiniBand是自定义的专用网线/交换机和协议。一般来说RoCE比InfiniBand更通用，但是延迟大一些。

RDMA支持多种服务类型，包括

| 类型 | 描述 | 适用 |
| ----- | ----- | ----- |
| Reliable Connected（RC） | 可靠、有序、点对点 | 最常用 |
| Unreliable Connected（UC） | 不可靠、有序、会丢包 |  |
| Reliable Datagram（RD） | 可靠、无序 | multicast |
| Unreliable Datagram（UD） | 不可靠、无序 | 路由管理 |

NVSHMEM默认的[IBRC](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=IBRC&zhida_source=entity)，意思就是用[IB协议](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=IB%E5%8D%8F%E8%AE%AE&zhida_source=entity)的RC。

## RDMA核心组件

在RDMA理念中，核心的两个抽象是HCA（host channel adapter，也就是[网卡](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E7%BD%91%E5%8D%A1&zhida_source=entity)）和QP（队列对）。其中HCA指的是硬件，QP则是软件。HCA内部通过硬件独立处理[网络协议栈](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E7%BD%91%E7%BB%9C%E5%8D%8F%E8%AE%AE%E6%A0%88&zhida_source=entity)，内存注册，数据包发送和接收。HCA可以通过[PCIe总线](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=PCIe%E6%80%BB%E7%BA%BF&zhida_source=entity)连接到host或者device的内存，应用程序通过HCA暴露的内存[映射](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E6%98%A0%E5%B0%84&zhida_source=entity)寄存器（MMIO）控制HCA。

QP这个软件概念，一般是由两部分组成，SQ和RQ（send queue和receive queue）。通信之前，机器之间通过某些建联机制建立起QP，交换各自的QP信息，SQ和RQ的关系并非是想象那样从SQ出发的数据进入对方的RQ，这两个queue之间没有直接的数据交流，而是通过实际传输的网络包构建的逻辑关系。发送方对自己的SQ放入信息，告诉网卡要发送的数据在哪里，网卡把数据打包发出去。接收方则是把自己要接收的数据信息放在RQ告诉网卡，网卡会预留内存等待数据到达，这个就是双边通信。[单边通信](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E5%8D%95%E8%BE%B9%E9%80%9A%E4%BF%A1&zhida_source=entity)则是不需要接收方在RQ中放WR。

应用程序和HCA之间沟通的结构称为WR（work request），而HCA实际操控硬件单元完成各种操作，需要的是更贴近硬件的[数据结构](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84&zhida_source=entity)，称为[WQE](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=WQE&zhida_source=entity)（work queue entry）。

## RDMA和host/device的交互

RDMA虽然想要bypass CPU，但是不能完全不和CPU交互。和CPU交互主要有两个机制：反馈和内存注册。反馈通过CQ（completion queue）完成。SQ/RQ上的WQE被完成了，HCA都会生成一个CQE（completion queue element）放在CQ中，用于记录哪个QP上的操作完成了。应用程序想要知道自己的通信是否完成，就要不断地poll CQ查看是否有新的CQE（CPU自旋在这个等待上）。HCA也支持[中断事件](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E4%B8%AD%E6%96%AD%E4%BA%8B%E4%BB%B6&zhida_source=entity)，在CQ出现新的CQE的时候就去中断一下CPU（CPU之前可以异步作别的事情）。两种方法各有优缺点，polling方式浪费CPU，但是获取CQE延迟最小。[中断方式](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E4%B8%AD%E6%96%AD%E6%96%B9%E5%BC%8F&zhida_source=entity)解放CPU，但是获取CQE延迟大。

RDMA需要能访问[host内存](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=host%E5%86%85%E5%AD%98&zhida_source=entity)，内存注册（memory registration, MR）提供了安全的访问机制。注册通过HCA驱动完成，这块内存需要物理上被锁住（禁止换页），这块内存对应生成多个TPT（translation and protection table）条目，记录[虚拟地址](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E8%99%9A%E6%8B%9F%E5%9C%B0%E5%9D%80&zhida_source=entity)、物理地址、[访问权限](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=%E8%AE%BF%E9%97%AE%E6%9D%83%E9%99%90&zhida_source=entity)。注册后，会生成一个Memory Key返回给应用程序。Key有两个：LKey（local key）和RKey（remote key）,LKey是自己访问自己的内存，需要放在WR里的，RKey是远端访问本地内存需要提供的key。

对于device来说，也可以直接由HCA注册内存，这样通信就无需把数据从device内存搬运到host内存。比如NV提供的[GPUDirect](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=GPUDirect&zhida_source=entity)支持，就是一项允许HCA直接访问GPU内存的技术。

## RDMA编程基础

通信时，资源的创建必须按照顺序：

| 顺序 | 资源名称 | 作用 | API/概念 |
| ----- | ----- | ----- | ----- |
| 1 | 上下文 (Context) | 表示 HCA 本身，是所有操作的锚点。 | ibv_context (通过 ibv_open_device 获取) |
| 2 | 保护域 (PD) | 内存安全沙箱，所有内存和队列资源必须绑定到 PD。 | ibv_alloc_pd |
| 3 | 完成队列 (CQ) | HCA 用来报告事件的队列。 | ibv_create_cq |
| 4 | 内存区域 (MR) | 注册 GPU 或主机内存，实现零拷贝。 | ibv_reg_mr |
| 5 | 队列对 (QP) | 双向通信通道 (SQ + RQ)。 | ibv_create_qp |

下面的代码作为一个例子，介绍如何创建这些资源

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <infiniband/verbs.h>

// 宏定义：队列大小
#define CQ_SIZE 10 // 完成队列深度
#define QP_SIZE 10 // 队列对深度 (SQ + RQ)

// 结构体：存储本地 RDMA 资源
struct RDMA_Resources {
    struct ibv_context *context; // HCA/RNIC 上下文
    struct ibv_pd *pd;           // 保护域
    struct ibv_cq *cq;           // 完成队列
    struct ibv_qp *qp;           // 队列对
    struct ibv_mr *mr;           // 内存区域 (Memory Region) 句柄
    char *buffer;                // 指向 Host 内存缓冲区的指针
    size_t buf_size;             // 缓冲区大小
};

// 全局资源对象
struct RDMA_Resources res;

#define BUF_SIZE 1024 // 缓冲区大小 1KB

int reg_memory() {
    // 1. 分配 Host 内存
    res.buf_size = BUF_SIZE;
    // 使用 aligned_alloc 以确保内存对齐，尽管 RDMA 不强制要求，但这是一个好的实践
    if (posix_memalign((void **)&res.buffer, sysconf(_SC_PAGESIZE), res.buf_size)) {
        fprintf(stderr, "错误: 无法分配对齐的主机内存。\n");
        return 1;
    }

    // 简单填充缓冲区，便于验证
    snprintf(res.buffer, res.buf_size, "Hello, RDMA World! Data from Host.");
    printf("✅ 主机缓冲区分配成功，大小: %zu 字节。\n", res.buf_size);

    // 2. 注册内存区域 (MR)
    // ibv_reg_mr(pd, addr, length, access_flags)
    // IBV_ACCESS_LOCAL_WRITE: 允许 HCA DMA 写入本地内存 (用于接收)
    // IBV_ACCESS_REMOTE_WRITE: 允许远程机器 RDMA 写入此内存
    // IBV_ACCESS_REMOTE_READ: 允许远程机器 RDMA 读取此内存
    int access_flags = IBV_ACCESS_LOCAL_WRITE |
                       IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ;

    res.mr = ibv_reg_mr(res.pd, res.buffer, res.buf_size, access_flags);
    if (!res.mr) {
        fprintf(stderr, "错误: 无法注册内存区域 (MR)。\n");
        // 注册失败，需要释放已分配的内存
        free(res.buffer);
        return 1;
    }

    printf("✅ 内存区域注册成功。\n");
    printf("   LKey (本地键): 0x%x\n", res.mr->lkey);
    printf("   RKey (远程键): 0x%x\n", res.mr->rkey);
    printf("   Buffer: %s\n", res.buffer);
    
    return 0;
}

int init_resources(const char *dev_name) {
    struct ibv_device **dev_list = NULL;
    struct ibv_device *ib_dev = NULL;
    int num_devices = 0;
    int i;

    // 1. 获取 HCA 设备列表
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices <= 0) {
        fprintf(stderr, "错误: 未找到任何 InfiniBand/RDMA 设备。\n");
        return 1;
    }

    // 查找指定设备 (如果 dev_name 为 NULL, 默认使用第一个)
    if (dev_name) {
        for (i = 0; i < num_devices; ++i) {
            if (!strcmp(ibv_get_device_name(dev_list[i]), dev_name)) {
                ib_dev = dev_list[i];
                break;
            }
        }
    } else {
        ib_dev = dev_list[0];
    }

    if (!ib_dev) {
        fprintf(stderr, "错误: 未找到设备 %s\n", dev_name ? dev_name : "第一个设备");
        ibv_free_device_list(dev_list);
        return 1;
    }

    // 2. 打开 HCA 设备并创建上下文 (Context)
    res.context = ibv_open_device(ib_dev);
    ibv_free_device_list(dev_list); // 列表已不再需要
    if (!res.context) {
        fprintf(stderr, "错误: 无法打开设备上下文。\n");
        return 1;
    }

    // 3. 分配保护域 (PD)
    res.pd = ibv_alloc_pd(res.context);
    if (!res.pd) {
        fprintf(stderr, "错误: 无法分配保护域。\n");
        return 1;
    }

    // 4. 创建完成队列 (CQ)
    // ibv_create_cq(context, cqe, cq_context, channel, comp_vector)
    // cqe: 完成队列深度，必须大于 (SQ_SIZE + RQ_SIZE)
    res.cq = ibv_create_cq(res.context, CQ_SIZE, NULL, NULL, 0);
    if (!res.cq) {
        fprintf(stderr, "错误: 无法创建完成队列。\n");
        return 1;
    }

    // 5. 创建队列对 (QP)
    struct ibv_qp_init_attr qp_init_attr = {
        .qp_context = NULL,
        .send_cq = res.cq, // 发送和接收都使用同一个 CQ
        .recv_cq = res.cq,
        .cap = {
            .max_send_wr  = QP_SIZE, // SQ 深度
            .max_recv_wr  = QP_SIZE, // RQ 深度
            .max_send_sge = 1,       // 每次操作的最大 Scatter/Gather 元素数
            .max_recv_sge = 1
        },
        .qp_type = IBV_QPT_RC // 使用 Reliable Connected (RC) 类型
    };

    res.qp = ibv_create_qp(res.pd, &qp_init_attr);
    if (!res.qp) {
        fprintf(stderr, "错误: 无法创建队列对 (QP)。\n");
        return 1;
    }

    printf("✅ 资源初始化成功: Context, PD, CQ, QP 已创建。\n");

    reg_memory();
    return 0;
}

void destroy_resources() {
    // 1. 销毁 MR 和释放缓冲区
    if (res.mr) {
        ibv_dereg_mr(res.mr);
        printf("🧹 MR 已注销。\n");
    }
    if (res.buffer) {
        free(res.buffer);
        printf("🧹 Host 缓冲区已释放。\n");
    }
    if (res.qp) ibv_destroy_qp(res.qp);
    if (res.cq) ibv_destroy_cq(res.cq);
    if (res.pd) ibv_dealloc_pd(res.pd);
    if (res.context) ibv_close_device(res.context);
    printf("🧹 所有 RDMA 资源已清理。\n");
}


int main(int argc, char *argv[]) {
    // 您可以通过命令行参数指定 HCA 设备名称，例如 mlx5_0
    const char *device_name = (argc > 1) ? argv[1] : NULL;

    if (init_resources(device_name)) {
        fprintf(stderr, "🔴 初始化失败，程序退出。\n");
        return 1;
    }
    
    // 可以在这里插入 QP 状态转换和通信代码 (后续单元将实现)
    printf("QP 号码 (QPN): 0x%x\n", res.qp->qp_num);

    destroy_resources();
    return 0;
}
```

这段代码可以直接在本地编译运行（还没涉及到建立连接的部分）。

```console
g++ test.cc -o test -libverbs
./test
# 输出
✅ 资源初始化成功: Context, PD, CQ, QP 已创建。
✅ 主机缓冲区分配成功，大小: 1024 字节。
✅ 内存区域注册成功。
   LKey (本地键): 0x10400
   RKey (远程键): 0x10400
   Buffer: Hello, RDMA World! Data from Host.
QP 号码 (QPN): 0x9a
🧹 MR 已注销。
🧹 Host 缓冲区已释放。
🧹 所有 RDMA 资源已清理。
```

一个QP必须经历四个状态转换才能进行通信：

| 状态 | 简称 | 描述 | 关键 API |
| ----- | ----- | ----- | ----- |
| Reset | INIT | 初始/重置：QP 刚被创建时的状态。 | ibv_create_qp |
| Init | INIT | 初始化：配置了 QP 的端口、保护域和访问权限。 | ibv_modify_qp (to INIT) |
| Ready to Receive | RTR | 准备接收：接收方 QP 已经知道对端信息，并准备接收数据。 | ibv_modify_qp (to RTR) |
| Ready to Send | RTS | 准备发送：QP 已经准备好开始发送数据。 | ibv_modify_qp (to RTS) |

为了建立连接，通信双方需要交换彼此连接信息：QPN（queue pair number，qp的唯一标志符），LID（local identifier，本地RNIC的端口地址），RKey（远端机器访问本地内存的key）。这些元信息的交换，需要借助额外的通信机制（有点类似鸡生蛋、蛋生鸡的问题），而这个额外的通信，一般都会用传统的[TCP](https://zhida.zhihu.com/search?content_id=267528734&content_type=Article&match_order=1&q=TCP&zhida_source=entity)/IP socket或者RDMA CM（connection manager）

下面的例子展示QP状态转换与连接的建立与通信：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <endian.h>
#include <sys/time.h>
#include <infiniband/verbs.h>

// ========== 宏定义和常量 ==========
#define RC_PORT "18515"     // TCP 连接端口
#define BUF_SIZE 1024       // RDMA 缓冲区大小 1KB
#define CQ_DEPTH 2         // CQ 深度
#define QP_DEPTH 1          // QP (SQ/RQ) 深度

// ========== 结构体定义 ==========

// 1. 本地和远程连接信息（通过 TCP 交换）
struct comm_info {
    uint32_t qp_num;        // QP 号码
    uint16_t lid;           // 本地 LID (InfiniBand)
    uint32_t rkey;          // 远程访问密钥
    uint64_t vaddr;         // 注册内存的起始虚拟地址
    union ibv_gid gid;      // GID (RoCE 必需)
};

// 2. 本地 RDMA 资源结构体
struct RDMA_Resources {
    struct ibv_context *context; // HCA/RNIC 上下文
    struct ibv_pd *pd;           // 保护域
    struct ibv_cq *cq;           // 完成队列
    struct ibv_qp *qp;           // 队列对
    struct ibv_mr *mr;           // 内存区域句柄
    char *buffer;                // 缓冲区指针
    size_t buf_size;             // 缓冲区大小
    int sock_fd;                 // TCP Socket 文件描述符
};

// 全局资源对象
struct RDMA_Resources res;


// ========== 辅助函数（TCP/RDMA 状态转换） ==========

/**
 * @brief 完整的 TCP Socket 连接函数（客户端/服务器模式）
 */
int sock_connect(const char *server_name, const char *port) {
    struct addrinfo hints;
    struct addrinfo *res_addr = NULL;
    int sockfd = -1;
    int ret;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    if (getaddrinfo(server_name, port, &hints, &res_addr) != 0) {
        fprintf(stderr, "错误: getaddrinfo 失败。\n");
        return -1;
    }

    struct addrinfo *t;
    for (t = res_addr; t != NULL; t = t->ai_next) {
        sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sockfd < 0) continue;

        if (server_name) { // 客户端模式
            if (connect(sockfd, t->ai_addr, t->ai_addrlen) == 0) break;
            close(sockfd);
            sockfd = -1;
        } else { // 服务器模式
            int opt = 1;
            setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            if (bind(sockfd, t->ai_addr, t->ai_addrlen) == 0) break;
            close(sockfd);
            sockfd = -1;
        }
    }

    if (sockfd == -1) {
        fprintf(stderr, "错误: 无法创建或连接 Socket。\n");
        freeaddrinfo(res_addr);
        return -1;
    }
    
    freeaddrinfo(res_addr);
    
    if (!server_name) { // 服务器模式：监听并接受
        if (listen(sockfd, 1) < 0) { perror("监听失败"); close(sockfd); return -1; }
        printf("服务器: 正在监听端口 %s...\n", port);
        
        int established_sockfd = accept(sockfd, NULL, NULL);
        close(sockfd); // 关闭监听 Socket
        
        if (established_sockfd < 0) { perror("接受连接失败"); return -1; }
        printf("服务器: 客户端连接成功建立。\n");
        return established_sockfd;
    }
    
    printf("客户端: 连接成功建立。\n");
    return sockfd; // 客户端返回连接好的 Socket
}

/**
 * @brief 交换本地和远程的 RDMA 连接信息
 */
int sock_sync_data(int sock, struct comm_info *local_info, struct comm_info *remote_info) {
    struct comm_info tmp_local;
    
    // 1. 转换为网络字节序
    tmp_local.qp_num = htonl(local_info->qp_num);
    tmp_local.lid = htons(local_info->lid);
    tmp_local.rkey = htonl(local_info->rkey);
    tmp_local.vaddr = htobe64(local_info->vaddr);
    // GID 已经是网络字节序，直接拷贝
    memcpy(&tmp_local.gid, &local_info->gid, sizeof(union ibv_gid));

    // 2. 发送本地信息
    if (send(sock, &tmp_local, sizeof(tmp_local), 0) != sizeof(tmp_local)) {
        fprintf(stderr, "错误: 发送本地信息失败。\n");
        return 1;
    }

    // 3. 接收远程信息
    if (recv(sock, remote_info, sizeof(*remote_info), MSG_WAITALL) != sizeof(*remote_info)) {
        fprintf(stderr, "错误: 接收远程信息失败。\n");
        return 1;
    }
    
    // 4. 转换为本地字节序 (GID 保持原样)
    remote_info->qp_num = ntohl(remote_info->qp_num);
    remote_info->lid = ntohs(remote_info->lid);
    remote_info->rkey = ntohl(remote_info->rkey);
    remote_info->vaddr = be64toh(remote_info->vaddr);

    return 0;
}

/**
 * @brief 查询本地 HCA 端口的 LID
 */
static uint16_t get_local_lid(struct ibv_context *context) {
    struct ibv_port_attr port_attr;
    // 默认使用端口 1
    if (ibv_query_port(context, 1, &port_attr)) { 
        fprintf(stderr, "错误: 无法查询端口属性。\n");
        return 0;
    }
    return port_attr.lid;
}

/**
 * @brief 查询本地 HCA 端口的 GID (RoCE 必需)
 */
static int get_local_gid(struct ibv_context *context, union ibv_gid *gid) {
    // 使用端口 1，GID 索引 0 (通常是 RoCE v1) 或索引 1 (RoCE v2)
    // 尝试索引 0
    if (ibv_query_gid(context, 1, 0, gid)) {
        fprintf(stderr, "错误: 无法查询 GID。\n");
        return 1;
    }
    return 0;
}

/**
 * @brief 将 QP 状态转换到 INIT
 */
static int modify_qp_to_init(struct ibv_qp *qp) {
    struct ibv_qp_attr attr;
    int flags;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state        = IBV_QPS_INIT;
    attr.port_num        = 1;
    attr.pkey_index      = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | 
                           IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ;

    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    if (ibv_modify_qp(qp, &attr, flags)) {
        fprintf(stderr, "错误: 无法将 QP 转换到 INIT 状态。\n");
        return 1;
    }
    printf("  [QP] 状态已转换为 INIT。\n");
    return 0;
}

/**
 * @brief 将 QP 状态转换到 RTR (Ready to Receive)
 * @note 支持 InfiniBand (LID) 和 RoCE (GID)
 */
static int modify_qp_to_rtr(struct ibv_qp *qp, uint16_t remote_lid, uint32_t remote_qpn, union ibv_gid *remote_gid) {
    struct ibv_qp_attr attr;
    int flags;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state           = IBV_QPS_RTR;
    attr.path_mtu           = IBV_MTU_1024; // 使用 1KB MTU (更保守)
    attr.dest_qp_num        = remote_qpn;
    attr.rq_psn             = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer      = 12; // RNR NAK 超时

    attr.ah_attr.dlid       = remote_lid;
    attr.ah_attr.sl         = 0;
    attr.ah_attr.port_num   = 1;
    
    // RoCE 需要设置 GRH (Global Routing Header)
    attr.ah_attr.is_global  = 1;
    attr.ah_attr.grh.dgid   = *remote_gid;
    attr.ah_attr.grh.sgid_index = 0;  // 本地 GID 索引
    attr.ah_attr.grh.hop_limit  = 64;
    attr.ah_attr.grh.traffic_class = 0;

    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    if (ibv_modify_qp(qp, &attr, flags)) {
        fprintf(stderr, "错误: 无法将 QP 转换到 RTR 状态。errno: %d (%s)\n", errno, strerror(errno));
        return 1;
    }
    printf("  [QP] 状态已转换为 RTR (Remote QPN: 0x%x)。\n", remote_qpn);
    return 0;
}

/**
 * @brief 将 QP 状态转换到 RTS (Ready to Send)
 */
static int modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr;
    int flags;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state       = IBV_QPS_RTS;
    attr.timeout        = 14;  // 传输超时
    attr.retry_cnt      = 7;   // 重试次数
    attr.rnr_retry      = 7;   // RNR 重试
    attr.sq_psn         = 0;   // SQ PSN
    attr.max_rd_atomic  = 1;   // RC QP 需要设置此属性

    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

    if (ibv_modify_qp(qp, &attr, flags)) {
        fprintf(stderr, "错误: 无法将 QP 转换到 RTS 状态。errno: %d (%s)\n", errno, strerror(errno));
        return 1;
    }
    printf("  [QP] 状态已转换为 RTS (Ready to Send)。\n");
    return 0;
}


// ========== 核心操作函数 ==========

/**
 * @brief 投递接收请求 (RR) 到 RQ
 */
static int post_receive(struct ibv_qp *qp, char *buffer, size_t length, uint32_t lkey, uint64_t wr_id) {
    struct ibv_sge list_sge;
    struct ibv_recv_wr recv_wr;
    struct ibv_recv_wr *bad_wr;

    list_sge.addr   = (uintptr_t)buffer;
    list_sge.length = (uint32_t)length;
    list_sge.lkey   = lkey;

    recv_wr.wr_id      = wr_id;
    recv_wr.next       = NULL;
    recv_wr.sg_list    = &list_sge;
    recv_wr.num_sge    = 1;

    if (ibv_post_recv(qp, &recv_wr, &bad_wr)) {
        fprintf(stderr, "错误: 投递接收请求失败。errno: %d (%s)\n", errno, strerror(errno));
        return 1;
    }
    return 0;
}

/**
 * @brief 投递发送请求 (SR) 到 SQ
 */
static int post_send(struct ibv_qp *qp, char *buffer, size_t length, uint32_t lkey, uint64_t wr_id) {
    struct ibv_sge list_sge;
    struct ibv_send_wr send_wr;
    struct ibv_send_wr *bad_wr;

    list_sge.addr   = (uintptr_t)buffer;
    list_sge.length = (uint32_t)length;
    list_sge.lkey   = lkey;

    send_wr.wr_id      = wr_id;
    send_wr.next       = NULL;
    send_wr.sg_list    = &list_sge;
    send_wr.num_sge    = 1;
    send_wr.opcode     = IBV_WR_SEND;
    // 标记 IBV_SEND_SIGNALED 确保完成后会生成 CQE
    send_wr.send_flags = IBV_SEND_SIGNALED; 

    if (ibv_post_send(qp, &send_wr, &bad_wr)) {
        fprintf(stderr, "错误: 投递发送请求失败。\n");
        return 1;
    }
    return 0;
}

/**
 * @brief 轮询 CQ 等待操作完成
 */
static int poll_completion(struct ibv_cq *cq, int expected_completions) {
    struct ibv_wc wc[CQ_DEPTH];
    int num_comp = 0;
    int ret;

    while (num_comp < expected_completions) {
        ret = ibv_poll_cq(cq, 1, wc);
        if (ret < 0) {
            fprintf(stderr, "错误: 轮询 CQ 失败。\n");
            return 1;
        }

        if (ret == 0) continue; // 没有事件，继续轮询

        if (wc[0].status != IBV_WC_SUCCESS) {
            fprintf(stderr, "错误: WR 完成失败，状态: %s\n", ibv_wc_status_str(wc[0].status));
            return 1;
        }

        num_comp += ret;
    }
    return 0;
}


// ========== 资源管理函数 ==========

/**
 * @brief 内存分配和注册
 */
static int reg_memory() {
    res.buf_size = BUF_SIZE;
    
    // 1. 分配页对齐的 Host 内存
    if (posix_memalign((void **)&res.buffer, sysconf(_SC_PAGESIZE), res.buf_size)) {
        fprintf(stderr, "错误: 无法分配对齐的主机内存。\n");
        return 1;
    }
    memset(res.buffer, 0, res.buf_size);

    // 2. 注册内存区域 (MR)
    int access_flags = IBV_ACCESS_LOCAL_WRITE |
                       IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ;

    res.mr = ibv_reg_mr(res.pd, res.buffer, res.buf_size, access_flags);
    if (!res.mr) {
        fprintf(stderr, "错误: 无法注册内存区域 (MR)。\n");
        free(res.buffer);
        return 1;
    }
    printf("  [MR] 内存注册成功。LKey: 0x%x, RKey: 0x%x\n", res.mr->lkey, res.mr->rkey);
    return 0;
}

/**
 * @brief 初始化所有 RDMA 资源 (Context, PD, CQ, QP)
 * @param dev_name 可选的设备名称，如 "mlx5_1"，为 NULL 时自动选择 InfiniBand 设备
 */
static int init_rdma_resources(const char *dev_name) {
    struct ibv_device **dev_list = NULL;
    struct ibv_device *ib_dev = NULL;
    int num_devices = 0;

    // 1. 查找设备
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices <= 0) {
        fprintf(stderr, "错误: 未找到任何 InfiniBand/RDMA 设备。\n");
        return 1;
    }

    // 2. 选择设备：优先指定设备名，其次选择 InfiniBand 设备
    if (dev_name) {
        // 按名称查找
        for (int i = 0; i < num_devices; i++) {
            if (strcmp(ibv_get_device_name(dev_list[i]), dev_name) == 0) {
                ib_dev = dev_list[i];
                break;
            }
        }
        if (!ib_dev) {
            fprintf(stderr, "错误: 未找到指定设备 '%s'。\n", dev_name);
            ibv_free_device_list(dev_list);
            return 1;
        }
    } else {
        // 自动选择：优先 InfiniBand (有有效 LID)，其次 RoCE
        for (int i = 0; i < num_devices; i++) {
            struct ibv_context *tmp_ctx = ibv_open_device(dev_list[i]);
            if (!tmp_ctx) continue;
            
            struct ibv_port_attr port_attr;
            if (ibv_query_port(tmp_ctx, 1, &port_attr) == 0) {
                // 检查是否为 InfiniBand (link_layer == IBV_LINK_LAYER_INFINIBAND)
                if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND && port_attr.lid != 0) {
                    ib_dev = dev_list[i];
                    ibv_close_device(tmp_ctx);
                    printf("   [INFO] 自动选择 InfiniBand 设备: %s (LID: %u)\n", 
                           ibv_get_device_name(ib_dev), port_attr.lid);
                    break;
                }
            }
            ibv_close_device(tmp_ctx);
        }
        // 回退到第一个设备
        if (!ib_dev) {
            ib_dev = dev_list[0];
            printf("   [INFO] 使用默认设备: %s\n", ibv_get_device_name(ib_dev));
        }
    }

    // 3. 打开设备上下文
    res.context = ibv_open_device(ib_dev);
    ibv_free_device_list(dev_list);
    if (!res.context) {
        fprintf(stderr, "错误: 无法打开设备上下文。\n");
        return 1;
    }
    printf("✅ HCA 设备上下文已打开 (%s)。\n", ibv_get_device_name(ib_dev));

    // 3. 分配保护域 (PD)
    res.pd = ibv_alloc_pd(res.context);
    if (!res.pd) {
        fprintf(stderr, "错误: 无法分配保护域。\n");
        return 1;
    }
    printf("✅ 保护域 PD 已分配。\n");

    // 4. 创建完成队列 (CQ)
    res.cq = ibv_create_cq(res.context, CQ_DEPTH, NULL, NULL, 0);
    if (!res.cq) {
        fprintf(stderr, "错误: 无法创建完成队列。\n");
        return 1;
    }
    printf("✅ 完成队列 CQ 已创建。\n");

    // 5. 创建队列对 (QP)
    struct ibv_qp_init_attr qp_init_attr = {
        .qp_context = NULL,
        .send_cq = res.cq, 
        .recv_cq = res.cq,
        .cap = {
            .max_send_wr  = QP_DEPTH,
            .max_recv_wr  = QP_DEPTH,
            .max_send_sge = 1,
            .max_recv_sge = 1
        },
        .qp_type = IBV_QPT_RC // 可靠连接模式
    };

    res.qp = ibv_create_qp(res.pd, &qp_init_attr);
    if (!res.qp) {
        fprintf(stderr, "错误: 无法创建队列对 (QP)。\n");
        return 1;
    }
    printf("✅ 队列对 QP 已创建 (QPN: 0x%x)。\n", res.qp->qp_num);

    // 6. 内存注册
    if (reg_memory()) return 1;

    return 0;
}

/**
 * @brief 清理所有资源
 */
static void destroy_resources() {
    printf("\n🧹 正在清理资源...\n");
    if (res.buffer) {
        if (res.mr) ibv_dereg_mr(res.mr);
        free(res.buffer);
    }
    if (res.qp) ibv_destroy_qp(res.qp);
    if (res.cq) ibv_destroy_cq(res.cq);
    if (res.pd) ibv_dealloc_pd(res.pd);
    if (res.context) ibv_close_device(res.context);
    if (res.sock_fd != -1) close(res.sock_fd);
    printf("🧹 所有 RDMA 资源已清理。\n");
}


// ========== 主程序逻辑 ==========

/**
 * @brief 客户端（发送方）逻辑
 */
static int run_client(const char *server_name, const char *port) {
    struct comm_info local_info, remote_info;

    // 1. 初始化 RDMA 资源
    if (init_rdma_resources(NULL)) return 1;  // NULL = 自动选择 InfiniBand 设备

    // 2. 建立 TCP 连接并交换信息
    res.sock_fd = sock_connect(server_name, port);
    if (res.sock_fd < 0) return 1;

    // 3. 填充本地信息
    local_info.qp_num = res.qp->qp_num;
    local_info.lid = get_local_lid(res.context);
    local_info.rkey = res.mr->rkey;
    local_info.vaddr = (uint64_t)res.buffer;
    if (get_local_gid(res.context, &local_info.gid)) return 1;

    if (sock_sync_data(res.sock_fd, &local_info, &remote_info)) return 1;

    // 4. QP 状态转换 INIT -> RTR -> RTS
    if (modify_qp_to_init(res.qp)) return 1;
    if (modify_qp_to_rtr(res.qp, remote_info.lid, remote_info.qp_num, &remote_info.gid)) return 1;
    if (modify_qp_to_rts(res.qp)) return 1;
    printf("✅ QP 连接建立完成。\n");
    
    // 5. 准备发送数据
    snprintf(res.buffer, res.buf_size, "Hello, RDMA Server! I am Client. (%ld bytes)", res.buf_size);
    printf("   [TX] 准备发送数据: \"%s\"\n", res.buffer);
    
    // 6. 投递发送请求
    if (post_send(res.qp, res.buffer, res.buf_size, res.mr->lkey, 0x1)) return 1;
    printf("   [TX] 发送请求已提交到 SQ。\n");

    // 7. 轮询 CQ 等待完成
    if (poll_completion(res.cq, 1)) return 1;
    printf("✅ [TX] 发送操作完成。\n");

    return 0;
}

/**
 * @brief 服务器（接收方）逻辑
 */
static int run_server(const char *port) {
    struct comm_info local_info, remote_info;

    // 1. 初始化 RDMA 资源
    if (init_rdma_resources(NULL)) return 1;  // NULL = 自动选择 InfiniBand 设备

    // 2. 建立 TCP 连接并交换信息
    res.sock_fd = sock_connect(NULL, port); // server_name = NULL, 监听模式
    if (res.sock_fd < 0) return 1;

    // 3. 填充本地信息
    local_info.qp_num = res.qp->qp_num;
    local_info.lid = get_local_lid(res.context);
    local_info.rkey = res.mr->rkey;
    local_info.vaddr = (uint64_t)res.buffer;
    if (get_local_gid(res.context, &local_info.gid)) return 1;

    if (sock_sync_data(res.sock_fd, &local_info, &remote_info)) return 1;
    
    // 4. QP 状态转换: 先转到 INIT (post_recv 需要 QP 至少处于 INIT 状态)
    if (modify_qp_to_init(res.qp)) return 1;

    // 5. 预投递接收请求 (QP 已处于 INIT 状态，可以安全投递)
    if (post_receive(res.qp, res.buffer, res.buf_size, res.mr->lkey, 0x1)) return 1;
    printf("   [RX] 接收请求已预投递到 RQ。\n");

    // 6. QP 状态转换 RTR -> RTS
    if (modify_qp_to_rtr(res.qp, remote_info.lid, remote_info.qp_num, &remote_info.gid)) return 1;
    if (modify_qp_to_rts(res.qp)) return 1;
    printf("✅ QP 连接建立完成，等待数据...\n");

    // 7. 轮询 CQ 等待接收完成 (只需要 1 个完成事件)
    if (poll_completion(res.cq, 1)) return 1;
    
    // 8. 检查接收到的数据
    printf("✅ [RX] 接收操作完成。\n");
    printf("   [RX] 接收到的数据: \"%s\"\n", res.buffer);
    
    return 0;
}

int main(int argc, char *argv[]) {
    int ret = 0;
    res.sock_fd = -1;
    
    if (argc < 2) {
        fprintf(stderr, "用法:\n");
        fprintf(stderr, "  服务器端: %s -s [本地监听 IP，通常为 0.0.0.0]\n", argv[0]);
        fprintf(stderr, "  客户端:   %s -c <服务器 IP>\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "-s") == 0) {
        const char *ip_addr = (argc > 2) ? argv[2] : NULL;
        // 服务器模式，IP 参数可以为 NULL（表示监听所有接口）
        printf("--- 启动 RDMA Server ---\n");
        ret = run_server(RC_PORT);
    } else if (strcmp(argv[1], "-c") == 0) {
        if (argc < 3) {
            fprintf(stderr, "错误: 客户端模式需要指定服务器 IP。\n");
            return 1;
        }
        printf("--- 启动 RDMA Client ---\n");
        ret = run_client(argv[2], RC_PORT);
    } else {
        fprintf(stderr, "错误: 无效参数。\n");
        ret = 1;
    }
    
    destroy_resources();
    return ret;
}
```