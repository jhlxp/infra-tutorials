`libfabric` 是一套通用的高性能网络接口，它的风格与前文提到的 RDMA ibverbs 接口十分相像，但是更易用一点。应用程序只需要调用 `libfabric` 的上层接口，而具体的协议由不同的 Provider 实现。`libfabric` 官方的 Provider 包括了 `tcp`、`udp`、`shm`（共享内存）、`verbs`（即 RDMA `ibverbs`）以及对本文来说最重要的 `efa`。

## 概念及术语

`libfabric` 定义了非常多的概念和术语，并且包含了很多缩写。这里我用稍不严谨的语言概括一下。

### 软件对象模型

![libfabric 软件对象模型](images/001.png)

-   Fabric: 所有硬件资源和软件状态的集合，类似于用于保存全局状态的一个结构。
-   Domain: 类似于一张网卡，例如 `eth0`。一个 Fabric 可以拥有多个 Domain。
-   (EP) Endpoint: 数据收发的端点，一个 Domain 可以拥有多个端点，并且每个端点也有不同的类型。例如在同一张以太网卡上，可以监听多个 `ip:port`，可以使用 TCP 和 UDP 这样不同类型的协议。
-   (EQ) Event Queue：事件队列，用来报告控制平面操作的完成。在本文中没有用到。
-   (CQ) Completion Queue: 完成队列，报告数据平面操作的完成（`RECV` / `SEND` / `WRITE` / `READ`）。
-   (AV) Address Vector：存储解析过的网络地址。需要先将通信的另一方加入 AV 才能对其发起通信。
-   (MR) Memory Region：用于收发数据的缓冲区。无论是哪种 RDMA 操作，都需要指定 MR。注册 MR 需要经过操作系统内核，因为操作系统内核需要设置 CPU 页表以及其他 PCIe 设备的页表。

### 通信方式

1.  端点类型（Endpoint types）

    1.  `FI_EP_MSG`（Reliable-connected）：类似于 RDMA 中的 RC QP，基于连接的可靠传输。EFA 不支持这个类型。
    2.  `FI_EP_RDM`（Reliable-unconnected）：基于数据报的可靠传输。

        1.  包含重传，保证消息送达顺序。
        2.  可以传输任意大小的数据。可以使用单边 RDMA（One-sided RDMA，`WRITE` / `READ`）和双边 RDMA（Two-sided RDMA，`RECV` / `SEND`）。
        3.  EFA 使用亚马逊自研的 [SRD](https://aws.amazon.com/blogs/hpc/in-the-search-for-performance-theres-more-than-one-way-to-build-a-network/)（Scalable Reliable Datagram）协议，所以这是 EFA 的主力端点类型。
        4.  [verbs Provider](https://ofiwg.github.io/libfabric/v2.0.0/man/fi_verbs.7.html) 则不原生支持这一端点类型，因为 RDMA 本身并没有类似的 QP 类型。不过 `libfabric` 上可以通过 [RxM (RDM over MSG) Provider](https://ofiwg.github.io/libfabric/v2.0.0/man/fi_rxm.7.html) 来模拟这一端点类型。

    3.  `FI_EP_DGRAM`（Unreliable datagram）：类似于 RDMA 中的 UD QP，基于数据报的不可靠传输。在 EFA 上只能传输小于 MTU 的数据，并且只能使用双边 RDMA。在这里不多做介绍。

2.  端点能力（Endpoint Capability）

    1.  `FI_MSG` 双边 RDMA：`RECV` / `SEND`，见 [fi_msg(3)](https://ofiwg.github.io/libfabric/v2.0.0/man/fi_msg.3.html)
    2.  `FI_TAGGED`：类似于 `FI_MSG`，但是每条消息带有一个标记，接收方可以根据这个标记选择缓冲区。本文不多做介绍。见 [fi_tagged(3)](https://ofiwg.github.io/libfabric/v2.0.0/man/fi_tagged.3.html)
    3.  `FI_RMA` 单边 RDMA：`WRITE` / `WRITE_IMM` / `READ`，见 [fi_rma(3)](https://ofiwg.github.io/libfabric/v2.0.0/man/fi_rma.3.html)
    4.  `FI_ATOMIC` 单边 RDMA 原子操作。本文不多做介绍。见 [fi_atomic(3)](https://ofiwg.github.io/libfabric/v2.0.0/man/fi_atomic.3.html)

## 安装 `libfabric`

`libfabric` 依赖于 [rdma-core](https://github.com/linux-rdma/rdma-core) 来进行 RDMA 操作，依赖于 [GDRCopy](https://github.com/NVIDIA/gdrcopy) 来实现 GPUDirect DMA。另外 `libfabric` 提供了一套性能测试工具 `fabtests`。下面这一段脚本会在当前目录下面建立一个 `build/` 目录，下载这些库的代码到 `build/` 目录中，编译并安装这些库到 `build/` 的子目录当中。

```bash
mkdir -p build
cd build
BUILD_DIR=$(pwd)

# RDMA Core
sudo apt-get install rdma-core

# GDRCopy
wget -O gdrcopy-2.4.4.tar.gz https://github.com/NVIDIA/gdrcopy/archive/refs/tags\
/v2.4.4.tar.gz
cd gdrcopy-2.4.4/
make prefix="$BUILD_DIR/gdrcopy" \
    CUDA=/usr/local/cuda \
    -j$(nproc --all) all install
cd ..
export LD_LIBRARY_PATH="$BUILD_DIR/gdrcopy/lib:$LD_LIBRARY_PATH"

# libfabric
wget https://github.com/ofiwg/libfabric/releases/download/v2.0.0/libfabric-2.0.0.tar.bz2
tar xf libfabric-2.0.0.tar.bz2
cd libfabric-2.0.0
./configure --prefix="$BUILD_DIR/libfabric" \
    --with-cuda=/usr/local/cuda \
    --with-gdrcopy="$BUILD_DIR/gdrcopy"
make -j$(nproc --all)
make install
cd ..
export LD_LIBRARY_PATH="$BUILD_DIR/libfabric/lib:$LD_LIBRARY_PATH"

# fabtests
wget https://github.com/ofiwg/libfabric/releases/download/v2.0.0/fabtests-2.0.0.tar.bz2
tar xf fabtests-2.0.0.tar.bz2
cd fabtests-2.0.0
./configure --prefix="$BUILD_DIR/fabtests" \
    --with-cuda=/usr/local/cuda \
    --with-libfabric="$BUILD_DIR/libfabric"
make -j$(nproc --all)
make install
cd ..
```

## 运行示例程序

现在我们可以运行 `libfabric` 自带的示例程序，一来可以验证软件已经安装到位，二来可以对我们的硬件有个初步的了解。

### 获得基本的网卡信息

```bash
./build/libfabric/bin/fi_info --verbose
```

`fi_info` 会输出所有网卡的信息，我在上面贴出来了第一张 EFA 的全部信息以供参考。

```yaml
---
fi_info:
    caps: [ FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_READ, FI_WRITE, FI_RECV, FI_SEND, FI_REMOTE_READ, FI_REMOTE_WRITE, FI_MULTI_RECV, FI_LOCAL_COMM, FI_REMOTE_COMM, FI_SOURCE, FI_DIRECTED_RECV ]
    mode: [  ]
    addr_format: FI_ADDR_EFA
    src_addrlen: 32
    dest_addrlen: 0
    src_addr: fi_addr_efa://[fe80::8e7:efff:feee:e81d]:0:0
    dest_addr: (null)
    handle: (nil)
    fi_tx_attr:
        caps: [ FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_READ, FI_WRITE, FI_SEND ]
        mode: [  ]
        op_flags: [ FI_COMPLETION, FI_INJECT, FI_TRANSMIT_COMPLETE, FI_DELIVERY_COMPLETE ]
        msg_order: [ FI_ORDER_SAS, FI_ORDER_ATOMIC_RAR, FI_ORDER_ATOMIC_RAW, FI_ORDER_ATOMIC_WAR, FI_ORDER_ATOMIC_WAW ]
        inject_size: 4096
        size: 4096
        iov_limit: 4
        rma_iov_limit: 1
        tclass: 0x0
    fi_rx_attr:
        caps: [ FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_RECV, FI_REMOTE_READ, FI_REMOTE_WRITE, FI_MULTI_RECV, FI_SOURCE, FI_DIRECTED_RECV ]
        mode: [  ]
        op_flags: [ FI_MULTI_RECV, FI_COMPLETION ]
        msg_order: [ FI_ORDER_SAS, FI_ORDER_ATOMIC_RAR, FI_ORDER_ATOMIC_RAW, FI_ORDER_ATOMIC_WAR, FI_ORDER_ATOMIC_WAW ]
        size: 8192
        iov_limit: 4
    fi_ep_attr:
        type: FI_EP_RDM
        protocol: FI_PROTO_EFA
        protocol_version: 4
        max_msg_size: 18446744073709551615
        msg_prefix_size: 0
        max_order_raw_size: 8776
        max_order_war_size: 8776
        max_order_waw_size: 8776
        mem_tag_format: 0xaaaaaaaaaaaaaaaa
        tx_ctx_cnt: 1
        rx_ctx_cnt: 1
        auth_key_size: 0
    fi_domain_attr:
        domain: 0x0
        name: rdmap79s0-rdm
        threading: FI_THREAD_SAFE
        progress: FI_PROGRESS_AUTO
        resource_mgmt: FI_RM_ENABLED
        av_type: FI_AV_TABLE
        mr_mode: [ FI_MR_VIRT_ADDR, FI_MR_ALLOCATED, FI_MR_PROV_KEY, FI_MR_HMEM ]
        mr_key_size: 4
        cq_data_size: 4
        cq_cnt: 512
        ep_cnt: 256
        tx_ctx_cnt: 256
        rx_ctx_cnt: 256
        max_ep_tx_ctx: 1
        max_ep_rx_ctx: 1
        max_ep_stx_ctx: 0
        max_ep_srx_ctx: 0
        cntr_cnt: 0
        mr_iov_limit: 1
        caps: [ FI_LOCAL_COMM, FI_REMOTE_COMM ]
        mode: [  ]
        auth_key_size: 0
        max_err_data: 0
        mr_cnt: 262144
        tclass: 0x0
    fi_fabric_attr:
        name: efa
        prov_name: efa
        prov_version: 200.0
        api_version: 2.0
    nic:
        fi_device_attr:
            name: rdmap79s0
            device_id: 0xefa1
            device_version: 6
            vendor_id: 0x1d0f
            driver: efa
            firmware: 0.0.0.0
        fi_bus_attr:
            bus_type: FI_BUS_PCI
            fi_pci_attr:
                domain_id: 0
                bus_id: 79
                device_id: 0
                function_id: 0
        fi_link_attr:
            address: EFA-fe80::8e7:efff:feee:e81d
            mtu: 8760
            speed: 100000000000
            state: FI_LINK_UP
            network_type: Ethernet
---
...
```

从中我们可以看到一些关键点，在后面的几篇文章中会用到，我在下面特地摘出来：

```yaml
src_addrlen: 32                         # EFA 地址长度为 32 字节
fi_tx_attr:
    size: 4096                          # 发送队列能容纳 4096 个操作
    iov_limit: 4                        # 发送操作最多能指定 4 个本地缓冲区
    rma_iov_limit: 1                    # 发送操作只能指定一个远端缓冲区
fi_rx_attr:
    size: 8192                          # 接收队列能容纳 8192 个操作
    iov_limit: 4                        # 接收操作最多能指定 4 个本地缓冲区
fi_ep_attr:
    max_msg_size: 18446744073709551615  # 可以发送任意大的数据
fi_domain_attr:
    name: rdmap79s0-rdm                 # domain 名称
    mr_key_size: 4                      # MR 远程密钥大小为 4 字节
    cq_data_size: 4                     # WRITE_IMM 携带的立即数为 4 字节
    mr_iov_limit: 1                     # 注册MR时一次只能指定一块缓冲区
nic:
    fi_device_attr:
        name: rdmap79s0                 # 网卡名称
        driver: efa                     # EFA 网卡
    fi_link_attr:
        mtu: 8760                       # 单个数据包最大为 8760 字节
        speed: 100000000000             # 网卡的速率为 100 Gbps
```

### 带宽测试

我们还可以运行 `fabtests` 当中的带宽测试。下面的这个命令可以在两台机器之间进行 GPUDirect RDMA WRITE 操作。可以看到，最高速度达到了 11843.86 MB/sec，也就是 94.751 Gbps，几乎打满了带宽。

```text
ip-172-19-226-174$ ./build/fabtests/bin/fi_rma_bw -p efa -o write -E -D cuda -S all
ip-172-19-230-131$ ./build/fabtests/bin/fi_rma_bw -p efa -o write -E -D cuda -S all 172.19.226.174
bytes   iters   total       time     MB/sec    usec/xfer   Mxfers/sec
1       20k     19k         0.03s      0.75       1.33       0.75
2       20k     39k         0.03s      1.58       1.27       0.79
3       20k     58k         0.03s      2.36       1.27       0.79
4       20k     78k         0.03s      3.18       1.26       0.79
6       20k     117k        0.03s      4.72       1.27       0.79
8       20k     156k        0.03s      6.37       1.26       0.80
12      20k     234k        0.03s      9.48       1.27       0.79
16      20k     312k        0.03s     12.66       1.26       0.79
24      20k     468k        0.03s     19.18       1.25       0.80
32      20k     625k        0.03s     24.78       1.29       0.77
48      20k     937k        0.02s     38.75       1.24       0.81
64      20k     1.2m        0.02s     51.87       1.23       0.81
96      20k     1.8m        0.02s     77.48       1.24       0.81
128     20k     2.4m        0.02s    103.46       1.24       0.81
192     20k     3.6m        0.02s    155.55       1.23       0.81
256     20k     4.8m        0.02s    206.08       1.24       0.80
384     20k     7.3m        0.02s    307.40       1.25       0.80
512     20k     9.7m        0.02s    412.82       1.24       0.81
768     20k     14m         0.03s    614.11       1.25       0.80
1k      20k     19m         0.03s    814.12       1.26       0.80
1.5k    20k     29m         0.03s   1212.41       1.27       0.79
2k      20k     39m         0.03s   1590.49       1.29       0.78
3k      20k     58m         0.03s   2340.84       1.31       0.76
4k      20k     78m         0.03s   3068.05       1.34       0.75
6k      20k     117m        0.03s   4462.85       1.38       0.73
8k      20k     156m        0.03s   5440.12       1.51       0.66
12k     20k     234m        0.04s   6770.81       1.81       0.55
16k     20k     312m        0.04s   7921.86       2.07       0.48
24k     20k     468m        0.06s   8319.43       2.95       0.34
32k     20k     625m        0.07s   9470.25       3.46       0.29
48k     20k     937m        0.10s  10127.33       4.85       0.21
64k     2k      125m        0.01s  10465.67       6.26       0.16
96k     2k      187m        0.02s  10885.78       9.03       0.11
128k    2k      250m        0.03s  10215.66      12.83       0.08
192k    2k      375m        0.03s  11301.58      17.40       0.06
256k    2k      500m        0.05s  11501.58      22.79       0.04
384k    2k      750m        0.08s  10123.87      38.84       0.03
512k    2k      1000m       0.10s  10749.66      48.77       0.02
768k    2k      1.4g        0.13s  11843.86      66.40       0.02
1m      200     200m        0.02s  10968.94      95.60       0.01
1.5m    200     300m        0.03s  10837.99     145.12       0.01
2m      200     400m        0.04s  10204.87     205.51       0.00
3m      200     600m        0.06s  10522.41     298.95       0.00
4m      200     800m        0.08s  10826.94     387.39       0.00
6m      200     1.1g        0.11s  11488.20     547.65       0.00
8m      200     1.5g        0.16s  10504.60     798.57       0.00
```

现在我们已经验证了软硬件都已就绪，那么下一章我们就可以开始编写代码了。
