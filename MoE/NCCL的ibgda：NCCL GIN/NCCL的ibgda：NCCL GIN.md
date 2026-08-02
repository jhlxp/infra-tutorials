# MCCL 的 IBGDA：NCCL GIN

## 1. 背景

2.28的nccl发布之后，也支持了最近很火的ibgda，nccl中称ibgda为GIN，即GPU-Initiated Networking，这节我们看下nccl是怎么做的。

## 2. 用法

```c++
int main() {
  [...]
  memset(&reqs, 0, sizeof(ncclDevCommRequirements));
  int nCTAs = 1;
  reqs.railGinBarrierCount = nCTAs;
  reqs.ginSignalCount = 1;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
  [...]
}

template <typename T>
__global__ void ginAlltoAllKernel(ncclDevComm devComm, ncclWindow_t win,
                                  size_t inputOffset, size_t outputOffset, size_t count) {
  int ginContext = 0;
  ncclGinSignal_t signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamWorld(devComm),
                                           devComm.railGinBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nThreads = blockDim.x * gridDim.x;

  const size_t size = count * sizeof(T);
  for (int peer = tid; peer < nRanks; peer += nThreads) {
    gin.put(ncclTeamWorld(devComm), peer, win, outputOffset + rank * size,
            win, inputOffset + peer * size, size, ncclGin_SignalInc{signalIndex});
  }

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + nRanks);
  gin.flush(ncclCoopCta());
}
```

相对于对称内存，多了一步ncclDevCommCreate创建device端的comm，device端构造ncclGin结构体，执行put进行数据发送，通过waitSignal等待所有数据的接收，最后执行flush，保证当前rank执行的put对应的源数据buffer可以被重新使用，不保证数据在对端落盘，实现上用的pollcq。

## 3. Host 端初始化

### 3.1 创建 Context

```c++
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
	NCCLCHECK(ginState->ncclGin->init(&ginState->ginInstance, comm->commHash, ncclDebugLog));
	NCCLCHECK(ncclTopoGetLocalNets(comm->topo, comm->rank, localNets, &nLocalNets));
	ginState->ginCommCount = std::min<int>(NCCL_GIN_MAX_CONTEXTS, ncclParamGinNcontexts());
	ncclSpaceConstruct(&ginState->counterSpace);
  ncclSpaceConstruct(&ginState->signalSpace);
}
```

ginState存了gin相关所有信息。ncclGin的init就是获取机器所有可用的网卡，然后ncclTopoGetLocalNets获取当前gpu本地的网卡，ginCommCount表示用多少个context，gin暂时每个context都是一个qp，多qp是通过多context实现的，后续都假设为1。然后初始化counter和signal的space。

```c++
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm) {
  ...
  for (int n = 0; n < ginState->ginCommCount; n++) {
    void* listenComm;
    ginState->ncclGin->listen(ginState->ginInstance, localNets[n%nLocalNets],
                                allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank, &listenComm);
    bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE);
    ginState->ncclGin->connect(comm->ginContext, handles, comm->nRanks, comm->rank,
                                             listenComm, ginState->ginComms + n);
    ginState->ncclGin->createContext(
                    ginState->ginComms[n], ginState->signalSpaceSize, ginState->counterSpaceSize,
                    &ginState->ginCtx[n], &ginState->ginDevHandles[n]);
    NCCLCHECKGOTO(ginState->ncclGin->closeListen(listenComm), ret, fail);
    ...
  }
```

在创建context之前，首先要初始化ncclGinIbCollComm，即ginState->ginComms，就是gin的bootstrap网络，用于之后的元信息交换。 ncclGin->connect就是bootstrap网络的建链，建链用的就是ncclNetIb的方法，即之前介绍过的rdma建链，最后得到的bootstrap网络中组成改了一个环形网络，通过sendComm发消息给rank + 1，通过recvComm接收rank - 1的消息，使用这个环形网络可以执行allgather。进一步的，还建立了一个全连接的网络，即用于发送的fullSendComm和用于接收的fullRecvComm，用于执行all2all。

#### 3.1.1 Counter 与 Signal

在deepep中，通过put\_nbi接口将数据发送过去，然后通过atomic通知对端rank数据已经完成发送了。对于这种场景，gin抽象出来了counter和signal，signal就是deepep这个用法，用于通知对端rank，表示数据已经发送过去了。counter同样的作用，只不过是作用于当前rank，因此后续可以看到，在当前rank创建到其他peer的qp时，一个peer对应两个qp，称为qp\_main和qp\_companion，qp\_main用于向其他rank发送数据和signal，qp\_companion为self-loop qp，用于通知当前rank，然后我们先看下ncclGinGdakiCreateContext中关于signal和counter的操作。因为signal和counter流程一致，所以只看下counter。

```c++
ncclResult_t ncclGinGdakiCreateContext(void *collComm, int nSignals, int nCounters,
                                       void **outGinCtx, ncclNetDeviceHandle_v11_t **outDevHandle) {
	const int num_counters = nCounters;
  GdakiGlobalGPUBufferTable<uint64_t> *counters_table =
   new GdakiGlobalGPUBufferTable<uint64_t>(num_counters, nranks);
  NCCLCHECKGOTO(counters_table->register_mr(gdaki_ctx->ib_pd, true), status, out);
  NCCLCHECKGOTO(counters_table->exchange_info(cComm), status, out);
}
```

num\_counters由环境变量指定，默认是65536，GdakiGlobalGPUBufferTable维护了一块gpu内存，支持内存的注册和rkey的全局allgather，如下所示。

```c++
template <typename T>
class GdakiGlobalGPUBufferTable {
  CUmemGenericAllocationHandle cumemhandle;
  GdakiHostGPUMemHandle<__be32> rkeys_hd_mhandle;
  T *gpu_ptr;
  struct ibv_mr *mr;

  ncclResult_t allocate(unsigned int num_elements, unsigned int num_ranks) {
    NCCLCHECK(ncclCuMemAlloc((void **)&this->gpu_ptr, &this->cumemhandle, CU_MEM_HANDLE_TYPE_NONE,
                             num_elements * sizeof(T)));
    NCCLCHECK(this->rkeys_hd_mhandle.allocate(num_ranks));
  }
  GdakiGlobalGPUBufferTable()
    : gpu_ptr(nullptr), mr(nullptr), cumemhandle(nullptr), num_elements(0), next_unused_idx(0){};
  GdakiGlobalGPUBufferTable(unsigned int num_elements, unsigned int num_ranks) {
    this->allocate(num_elements, num_ranks);
  };
```

num\_elements就是num\_counters，表示counters\_table的大小，首先分配counters\_table到gpu\_ptr。rkeys\_hd\_mhandle用于保存所有rank gpu\_ptr的rkey，类型为GdakiHostGPUMemHandle，维护了一个hostbuf和devbuf，用于host和device的同步，可以理解为一个长度为nranks的数组，然后通过register\_mr注册gpu\_ptr。

```c++
static ncclResult_t gdakiRegMr(struct ibv_mr **mr, struct ibv_pd *pd, void *addr, size_t length,
                               int access, bool force_strict_ordering = false) {
  if (!force_strict_ordering && gdakiRelaxedOrderingEnabled())
    access |= IBV_ACCESS_RELAXED_ORDERING;
  NOWARN(status = gdakiRegMrDmaBuf(mr, pd, addr, length, access), NCCL_NET);
  if (status == ncclSuccess) return ncclSuccess;
  NCCLCHECK(wrap_ibv_reg_mr_iova2(mr, pd, addr, length, 0, access));
  return ncclSuccess;
}
```

gdakiRegMr负责注册gpu\_ptr，首先尝试dmabuf，如果不成功则尝试peermem，不只是counters\_table会通过这个函数注册内存，上节提到的window也会用gdakiRegMr注册，可以看到这里iova传的是0，因此，后续执行数据收发时填的地址是个相对于addr的offset。

因为counter和signal会使用atomic，因此需要一块buffer存储返回值，就是sink\_buffer，但是对于返回值不会使用，因此大小为sizeof(uint64\_t)。

```c++
ncclResult_t ncclGinGdakiCreateContext(...) {
  ncclCuMemAlloc((void **)&sink_buffer, &sink_buffer_mhandle, CU_MEM_HANDLE_TYPE_NONE,
                               sizeof(uint64_t));

  gdakiRegMr(&sink_buffer_mr, gdaki_ctx->ib_pd, sink_buffer, sizeof(uint64_t),
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                             IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
}
```

#### 3.1.2 QP 和 CQ 的创建

```c++
ncclResult_t ncclGinGdakiCreateContext(...) {
  const int ncontexts = 1;
  const int nqps_per_rank = ncontexts;
  const int nqps_for_comm = nqps_per_rank * nranks;  // Number of QPs for communication
  const int ncompanion_qps = nqps_for_comm * 2;      // Number of companion QPs for communication
                                                     // Double because we connect to self.
  const int nqps =
    nqps_per_rank * (nranks + 1);  // +1 for the local rank.
                                   // The last group is the responder of the local rank.

}
```

nic\_handler表示分配uar的方式，比如用BF还是NONCACHE，默认为DOCA\_GPUNETIO\_VERBS\_NIC\_HANDLER\_AUTO。 mreg\_type表示注册umem的方式，用dmabuf还是peermem，默认为DOCA\_GPUNETIO\_VERBS\_MEM\_REG\_TYPE\_DEFAULT。

```c++
doca_error_t doca_gpu_verbs_create_qp_group_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                                               struct doca_gpu_verbs_qp_group_hl **qpg) {
    if (qp_init_attr->nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    struct doca_gpu_verbs_qp_group_hl *qpg_ =
        (struct doca_gpu_verbs_qp_group_hl *)calloc(1, sizeof(struct doca_gpu_verbs_qp_group_hl));
}
```

值得注意的是现在gin不支持使用BF。然后分配qpg\_，qpg\_包含两个qp，即qp\_main和qp\_companion。

##### 3.1.2.1 分配 UAR

```c++
static doca_error_t create_uar(struct ibv_context *ibctx,
                               enum doca_gpu_dev_verbs_nic_handler nic_handler,
                               struct doca_verbs_uar **external_uar, bool bf_supported) {
    if (nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
        status = doca_verbs_uar_create(ibctx, DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED,
                                       external_uar);
        if (status != DOCA_SUCCESS) {
            doca_verbs_uar_create(ibctx, DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE, external_uar);
            return DOCA_SUCCESS;
        } else
            return DOCA_SUCCESS;
    }
}
```

首先尝试分配NONCACHE\_DEDICATED的uar，如果分配失败，则分配普通的NONCACHE的uar。doca\_verbs\_uar\_create就是通过mlx5dv\_devx\_alloc\_uar分配uar，然后将page\_id和reg\_addr记录到doca\_verbs*uar，返回给qpg*\->qp\_main.external\_uar。

##### 3.1.2.2 创建 CQ

然后开始创建cq。

```c++
static doca_error_t create_cq(...) {
	status = doca_verbs_cq_attr_create(&verbs_cq_attr);
	status = doca_gpu_mem_alloc(gpu_dev, external_umem_size, priv_get_page_size(),
                                DOCA_GPU_MEM_TYPE_GPU, (void **)gpu_umem_dev_ptr, NULL);
  cq_ring_haddr = (doca_gpunetio_ib_mlx5_cqe64 *)(calloc(external_umem_size, sizeof(uint8_t)));
  mlx5_init_cqes(cq_ring_haddr, ncqes);
  status_cuda = cudaMemcpy((*gpu_umem_dev_ptr), (void *)(cq_ring_haddr), external_umem_size,
                           cudaMemcpyDefault);

}
```

verbs\_cq\_attr记录了cq的各种属性，比如cq的uar，buf大小等，用于最后创建cq。 接着通过doca\_gpu\_mem\_alloc分配位于gpu的cq buff，即gpu\_umem\_dev\_ptr，doca\_gpu\_mem\_alloc就是cudaMalloc。 然后开始初始化gpu\_umem\_dev\_ptr，通过calloc分配位于cpu内存的cq\_ring\_haddr，即cq buff，初始化cpu的cq buff的op\_own为invalid，然后cudaMemcpy到gpu\_umem\_dev\_ptr。

```c++
static doca_error_t create_gpu_umem(...) {
    if (mreg_type == DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT) {
        status = doca_gpu_dmabuf_fd(gpu_dev, umem_ptr, umem_sz, &dmabuf_fd);
        if (status != DOCA_SUCCESS) {
            dmabuf_fd = DOCA_VERBS_DMABUF_INVALID_FD;
        }

        status = doca_verbs_umem_create(ibctx, umem_ptr, umem_sz, IBV_ACCESS_LOCAL_WRITE, dmabuf_fd,
                                        0, umem);
        if (status != DOCA_SUCCESS) {
            if (dmabuf_fd > 0) {
                status = doca_verbs_umem_create(ibctx, umem_ptr, umem_sz, IBV_ACCESS_LOCAL_WRITE,
                                                DOCA_VERBS_DMABUF_INVALID_FD, 0, umem);
                if (status != DOCA_SUCCESS) {
                    goto destroy_resources;
                }
            } else {
                goto destroy_resources;
            }
        }
}
```

然后执行create\_gpu\_umem对gpu\_umem\_dev\_ptr注册umem，支持dmabuf和peermem，如果是default，则首先尝试dmabuf，不成功回退到peermem。 doca\_gpu\_dmabuf\_fd会通过cuMemGetHandleForAddressRange获取这段显存的fd，doca\_verbs\_umem\_create通过执行mlx5dv\_devx\_umem\_reg\_ex进行umem的注册，如果输入的dmabuf\_fd有效，那么将通过dmabuf的方式注册，否则peer\_mem。

到现在就完成了cq buf的分配，初始化和umem的注册，然后同样的流程分配了dbr显存，初始化以及umem的注册。 有了cq buff，dbr，然后通过mlx5dv\_devx\_obj*create创建cq，保存到qpg*\->qp\_main.cq\_sq，和nvshmem的不同点是gin的cq不用collapsed。

##### 3.1.2.3 创建 QP

```c++
static doca_error_t create_qp(...) {
	status = doca_verbs_qp_init_attr_create(&verbs_qp_init_attr);
  status = doca_gpu_mem_alloc(gpu_dev, external_umem_size, priv_get_page_size(),
                             DOCA_GPU_MEM_TYPE_GPU, gpu_umem_dev_ptr, NULL);
	status =
        create_gpu_umem(gpu_dev, ibpd, mreg_type, external_umem_size, *gpu_umem_dev_ptr, gpu_umem);
  // dbr
  ...
  status = doca_verbs_qp_create(ibctx, verbs_qp_init_attr, &new_qp);
}
```

然后是qp的创建，和cq差不多，先通过doca\_gpu\_mem\_alloc分配qp buf，然后注册umem，同理对dbr。将相关信息set到qp\_init\_attr，如qp\_buf，dbr，uar，cq\_sq等，然后执行doca\_verbs\_qp*create创建qp，保存到qpg*\->qp\_main.qp，和nvshmem不同的在于现在gin不使用rq。 到这里就完成了qp的创建，结构体为doca\_verbs\_qp，因此还需要转换成device格式，这个操作就是通过doca\_gpu\_verbs\_export\_qp做的。

```c++
doca_error_t doca_gpu_verbs_export_qp(struct doca_gpu *gpu_dev, struct doca_verbs_qp *qp,
                                      enum doca_gpu_dev_verbs_nic_handler nic_handler,
                                      void *gpu_qp_umem_dev_ptr, struct doca_verbs_cq *cq_sq,
                                      struct doca_gpu_verbs_qp **qp_out) {
	*qp_out = (struct doca_gpu_verbs_qp *)calloc(1, sizeof(struct doca_gpu_verbs_qp));
	(*qp_out)->qp_cpu = (doca_gpu_dev_verbs_qp *)calloc(1, sizeof(struct doca_gpu_dev_verbs_qp));
	qp_cpu_ = (*qp_out)->qp_cpu;
	doca_verbs_qp_get_wq(qp, ...);
  uint32_t *dbrec = reinterpret_cast<uint32_t *>(doca_verbs_qp_get_dbr_addr(qp));
  qp_cpu_->sq_wqe_pi = 0;
  qp_cpu_->sq_rsvd_index = 0;
  qp_cpu_->sq_ready_index = 0;
  qp_cpu_->sq_lock = 0;
  qp_cpu_->sq_dbrec = (__be32 *)(dbrec + DOCA_GPUNETIO_IB_MLX5_SND_DBR);
	status = doca_gpu_verbs_export_uar(sq_db, (uint64_t **)&(qp_cpu_->sq_db));
	qp_cpu_->nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;

  doca_verbs_cq_get_wq(cq_sq, ...);
  doca_verbs_cq_get_dbr_addr(cq_sq, &uar_db_reg, (uint32_t **)&(cq_dbrec), &arm_dbr);
  qp_cpu_->cq_sq.dbrec = (__be32 *)cq_dbrec;
}
```

分配doca\_gpu\_verbs\_qp qp\_out，qp\_out有doca\_gpu\_dev\_verbs\_qp类型的qp\_cpu和qp\_gpu，接下来要做的事情就是通过qp初始化qp\_out->qp\_cpu，之后qp\_cpu会拷贝到device上的qp\_gpu提供给kernel使用。 通过doca\_verbs\_qp\_get\_wq获取qp相关信息，赋值到qp*cpu*，doca\_gpu\_verbs\_export\_uar会通过cudaHostRegister注册uar，然后通过cudaHostGetDevicePointer获取uar的device地址，之后kernel会通过这个device地址写uar。同理对于cq。

到这里就完成qp\_main的创建了，然后开始创建qp*companion，流程完全一致，创建cq到qpg*\->qp\_companion.cq*sq，创建qp到qpg*\->qp\_companion.qp，还是通过doca\_gpu\_verbs\_export\_qp将qp*companion export到qpg*\->qp\_companion.qp\_gverbs。在一个qpg\_中， qp\_main.cq\_sq

##### 3.1.2.4 建链

```c++
ncclResult_t ncclGinGdakiCreateContext(......)
	for (int rank_idx = 0; rank_idx < nranks; rank_idx++) {
      int qp_idx = rank_idx + ctx_idx * nranks;
      gdakiFillExchInfo(&local_exch_info[rank_idx], gdaki_ctx, gdaki_ctx->gqps[qp_idx]);
    }
    cComm->allToAll(cComm, local_exch_info, remote_exch_info, sizeof(struct gdaki_exch_info)),

    for (int rank_idx = 0; rank_idx < nranks; rank_idx++) {
      int qp_idx = rank_idx + ctx_idx * nranks;
      if (rank_idx == rank)
        gdakiFillExchInfo(&remote_exch_info[rank_idx], gdaki_ctx,
                          gdaki_ctx->gqps[nqps_for_comm + ctx_idx]);
      NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->gqps[qp_idx], &remote_exch_info[rank_idx]),
                    status, out);
    }
}
```

对qp\_main进行建链，对gqps的每个qp执行gdakiFillExchInfo，将lid，qpn等信息记录到local\_exch\_info，然后执行一个all2all获取其他rank对应的qp信息，然后开始执行建链，遍历gqps，如果rank\_idx为当前rank，那么将最后一个qp，即gqps\[nqps\_for\_comm\]的信息赋值到remote\_exch\_info\[rank\_idx\]，最后通过gdakiConnectQp进行建链，即modify\_qp，这里不再赘述。

```c++
ncclResult_t ncclGinGdakiCreateContext(...) {
  for (int qp_idx = 0; qp_idx < nqps_per_rank; qp_idx++) {
    int peer_qp_idx = nqps_for_comm + qp_idx;
    struct gdaki_exch_info exch_info;
    gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->gqps[qp_idx * nqps_per_rank + rank]);
    NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->gqps[peer_qp_idx], &exch_info), status, out)
  }
}
```

相反的，还需将gqps\[nqps\_for\_comm\]和gqps\[rank\]进行建链。

```c++
ncclResult_t ncclGinGdakiCreateContext(...) {
  for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
    int peer_qp_idx = nqps_for_comm + qp_idx;
    struct gdaki_exch_info exch_info;
    gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->companion_gqps[peer_qp_idx]);
    NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->companion_gqps[qp_idx], &exch_info), status,
                  out);
    gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->companion_gqps[qp_idx]);
    NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->companion_gqps[peer_qp_idx], &exch_info),
                  status, out);
  }
}
```

然后开始对qp\_companion进行建链，qp\_companion都是self-loop的，companion\_gqps\[i\]和companion\_gqps\[i + nqps\_for\_comm \]进行建链。

### 3.2 DevComm 的创建

建链完成之后，需要将位于host内存的qp，counters\_table等信息拷贝到device。

```c++
ncclResult_t ncclGinGdakiCreateContext(..., void **outGinCtx, ncclNetDeviceHandle_v11_t **outDevHandle) {
		struct ncclGinGdakiGPUContext *gin_gdaki_gpu_ctx =
      &gin_gdaki_gpu_ctx_hd_mhandle->host_buf[ctx_idx];
    tmp_qp = (struct doca_gpu_dev_verbs_qp *)calloc(nranks, sizeof(struct doca_gpu_dev_verbs_qp));
    tmp_qp_companion = (struct doca_gpu_dev_verbs_qp *)calloc(nranks, sizeof(struct doca_gpu_dev_verbs_qp));
    for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
      struct doca_gpu_dev_verbs_qp *qp_cpu =
        gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs->qp_cpu;
      memcpy(&tmp_qp[qp_idx], qp_cpu, sizeof(struct doca_gpu_dev_verbs_qp));

      qp_cpu = gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs->qp_cpu;
      memcpy(&tmp_qp_companion[qp_idx], qp_cpu, sizeof(struct doca_gpu_dev_verbs_qp));
    }

    doca_gpu_mem_alloc(gdaki_ctx->gdev, sizeof(struct doca_gpu_dev_verbs_qp) * nranks,
                            host_page_size, DOCA_GPU_MEM_TYPE_GPU,
                            (void **)&gin_gdaki_gpu_ctx->gdqp, nullptr);

    ncclCudaMemcpy<struct doca_gpu_dev_verbs_qp>(gin_gdaki_gpu_ctx->gdqp, tmp_qp, nranks);
    doca_gpu_mem_alloc(gdaki_ctx->gdev, sizeof(struct doca_gpu_dev_verbs_qp) * nranks,
                            host_page_size, DOCA_GPU_MEM_TYPE_GPU,
                            (void **)&gin_gdaki_gpu_ctx->companion_gdqp, nullptr);
    ncclCudaMemcpy<struct doca_gpu_dev_verbs_qp>(
                    gin_gdaki_gpu_ctx->companion_gdqp, tmp_qp_companion, nranks);
    gin_gdaki_gpu_ctx->counters_table.buffer = counters_table->gpu_ptr;
    gin_gdaki_gpu_ctx->counters_table.rkeys = counters_table->get_rkeys_d();
      devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;

	  devHandle->netDeviceVersion = NCCL_GIN_GDAKI_VERSION;
	  devHandle->handle = (void *)gin_gdaki_gpu_ctx_hd_mhandle->gpu_buf;
	  *outDevHandle = devHandle;
}
```

ncclGinGdakiCreateContext返回的是outDevHandle和outGinCtx，非proxy场景下我们只关注outDevHandle。gin\_gdaki\_gpu\_ctx\_hd\_mhandle同样是个GdakiHostGPUMemHandle，维护了hostbuf和devbuf， 对于qp，将qp\_main和qp\_companion拷贝到连续的cpu空间tmp\_qp和tmp\_qp\_companion，再拷贝到device上gin\_gdaki\_gpu\_ctx的gdqp和companion\_gdqp。 对于counters\_table，将counters\_tabl的gpu\_ptr和rkeys赋值给gin\_gdaki\_gpu\_ctx。

### 3.3 对称内存的注册

上节中介绍了对称内存，因为gin只能操作对称内存，那么还需对之前执行过ncclCommWindowRegister的内存进行rdma的注册。

```c++
ncclResult_t ncclDevrCommCreateInternal(...) {
  if (ginActivated) {
    NCCLCHECKGOTO(ncclGinConnectOnce(comm), ret, fail);
    for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
      NCCLCHECKGOTO(symMemoryRegisterGin(comm, mem), ret, fail);
    }
  }
static ncclResult_t symMemoryRegisterGin(struct ncclComm* comm, struct ncclDevrMemory* mem) {
  NCCLCHECK(ncclGinRegister(comm, mem->primaryAddr, mem->size, mem->ginHostWins, mem->ginDevWins));
}
```

这里就是遍历所有对称内存对应的mem，执行rdma注册，得到的handle存到mem的ginHostWins和ginDevWins，handle就是注册得到的mr相关的信息。

```c++
ncclResult_t ncclGinGdakiRegMrSym(void *collComm, void *data, size_t size, int type, void **mhandle,
                                  void **ginHandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct gdaki_context *gdaki_ctx = (struct gdaki_context *)cComm->ginCtx;
  GdakiHostGPUMemHandle<struct ncclGinGdakiMemHandle> *gdaki_mhandle_hd_mhandle =
    new GdakiHostGPUMemHandle<struct ncclGinGdakiMemHandle>(1);

  GdakiHostGPUMemHandle<__be32> *rkeys_hd_mhandle =
    new GdakiHostGPUMemHandle<__be32>(cComm->nranks);

  NCCLCHECK(gdakiRegMr(&mr, gdaki_ctx->ib_pd, data, size,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_ATOMIC));
  rkey = htobe32(mr->rkey);
  NCCLCHECK(cComm->allGather(cComm, &rkey, rkeys_hd_mhandle->host_buf, sizeof(__be32)));
  NCCLCHECK(rkeys_hd_mhandle->copy_h_to_d());

  gdaki_mhandle_hd_mhandle->host_buf->rkeys = rkeys_hd_mhandle->gpu_buf;
  gdaki_mhandle_hd_mhandle->host_buf->lkey = htobe32(mr->lkey);
  NCCLCHECK(gdaki_mhandle_hd_mhandle->copy_h_to_d());
  *ginHandle = (void *)gdaki_mhandle_hd_mhandle->gpu_buf;
}
```

通过gdakiRegMr对data进行注册，将rkey进行allgather到rkeys\_hd\_mhandle->host\_buf，然后将lkey和rkey填充到ginHandle的gpu\_buf，然后返回ginHandle。

如上一章节所述，window记录的是userPtr和userSize，mem记录的是userPtr对应的内存块的起始位置memPtr，rdma注册的是memPtr，因此这里更新了一下window的ginOffset4K，即userPtr - memPtr，然后记录ginWins，即lkey和rkey，最后再拷贝回device。

```c++
ncclResult_t ncclDevrCommCreateInternal(...) {
  if (ginActivated) {
    for (int i=0; i < devr->winSortedCount; i++) {
      struct ncclDevrWindow* win = devr->winSorted[i].win;
      struct ncclWindow_vidmem* winHost;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, win->vidmem, &winHost), ret, fail_stream);
      winHost->ginOffset4K = (win->bigOffset - win->memory->bigOffset)>>12;
      for (int i=0; i < NCCL_GIN_MAX_CONTEXTS; i++) {
        winHost->ginWins[i] = win->memory->ginDevWins[i];
      }
      CUDACHECKGOTO(cudaMemcpyAsync(win->vidmem, winHost, sizeof(struct ncclWindow_vidmem), cudaMemcpyHostToDevice, stream), ret, fail_stream);
    }
  }
}
```

### 3.4 分配 Counter 和 Signal

在创建context的时候分配了一片counter和signal的内存，这里要根据用户传入实际的counter数量进行分配

```c++
ncclResult_t ncclDevrCommCreateInternal(...) {
		outDevComm->ginContextCount = nGinContexts;
    outDevComm->ginSignalCount = ginSignalTotal;
    outDevComm->ginCounterCount = ginCounterTotal;
    NCCLCHECKGOTO(ncclGinAllocSignalsCounters(comm,
      ginSignalTotal, &outDevComm->ginSignalBase,
      ginCounterTotal, &outDevComm->ginCounterBase
    ), ret, fail_stream_mem_win);
}
ncclResult_t ncclGinAllocSignalsCounters(struct ncclComm* comm, int nSignals, uint32_t* outSignal0,
                                         int nCounters, uint32_t* outCounter0) {
    ncclSpaceAlloc(&ginState->counterSpace, ginState->counterSpaceSize, nCounters, 1, &start);
    *outCounter0 = (uint32_t)start;
}
```

对counter的管理也是通过Space，这里通过ncclSpaceAlloc分配出nCounters个counter，首地址为outCounter0。

最后再赋值给outDevComm，就可以在device端用到gin的所有信息了。

```c++
ncclResult_t ncclDevrCommCreateInternal(...) {
    for (int ctx=0; ctx < nGinContexts; ctx++) {
      outDevComm->ginTypes[ctx] = (int)comm->sharedRes->ginState.ginDevHandles[ctx]->netDeviceType;
      outDevComm->ginHandles[ctx] = comm->sharedRes->ginState.ginDevHandles[ctx]->handle;
    }
}
```

## 4. Device 执行

### 4.1 ncclGin

ncclGin为device端的结构体，gin相关操作如put之类的都是通过ncclGin的方法执行。

```c++
using ncclGin = ncclGin_BackendMask<NCCL_GIN_BACKEND_MASK_ALL>;
template<unsigned backendMask>
struct ncclGin_BackendMask {
	ncclDevComm const& comm;
  uint32_t nContexts:8, contextId:8, _ginBackend:8;
}
```

backendMask为支持哪些ncclNetDeviceType，默认是支持GIN\_PROXY和GIN\_GDAKI，我们只关注GDAKI就好。

```c++
template<unsigned beMask>
NCCL_DEVICE_INLINE ncclGin_BackendMask<beMask>::ncclGin_BackendMask(ncclDevComm const& comm, int contextIndex):
  comm(comm) {
  this->nContexts = comm.ginContextCount;
  this->contextId = comm.ginContextCount == 3
    ? uint32_t(contextIndex)%3 // 3 is only non power of 2
    : contextIndex & (comm.ginContextCount-1); // powers of 2

  this->_ginBackend = comm.ginTypes[this->contextId];
  this->_ginHandle = comm.ginHandles[this->contextId];
  this->_signalShadows = comm.ginSignalShadows + this->contextId*comm.ginSignalCount;
}
```

构造函数就是记录下devComm，保存对应的context到\_ginHandle。

#### 4.1.1 Put

首先看下put的定义

```c++
template<
    // Action to take on peer when put completes. If a signalling action is used
    // then that signal will be visible only after the payload of this put as well as
    // the payloads of preceding puts on this netContext to the same peer are settled.
    typename RemoteAction = ncclGin_None, // one of ncclGin_{None|SignalInc|SignalAdd|SignalSet}
    // Action to take locally when source has been consumed.
    typename LocalAction = ncclGin_None, // one of ncclGin_{None|CounterInc}
    // Set of threads participating in this put. Must be a subset of Coop.
    typename Coop = ncclCoopThread,
    // Optional smem descriptor space to use. Either ncclGin_{None|DescriptorSmem}
    typename DescriptorSmem = ncclGin_None
  >
  NCCL_DEVICE_INLINE void put(
    ncclTeam, int peer,
    ncclWindow_t dstWnd, size_t dstOffset,
    ncclWindow_t srcWnd, size_t srcOffset, size_t bytes,
    RemoteAction remoteAction = ncclGin_None{},
    LocalAction localAction = ncclGin_None{},
    Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope alreadyReleased = cuda::thread_scope_thread,
    cuda::thread_scope expected_scope = cuda::thread_scope_device
  ) const;
```

当put的数据在peer可见之后，会对peer执行RemoteAction的操作，就是signal，默认是ncclGin\_None，表示啥也不做，ncclGin\_SignalAdd 会对指定的signal加上value，ncclGin\_SignalInc 会对指定signal加一。当put的数据在当前rank被消费后，会对当前rank执行LocalAction的操作，默认是啥也不做，ncclGin\_CounterInc 会对指定的counter加一。

```c++
struct ncclGin_None {};
struct ncclGin_SignalAdd { ncclGinSignal_t signal; uint64_t value; };
struct ncclGin_SignalInc { ncclGinSignal_t signal; };
struct ncclGin_CounterInc { ncclGinCounter_t counter; };
```

然后看下put的实现。

```c++
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_GDAKI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(...) {
    coop.sync();
    if (coop.thread_rank() == 0) {
      ncclGinGdakiGPUContext* gdaki = (struct ncclGinGdakiGPUContext*)ctx.handle;
      doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
      doca_gpu_dev_verbs_qp* companion_qp;
      ncclGinGdakiMemHandle* dstMh = (ncclGinGdakiMemHandle*)dstWin;
      ncclGinGdakiMemHandle* srcMh = (ncclGinGdakiMemHandle*)srcWin;
      raddr.addr = dstOff;
      raddr.key = loadConst(loadConst(&dstMh->rkeys) + peer);
      laddr.addr = srcOff, laddr.key = loadConst(&srcMh->lkey);
      ...
    }
	}
}
```

coop表示协作组，put默认用thread，即单线程。gdaki就是初始化过程中存下来的ncclGinGdakiGPUContext，存储了qp，couters\_table等信息。 dstWin和srcWin就是对称内存注册时的lkey和rkey，前边有说过，通信时用的地址都是offset，因此设置laddr和raddr为srcOff和dstOff，win中获取lkey和rkey。

```c++
template <typename Coop>
  NCCL_DEVICE_INLINE static void call(...) {
    coop.sync();
    if (coop.thread_rank() == 0) {
      ncclGinGdakiGPUContext* gdaki = (struct ncclGinGdakiGPUContext*)ctx.handle;
      doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
      doca_gpu_dev_verbs_qp* companion_qp;
      ncclGinGdakiMemHandle* dstMh = (ncclGinGdakiMemHandle*)dstWin;
      ncclGinGdakiMemHandle* srcMh = (ncclGinGdakiMemHandle*)srcWin;
      raddr.addr = dstOff;
      raddr.key = loadConst(loadConst(&dstMh->rkeys) + peer);
      laddr.addr = srcOff, laddr.key = loadConst(&srcMh->lkey);
      ...
    }
	}
template <typename Coop>
  NCCL_DEVICE_INLINE static void call(...) {
    if (coop.thread_rank() == 0) {
			      doca_gpu_dev_verbs_addr sig_raddr, sig_laddr;
      if (hasSignal) {
        if (signalOp == ncclGinSignalInc) signalOpArg = 1;
        sig_raddr.addr = sizeof(uint64_t) * signalId;
        sig_raddr.key = loadConst(loadConst(&gdaki->signals_table.rkeys) + peer);
        sig_laddr.addr = 0;
        sig_laddr.key = loadConst(&gdaki->sink_buffer_lkey);
      }

      doca_gpu_dev_verbs_addr counter_raddr, counter_laddr;
      if (hasCounter) {
        companion_qp = loadConst(&gdaki->companion_gdqp) + peer;
        counter_raddr.addr = sizeof(uint64_t) * counterId;
        counter_raddr.key = loadConst(loadConst(&gdaki->counters_table.rkeys) + ctx.rank);
        counter_laddr.addr = 0;
        counter_laddr.key = loadConst(&gdaki->sink_buffer_lkey);
      }
      doca_gpu_dev_verbs_put_signal_counter<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD>(
	      qp, raddr, laddr, bytes, sig_raddr, sig_laddr, signalOpArg, companion_qp, counter_raddr,
	      counter_laddr, 1);

    }
	}
```

然后开始获取counter和signal相关信息。对于signal，通过signals\_table获取peer的rkey，通过用户指定的signalId获取sig\_raddr，同理对于counter，返回值均为sink\_buffer。唯一不同的是counter使用的是companion\_qp，即self-loop qp。

最后通过doca\_gpu\_dev\_verbs\_put\_signal\_counter完成数据的put。对wq，cq，db的同步过程和nvshmem几乎完全一样。

```c++
__device__ static __forceinline__ void doca_gpu_dev_verbs_put_signal_counter(...) {
	base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks + 1);
	for (uint64_t i = 0; i < num_chunks; i++) {
        wqe_idx = base_wqe_idx + i;
        size_ = remaining_size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    : remaining_size;
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

        [[likely]] if (size_ > 0) {
            doca_gpu_dev_verbs_wqe_prepare_write(...);
        } else {
            doca_gpu_dev_verbs_wqe_prepare_nop(qp, wqe_ptr, wqe_idx,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
        }
        remaining_size -= size_;
  }
  ++wqe_idx;
  wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
  doca_gpu_dev_verbs_wqe_prepare_atomic(...);
  doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);
}
```

非常眼熟，doca\_gpu\_dev\_verbs\_reserve\_wq\_slots就是在qp中通过atomic预留num\_chunks + 1个wqe，num\_chunks为write wqe，1个为atomic wqe。 写wqe完成之后，开始执行doca\_gpu\_dev\_verbs\_mark\_wqes\_ready，就是nvshmem的ready\_idx，表示这段连续的wqe填充完成。

```c++
__device__ static __forceinline__ void doca_gpu_dev_verbs_mark_wqes_ready(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t from_wqe_idx, uint64_t to_wqe_idx) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE)
        qp->sq_ready_index = to_wqe_idx + 1;
    else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
        cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ready_index_aref(qp->sq_ready_index);
        while (ready_index_aref.load(cuda::memory_order_relaxed) != from_wqe_idx) continue;
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
        ready_index_aref.store(to_wqe_idx + 1, cuda::memory_order_relaxed);
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ready_index_aref(qp->sq_ready_index);
        while (ready_index_aref.load(cuda::memory_order_relaxed) != from_wqe_idx) continue;
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        ready_index_aref.store(to_wqe_idx + 1, cuda::memory_order_relaxed);
    }
}
```

对于单线程的qp，直接对sq\_ready\_index赋值即可，对于多线程共享的，需要先等待sq\_ready\_index等于from\_wqe\_idx，然后才可以写入to\_wqe\_idx。

```c++
__device__ static __forceinline__ void doca_gpu_dev_verbs_put_signal_counter(...) {
		uint64_t companion_base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2);
    uint64_t companion_wqe_idx = companion_base_wqe_idx;

    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp, wqe_ptr, companion_wqe_idx,
                                        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, wqe_idx,
                                        qp->cq_sq.cq_num);

    ++companion_wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companion_qp, wqe_ptr, companion_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, counter_raddr.addr, counter_raddr.key,
        counter_laddr.addr, counter_laddr.key, sizeof(uint64_t), counter_val, 0);
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp, companion_base_wqe_idx,
                                                              companion_wqe_idx);

    doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
    uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};
    doca_gpu_dev_verbs_submit_multi_qps<num_qps, resource_sharing_mode,
                                        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(
        qps, prod_indices);
}
```

然后开始写counter，这里用了rdma wait，max\_index为qp\_main的wqe\_idx，因此当qp\_main的atomic完成之后，qp\_companion的wait完成，qp\_companion的atomic开始执行，之后写入当前rank的counter。最后写db触发执行，这个过程和nvshmem一致，不再赘述。

#### 4.1.2 waitSignal

waitSignal就是while轮询直到signal等于expect。

#### 4.1.3 Flush

flush就是pollcq，通过pollcq保证所有操作执行完成，最后更新cons\_index + 1到cqe\_ci。

```c++
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    int status = doca_priv_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
    if (status == 0) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
    }
    return status;
}
```

pollcq的过程就是等待cons\_index位置的cqe的op\_own。

```c++
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_cq_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe =
        (struct doca_gpunetio_ib_mlx5_cqe64 *)__ldg((uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = __ldg(&cq->cqe_num);
    uint32_t idx = cons_index & (cqe_num - 1);
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 = &cqe[idx];
    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci)
            return 0;
        opown = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
    } while ((cons_index >= cqe_ci + cqe_num) ||
             ((cqe_ci <= cons_index) &&
              ((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num))));
    ...
}
```

## 5. 总结

2.28之后，nccl也拥有了nvshmem的所有特色，后续应该会有gin实现的allreduce等接口，另外感觉也可以对外暴露device的ring，tree等结构，方便用户通过device api利用nccl的这些能力。
