from __future__ import annotations

import math
import platform
import time
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import RLock
from threading import Thread

import torch

from jev_like.engine.backend import FlashInferPagedBackend, ReferencePagedBackend
from jev_like.engine.kv import KVCacheManager
from jev_like.engine.metrics import EngineStats
from jev_like.engine.request import (
    CandidateResult,
    DecisionResponse,
    DecisionResult,
    SubmitReceipt,
)
from jev_like.engine.runner import ModelRunner, PointerRunner
from jev_like.engine.storage import PagedKVStore, ReprStore
from jev_like.model.decision_model import NerivDecisionModel
from jev_like.model.contract import CompiledDecisionRequest

from .batch_plan import BatchPlan, Stage
from .graph import RequestCompiler
from .request import RequestContext, RequestStatus
from .scheduler import Scheduler
from .tables import NodeStatus, RuntimeTables


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """引擎资源上限；所有容量在启动时确定。"""

    max_inflight_requests: int = 64
    max_queued_requests: int = 64
    max_batch_tokens: int = 4096
    max_kv_pages: int = 512
    page_size: int = 16
    query_repr_slots: int = 256
    candidate_repr_slots: int = 2048
    attention_backend: str = "auto"
    numerical_profile: str = "auto"
    prefix_cache: bool = True
    cache_epoch: str = ""


class EngineRejected(RuntimeError):
    """携带确定状态码的预期提交拒绝。"""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


class NerivEngine:
    """Prefill-only、DAG-aware、基于紧凑表的单 GPU 决策引擎。"""

    def __init__(self, model: NerivDecisionModel, config: EngineConfig | None = None,
                 device: torch.device | str | None = None) -> None:
        self.config = config or EngineConfig()
        if min(self.config.max_inflight_requests, self.config.max_queued_requests,
               self.config.max_batch_tokens, self.config.max_kv_pages,
               self.config.page_size, self.config.query_repr_slots,
               self.config.candidate_repr_slots) <= 0:
            raise ValueError("EngineConfig 容量必须大于 0")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.tables = RuntimeTables()
        if model.contract is None:
            raise ValueError("Engine 需要带 Model Contract 的模型")
        self.contract_hash = model.contract.contract_hash
        self.graph = RequestCompiler(self.tables)
        self.scheduler = Scheduler(self.tables)
        backbone_config = model.backbone.config
        dtype = next(model.backbone.parameters()).dtype
        self.numerical_profile = self.config.numerical_profile
        if self.numerical_profile == "auto":
            self.numerical_profile = "bf16-cuda-v1" if self.device.type == "cuda" and dtype == torch.bfloat16 else "fp32-reference"
        if self.numerical_profile == "bf16-cuda-v1" and (self.device.type != "cuda" or dtype != torch.bfloat16):
            raise ValueError("bf16-cuda-v1 需要 CUDA BF16 模型")
        if self.numerical_profile == "fp32-reference" and dtype != torch.float32:
            raise ValueError("fp32-reference 需要 FP32 模型")
        if self.numerical_profile not in {"bf16-cuda-v1", "fp32-reference"}:
            raise ValueError(f"未知数值 Profile: {self.numerical_profile}")
        self.kv_store = PagedKVStore(
            self.config.max_kv_pages,
            self.config.page_size,
            backbone_config.num_hidden_layers,
            backbone_config.num_key_value_heads,
            backbone_config.head_dim,
            self.device,
            dtype,
        )
        self.kv = KVCacheManager(self.kv_store)
        self._cache_epoch = (f"{self.config.cache_epoch or id(model)}:{self.contract_hash}:"
                             f"{dtype}:{self.config.page_size}:{self.numerical_profile}")
        hidden_size = model.backbone.config.hidden_size
        self.reprs = ReprStore(
            self.config.query_repr_slots,
            self.config.candidate_repr_slots,
            hidden_size,
            self.device,
            torch.float32,
        )
        backend_name = self.config.attention_backend
        if backend_name == "auto":
            backend_name = "flashinfer" if self.numerical_profile == "bf16-cuda-v1" and platform.system() == "Linux" else "reference"
        if backend_name not in {"flashinfer", "reference"}:
            raise ValueError(f"未知 Engine backend: {backend_name}")
        if backend_name == "flashinfer" and self.numerical_profile != "bf16-cuda-v1":
            raise ValueError("FlashInfer 只支持已验证的 bf16-cuda-v1 Profile")
        backend = (
            FlashInferPagedBackend(model.backbone, self.device)
            if backend_name == "flashinfer"
            else ReferencePagedBackend(model.backbone, self.device)
        )
        self.model_runner = ModelRunner(backend, self.kv, self.kv_store)
        self.pointer_runner = PointerRunner(model.pointer, self.reprs, self.tables, model.temperature)
        self.stats = EngineStats()
        self.contexts: dict[int, RequestContext] = {}
        self._public_ids: dict[str, int] = {}
        self._page_reservations: dict[int, int] = {}
        self._reserved_pages = 0
        self._next_request_id = 0
        self._pointer_ready: list[int] = []
        self._shutdown = False
        self._lock = RLock()
        self._ingress: Queue[tuple[str, object]] = Queue(maxsize=self.config.max_queued_requests)
        self._completions: Queue[DecisionResponse] = Queue()
        self._awaiters: dict[str, Future[DecisionResponse]] = {}
        self._snapshots: dict[str, str] = {}
        self._finished: dict[str, DecisionResponse] = {}
        self._synchronous: set[str] = set()
        self._gpu_jobs: Queue[tuple[str, object]] = Queue(maxsize=1)
        self._gpu_results: Queue[tuple[str, object, object, Exception | None]] = Queue()
        self._gpu_busy = False
        self._inflight_requests: set[int] = set()
        self._deferred_errors: dict[int, Exception] = {}
        self._gpu_started_at = 0.0
        self._gpu_worker = Thread(target=self._run_gpu, name="neriv-gpu", daemon=True)
        self._gpu_worker.start()
        self._worker = Thread(target=self._run, name="neriv-engine", daemon=True)
        self._worker.start()

    def submit(self, request: CompiledDecisionRequest) -> SubmitReceipt:
        """校验输入并非阻塞地入队；GPU 与运行时表由后台单线程独占。"""

        return self._enqueue(request, synchronous=False)

    def _enqueue(self, request: CompiledDecisionRequest, synchronous: bool) -> SubmitReceipt:
        """共用同步与异步入口的原子入队逻辑。"""

        status = self._validate(request)
        with self._lock:
            if self._shutdown:
                status = "CANCELLED"
            elif request.request_id in self._awaiters or request.request_id in self._finished:
                raise ValueError(f"重复 request_id: {request.request_id}")
            if status != "OK":
                return SubmitReceipt(False, request.request_id, status)
            future: Future[DecisionResponse] = Future()
            self._awaiters[request.request_id] = future
            self._snapshots[request.request_id] = request.snapshot_id
            if synchronous:
                self._synchronous.add(request.request_id)
            try:
                self._ingress.put_nowait(("submit", request))
            except Full:
                self._awaiters.pop(request.request_id)
                self._snapshots.pop(request.request_id)
                self._synchronous.discard(request.request_id)
                self.stats.increment("backpressure_rejections")
                return SubmitReceipt(False, request.request_id, "OVERLOADED")
        return SubmitReceipt(True, request.request_id, "OK")

    def _validate(self, request: CompiledDecisionRequest) -> str:
        """在入队前校验协议与容量；非法 scope 作为请求错误抛出。"""

        if request.contract_hash != self.contract_hash:
            return "MODEL_CONTRACT_MISMATCH"
        if not request.cache_scope_id.strip():
            raise ValueError("cache_scope_id 不能为空")
        if request.deadline_ns and request.deadline_ns <= time.monotonic_ns():
            return "DEADLINE_EXCEEDED"
        if len(request.state_tokens) > 8192 or not request.questions:
            return "INPUT_TOO_LARGE"
        for question in request.questions:
            if not 2 <= len(question.candidate_ids) <= 16 or len(question.question_tokens) > 512:
                return "INPUT_TOO_LARGE"
            if any(len(tokens) > 256 or len(request.state_tokens) + len(question.question_tokens) + len(tokens) > 9216
                   for tokens in question.candidate_tokens):
                return "INPUT_TOO_LARGE"
        return "OK"

    def _run(self) -> None:
        """后台推进入队、调度、GPU 执行和完成事件。"""

        while not self._shutdown or self.contexts or not self._ingress.empty() or self._gpu_busy:
            try:
                kind, payload = self._ingress.get(timeout=0.001 if self.contexts else 0.05)
            except Empty:
                kind = ""
                payload = None
            for _ in range(self.config.max_queued_requests):
                self._process_command(kind, payload)
                try:
                    kind, payload = self._ingress.get_nowait()
                except Empty:
                    break
            try:
                kind, item, result, error = self._gpu_results.get_nowait()
                self._complete_gpu(kind, item, result, error)
            except Empty:
                pass
            self._expire_requests()
            if self._shutdown:
                for request_id in tuple(self._public_ids):
                    self._cancel(request_id)
            if not self._gpu_busy:
                try:
                    self._dispatch_gpu()
                except Exception as error:
                    self._fail_requests(set(self.contexts), error)
        self._gpu_jobs.put(("stop", None))

    def _run_gpu(self) -> None:
        """专用工作线程串行执行模型与 Pointer 算子。"""

        while True:
            kind, item = self._gpu_jobs.get()
            if kind == "stop":
                return
            try:
                result = self.model_runner.run(item) if kind == "model" else self.pointer_runner.score(item)
                self._gpu_results.put((kind, item, result, None))
            except Exception as error:
                self._gpu_results.put((kind, item, None, error))

    def _dispatch_gpu(self) -> None:
        """选择一个同阶段批次，Pointer 完成批次优先。"""

        if self._pointer_ready:
            ready = [key for key in self._pointer_ready
                     if self.tables.questions[key].status != NodeStatus.CANCELLED
                     and self.tables.questions[key].request_id in self.contexts]
            self._pointer_ready = []
            if ready:
                request_ids = {self.tables.questions[key].request_id for key in ready}
                kind, item = "pointer", ready
            else:
                return
        else:
            plan = self.scheduler.schedule(self.config.max_batch_tokens)
            if plan is None:
                return
            self._mark_started(plan)
            request_ids = self._request_ids(plan)
            kind, item = "model", plan
        self._gpu_busy = True
        self._inflight_requests = request_ids
        self._gpu_started_at = time.monotonic()
        self._gpu_jobs.put_nowait((kind, item))

    def _complete_gpu(self, kind: str, item: object, result: object, error: Exception | None) -> None:
        """提交 GPU 结果；取消请求只在算子完成后释放其物理页。"""

        self.stats.observe("set_pointer_latency" if kind == "pointer" else f"{item.stage.name.lower()}_latency",
                           time.monotonic() - self._gpu_started_at)
        inflight = self._inflight_requests
        self._inflight_requests = set()
        self._gpu_busy = False
        for request_id in inflight:
            deferred = self._deferred_errors.pop(request_id, None)
            context = self.contexts.get(request_id)
            if deferred is not None and context is not None:
                self._cleanup(context, RequestStatus.CANCELLED, deferred)
        if error is not None:
            self._fail_requests(inflight, error)
            return
        if not self.contexts:
            return
        try:
            if kind == "model":
                self.stats.increment("batch_tokens", item.token_count)
                self.stats.increment("batch_sequences", item.sequence_count)
                self._complete(item, result)
            else:
                for question_id in item:
                    if question_id < len(self.tables.questions) and self.tables.questions[question_id].request_id in self.contexts:
                        self._finish_question(question_id, result[question_id])
        except Exception as failure:
            self._fail_requests(inflight, failure)

    def _process_command(self, kind: str, payload: object) -> None:
        """把公共命令转换为单所有者 Runtime Table 变更。"""

        if kind == "cancel":
            self._cancel(payload)
        elif kind == "submit":
            request = payload
            if self._shutdown:
                self._publish_status(request.request_id, request.snapshot_id, "CANCELLED")
                return
            try:
                internal_id = self._submit(request)
                context = self.contexts[internal_id]
                context.future.add_done_callback(
                    lambda completed, key=request.request_id, snapshot=request.snapshot_id:
                    self._publish(key, snapshot, completed)
                )
            except EngineRejected as error:
                self._publish_status(request.request_id, request.snapshot_id, error.status)
            except MemoryError:
                self._publish_status(request.request_id, request.snapshot_id, "OVERLOADED")
            except Exception:
                self._publish_status(request.request_id, request.snapshot_id, "INTERNAL_ERROR")

    def _publish(self, request_id: str, snapshot_id: str, future: Future[DecisionResponse]) -> None:
        try:
            response = future.result()
        except TimeoutError:
            response = DecisionResponse(request_id, snapshot_id, (), "DEADLINE_EXCEEDED")
        except CancelledError:
            response = DecisionResponse(request_id, snapshot_id, (), "CANCELLED")
        except Exception:
            response = DecisionResponse(request_id, snapshot_id, (), "INTERNAL_ERROR")
        self._publish_response(response)

    def _publish_status(self, request_id: str, snapshot_id: str, status: str) -> None:
        response = DecisionResponse(request_id, snapshot_id, (), status)
        self._publish_response(response)

    def _publish_response(self, response: DecisionResponse) -> None:
        """只完成一次公开请求，避免关闭后的迟到 GPU 回调重复发布。"""

        request_id = response.request_id
        with self._lock:
            awaiter = self._awaiters.pop(request_id, None)
            if awaiter is None:
                return
            self._snapshots.pop(request_id, None)
            self._finished[request_id] = response
            if request_id not in self._synchronous:
                self._completions.put(response)
            self._synchronous.discard(request_id)
            if not awaiter.done():
                awaiter.set_result(response)

    def poll_completion(self, max_items: int = 64) -> list[DecisionResponse]:
        """非阻塞地读取已完成响应；不会执行模型计算。"""

        items = []
        for _ in range(max_items):
            try:
                response = self._completions.get_nowait()
                items.append(response)
                with self._lock:
                    self._finished.pop(response.request_id, None)
            except Empty:
                break
        return items

    def _submit(self, request: CompiledDecisionRequest) -> int:
        """先保留命中页，再按本次新增页需求决定是否接纳。

        活跃缓存物理页只计一次；每个请求仅预留未命中页预算。
        """

        if self._shutdown:
            raise EngineRejected("CANCELLED")
        if request.contract_hash != self.contract_hash:
            raise EngineRejected("MODEL_CONTRACT_MISMATCH")
        if request.request_id in self._public_ids:
            raise ValueError(f"重复 request_id: {request.request_id}")
        active = sum(context.status not in {RequestStatus.DONE, RequestStatus.CANCELLED, RequestStatus.FAILED}
                     for context in self.contexts.values())
        queued = sum(context.status == RequestStatus.QUEUED for context in self.contexts.values())
        if active >= self.config.max_inflight_requests or queued >= self.config.max_queued_requests:
            self.stats.increment("backpressure_rejections")
            raise EngineRejected("OVERLOADED")
        question_count = len(request.questions)
        candidate_count = sum(len(question.candidate_ids) for question in request.questions)
        if question_count > self.reprs.free_query_count or candidate_count > self.reprs.free_candidate_count:
            self.stats.increment("backpressure_rejections")
            raise EngineRejected("OVERLOADED")
        required_pages = self._required_pages(request)
        if self.config.prefix_cache:
            state_prefix, cached_tokens = self.kv.lookup_state_prefix(
                request.state_tokens, request.cache_scope_id, self._cache_epoch,
            )
            required_free = required_pages - cached_tokens // self.config.page_size
        else:
            state_prefix, cached_tokens = self.kv.create_prefix([], 0), 0
            required_free = required_pages
        if self._reserved_pages + self.kv.active_cached_page_count + required_free > self.config.max_kv_pages:
            self.kv.release(state_prefix)
            self.stats.increment("backpressure_rejections")
            raise EngineRejected("OVERLOADED")
        if self.config.prefix_cache:
            self.stats.increment("cache_evictions", self.kv.evict_cached_until_free(required_free))
        if self.kv_store.free_page_count < required_free:
            self.kv.release(state_prefix)
            self.stats.increment("backpressure_rejections")
            raise EngineRejected("OVERLOADED")
        internal_id = self._next_request_id
        self._next_request_id += 1
        deadline = request.deadline_ns / 1_000_000_000 if request.deadline_ns else None
        try:
            context = self.graph.compile(internal_id, request, deadline)
        except Exception:
            self.kv.release(state_prefix)
            raise
        try:
            query_slots = self.reprs.allocate_query_slots(question_count)
        except Exception:
            self.kv.release(state_prefix)
            raise
        try:
            candidate_slots = self.reprs.allocate_candidate_slots(candidate_count)
        except Exception:
            self.reprs.free_query_slots(query_slots)
            self.kv.release(state_prefix)
            raise
        candidate_slot = iter(candidate_slots)
        for offset, question_id in enumerate(range(context.question_begin, context.question_begin + question_count)):
            question = self.tables.questions[question_id]
            question.query_repr_slot = query_slots[offset]
            for candidate_offset in range(question.candidate_count):
                self.tables.candidates[question.candidate_begin + candidate_offset].repr_slot = next(candidate_slot)
        self.contexts[internal_id] = context
        self._public_ids[request.request_id] = internal_id
        self._page_reservations[internal_id] = required_free
        self._reserved_pages += required_free
        state = self.tables.states[context.state_id]
        state.prefix_id, state.token_progress = state_prefix, cached_tokens
        if self.config.prefix_cache:
            self.stats.increment("cache_hits" if cached_tokens else "cache_misses")
            self.stats.increment("cached_state_tokens", cached_tokens)
        try:
            if state.token_progress == state.token_len:
                self._activate_questions(context)
            else:
                self.scheduler.enqueue_state(context.state_id)
        except Exception as error:
            self._cleanup(context, RequestStatus.FAILED, error)
            raise
        self.stats.increment("submitted_requests")
        return internal_id

    def decide(self, request: CompiledDecisionRequest, timeout_seconds: float | None = None) -> DecisionResponse:
        """同步等待已提交请求；时限由单进程 monotonic deadline 控制。"""

        receipt = self._enqueue(request, synchronous=True)
        if not receipt.accepted:
            return DecisionResponse(request.request_id, request.snapshot_id, (), receipt.status)
        with self._lock:
            future = self._awaiters.get(request.request_id)
            completed = self._finished.pop(request.request_id, None)
        if completed is not None:
            return completed
        if future is None:
            raise RuntimeError("请求结果已被其他消费者读取")
        try:
            response = future.result(timeout=timeout_seconds)
            with self._lock:
                self._finished.pop(request.request_id, None)
            return response
        except TimeoutError:
            self.cancel(request.request_id)
            return DecisionResponse(request.request_id, request.snapshot_id, (), "DEADLINE_EXCEEDED")

    def decide_batch(
        self,
        requests: list[CompiledDecisionRequest],
        timeout_seconds: float | None = None,
    ) -> list[DecisionResponse]:
        """先提交整批请求，再由同一调度循环连续批处理。

        @param requests 待决策请求，返回顺序与该列表一致。
        @param timeout_seconds 每个请求共享的相对超时时间。
        @return 按输入顺序排列的决策结果。
        """

        receipts = [self._enqueue(request, synchronous=True) for request in requests]
        with self._lock:
            pending = {request.request_id: self._awaiters.get(request.request_id) for request in requests}
            completed = {request.request_id: self._finished.get(request.request_id) for request in requests}
        responses = []
        for request, receipt in zip(requests, receipts):
            if not receipt.accepted:
                responses.append(DecisionResponse(request.request_id, request.snapshot_id, (), receipt.status))
            elif completed[request.request_id] is not None:
                responses.append(completed[request.request_id])
            else:
                try:
                    responses.append(pending[request.request_id].result(timeout=timeout_seconds))
                except TimeoutError:
                    self.cancel(request.request_id)
                    responses.append(DecisionResponse(request.request_id, request.snapshot_id, (), "DEADLINE_EXCEEDED"))
        with self._lock:
            for request in requests:
                self._finished.pop(request.request_id, None)
        return responses

    def cancel(self, request_id: str) -> bool:
        """非阻塞地申请取消尚未完成的请求。"""

        with self._lock:
            if request_id not in self._awaiters:
                return False
        try:
            self._ingress.put_nowait(("cancel", request_id))
        except Full:
            return False
        return True

    def _cancel(self, request_id: str) -> bool:
        """惰性取消请求；队列中的 ID 在取出时通过状态跳过。"""

        internal_id = self._public_ids.get(request_id)
        context = self.contexts.get(internal_id) if internal_id is not None else None
        if context is None or context.future.done() or context.cancelled:
            return False
        context.cancelled = True
        if internal_id in self._inflight_requests:
            self._deferred_errors[internal_id] = CancelledError("请求已取消")
            return True
        self._cleanup(context, RequestStatus.CANCELLED, CancelledError("请求已取消"))
        self.stats.increment("cancelled_requests")
        return True

    def shutdown(self, timeout_seconds: float = 5.0) -> None:
        """停接新请求并终结全部等待者；GPU 中的资源延后安全回收。"""

        with self._lock:
            self._shutdown = True
            for request_id, snapshot_id in tuple(self._snapshots.items()):
                self._publish_status(request_id, snapshot_id, "CANCELLED")
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        self._worker.join(timeout=max(0.0, deadline - time.monotonic()))
        self._gpu_worker.join(timeout=max(0.0, deadline - time.monotonic()))

    def metrics(self) -> dict[str, float | int | str]:
        values = self.stats.snapshot()
        values.update(self.scheduler.counts())
        values["free_pages"] = self.kv_store.free_page_count
        values["used_pages"] = self.config.max_kv_pages - self.kv_store.free_page_count
        values["reserved_pages"] = self._reserved_pages
        values["cached_pages"] = self.kv.cached_page_count
        values["ingress_depth"] = self._ingress.qsize()
        values["numerical_profile"] = self.numerical_profile
        return values

    def _required_pages(self, request: CompiledDecisionRequest) -> int:
        """计算 State/Question/Candidate 分叉后的最坏物理页数。

        未对齐页在 fork 时执行 COW，因此每个子分支都需要计入一个私有尾页。
        """

        page_size = self.config.page_size

        def append_pages(parent_length: int, token_count: int) -> int:
            if token_count <= 0:
                return 0
            copied_tail = 1 if parent_length % page_size else 0
            before = (parent_length + page_size - 1) // page_size
            after = (parent_length + token_count + page_size - 1) // page_size
            return copied_tail + max(0, after - before)

        state_length = len(request.state_tokens)
        required = (state_length + page_size - 1) // page_size
        for question in request.questions:
            question_length = len(question.question_tokens)
            required += append_pages(state_length, question_length)
            prefix_length = state_length + question_length
            required += sum(
                append_pages(prefix_length, len(tokens)) for tokens in question.candidate_tokens
            )
        return required

    def _mark_started(self, plan: BatchPlan) -> None:
        for node_id in plan.node_ids:
            if plan.stage == Stage.STATE:
                request_id = self.tables.states[node_id].request_id
            elif plan.stage == Stage.QUESTION:
                request_id = self.tables.questions[node_id].request_id
            else:
                question_id = self.tables.candidates[node_id].question_id
                request_id = self.tables.questions[question_id].request_id
            context = self.contexts[request_id]
            if context.started_at is None:
                context.started_at = time.monotonic()
                context.status = RequestStatus.RUNNING
                self.stats.observe("queue_latency", context.started_at - context.submitted_at)

    def _request_ids(self, plan: BatchPlan) -> set[int]:
        if plan.stage == Stage.STATE:
            return {self.tables.states[node_id].request_id for node_id in plan.node_ids}
        if plan.stage == Stage.QUESTION:
            return {self.tables.questions[node_id].request_id for node_id in plan.node_ids}
        return {
            self.tables.questions[self.tables.candidates[node_id].question_id].request_id
            for node_id in plan.node_ids
        }

    def _fail_requests(self, request_ids: set[int], error: Exception) -> None:
        """计算失败时释放受影响请求的全部资源。"""

        for request_id in request_ids:
            context = self.contexts.get(request_id)
            if context is not None and not context.future.done():
                self._cleanup(context, RequestStatus.FAILED, error)
                self.stats.increment("failed_requests")

    def _complete(self, plan: BatchPlan, result) -> None:
        for row, node_id in enumerate(plan.node_ids):
            if plan.stage == Stage.STATE:
                request_id = self.tables.states[node_id].request_id
            elif plan.stage == Stage.QUESTION:
                request_id = self.tables.questions[node_id].request_id
            else:
                request_id = self.tables.questions[self.tables.candidates[node_id].question_id].request_id
            if request_id not in self.contexts:
                continue
            if plan.stage == Stage.STATE:
                self._complete_state(plan, row, node_id)
            elif plan.stage == Stage.QUESTION:
                self._complete_question(node_id, result.representations[row])
            else:
                self._complete_candidate(node_id, result.representations[row])

    def _complete_state(self, plan: BatchPlan, row: int, state_id: int) -> None:
        state = self.tables.states[state_id]
        consumed = int(plan.token_indptr[row + 1] - plan.token_indptr[row])
        state.token_progress += consumed
        if state.token_progress < state.token_len:
            self.scheduler.enqueue_state(state_id)
            return
        context = self.contexts[state.request_id]
        if self.config.prefix_cache:
            published = self.kv.publish_state_full_blocks(
                state.prefix_id, context.request.state_tokens, context.request.cache_scope_id, self._cache_epoch,
            )
            self._page_reservations[context.internal_id] -= published
            self._reserved_pages -= published
        self._activate_questions(context)

    def _activate_questions(self, context: RequestContext) -> None:
        """State 完成或整页命中后启动隔离的 Question 分支。"""

        state = self.tables.states[context.state_id]
        state.status = NodeStatus.DONE
        for question_id in range(context.question_begin, context.question_begin + context.question_count):
            self.tables.questions[question_id].prefix_id = self.kv.fork_prefix(state.prefix_id)
            self.scheduler.enqueue_question(question_id)

    def _complete_question(self, question_id: int, representation) -> None:
        question = self.tables.questions[question_id]
        self.reprs.write_query(question.query_repr_slot, representation.float())
        question.status = NodeStatus.DONE
        for candidate_id in range(question.candidate_begin, question.candidate_begin + question.candidate_count):
            self.tables.candidates[candidate_id].prefix_id = self.kv.fork_prefix(question.prefix_id)
            self.scheduler.enqueue_candidate(candidate_id)

    def _complete_candidate(self, candidate_id: int, representation) -> None:
        candidate = self.tables.candidates[candidate_id]
        self.reprs.write_candidate(candidate.repr_slot, representation.float())
        self.kv.release(candidate.prefix_id)
        candidate.prefix_id = -1
        candidate.status = NodeStatus.DONE
        question = self.tables.questions[candidate.question_id]
        question.pending_candidates -= 1
        if question.pending_candidates == 0:
            self._pointer_ready.append(candidate.question_id)

    def _finish_question(self, question_id: int, probabilities: torch.Tensor) -> None:
        question = self.tables.questions[question_id]
        context = self.contexts[question.request_id]
        offset = question_id - context.question_begin
        request_question = context.request.questions[offset]
        ordinary = probabilities[:-1]
        reject_probability = float(probabilities[-1])
        top_values, top_indices = ordinary.topk(min(2, len(ordinary)))
        maximum = float(top_values[0])
        margin = maximum - float(top_values[1]) if len(top_values) > 1 else maximum
        entropy = float(-(probabilities * probabilities.clamp_min(1e-8).log()).sum() / math.log(len(probabilities)))
        selected = int(top_indices[0])
        result = DecisionResult(
            context.request.request_id,
            context.request.snapshot_id,
            request_question.question_id,
            tuple(CandidateResult(candidate_id, float(ordinary[index]))
                  for index, candidate_id in enumerate(request_question.candidate_ids)),
            reject_probability,
            maximum,
            margin,
            entropy,
            request_question.candidate_ids[selected],
        )
        context.results[offset] = result
        candidate_records = self.tables.candidates[
            question.candidate_begin:question.candidate_begin + question.candidate_count
        ]
        self.reprs.free_candidate_slots([record.repr_slot for record in candidate_records])
        for record in candidate_records:
            record.repr_slot = -1
        self.reprs.free_query_slots([question.query_repr_slot])
        question.query_repr_slot = -1
        self.kv.release(question.prefix_id)
        question.prefix_id = -1
        state = self.tables.states[question.state_id]
        state.pending_questions -= 1
        context.pending_questions -= 1
        if context.pending_questions == 0:
            self.kv.release(state.prefix_id)
            state.prefix_id = -1
            self._release_reservation(context.internal_id)
            context.status = RequestStatus.DONE
            response = DecisionResponse(
                context.request.request_id,
                context.request.snapshot_id,
                tuple(context.results[index] for index in range(context.question_count)),
            )
            self.stats.observe("request_latency", time.monotonic() - context.submitted_at)
            self.stats.increment("completed_requests")
            self._retire_context(context.internal_id)
            context.future.set_result(response)

    def _expire_requests(self) -> None:
        now = time.monotonic()
        for context in list(self.contexts.values()):
            if not context.future.done() and not context.cancelled and context.deadline is not None and now >= context.deadline:
                context.cancelled = True
                if context.internal_id in self._inflight_requests:
                    self._deferred_errors[context.internal_id] = TimeoutError("请求超时")
                else:
                    self._cleanup(context, RequestStatus.CANCELLED, TimeoutError("请求超时"))
                self.stats.increment("timed_out_requests")

    def _cleanup(self, context: RequestContext, status: RequestStatus, error: Exception) -> None:
        state = self.tables.states[context.state_id]
        state.status = NodeStatus.CANCELLED
        candidate_slots: list[int] = []
        query_slots: list[int] = []
        for question_id in range(context.question_begin, context.question_begin + context.question_count):
            question = self.tables.questions[question_id]
            question.status = NodeStatus.CANCELLED
            if question.query_repr_slot >= 0:
                query_slots.append(question.query_repr_slot)
                question.query_repr_slot = -1
            if question.prefix_id >= 0:
                self.kv.release(question.prefix_id)
                question.prefix_id = -1
            for candidate_id in range(question.candidate_begin, question.candidate_begin + question.candidate_count):
                candidate = self.tables.candidates[candidate_id]
                candidate.status = NodeStatus.CANCELLED
                if candidate.prefix_id >= 0:
                    self.kv.release(candidate.prefix_id)
                    candidate.prefix_id = -1
                if candidate.repr_slot >= 0:
                    candidate_slots.append(candidate.repr_slot)
                    candidate.repr_slot = -1
        if query_slots:
            self.reprs.free_query_slots(query_slots)
        if candidate_slots:
            self.reprs.free_candidate_slots(candidate_slots)
        if state.prefix_id >= 0:
            self.kv.release(state.prefix_id)
            state.prefix_id = -1
        self._release_reservation(context.internal_id)
        context.status = status
        self._retire_context(context.internal_id)
        if not context.future.done():
            context.future.set_exception(error)

    def _release_reservation(self, request_id: int) -> None:
        """归还请求提交时预留的未命中 KV 页额度。"""

        self._reserved_pages -= self._page_reservations.pop(request_id, 0)

    def _retire_context(self, request_id: int) -> None:
        """移除控制对象；空闲时重建追加表，阻止常驻服务主存增长。"""

        context = self.contexts.pop(request_id, None)
        if context is not None:
            self._public_ids.pop(context.request.request_id, None)
        if self.contexts:
            return
        self.tables = RuntimeTables()
        self.graph.tables = self.tables
        self.scheduler = Scheduler(self.tables)
        self.pointer_runner.tables = self.tables
        self._pointer_ready.clear()
