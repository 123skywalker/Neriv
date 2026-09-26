/* App.tsx, 决策看板：一列一个服务端 Session，显示名与 session_id 分离 */

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Activity, ArrowRight, Check, Loader2, Plus, RotateCcw, Trash2 } from "lucide-react";
import { api, type DecisionResponse, type DecisionResult, type Session } from "./api";
import { Badge, Button, Input, Textarea } from "./components/ui";

type QuestionType = "CHOICE" | "SCORE" | "NOUL";
/** 本轮或历史中的一道 typed decision。 */
type Question = {
  id: string;
  type: QuestionType;
  instructions: string;
  candidates: string[];
  result?: DecisionResult;
  status?: string;
  error?: string;
  loading?: boolean;
};
/**
 * 看板列。
 * @property name 仅用于界面展示，创建会话后仍可改，不作为服务端主键
 * @property sessionId 服务端 session_id，首次创建后固定
 */
type Column = {
  id: string;
  name: string;
  sessionId?: string;
  notes: string;
  syncedNotes?: string;
  history: Question[][];
  turn: Question[];
  turnError?: string;
  loading?: boolean;
  deleting?: boolean;
};

const defaults: Record<QuestionType, string[]> = {
  CHOICE: ["", ""],
  SCORE: ["1", "2", "3", "4", "5"],
  NOUL: ["是", "否"],
};

/** 题卡与轮次只保存在本页内存。 */
function newQuestion(): Question {
  return { id: crypto.randomUUID(), type: "CHOICE", instructions: "", candidates: [...defaults.CHOICE] };
}

/** @returns 空列，尚未绑定服务端会话 */
function newColumn(): Column {
  return { id: crypto.randomUUID(), name: "", notes: "", history: [], turn: [newQuestion()] };
}

/**
 * 发送前逐题校验，非法题不会进入本轮 decisions。
 * @returns 错误文案；通过时为 undefined
 */
function validate(question: Question): string | undefined {
  if (!question.instructions.trim()) return "请填写题干。";
  if (question.candidates.length < 2 || question.candidates.length > 16) return "选项数量须为 2–16 个。";
  if (question.candidates.some((item) => !item.trim())) return "选项内容不能为空。";
}

/** @returns 本题是否已拿到可展示的 OK 结果 */
function answered(question: Question): boolean {
  return question.status === "OK" && Boolean(question.result);
}

/** @returns 选中候选的展示文案；拒答时为 Reject */
function selectedLabel(question: Question): string {
  const selected = question.result?.selected_candidate_id;
  if (selected == null) return "Reject";
  const index = Number(selected);
  return Number.isInteger(index) ? question.candidates[index] ?? selected : selected;
}

/** @returns 题卡状态徽标配色 */
function statusVariant(status?: string): "success" | "warning" | "destructive" | "outline" {
  if (status === "OK") return "success";
  if (status === "OVERLOADED") return "warning";
  return status ? "destructive" : "outline";
}

/** 历史轮只显示已归档答案，不重新进入 Engine。 */
function HistoryCard({ question }: { question: Question }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <button type="button" onClick={() => setExpanded(!expanded)} className="w-full rounded-lg border border-zinc-800 bg-zinc-900/80 p-3 text-left transition-colors hover:border-zinc-700">
      <div className="flex items-center justify-between gap-2"><span className="section-label">{question.type}</span><Badge variant="success">OK</Badge></div>
      <p className="mt-2 line-clamp-2 text-sm text-zinc-300">{question.instructions}</p>
      <div className="mt-2 flex items-center gap-2 text-xs text-zinc-500"><Check size={13} className="text-cyan-400" /><span className="truncate text-cyan-300">{selectedLabel(question)}</span><span className="ml-auto">{expanded ? "收起" : "展开"}</span></div>
      {expanded && question.result && <div className="mt-3 space-y-1 border-t border-zinc-800 pt-3 text-xs text-zinc-400">
        {question.result.candidates.map((candidate) => <div key={candidate.candidate_id} className="flex justify-between gap-2"><span className="truncate">{question.candidates[Number(candidate.candidate_id)] ?? candidate.candidate_id}</span><span className="mono">{(candidate.probability * 100).toFixed(1)}%</span></div>)}
        <div className="flex justify-between"><span>Reject</span><span className="mono">{(question.result.reject_probability * 100).toFixed(1)}%</span></div>
      </div>}
    </button>
  );
}

/** 本轮题卡独立编辑，任何字段变化都会使旧答案失效。 */
function QuestionCard({ question, index, total, disabled, onChange, onDelete }: {
  question: Question;
  index: number;
  total: number;
  disabled: boolean;
  /** 合并字段变更；调用方负责写回列状态 */
  onChange: (change: (question: Question) => Question) => void;
  onDelete: () => void;
}) {
  const edit = (change: (question: Question) => Question) => {
    onChange((old) => ({ ...change(old), result: undefined, status: undefined, error: undefined }));
  };
  return <div className="rounded-xl border border-zinc-700 bg-[#1c1c1f] p-4 shadow-[0_8px_32px_rgba(0,0,0,0.16)]">
    <div className="mb-4 flex items-center justify-between gap-2">
      <div><div className="section-label">本轮并行题 / {index + 1} of {total}</div><div className="mt-1 text-sm font-semibold text-zinc-100">决策输入</div></div>
      <Badge variant={statusVariant(question.status)}>{question.loading ? "运行中" : question.status ?? "待提交"}</Badge>
        </div>
    <label className="field-label" htmlFor={`${question.id}-type`}>类型</label>
    <select id={`${question.id}-type`} className="form-select mb-4" value={question.type} disabled={disabled} onChange={(event) => {
      const type = event.target.value as QuestionType;
      edit((old) => ({ ...old, type, candidates: [...defaults[type]] }));
    }}>
      <option value="CHOICE">CHOICE · 选择</option><option value="SCORE">SCORE · 评分</option><option value="NOUL">NOUL · 是非</option>
    </select>
    <label className="field-label" htmlFor={`${question.id}-prompt`}>题干</label>
    <Textarea id={`${question.id}-prompt`} placeholder="输入这一问要判断的内容…" value={question.instructions} disabled={disabled} onChange={(event) => edit((old) => ({ ...old, instructions: event.target.value }))} className="mb-4 min-h-24 resize-y" />
    <div className="mb-2 flex items-center justify-between"><span className="field-label mb-0">选项 <span className="mono text-zinc-500">{question.candidates.length}/16</span></span><Button variant="ghost" size="sm" disabled={disabled || question.candidates.length >= 17} onClick={() => edit((old) => ({ ...old, candidates: [...old.candidates, ""] }))}><Plus size={13} /> 添加</Button></div>
    <div className="space-y-2">{question.candidates.map((candidate, candidateIndex) => <div key={`${question.id}-${candidateIndex}`} className="flex items-center gap-2">
      <span className="mono w-4 shrink-0 text-xs text-zinc-500">{candidateIndex + 1}</span>
      <Input aria-label={`第 ${index + 1} 题选项 ${candidateIndex + 1}`} value={candidate} disabled={disabled} placeholder={`选项 ${candidateIndex + 1}`} onChange={(event) => edit((old) => ({ ...old, candidates: old.candidates.map((value, item) => item === candidateIndex ? event.target.value : value) }))} className="h-8 min-w-0 text-xs" />
      <Button aria-label={`删除第 ${index + 1} 题选项 ${candidateIndex + 1}`} variant="ghost" size="icon" disabled={disabled} onClick={() => edit((old) => ({ ...old, candidates: old.candidates.filter((_, item) => item !== candidateIndex) }))} className="shrink-0 text-zinc-500"><Trash2 size={14} /></Button>
    </div>)}</div>
    <div className="mt-5 border-t border-zinc-700/70 pt-4">
      <div className="mb-3 flex items-center justify-between"><span className="section-label">答案</span>{answered(question) && <span className="text-xs text-cyan-300">{selectedLabel(question)}</span>}</div>
      {question.loading ? <div className="flex items-center gap-2 text-xs text-zinc-400"><Loader2 size={15} className="animate-spin text-cyan-400" />本轮决策中…</div> :
        question.error ? <p role="alert" className="rounded-md border border-amber-500/25 bg-amber-500/5 px-3 py-2 text-xs leading-5 text-amber-200">{question.status && <span className="mono mr-2">{question.status}</span>}{question.error}</p> :
        answered(question) && question.result ? <div className="space-y-3">
          {question.result.candidates.map((candidate) => <div key={candidate.candidate_id}>
            <div className="mb-1 flex items-center justify-between gap-2 text-xs"><span className={`truncate ${question.result?.selected_candidate_id === candidate.candidate_id ? "text-cyan-300" : "text-zinc-400"}`}>{question.candidates[Number(candidate.candidate_id)] ?? candidate.candidate_id}</span><span className="mono text-zinc-400">{(candidate.probability * 100).toFixed(1)}%</span></div>
            <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800"><div className={`h-full rounded-full ${question.result?.selected_candidate_id === candidate.candidate_id ? "bg-cyan-400" : "bg-zinc-500"}`} style={{ width: `${candidate.probability * 100}%` }} /></div>
          </div>)}
          <div><div className="mb-1 flex justify-between text-xs text-zinc-500"><span>Reject</span><span className="mono">{(question.result.reject_probability * 100).toFixed(1)}%</span></div><div className="h-1.5 rounded-full bg-zinc-800"><div className="h-full rounded-full bg-amber-400/80" style={{ width: `${question.result.reject_probability * 100}%` }} /></div></div>
        </div> : <p className="text-xs leading-5 text-zinc-500">本轮题目在同一次请求中并行决策。</p>}
      </div>
    <div className="mt-3 flex justify-end border-t border-zinc-700/50 pt-2">
      <Button aria-label="删除本题" title={disabled ? "决策中不能删除本题" : total === 1 ? "本轮至少保留一题" : "删除本题"} variant="ghost" size="icon" disabled={disabled || total === 1} onClick={onDelete} className="text-zinc-500 hover:text-red-300"><Trash2 size={14} /></Button>
      </div>
  </div>;
}

type ColumnViewProps = {
  column: Column;
  selected: boolean;
  onSelect: () => void;
  onColumnChange: (change: (column: Column) => Column) => void;
  onQuestionChange: (questionId: string, change: (question: Question) => Question) => void;
  onDelete: (questionId: string) => void;
  onDeleteColumn: () => void;
  canDeleteColumn: boolean;
  onDecision: () => void;
  onAdd: () => void;
  onNext: () => void;
};

/** 固定列头标记整轮题数；视口仅承载纵向历史与本轮题卡。 */
function ColumnView({ column, selected, onSelect, onColumnChange, onQuestionChange, onDelete, onDeleteColumn, canDeleteColumn, onDecision, onAdd, onNext }: ColumnViewProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const previousTurn = useRef({ firstId: column.turn[0].id, length: column.turn.length });
  const deletedScrollTarget = useRef<number | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const allAnswered = column.turn.every(answered);

  /** 增题、换轮或删题后定位目标卡，普通尺寸变化保留阅读位置。 */
  useLayoutEffect(() => {
    const previous = previousTurn.current;
    const nextIndex = deletedScrollTarget.current ?? (previous.firstId !== column.turn[0].id ? 0 : column.turn.length > previous.length ? column.turn.length - 1 : -1);
    deletedScrollTarget.current = null;
    previousTurn.current = { firstId: column.turn[0].id, length: column.turn.length };
    if (nextIndex < 0) return;
    const frame = requestAnimationFrame(() => scrollToQuestion(nextIndex));
    return () => cancelAnimationFrame(frame);
  }, [column.turn[0].id, column.turn.length]);

  function scrollToQuestion(index: number) {
    const viewport = viewportRef.current;
    const card = viewport?.querySelector<HTMLElement>(`[data-turn-index="${index}"]`);
    if (!viewport || !card) return;
    viewport.scrollTo({ top: viewport.scrollTop + card.getBoundingClientRect().top - viewport.getBoundingClientRect().top, behavior: "smooth" });
    setActiveIndex(index);
  }

  function deleteQuestion(index: number) {
    if (column.loading || column.deleting || column.turn.length === 1) return;
    deletedScrollTarget.current = Math.min(index, column.turn.length - 2);
    onDelete(column.turn[index].id);
  }

  function syncActive() {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const top = viewport.getBoundingClientRect().top;
    const cards = viewport.querySelectorAll<HTMLElement>("[data-turn-index]");
    let nearest = 0;
    let distance = Infinity;
    cards.forEach((card, index) => {
      const current = Math.abs(card.getBoundingClientRect().top - top);
      if (current < distance) { nearest = index; distance = current; }
    });
    setActiveIndex(nearest);
  }

  return <section className={`session-column ${selected ? "session-column-selected" : ""}`} onClick={onSelect}>
    <div className="space-y-3 border-b border-zinc-800 p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="section-label">Session / 第 {column.history.length + 1} 轮</span>
        <div className="flex items-center gap-1.5">
          <Badge variant={column.sessionId ? "success" : "outline"}>{column.sessionId ? "已创建" : "临时"}</Badge>
          <Button aria-label="删除本列" title={!canDeleteColumn ? "至少保留一列，先点右侧「新建一列」" : column.loading ? "决策中不能删除本列" : column.deleting ? "正在删除本列" : "删除本列"} variant="outline" size="sm" disabled={!canDeleteColumn || column.loading || column.deleting} onClick={onDeleteColumn} className="shrink-0 text-zinc-300 hover:border-red-400/50 hover:text-red-300"><Trash2 size={13} />删除</Button>
                </div>
              </div>
      <Input aria-label="会话名" placeholder="会话名 · 回车创建" value={column.name} disabled={column.deleting} onChange={(event) => onColumnChange((old) => ({ ...old, name: event.target.value }))} className="font-medium" />
      <Textarea aria-label="该列背景" placeholder="该列的背景或上下文，只供这一列使用…" value={column.notes} disabled={column.loading || column.deleting} onChange={(event) => onColumnChange((old) => ({ ...old, notes: event.target.value }))} className="min-h-20 resize-y text-xs leading-relaxed" />
      <div className="flex items-center justify-between gap-2 border-t border-zinc-800 pt-3"><span className="text-xs font-medium text-zinc-300">本轮 {column.turn.length} 问</span><div className="flex items-center gap-1.5" aria-label={`本轮 ${column.turn.length} 问`}>{column.turn.map((question, index) => <button key={question.id} type="button" aria-label={`查看本轮第 ${index + 1} 题`} title={`第 ${index + 1} 题 · ${question.status ?? "待提交"}`} onClick={() => scrollToQuestion(index)} className={`turn-dot ${activeIndex === index ? "turn-dot-active" : ""} ${answered(question) ? "turn-dot-done" : ""}`}>{answered(question) && <Check size={10} strokeWidth={3} />}</button>)}</div></div>
            </div>
    <div ref={viewportRef} onScroll={syncActive} className="question-viewport scroll-thin">
      {column.history.map((round, index) => <div key={round[0].id} className="history-round px-3 pt-4"><div className="section-label mb-2">历史 · 第 {index + 1} 轮 / {round.length} 问</div><div className="space-y-2">{round.map((question) => <HistoryCard key={question.id} question={question} />)}</div></div>)}
      {column.turn.map((question, index) => <div key={question.id} data-turn-index={index} className="question-slide"><QuestionCard question={question} index={index} total={column.turn.length} disabled={Boolean(column.loading || column.deleting)} onChange={(change) => onQuestionChange(question.id, change)} onDelete={() => deleteQuestion(index)} /></div>)}
              </div>
    {column.turnError && <div role="alert" className="border-t border-amber-500/20 bg-amber-500/5 px-4 py-2 text-xs leading-5 text-amber-200">{column.turnError}</div>}
    <div className="flex gap-2 border-t border-zinc-800 px-3 py-3">
      <Button variant="outline" size="sm" disabled={column.loading || column.deleting} onClick={onAdd}><Plus size={13} />加一道</Button>
      <Button size="sm" disabled={column.loading || column.deleting} onClick={onDecision} className="flex-1"><Activity size={13} />决策</Button>
      <Button variant="outline" size="sm" disabled={!allAnswered || column.loading || column.deleting || Boolean(column.turnError)} onClick={onNext} title={!allAnswered ? "本轮全部题成功显示答案后才能进入下一轮" : "归档整轮并创建新题"}>下一轮 <ArrowRight size={13} /></Button>
    </div>
  </section>;
}

/** 一列一次 Engine 请求；多列请求仅在 UI 层并发。 */
export default function App() {
  const initial = useRef<Column[]>([newColumn()]);
  const columnsRef = useRef(initial.current);
  const [columns, setColumns] = useState(initial.current);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const [engine, setEngine] = useState("检查中");
  const [running, setRunning] = useState(0);
  const runningIds = useRef(new Set<string>());
  const creating = useRef(new Map<string, Promise<string>>());
  const boardRef = useRef<HTMLDivElement>(null);

  /** 同步 ref 与 React 状态，供并发请求读取最新列。 */
  const update = (id: string, change: (column: Column) => Column) => {
    const next = columnsRef.current.map((column) => column.id === id ? change(column) : column);
    columnsRef.current = next;
    setColumns(next);
  };
  const updateQuestion = (id: string, questionId: string, change: (question: Question) => Question) => {
    update(id, (column) => ({ ...column, turn: column.turn.map((question) => question.id === questionId ? change(question) : question) }));
  };

  useEffect(() => {
    let active = true;
    api<unknown>("/api/v1/health").then(() => { if (active) setEngine("在线"); }).catch(() => { if (active) setEngine("离线"); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const element = boardRef.current;
    if (!element) return;
    /** 题目视口保留原生纵向滚动，只有列外区域转为看板横向滚动。 */
    const wheel = (event: WheelEvent) => {
      if (event.target instanceof Element && event.target.closest(".question-viewport")) return;
      const scale = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 32 : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? element.clientWidth : 1;
      const delta = (event.deltaX || event.deltaY) * scale;
      if (element.scrollWidth <= element.clientWidth + 1 ||
          (delta < 0 && element.scrollLeft <= 0) ||
          (delta > 0 && element.scrollLeft + element.clientWidth >= element.scrollWidth - 1)) return;
      event.preventDefault();
      element.scrollLeft += delta;
    };
    element.addEventListener("wheel", wheel, { passive: false });
    return () => element.removeEventListener("wheel", wheel);
  }, []);

  /**
   * 本列首次决策时创建服务端会话。
   * session_id 使用 UUID 且不再随显示名变化；无显示名也会创建，以便写入 history_summary。
   * @param id 列的前端标识
   * @returns 绑定后的 session_id；列正在删除时为空
   */
  async function ensureSession(id: string): Promise<string | undefined> {
    const column = columnsRef.current.find((item) => item.id === id);
    if (!column || column.deleting) return;
    if (column.sessionId) return column.sessionId;
    const pending = creating.current.get(id);
    if (pending) return pending;
    const sessionId = crypto.randomUUID();
    const promise = api<Session>("/api/v1/sessions", token, "POST", { session_id: sessionId, state: { notes: column.notes } })
      .then((session) => {
        update(id, (old) => ({ ...old, sessionId: session.session_id, syncedNotes: column.notes }));
        return session.session_id;
      });
    creating.current.set(id, promise);
    try { return await promise; } finally { creating.current.delete(id); }
  }

  /** 校验整轮后一次提交所有合法题，并按 question_id 分发结果。 */
  async function submit(id: string, retried = false): Promise<string | undefined> {
    const column = columnsRef.current.find((item) => item.id === id);
    if (!column || column.deleting || runningIds.current.has(id)) return;
    const eligible = column.turn.filter((question) => !validate(question));
    const submitted = new Set(eligible.map((question) => question.id));
    update(id, (old) => ({ ...old, turnError: undefined, turn: old.turn.map((question) => {
      const error = validate(question);
      return error ? { ...question, status: "INVALID", result: undefined, error } : question;
    }) }));
    if (!eligible.length) {
      update(id, (old) => ({ ...old, turnError: "本轮没有可提交的题，请先填写题干和 2–16 个选项。" }));
      return;
    }
    runningIds.current.add(id);
    setRunning(runningIds.current.size);
    update(id, (old) => ({ ...old, loading: true, turn: old.turn.map((question) => submitted.has(question.id) ? { ...question, loading: true, error: undefined } : question) }));
    try {
      const sessionId = await ensureSession(id);
      const latest = columnsRef.current.find((item) => item.id === id);
      if (!latest || latest.turn[0].id !== column.turn[0].id) return;
      if (sessionId && latest.syncedNotes !== latest.notes) {
        await api<Session>(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, token, "PATCH", { state: { notes: latest.notes } });
        update(id, (old) => ({ ...old, syncedNotes: latest.notes }));
      }
      const response = await api<DecisionResponse>("/api/v1/decide", token, "POST", {
        ...(sessionId ? { session_id: sessionId } : { state: { notes: latest.notes } }),
        execution_mode: "EVALUATE",
        timeout_ms: 30000,
        decisions: eligible.map((question) => ({
          decision_id: question.id,
          ad_hoc: { type: question.type, instructions: question.instructions.trim(), candidates: question.candidates.map((candidate) => candidate.trim()) },
        })),
      });
      if (response.status === "OK") {
        const results = new Map(response.decisions.filter((item) => item.result).map((item) => [item.result!.question_id, item.result!]));
        const missing = eligible.filter((question) => !results.has(question.id));
        update(id, (old) => ({ ...old, turnError: missing.length ? "部分题未返回答案，请检查后重试。" : undefined, turn: old.turn.map((question) => {
          if (!submitted.has(question.id)) return question;
          const result = results.get(question.id);
          return result ? { ...question, status: "OK", result, error: undefined } : { ...question, status: "INTERNAL_ERROR", result: undefined, error: "当前题未返回答案。" };
        }) }));
        return missing.length ? "INTERNAL_ERROR" : "OK";
      }
      const message = response.status === "OVERLOADED" ? retried ? "重试后仍过载，请稍后手动再试。" : "引擎过载；本轮结束后只重试此列一次。" : "本轮决策未完成，请检查状态后重试。";
      update(id, (old) => ({ ...old, turnError: `${response.status} · ${message}` }));
      return response.status;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      update(id, (old) => ({ ...old, turnError: message }));
    } finally {
      update(id, (old) => ({ ...old, loading: false, turn: old.turn.map((question) => submitted.has(question.id) ? { ...question, loading: false } : question) }));
      runningIds.current.delete(id);
      setRunning(runningIds.current.size);
    }
  }

  /** 各列并发、每列一个请求；只对整轮过载的列再试一次。 */
  async function run(ids: string[]) {
    const targets = ids.filter((id) => !runningIds.current.has(id));
    const first = await Promise.all(targets.map(async (id) => ({ id, status: await submit(id) })));
    const retry = first.filter(({ id, status }) => status === "OVERLOADED" && columnsRef.current.find((column) => column.id === id)?.turnError?.startsWith("OVERLOADED")).map(({ id }) => id);
    if (retry.length) await Promise.all(retry.map(async (id) => { await submit(id, true); }));
  }

  function addColumn() {
    const column = newColumn();
    const next = [...columnsRef.current, column];
    columnsRef.current = next;
    setColumns(next);
    setSelectedId(column.id);
    requestAnimationFrame(() => boardRef.current?.scrollTo({ left: boardRef.current.scrollWidth, behavior: "smooth" }));
  }

  /** 已创建的会话先在服务端关闭；失败时保留整列与错误提示。 */
  async function deleteColumn(id: string) {
    const current = columnsRef.current.find((column) => column.id === id);
    if (!current || columnsRef.current.filter((column) => !column.deleting).length <= 1 || current.loading || current.deleting) return;
    update(id, (column) => ({ ...column, deleting: true, turnError: undefined }));
    try {
      const sessionId = await creating.current.get(id) ?? columnsRef.current.find((column) => column.id === id)?.sessionId;
      if (sessionId) await api<unknown>(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, token, "DELETE");
      const next = columnsRef.current.filter((column) => column.id !== id);
      columnsRef.current = next;
      setColumns(next);
      setSelectedId((selected) => selected === id ? next[0].id : selected);
    } catch (error) {
      update(id, (column) => ({ ...column, deleting: false, turnError: error instanceof Error ? error.message : String(error) }));
    }
  }

  const canRunAny = columns.some((column) => column.turn.some((question) => !validate(question)));
  return <main className="console-shell min-h-screen">
    <header className="border-b border-zinc-800 bg-[#111113] px-5 py-4 md:px-7"><div className="mx-auto flex max-w-[1800px] flex-wrap items-center justify-between gap-4">
      <div className="flex items-center gap-4"><div className="flex items-center gap-2.5"><div className="flex h-8 w-8 items-center justify-center rounded-md border border-cyan-400/30 bg-cyan-400/10 text-sm font-bold text-cyan-300">N</div><span className="text-sm font-bold tracking-[0.18em] text-zinc-100">NERIV</span></div><div className="h-5 w-px bg-zinc-800" /><span className="text-xs text-zinc-500">决策看板</span><span className="flex items-center gap-1.5 text-xs text-zinc-400"><span className={`h-1.5 w-1.5 rounded-full ${engine === "在线" ? "bg-emerald-400" : engine === "离线" ? "bg-red-400" : "bg-zinc-500"}`} />引擎{engine}</span></div>
      <div className="flex items-center gap-3"><span className="hidden text-xs text-zinc-500 sm:inline">{columns.length} 列 · {running ? `${running} 运行中` : "就绪"}</span><Input aria-label="Bearer token" type="password" autoComplete="off" placeholder="Bearer token · 可空" value={token} onChange={(event) => setToken(event.target.value)} className="h-8 w-44 text-xs md:w-56" /></div>
    </div></header>
    <div className="mx-auto max-w-[1800px] px-5 pb-6 pt-7 md:px-7">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3"><div><div className="section-label mb-2">WORKSPACE / SESSION BOARD</div><h1 className="text-xl font-semibold tracking-tight text-zinc-100">独立上下文，并行决策</h1><p className="mt-1 text-xs text-zinc-500">一列一份背景；一轮多题，同次请求；旧轮仅供查阅。</p></div><span className="text-xs text-zinc-600">列内上下翻题 · 列外横向换 Session</span></div>
      <div ref={boardRef} className="board-scroll scroll-thin">
        {columns.map((column) => <ColumnView key={column.id} column={column} selected={selectedId === column.id} onSelect={() => setSelectedId(column.id)}
          onColumnChange={(change) => update(column.id, change)}
          onQuestionChange={(questionId, change) => updateQuestion(column.id, questionId, change)}
          onDelete={(questionId) => update(column.id, (old) => old.loading || old.turn.length === 1 ? old : { ...old, turn: old.turn.filter((question) => question.id !== questionId), turnError: undefined })}
          onDeleteColumn={() => { void deleteColumn(column.id); }} canDeleteColumn={columns.filter((item) => !item.deleting).length > 1}
          onDecision={() => { void run([column.id]); }}
          onAdd={() => update(column.id, (old) => ({ ...old, turn: [...old.turn, newQuestion()], turnError: undefined }))}
          onNext={() => update(column.id, (old) => old.turn.every(answered) && !old.turnError ? { ...old, history: [...old.history, old.turn], turn: [newQuestion()], turnError: undefined } : old)}
        />)}
        <button type="button" onClick={addColumn} className="add-column flex h-28 shrink-0 flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-zinc-700 text-xs text-zinc-500 transition-colors hover:border-cyan-500/60 hover:text-cyan-300"><Plus size={20} />新建一列</button>
      </div>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-zinc-800 bg-[#171719] px-4 py-3"><div><div className="text-sm font-medium text-zinc-200">全部决策</div><p className="mt-0.5 text-xs text-zinc-500">每列整轮各提交一次；已有答案会覆盖，过载列只重试一次。</p></div><Button disabled={running > 0 || !canRunAny} title={!canRunAny ? "没有可提交的当前轮题目" : undefined} onClick={() => { void run(columnsRef.current.map((column) => column.id)); }}><RotateCcw size={14} />全部决策</Button></div>
      <p className="mt-4 text-center text-[11px] text-zinc-600">题目与历史仅保存在本次页面内存；刷新后不会恢复。</p>
    </div>
  </main>;
}
