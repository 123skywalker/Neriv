/* api.ts, 控制台 REST 客户端与响应类型 */

/** 服务端会话快照；session_id 创建后不可改。 */
export interface Session {
  session_id: string;
  snapshot_id: string;
  state_version: number;
  state: Record<string, unknown>;
  status: string;
}
/** 单候选概率。 */
export interface Candidate {
  candidate_id: string;
  probability: number;
}
/** 单题指针与拒答结果。 */
export interface DecisionResult {
  question_id: string;
  candidates: Candidate[];
  reject_probability: number;
  selected_candidate_id: string | null;
}
/** decide 里一道题的结果槽。 */
export interface DecisionOutcome {
  result: DecisionResult | null;
}
/** 整轮决策响应。 */
export interface DecisionResponse {
  request_id: string;
  snapshot_id: string;
  status: string;
  decisions: DecisionOutcome[];
}
/** HTTP 非 2xx 时抛出，detail 来自响应体。 */
export class ApiError extends Error {
  constructor(
    public status: number,
    detail: unknown,
  ) {
    super(
      typeof detail === "string"
        ? detail
        : JSON.stringify(detail ?? `HTTP ${status}`),
    );
  }
}

/** 统一调用 REST；身份只通过 Bearer 头传递。 */
export async function api<T>(
  path: string,
  token?: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: {
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      data && typeof data === "object" && "detail" in data ? data.detail : data;
    throw new ApiError(response.status, detail);
  }
  return data as T;
}
