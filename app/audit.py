"""资格审计（POST /conflict-audit）。

审计每个 (稿件, 评审人) 配对是否具备评审资格，规则与 /assign 系列接口保持一致：

- 自稿冲突：评审人是稿件作者；
- 显式回避：评审人 conflicted_authors 命中稿件作者，或 conflicted_manuscripts
  命中稿件 ID；
- 主题匹配：稿件 topics 非空时，评审人 topics 须与其有交集（主题不匹配单独
  标记，不属于利益冲突原因）；
- 合作链利益冲突：在由人员 ID 组成的无向合作图上，评审人到稿件任一作者存在
  长度不超过 max_hops 跳的路径。

合作链冲突返回「最短人员路径」；最短路径有多条等长路径时，按完整 ID 序列
（含评审人与作者端点）字典序取最小者。合作图由无向边集合构造：重复边、反向
边与自环一律忽略，结果只取决于边的集合与业务输入的 ID 集合——重复边及输入
列表重排不改变任何输出。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .solver import ManuscriptSpec, ReviewerSpec

# 利益冲突原因在输出中的固定顺序（主题不匹配不在此列，单独标记）
REASON_ORDER: tuple[str, ...] = (
    "self_conflict",
    "author_conflict",
    "manuscript_conflict",
    "collaboration_chain_conflict",
)
_REASON_RANK = {reason: i for i, reason in enumerate(REASON_ORDER)}


@dataclass(frozen=True)
class IneligiblePair:
    reviewer_id: str
    reasons: tuple[str, ...]
    topic_mismatch: bool
    # 合作链冲突时的最短人员路径 [评审人, ..., 作者]；无该原因为 None
    collaboration_path: tuple[str, ...] | None
    # 最短合作链跳数（= len(collaboration_path) - 1）；无该原因为 None
    hops: int | None


@dataclass(frozen=True)
class ManuscriptAudit:
    manuscript_id: str
    eligible_reviewers: tuple[str, ...]
    ineligible_pairs: tuple[IneligiblePair, ...]


@dataclass(frozen=True)
class AuditResult:
    max_hops: int
    audits: tuple[ManuscriptAudit, ...]


def _build_graph(
    edges: frozenset[tuple[str, str]],
) -> dict[str, set[str]]:
    """把无向边集合展开为邻接表。自环（两端相同）忽略；重复/反向边天然去重。"""
    graph: dict[str, set[str]] = {}
    for a, b in edges:
        if a == b:
            continue
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph


def _bfs_layers(
    graph: dict[str, set[str]],
    start: str,
    max_hops: int,
) -> list[set[str]]:
    """从 start 出发在无向图上做 1～max_hops 跳的分层 BFS。

    返回 layers[d]（d = 1..max_hops）：距 start 恰好 d 跳的人员集合。
    邻接顺序不影响距离集合（路径选取在 _shortest_chain_path 中按 ID 字典序
    确定性完成），故结果与边输入顺序无关。
    """
    layers: list[set[str]] = [set() for _ in range(max_hops + 1)]
    if start not in graph:
        return layers
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        d = dist[u]
        if d >= max_hops:
            continue
        for v in graph[u]:
            if v not in dist:
                dist[v] = d + 1
                layers[d + 1].add(v)
                q.append(v)
    return layers


def _shortest_chain_path(
    graph: dict[str, set[str]],
    start: str,
    authors: set[str],
    layers: list[set[str]],
    shortest: int,
) -> tuple[str, ...]:
    """返回 start -> 某作者的最短人员路径（跳数 shortest）。

    等长路径按完整 ID 序列字典序取最小：自起点逐跳选取「能在剩余跳数内到达
    某个最短距离作者」的最小 ID 邻点。可到达性经反向分层集合（good）判定，
    贪心每步取最小 ID 恰好得到字典序最小的完整序列。无向图上的 BFS 简单路径
    天然无环（每个后续节点严格更接近目标层）。
    """
    # good[d]：第 d 跳可落在的人员集合——到某个「恰为 shortest 跳」作者仍有
    # shortest - d 跳路径的节点
    target = authors & layers[shortest]
    good: list[set[str]] = [set() for _ in range(shortest + 1)]
    good[shortest] = set(target)
    for d in range(shortest - 1, -1, -1):
        good[d] = {
            u
            for u in layers[d] | ({start} if d == 0 else set())
            if graph.get(u, set()) & good[d + 1]
        }

    path = [start]
    for d in range(shortest):
        cur = path[-1]
        nxt = min(v for v in graph.get(cur, set()) if v in good[d + 1])
        path.append(nxt)
    return tuple(path)


def audit_conflicts(
    manuscripts: list[ManuscriptSpec],
    reviewers: list[ReviewerSpec],
    collaboration_edges: frozenset[tuple[str, str]],
    max_hops: int,
) -> AuditResult:
    """审计全部 (稿件, 评审人) 配对的资格。

    输出按稿件 ID、评审人 ID 确定排序；同一业务输入（含重复边、列表乱序）
    必然产生完全相同的审计结果。本函数只做资格判定，不涉及任何分配结果。
    """
    ms = sorted(manuscripts, key=lambda m: m.id)
    rs = sorted(reviewers, key=lambda r: r.id)
    graph = _build_graph(collaboration_edges)

    # 每个评审人只做一次分层 BFS，供其涉及的全部稿件复用
    layers_by_reviewer = {
        r.id: _bfs_layers(graph, r.id, max_hops) for r in rs
    }

    audits: list[ManuscriptAudit] = []
    for m in ms:
        authors = set(m.authors)
        m_topics = set(m.topics)
        eligible: list[str] = []
        ineligible: list[IneligiblePair] = []
        for r in rs:
            reasons: list[str] = []
            if r.id in authors:
                reasons.append("self_conflict")
            if authors.intersection(r.conflicted_authors):
                reasons.append("author_conflict")
            if m.id in r.conflicted_manuscripts:
                reasons.append("manuscript_conflict")

            # 合作链冲突：评审人到稿件任一作者的最短合作链不超过 max_hops 跳。
            # 作者就是评审人本人（0 跳、自稿）的情形由 self_conflict 覆盖，
            # 合作链只计 1 跳及以上的路径，故作者集合排除评审人自身。
            chain_path: tuple[str, ...] | None = None
            chain_hops: int | None = None
            other_authors = authors - {r.id}
            if other_authors:
                layers = layers_by_reviewer[r.id]
                shortest = next(
                    (
                        d
                        for d in range(1, max_hops + 1)
                        if other_authors & layers[d]
                    ),
                    None,
                )
                if shortest is not None:
                    reasons.append("collaboration_chain_conflict")
                    chain_path = _shortest_chain_path(
                        graph, r.id, other_authors, layers, shortest
                    )
                    chain_hops = shortest

            topic_mismatch = bool(
                m_topics and not m_topics.intersection(r.topics)
            )

            if not reasons and not topic_mismatch:
                eligible.append(r.id)
            else:
                ineligible.append(IneligiblePair(
                    reviewer_id=r.id,
                    reasons=tuple(sorted(reasons, key=_REASON_RANK.__getitem__)),
                    topic_mismatch=topic_mismatch,
                    collaboration_path=chain_path,
                    hops=chain_hops,
                ))
        audits.append(ManuscriptAudit(
            manuscript_id=m.id,
            eligible_reviewers=tuple(eligible),
            ineligible_pairs=tuple(ineligible),
        ))

    return AuditResult(max_hops=max_hops, audits=tuple(audits))
