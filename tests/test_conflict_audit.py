"""POST /conflict-audit 的端到端测试。"""

from __future__ import annotations

import copy
import random

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _audit(body: dict) -> dict:
    resp = client.post("/conflict-audit", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _index(data: dict) -> dict:
    """返回 {稿件: {评审人: 不合格原因条目}}，并校验结构与排序约束。"""
    out: dict[str, dict[str, dict]] = {}
    mids = [a["manuscript_id"] for a in data["audits"]]
    assert mids == sorted(mids), "稿件未按 ID 排序"
    for a in data["audits"]:
        eligible = a["eligible_reviewer_ids"]
        assert eligible == sorted(eligible), "合格评审人未按 ID 排序"
        pairs = a["ineligible_pairs"]
        rids = [p["reviewer_id"] for p in pairs]
        assert rids == sorted(rids), "不合格配对未按评审人 ID 排序"
        assert not set(eligible) & set(rids), "同一评审人同时合格与不合格"
        out[a["manuscript_id"]] = {p["reviewer_id"]: p for p in pairs}
    return out


# ---------- 基础规则：自稿 / 显式回避 / 主题 ----------


def test_basic_rules():
    body = {
        "max_hops": 1,
        "manuscripts": [
            {"id": "m1", "authors": ["alice"], "topics": ["ml"]},
            {"id": "m2", "authors": ["bob"], "topics": ["db"]},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["ml"], "capacity": 2},
            {"id": "r2", "topics": ["ml", "db"], "capacity": 1,
             "conflicted_authors": ["alice"]},
            {"id": "r3", "topics": ["ml", "db"], "capacity": 1,
             "conflicted_manuscripts": ["m2"]},
            {"id": "alice", "topics": ["ml"], "capacity": 1},
        ],
        "collaboration_edges": [],
    }
    data = _audit(body)
    assert data["max_hops"] == 1
    pairs = _index(data)

    assert pairs["m1"]["r2"]["reasons"] == ["author_conflict"]
    assert pairs["m1"]["r2"]["topic_mismatch"] is False
    assert pairs["m1"]["r2"]["collaboration_path"] is None
    assert pairs["m1"]["r2"]["hops"] is None
    # 自稿冲突
    assert pairs["m1"]["alice"]["reasons"] == ["self_conflict"]
    # r1 主题 ml 与 m2 的 db 不匹配：无利益冲突原因，仅单独标记
    p = pairs["m2"]["r1"]
    assert p["reasons"] == []
    assert p["topic_mismatch"] is True
    assert p["collaboration_path"] is None
    assert p["hops"] is None
    # 显式声明回避该稿件
    assert pairs["m2"]["r3"]["reasons"] == ["manuscript_conflict"]

    eligible_m1 = next(a for a in data["audits"] if a["manuscript_id"] == "m1")
    assert eligible_m1["eligible_reviewer_ids"] == ["r1", "r3"]


def test_topic_match_rules():
    """稿件 topics 为空时不做主题限制；有交集即匹配；全不匹配单独标记。"""
    body = {
        "max_hops": 1,
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": []},
            {"id": "m2", "authors": [], "topics": ["ml", "nlp"]},
        ],
        "reviewers": [
            {"id": "r1", "topics": [], "capacity": 1},
            {"id": "r2", "topics": ["nlp"], "capacity": 1},
            {"id": "r3", "topics": ["db"], "capacity": 1},
        ],
        "collaboration_edges": [],
    }
    pairs = _index(_audit(body))
    # 无主题稿件：评审人 topics 为空也合格
    assert "r1" not in pairs["m1"] and "r3" not in pairs["m1"]
    # 交集即合格
    assert "r2" not in pairs["m2"]
    p = pairs["m2"]["r3"]
    assert p["reasons"] == [] and p["topic_mismatch"] is True


def test_topic_mismatch_accumulates_with_conflict():
    """主题不匹配与利益冲突并存时：原因列出冲突，topic_mismatch 同时为真。"""
    body = {
        "max_hops": 1,
        "manuscripts": [
            {"id": "m1", "authors": ["alice"], "topics": ["ml"]},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["db"], "capacity": 1,
             "conflicted_authors": ["alice"]},
        ],
        "collaboration_edges": [],
    }
    pairs = _index(_audit(body))
    p = pairs["m1"]["r1"]
    assert p["reasons"] == ["author_conflict"]
    assert p["topic_mismatch"] is True


# ---------- 合作链利益冲突 ----------


def test_collaboration_chain_hops():
    body = {
        "max_hops": 2,
        "manuscripts": [
            {"id": "m1", "authors": ["alice"], "topics": ["ml"]},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["ml"], "capacity": 1},
            {"id": "r2", "topics": ["ml"], "capacity": 1},
            {"id": "r3", "topics": ["ml"], "capacity": 1},
        ],
        # r1 -- x -- alice（2 跳）；r2 --- y-z-alice（3 跳）；r3 直接合作（1 跳）
        "collaboration_edges": [
            ["r1", "x"], ["x", "alice"],
            ["r2", "y"], ["y", "z"], ["z", "alice"],
            ["r3", "alice"],
        ],
    }
    pairs = _index(_audit(body))
    assert pairs["m1"]["r1"]["reasons"] == ["collaboration_chain_conflict"]
    assert pairs["m1"]["r1"]["collaboration_path"] == ["r1", "x", "alice"]
    assert pairs["m1"]["r1"]["hops"] == 2
    assert pairs["m1"]["r3"]["collaboration_path"] == ["r3", "alice"]
    assert pairs["m1"]["r3"]["hops"] == 1
    # r2 需 3 跳 > max_hops=2，仍合格
    eligible = next(a for a in _audit(body)["audits"] if a["manuscript_id"] == "m1")
    assert eligible["eligible_reviewer_ids"] == ["r2"]


def test_chain_hops_boundaries():
    edges = [["s", "p1"], ["p1", "p2"], ["p2", "p3"],
             ["p3", "p4"], ["p4", "p5"], ["p5", "alice"]]
    base = {
        "manuscripts": [
            {"id": "m1", "authors": ["alice"], "topics": ["ml"]},
        ],
        "reviewers": [{"id": "s", "topics": ["ml"], "capacity": 1}],
        "collaboration_edges": edges,
    }

    # max_hops=1..5：6 跳链不构成冲突
    for hops in range(1, 6):
        body = {**base, "max_hops": hops}
        pairs = _index(_audit(body))
        assert "s" not in pairs["m1"], f"{hops} 跳不应判冲突（链长 6）"

    # max_hops=6：恰好 6 跳构成冲突
    body = {**base, "max_hops": 6}
    pairs = _index(_audit(body))
    p = pairs["m1"]["s"]
    assert p["reasons"] == ["collaboration_chain_conflict"]
    assert p["hops"] == 6
    assert p["collaboration_path"] == ["s", "p1", "p2", "p3", "p4", "p5", "alice"]


def test_max_hops_validation():
    base = {
        "manuscripts": [{"id": "m1", "authors": [], "topics": []}],
        "reviewers": [{"id": "r1", "topics": [], "capacity": 1}],
        "collaboration_edges": [],
    }
    for bad in (0, 7, -1):
        resp = client.post("/conflict-audit", json={**base, "max_hops": bad})
        assert resp.status_code == 422, bad
    resp = client.post("/conflict-audit", json={**base, "max_hops": 1})
    assert resp.status_code == 200


def test_chain_lexicographic_tie_break():
    """等长最短路径按完整 ID 序列字典序选取。"""
    # 首跳更小的路径字典序更小（无论作者 ID 大小）
    body = {
        "max_hops": 2,
        "manuscripts": [{"id": "m1", "authors": ["t0", "t1"], "topics": []}],
        "reviewers": [{"id": "s", "topics": [], "capacity": 1}],
        "collaboration_edges": [
            ["s", "a"], ["s", "b"], ["a", "t1"], ["b", "t0"],
        ],
    }
    pairs = _index(_audit(body))
    assert pairs["m1"]["s"]["collaboration_path"] == ["s", "a", "t1"]

    # 贪心陷阱：最小首跳 a 到不了任何作者，必须经 b
    body["collaboration_edges"] = [
        ["s", "a"], ["s", "b"], ["a", "u"], ["b", "t0"], ["u", "w"],
    ]
    pairs = _index(_audit(body))
    assert pairs["m1"]["s"]["collaboration_path"] == ["s", "b", "t0"]

    # 首跳相同：次跳字典序更小（3 跳链，需放宽 max_hops）
    body["max_hops"] = 3
    body["collaboration_edges"] = [
        ["s", "a"], ["a", "q"], ["a", "z"], ["q", "alice"], ["z", "alice"],
    ]
    body["manuscripts"] = [{"id": "m1", "authors": ["alice"], "topics": []}]
    pairs = _index(_audit(body))
    assert pairs["m1"]["s"]["collaboration_path"] == ["s", "a", "q", "alice"]


def test_chain_uses_shortest_over_lexicographic():
    """先取最短，再在等长路径中字典序：2 跳路径优先于 1 跳不存在时的长路径。"""
    body = {
        "max_hops": 3,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": []}],
        "reviewers": [{"id": "s", "topics": [], "capacity": 1}],
        # s->alice 有 3 跳经 a；另有 s->b->alice 的 2 跳（b 的 ID 更大）
        "collaboration_edges": [
            ["s", "a"], ["a", "x"], ["x", "alice"],
            ["s", "b"], ["b", "alice"],
        ],
    }
    pairs = _index(_audit(body))
    assert pairs["m1"]["s"]["hops"] == 2
    assert pairs["m1"]["s"]["collaboration_path"] == ["s", "b", "alice"]


def test_chain_to_any_author():
    """合作链只需到达稿件任一作者；多作者时选（最短, 字典序）综合最优路径。"""
    body = {
        "max_hops": 3,
        "manuscripts": [
            {"id": "m1", "authors": ["alice", "bob"], "topics": []},
        ],
        "reviewers": [{"id": "s", "topics": [], "capacity": 1}],
        # 到 alice 2 跳，到 bob 1 跳 -> 选更短的 bob 路径
        "collaboration_edges": [
            ["s", "x"], ["x", "alice"], ["s", "bob"],
        ],
    }
    pairs = _index(_audit(body))
    assert pairs["m1"]["s"]["collaboration_path"] == ["s", "bob"]
    assert pairs["m1"]["s"]["hops"] == 1


def test_self_author_not_counted_as_zero_hop_chain():
    """评审人本人是作者时走 self_conflict；合作链从 1 跳起算，不把本人当目标。"""
    # x 是作者之一，且 x -- bob（另一作者）有 1 跳合作：两原因并存
    body = {
        "max_hops": 2,
        "manuscripts": [{"id": "m1", "authors": ["x", "bob"], "topics": []}],
        "reviewers": [{"id": "x", "topics": [], "capacity": 1}],
        "collaboration_edges": [["x", "bob"]],
    }
    pairs = _index(_audit(body))
    p = pairs["m1"]["x"]
    assert p["reasons"] == ["self_conflict", "collaboration_chain_conflict"]
    assert p["collaboration_path"] == ["x", "bob"]
    assert p["hops"] == 1

    # x 是唯一作者，只有自稿冲突（无 0 跳合作链原因）
    body["manuscripts"] = [{"id": "m1", "authors": ["x"], "topics": []}]
    pairs = _index(_audit(body))
    assert pairs["m1"]["x"]["reasons"] == ["self_conflict"]
    assert pairs["m1"]["x"]["collaboration_path"] is None


def test_explicit_and_chain_conflict_coexist():
    """直接合作者同时被显式声明回避：两个原因都列出，路径仍返回。"""
    body = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": []}],
        "reviewers": [
            {"id": "r1", "topics": [], "capacity": 1,
             "conflicted_authors": ["alice"]},
        ],
        "collaboration_edges": [["r1", "alice"]],
    }
    pairs = _index(_audit(body))
    p = pairs["m1"]["r1"]
    assert p["reasons"] == ["author_conflict", "collaboration_chain_conflict"]
    assert p["collaboration_path"] == ["r1", "alice"]
    assert p["hops"] == 1


def test_reasons_fixed_order():
    """多原因并存时按 self / author / manuscript / chain 的固定顺序输出。"""
    body = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": ["ml"]}],
        "reviewers": [
            {"id": "alice", "topics": ["db"], "capacity": 1,
             "conflicted_authors": ["alice"], "conflicted_manuscripts": ["m1"]},
        ],
        "collaboration_edges": [["alice", "alice"]],
    }
    pairs = _index(_audit(body))
    p = pairs["m1"]["alice"]
    assert p["reasons"] == [
        "self_conflict",
        "author_conflict",
        "manuscript_conflict",
    ]
    assert p["topic_mismatch"] is True


def test_intermediary_need_not_be_author_or_reviewer():
    """合作边中的中间人员无需是作者或评审人。"""
    body = {
        "max_hops": 2,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": []}],
        "reviewers": [{"id": "r1", "topics": [], "capacity": 1}],
        "collaboration_edges": [["r1", "someone-else"], ["someone-else", "alice"]],
    }
    pairs = _index(_audit(body))
    assert pairs["m1"]["r1"]["collaboration_path"] == ["r1", "someone-else", "alice"]


# ---------- 确定性：重复边、反向边、输入重排 ----------


def test_deterministic_under_duplicate_and_shuffle():
    body = {
        "max_hops": 3,
        "manuscripts": [
            {"id": "m1", "authors": ["alice", "carol"], "topics": ["ml", "db"]},
            {"id": "m2", "authors": ["bob"], "topics": ["nlp"]},
            {"id": "m3", "authors": ["dave"], "topics": []},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["ml"], "capacity": 2,
             "conflicted_authors": ["bob"]},
            {"id": "r2", "topics": ["nlp", "db"], "capacity": 1,
             "conflicted_manuscripts": ["m1"]},
            {"id": "r3", "topics": [], "capacity": 1},
            {"id": "r4", "topics": ["ml", "nlp"], "capacity": 1},
        ],
        "collaboration_edges": [
            ["r1", "x"], ["x", "alice"],
            ["r2", "y"], ["y", "z"], ["z", "carol"],
            ["r3", "dave"],
            ["u", "r4"], ["bob", "u"],
            # 重复边与反向边
            ["r1", "x"], ["x", "r1"], ["r1", "r1"],
            ["z", "y"], ["y", "r2"],
        ],
    }
    baseline = _audit(body)

    shuffled = copy.deepcopy(body)
    rng = random.Random(20260927)
    for _ in range(20):
        rng.shuffle(shuffled["manuscripts"])
        rng.shuffle(shuffled["reviewers"])
        rng.shuffle(shuffled["collaboration_edges"])
        assert _audit(shuffled) == baseline, "输入重排改变了审计结果"

    # 无合作边时结果与上述（去掉链原因后）不同对象但结构稳定：重复请求幂等
    again = _audit(copy.deepcopy(body))
    assert again == baseline


def test_response_sorted_by_id():
    body = {
        "max_hops": 1,
        "manuscripts": [
            {"id": "m9", "authors": [], "topics": []},
            {"id": "m1", "authors": [], "topics": []},
            {"id": "m5", "authors": [], "topics": []},
        ],
        "reviewers": [
            {"id": "r9", "topics": [], "capacity": 1},
            {"id": "r1", "topics": [], "capacity": 1},
        ],
        "collaboration_edges": [],
    }
    data = _audit(body)
    assert [a["manuscript_id"] for a in data["audits"]] == ["m1", "m5", "m9"]
    for a in data["audits"]:
        assert a["eligible_reviewer_ids"] == ["r1", "r9"]
        assert a["ineligible_pairs"] == []


# ---------- 输入校验 ----------


def test_duplicate_ids_rejected():
    body = {
        "max_hops": 1,
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": []},
            {"id": "m1", "authors": [], "topics": []},
        ],
        "reviewers": [{"id": "r1", "topics": [], "capacity": 1}],
        "collaboration_edges": [],
    }
    assert client.post("/conflict-audit", json=body).status_code == 422

    body["manuscripts"] = [{"id": "m1", "authors": [], "topics": []}]
    body["reviewers"] = [
        {"id": "r1", "topics": [], "capacity": 1},
        {"id": "r1", "topics": [], "capacity": 1},
    ]
    assert client.post("/conflict-audit", json=body).status_code == 422


def test_bad_edge_shape_rejected():
    base = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": [], "topics": []}],
        "reviewers": [{"id": "r1", "topics": [], "capacity": 1}],
    }
    # 边必须恰好 2 个端点
    for edges in ([["r1"]], [["r1", "a", "b"]], [[""]]):
        resp = client.post(
            "/conflict-audit", json={**base, "collaboration_edges": edges}
        )
        assert resp.status_code == 422, edges
    # 空字符串 ID 不允许
    resp = client.post(
        "/conflict-audit",
        json={**base, "collaboration_edges": [["r1", ""]]},
    )
    assert resp.status_code == 422


def test_extra_field_rejected():
    body = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": [], "topics": [], "bogus": 1}],
        "reviewers": [{"id": "r1", "topics": [], "capacity": 1}],
        "collaboration_edges": [],
    }
    assert client.post("/conflict-audit", json=body).status_code == 422


# ---------- 审计不依赖容量 / 不改动分配 ----------


def test_capacity_ignored_by_audit():
    """资格审计不考虑容量：capacity=0 不影响合格判定。"""
    body = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": [], "topics": ["ml"]}],
        "reviewers": [
            {"id": "r1", "topics": ["ml"], "capacity": 0},
            {"id": "r2", "topics": ["ml"], "capacity": 0,
             "track_capacities": {"poster": 0}},
        ],
        "collaboration_edges": [],
    }
    data = _audit(body)
    a = next(x for x in data["audits"] if x["manuscript_id"] == "m1")
    assert a["eligible_reviewer_ids"] == ["r1", "r2"]
    assert a["ineligible_pairs"] == []


def test_audit_does_not_mutate_assignment_inputs():
    """审计是只读接口：不会因任何审计字段产生分配结果字段。"""
    body = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": [], "topics": []}],
        "reviewers": [{"id": "r1", "topics": [], "capacity": 0}],
        "collaboration_edges": [],
    }
    data = _audit(body)
    assert set(data.keys()) == {"max_hops", "audits"}


def test_empty_inputs():
    body = {"max_hops": 6, "manuscripts": [], "reviewers": [],
            "collaboration_edges": []}
    data = _audit(body)
    assert data == {"max_hops": 6, "audits": []}

    # 有稿件无评审人
    body["manuscripts"] = [{"id": "m1", "authors": [], "topics": []}]
    data = _audit(body)
    assert data["audits"] == [{
        "manuscript_id": "m1",
        "eligible_reviewer_ids": [],
        "ineligible_pairs": [],
    }]
