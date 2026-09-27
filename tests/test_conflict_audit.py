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


def _ineligible_map(data: dict) -> dict[str, dict[str, dict]]:
    """响应 -> {manuscript_id: {reviewer_id: pair}}。"""
    return {
        a["manuscript_id"]: {p["reviewer_id"]: p for p in a["ineligible"]}
        for a in data["manuscripts"]
    }


def _eligible_map(data: dict) -> dict[str, list[str]]:
    return {a["manuscript_id"]: list(a["eligible_reviewer_ids"]) for a in data["manuscripts"]}


def _reviewer(topics=None, **kw) -> dict:
    payload = {"id": kw.pop("id"),
               "topics": sorted(topics) if topics is not None else ["t"],
               "capacity": 1}
    payload.update(kw)
    return payload


# ---------- 无合作边时：自稿 / 显式回避 / 主题匹配规则与 /assign 一致 ----------


def test_base_rules_without_edges():
    body = {
        "max_hops": 1,
        "collaboration_edges": [],
        "manuscripts": [
            {"id": "m1", "authors": ["alice"], "topics": ["ml"]},
        ],
        "reviewers": [
            _reviewer({"ml"}, id="r1"),
            _reviewer({"ml"}, id="alice"),
            _reviewer({"ml"}, id="r2", conflicted_authors=["alice"]),
            _reviewer({"ml"}, id="r3", conflicted_manuscripts=["m1"]),
            _reviewer({"db"}, id="r4"),
            _reviewer({"db"}, id="alice2", conflicted_authors=["alice"],
                      conflicted_manuscripts=["m1"]),
        ],
    }
    data = _audit(body)
    audit = data["manuscripts"][0]
    assert audit["eligible_reviewer_ids"] == ["r1"]
    pairs = {p["reviewer_id"]: p for p in audit["ineligible"]}
    assert pairs["alice"]["reasons"] == ["self_conflict"]
    assert pairs["alice"]["topic_mismatch"] is False
    assert pairs["alice"]["collaboration_path"] is None
    assert pairs["r2"]["reasons"] == ["author_conflict"]
    assert pairs["r3"]["reasons"] == ["manuscript_conflict"]
    assert pairs["r4"]["reasons"] == []
    assert pairs["r4"]["topic_mismatch"] is True
    assert pairs["alice2"]["reasons"] == ["author_conflict", "manuscript_conflict"]
    assert pairs["alice2"]["topic_mismatch"] is True
    assert data == {
        "manuscripts": data["manuscripts"],
        "total_pairs": 6,
        "eligible_pairs": 1,
        "ineligible_pairs": 5,
    }


def test_manuscript_without_topics_never_topic_mismatch():
    body = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": [], "topics": []}],
        "reviewers": [_reviewer([], id="r1"), _reviewer({"db"}, id="r2")],
    }
    data = _audit(body)
    audit = data["manuscripts"][0]
    assert audit["eligible_reviewer_ids"] == ["r1", "r2"]
    assert audit["ineligible"] == []


# ---------- 合作链冲突 ----------


def test_direct_edge_one_hop_conflict():
    body = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": ["t"]}],
        "reviewers": [_reviewer(id="r1"), _reviewer(id="r2")],
        "collaboration_edges": [{"person_ids": ["r1", "alice"]}],
    }
    data = _audit(body)
    pairs = _ineligible_map(data)["m1"]
    assert pairs["r1"]["reasons"] == ["collaboration_chain_conflict"]
    assert pairs["r1"]["collaboration_path"] == {
        "person_ids": ["r1", "alice"], "hops": 1,
    }
    assert "r2" not in pairs
    assert _eligible_map(data)["m1"] == ["r2"]


def test_chain_length_boundary():
    """链长恰好 max_hops 算冲突；多 1 跳则不算。"""
    # r1 - x - y - alice（3 跳）
    edges = [
        {"person_ids": ["r1", "x"]},
        {"person_ids": ["x", "y"]},
        {"person_ids": ["y", "alice"]},
    ]
    base = {
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": ["t"]}],
        "reviewers": [_reviewer(id="r1")],
        "collaboration_edges": edges,
    }
    body = {**base, "max_hops": 3}
    pairs = _ineligible_map(_audit(body))["m1"]
    assert pairs["r1"]["collaboration_path"] == {
        "person_ids": ["r1", "x", "y", "alice"], "hops": 3,
    }
    body = {**base, "max_hops": 2}
    assert _eligible_map(_audit(body))["m1"] == ["r1"]
    body = {**base, "max_hops": 1}
    assert _eligible_map(_audit(body))["m1"] == ["r1"]


def test_shortest_path_preferred_over_lex_smaller_longer_path():
    # [r1, z, alice]：2 跳；[r1, a, b, alice]：3 跳（字典序更小但更长）
    body = {
        "max_hops": 3,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": ["t"]}],
        "reviewers": [_reviewer(id="r1")],
        "collaboration_edges": [
            {"person_ids": ["r1", "z"]}, {"person_ids": ["z", "alice"]},
            {"person_ids": ["r1", "a"]}, {"person_ids": ["a", "b"]},
            {"person_ids": ["b", "alice"]},
        ],
    }
    pairs = _ineligible_map(_audit(body))["m1"]
    assert pairs["r1"]["collaboration_path"] == {
        "person_ids": ["r1", "z", "alice"], "hops": 2,
    }


def test_equal_length_paths_lex_by_full_id_sequence():
    # 三条等长 2 跳链，经过 a / b / c；按完整 ID 序列字典序取经过 a 的
    body = {
        "max_hops": 2,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": ["t"]}],
        "reviewers": [_reviewer(id="r1")],
        "collaboration_edges": [
            {"person_ids": ["r1", "c"]}, {"person_ids": ["c", "alice"]},
            {"person_ids": ["r1", "b"]}, {"person_ids": ["b", "alice"]},
            {"person_ids": ["r1", "a"]}, {"person_ids": ["a", "alice"]},
        ],
    }
    pairs = _ineligible_map(_audit(body))["m1"]
    assert pairs["r1"]["collaboration_path"] == {
        "person_ids": ["r1", "a", "alice"], "hops": 2,
    }


def test_lex_tie_break_on_full_sequence_not_first_neighbor_only():
    # 两条等长 3 跳链共享前缀 [r1, x]，在第二跳分岔：a < b，
    # 必须按完整 ID 序列比较（仅按第一跳排序无法区分）
    body = {
        "max_hops": 3,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": ["t"]}],
        "reviewers": [_reviewer(id="r1")],
        "collaboration_edges": [
            {"person_ids": ["r1", "x"]},
            {"person_ids": ["x", "b"]}, {"person_ids": ["b", "alice"]},
            {"person_ids": ["x", "a"]}, {"person_ids": ["a", "alice"]},
        ],
    }
    pairs = _ineligible_map(_audit(body))["m1"]
    assert pairs["r1"]["collaboration_path"] == {
        "person_ids": ["r1", "x", "a", "alice"], "hops": 3,
    }


def test_multiple_authors_choose_closest_author():
    # 到 bob 2 跳，到 alice 3 跳 -> 取到 bob 的最短链
    body = {
        "max_hops": 3,
        "manuscripts": [{"id": "m1", "authors": ["alice", "bob"], "topics": ["t"]}],
        "reviewers": [_reviewer(id="r1")],
        "collaboration_edges": [
            {"person_ids": ["r1", "x"]}, {"person_ids": ["x", "bob"]},
            {"person_ids": ["r1", "y"]}, {"person_ids": ["y", "z"]},
            {"person_ids": ["z", "alice"]},
        ],
    }
    pairs = _ineligible_map(_audit(body))["m1"]
    assert pairs["r1"]["collaboration_path"] == {
        "person_ids": ["r1", "x", "bob"], "hops": 2,
    }


def test_self_loop_edge_ignored():
    body = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": ["t"]}],
        "reviewers": [_reviewer(id="r1")],
        "collaboration_edges": [{"person_ids": ["r1", "r1"]}],
    }
    assert _eligible_map(_audit(body))["m1"] == ["r1"]


def test_self_conflict_is_zero_hops_not_chain():
    # 评审人本人就是作者（0 跳）：只报 self_conflict，不产生合作链
    body = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": ["r1", "alice"], "topics": ["t"]}],
        "reviewers": [_reviewer(id="r1")],
        "collaboration_edges": [{"person_ids": ["r1", "alice"]}],
    }
    pairs = _ineligible_map(_audit(body))["m1"]
    assert pairs["r1"]["reasons"] == ["self_conflict"]
    assert pairs["r1"]["collaboration_path"] is None


def test_chain_conflict_and_topic_mismatch_marked_separately():
    body = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": ["ml"]}],
        "reviewers": [
            _reviewer({"db"}, id="r1"),  # 主题不匹配且 1 跳合作链
            _reviewer({"ml"}, id="r2", conflicted_authors=["alice"]),  # 显式冲突 + 合作链
        ],
        "collaboration_edges": [
            {"person_ids": ["r1", "alice"]},
            {"person_ids": ["r2", "alice"]},
        ],
    }
    pairs = _ineligible_map(_audit(body))["m1"]
    assert pairs["r1"]["reasons"] == ["collaboration_chain_conflict"]
    assert pairs["r1"]["topic_mismatch"] is True
    assert pairs["r1"]["collaboration_path"] == {
        "person_ids": ["r1", "alice"], "hops": 1,
    }
    # 显式 author_conflict 与合作链冲突并列，原因顺序固定
    assert pairs["r2"]["reasons"] == [
        "author_conflict", "collaboration_chain_conflict",
    ]
    assert pairs["r2"]["topic_mismatch"] is False


def test_chain_through_unknown_person_ids_allowed():
    """合作边可以引用既非评审人也非作者的人员 ID。"""
    body = {
        "max_hops": 6,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": ["t"]}],
        "reviewers": [_reviewer(id="r1")],
        "collaboration_edges": [
            {"person_ids": ["r1", "p1"]},
            {"person_ids": ["p1", "p2"]},
            {"person_ids": ["p2", "p3"]},
            {"person_ids": ["p3", "alice"]},
        ],
    }
    pairs = _ineligible_map(_audit(body))["m1"]
    assert pairs["r1"]["collaboration_path"]["person_ids"] == [
        "r1", "p1", "p2", "p3", "alice",
    ]
    # max_hops=3 时 4 跳链不算
    body["max_hops"] = 3
    assert _eligible_map(_audit(body))["m1"] == ["r1"]


# ---------- 排序、去重与确定性 ----------


def test_response_sorted_by_id():
    body = {
        "max_hops": 1,
        "manuscripts": [
            {"id": "m3", "authors": [], "topics": ["t"]},
            {"id": "m1", "authors": [], "topics": ["t"]},
            {"id": "m2", "authors": [], "topics": ["t"]},
        ],
        "reviewers": [
            _reviewer(id="r9"), _reviewer(id="r2"), _reviewer({"x"}, id="r5"),
        ],
    }
    data = _audit(body)
    assert [a["manuscript_id"] for a in data["manuscripts"]] == ["m1", "m2", "m3"]
    for audit in data["manuscripts"]:
        assert audit["eligible_reviewer_ids"] == ["r2", "r9"]
        assert [p["reviewer_id"] for p in audit["ineligible"]] == ["r5"]


def test_duplicate_reversed_and_reordered_edges_give_same_result():
    manuscripts = [{"id": "m1", "authors": ["alice"], "topics": ["t"]}]
    reviewers = [_reviewer(id="r1"), _reviewer(id="r2")]
    edges = [
        {"person_ids": ["r1", "x"]}, {"person_ids": ["x", "alice"]},
        {"person_ids": ["r2", "y"]}, {"person_ids": ["y", "z"]},
        {"person_ids": ["z", "alice"]},
    ]
    body = {"max_hops": 3, "manuscripts": manuscripts, "reviewers": reviewers,
            "collaboration_edges": edges}
    expected = _audit(body)

    # 重复边（含方向互换）、自环、边顺序完全重排
    shuffled_edges = [
        {"person_ids": ["alice", "x"]},          # 方向互换
        {"person_ids": ["z", "y"]},
        {"person_ids": ["r1", "x"]},             # 重复
        {"person_ids": ["x", "r1"]},             # 重复且方向互换
        {"person_ids": ["y", "z"]},
        {"person_ids": ["z", "alice"]},
        {"person_ids": ["r2", "y"]},
        {"person_ids": ["x", "x"]},              # 自环
    ]
    random.Random(7).shuffle(shuffled_edges)
    body2 = {**body, "collaboration_edges": shuffled_edges}
    assert _audit(body2) == expected


def test_input_reorder_does_not_change_result():
    body = {
        "max_hops": 2,
        "manuscripts": [
            {"id": "m2", "authors": ["bob", "carol"], "topics": ["db", "ml"]},
            {"id": "m1", "authors": ["alice"], "topics": ["ml"]},
        ],
        "reviewers": [
            _reviewer({"ml"}, id="r3", conflicted_authors=["bob"]),
            _reviewer({"ml", "db"}, id="r1"),
            _reviewer({"db"}, id="r2"),
        ],
        "collaboration_edges": [
            {"person_ids": ["r1", "x"]}, {"person_ids": ["x", "alice"]},
            {"person_ids": ["r2", "u"]}, {"person_ids": ["u", "v"]},
            {"person_ids": ["v", "bob"]}, {"person_ids": ["r3", "carol"]},
        ],
    }
    expected = _audit(body)
    rng = random.Random(42)
    for _ in range(5):
        shuffled = copy.deepcopy(body)
        rng.shuffle(shuffled["manuscripts"])
        rng.shuffle(shuffled["reviewers"])
        rng.shuffle(shuffled["collaboration_edges"])
        for m in shuffled["manuscripts"]:
            rng.shuffle(m["authors"])
            rng.shuffle(m["topics"])
        for r in shuffled["reviewers"]:
            rng.shuffle(r["topics"])
            if "conflicted_authors" in r:
                rng.shuffle(r["conflicted_authors"])
        assert _audit(shuffled) == expected


def test_collaboration_edges_defaults_to_empty():
    body = {
        "max_hops": 1,
        "manuscripts": [{"id": "m1", "authors": [], "topics": ["t"]}],
        "reviewers": [_reviewer(id="r1")],
    }
    data = _audit(body)
    assert data["manuscripts"][0]["eligible_reviewer_ids"] == ["r1"]


def test_empty_inputs():
    data = _audit({"max_hops": 1, "manuscripts": [], "reviewers": []})
    assert data == {
        "manuscripts": [],
        "total_pairs": 0,
        "eligible_pairs": 0,
        "ineligible_pairs": 0,
    }


# ---------- 只读：不影响 /assign 分配结果 ----------


def test_audit_does_not_change_assignment():
    body = {
        "required_reviewers": 1,
        "manuscripts": [{"id": "m1", "authors": ["alice"], "topics": ["t"]}],
        "reviewers": [_reviewer(id="r1")],
    }
    assign_before = client.post("/assign", json=body).json()
    audit_body = {
        "max_hops": 1,
        "manuscripts": body["manuscripts"],
        "reviewers": body["reviewers"],
        "collaboration_edges": [{"person_ids": ["r1", "alice"]}],
    }
    audit = _audit(audit_body)
    # 审计判定合作链冲突……
    assert audit["manuscripts"][0]["ineligible"][0]["reasons"] == [
        "collaboration_chain_conflict",
    ]
    # ……但 /assign 不接收合作边，分配结果不变：r1 仍可评审 m1
    assign_after = client.post("/assign", json=body).json()
    assert assign_after == assign_before
    assert assign_after["assignments"] == [{"manuscript_id": "m1", "reviewer_ids": ["r1"]}]


# ---------- 请求校验 ----------


def test_max_hops_out_of_range_422():
    base = {"manuscripts": [], "reviewers": []}
    for bad in (0, -1, 7, 100):
        resp = client.post("/conflict-audit", json={**base, "max_hops": bad})
        assert resp.status_code == 422, bad


def test_max_hops_required_422():
    resp = client.post("/conflict-audit", json={"manuscripts": [], "reviewers": []})
    assert resp.status_code == 422


def test_edge_must_have_exactly_two_person_ids():
    base = {
        "max_hops": 1,
        "manuscripts": [],
        "reviewers": [],
    }
    for bad in ([], ["r1"], ["r1", "r2", "r3"], [""]):
        resp = client.post(
            "/conflict-audit", json={**base, "collaboration_edges": [{"person_ids": bad}]}
        )
        assert resp.status_code == 422, bad


def test_duplicate_ids_422():
    resp = client.post("/conflict-audit", json={
        "max_hops": 1,
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": []},
            {"id": "m1", "authors": [], "topics": []},
        ],
        "reviewers": [],
    })
    assert resp.status_code == 422
    resp = client.post("/conflict-audit", json={
        "max_hops": 1,
        "manuscripts": [],
        "reviewers": [_reviewer(id="r1"), _reviewer(id="r1")],
    })
    assert resp.status_code == 422


def test_extra_field_422():
    resp = client.post("/conflict-audit", json={
        "max_hops": 1,
        "manuscripts": [],
        "reviewers": [],
        "bogus": 1,
    })
    assert resp.status_code == 422
    resp = client.post("/conflict-audit", json={
        "max_hops": 1,
        "manuscripts": [],
        "reviewers": [],
        "collaboration_edges": [{"person_ids": ["a", "b"], "weight": 1}],
    })
    assert resp.status_code == 422


# ---------- 随机用例：与穷举简单路径的参考实现逐字段比对 ----------


_TOPICS = ["ml", "db", "t1", "t2"]


def _ref_best_path(edges, source, targets, max_hops):
    """DFS 穷举长度 <= max_hops 的全部简单路径，返回 (链长, 完整 ID 序列) 最小者。"""
    adj: dict[str, set[str]] = {}
    for a, b in edges:
        if a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    best = None
    stack = [(source, [source])]
    while stack:
        node, path = stack.pop()
        if len(path) - 1 >= max_hops:
            continue
        for nb in adj.get(node, ()):
            if nb in path:
                continue
            npath = path + [nb]
            if nb in targets:
                if best is None or (len(npath), npath) < (len(best), best):
                    best = npath
            stack.append((nb, npath))
    return best


def _ref_audit(body: dict) -> dict:
    """独立参考实现：基础规则 + 穷举最短合作链。"""
    edges = [tuple(e["person_ids"]) for e in body["collaboration_edges"]]
    out = []
    eligible_pairs = ineligible_pairs = 0
    for m in sorted(body["manuscripts"], key=lambda x: x["id"]):
        authors = set(m["authors"])
        eligible: list[str] = []
        ineligible: list[dict] = []
        for r in sorted(body["reviewers"], key=lambda x: x["id"]):
            reasons = []
            if r["id"] in authors:
                reasons.append("self_conflict")
            if authors & set(r.get("conflicted_authors", [])):
                reasons.append("author_conflict")
            if m["id"] in r.get("conflicted_manuscripts", []):
                reasons.append("manuscript_conflict")
            topic_mismatch = bool(m.get("topics")) and not (
                set(m["topics"]) & set(r["topics"])
            )
            path = None
            if r["id"] not in authors:
                best = _ref_best_path(
                    edges, r["id"], authors - {r["id"]}, body["max_hops"]
                )
                if best is not None:
                    reasons.append("collaboration_chain_conflict")
                    path = {"person_ids": best, "hops": len(best) - 1}
            pair = {
                "reviewer_id": r["id"],
                "reasons": reasons,
                "topic_mismatch": topic_mismatch,
                "collaboration_path": path,
            }
            if not reasons and not topic_mismatch:
                eligible.append(r["id"])
                eligible_pairs += 1
            else:
                ineligible.append(pair)
                ineligible_pairs += 1
        out.append({
            "manuscript_id": m["id"],
            "eligible_reviewer_ids": eligible,
            "ineligible": ineligible,
        })
    return {
        "manuscripts": out,
        "total_pairs": eligible_pairs + ineligible_pairs,
        "eligible_pairs": eligible_pairs,
        "ineligible_pairs": ineligible_pairs,
    }


def _random_body(seed: int) -> dict:
    rng = random.Random(seed)
    persons = [f"p{i}" for i in range(8)]
    reviewer_ids = rng.sample(persons, 5)
    m_ids = [f"m{i}" for i in range(rng.randint(1, 4))]

    manuscripts = []
    for mid in m_ids:
        n_authors = rng.randint(0, 3)
        topics = (
            [] if rng.random() < 0.2
            else rng.sample(_TOPICS, rng.randint(1, 2))
        )
        manuscripts.append({
            "id": mid,
            "authors": rng.sample(persons, n_authors),
            "topics": topics,
        })

    reviewers = []
    for rid in reviewer_ids:
        reviewers.append(_reviewer(
            id=rid,
            topics=rng.sample(_TOPICS, rng.randint(0, 3)),
            conflicted_authors=rng.sample(persons, rng.randint(0, 2)),
            conflicted_manuscripts=rng.sample(m_ids, rng.randint(0, 1)),
        ))

    edges_raw = []
    for _ in range(rng.randint(0, 14)):
        a, b = rng.choice(persons), rng.choice(persons)  # 允许自环
        if rng.random() < 0.5:
            a, b = b, a
        edges_raw.append((a, b))
    for _ in range(3):  # 注入重复边（可能反向）
        if edges_raw:
            edges_raw.append(rng.choice(edges_raw))
    rng.shuffle(edges_raw)

    return {
        "max_hops": rng.randint(1, 6),
        "manuscripts": manuscripts,
        "reviewers": reviewers,
        "collaboration_edges": [{"person_ids": [a, b]} for a, b in edges_raw],
    }


def test_random_against_reference_implementation():
    for seed in range(80):
        body = _random_body(seed)
        data = _audit(body)
        assert data == _ref_audit(body), seed
        # 路径结构自洽：端点正确、hops 一致、链长不超过 max_hops
        for audit in data["manuscripts"]:
            for pair in audit["ineligible"]:
                path = pair["collaboration_path"]
                if path is None:
                    assert "collaboration_chain_conflict" not in pair["reasons"]
                    continue
                assert "collaboration_chain_conflict" in pair["reasons"]
                assert path["hops"] == len(path["person_ids"]) - 1
                assert 1 <= path["hops"] <= body["max_hops"]


def test_random_bodies_are_deterministic_under_reorder():
    body = _random_body(123)
    expected = _audit(body)
    rng = random.Random(999)
    shuffled = copy.deepcopy(body)
    rng.shuffle(shuffled["manuscripts"])
    rng.shuffle(shuffled["reviewers"])
    rng.shuffle(shuffled["collaboration_edges"])
    assert _audit(shuffled) == expected
