"""Coverage diagnostics must remain honest and independent of a particular discipline."""

from src.study_loop.formula_quality import (
    QUALITY_CHECKS,
    REVIEW_QUALITY_POLICY,
    audit_summary,
    formula_inventory,
    quoted_transcript_evidence,
    source_table_inventory,
)


def test_inventory_covers_tables_colored_math_and_display_but_not_literal_code():
    text = '''| 项目 | 关系 |
|---|---|
| 合成 | <span style="color:#D62828">$q=a+b$</span> |

<span style="color:#2563EB">$$
q=\\frac{a}{b}
$$</span>

`$literal$`
```python
text = "$not_a_study_formula$"
```
'''
    assert formula_inventory(text) == ["q=a+b", r"q=\frac{a}{b}"]


def test_inventory_preserves_repeated_forms_instead_of_claiming_one_was_checked():
    values = formula_inventory("$q=a+b$，另有 $q=c$；表中再写 $q=a+b$。")
    assert values == ["q=a+b", "q=c", "q=a+b"]


def test_diagnostics_never_turn_scanning_into_a_proof_or_confirmation():
    result = audit_summary("核心关系 $q=a/b$，其中 $b\\ne0$。")
    assert result["formula_count"] == 2
    assert result["strictly_proven"] is False
    assert result["status"] == "待人工核验"
    assert len(result["checks"]) == 8
    assert {item["name"] for item in result["checks"]} == set(QUALITY_CHECKS)
    assert all(item["status"] == "待独立核验" for item in result["checks"])


def test_plain_notes_do_not_invent_formula_verification():
    assert audit_summary("这是一段普通学习说明。")["formula_count"] == 0
    assert audit_summary("这是一段普通学习说明。")["status"] == "未包含公式"


def test_shared_policy_is_subject_neutral_and_covers_independent_review_limits():
    required = ("量纲", "单位", "符号", "取值域", "近似", "一致性", "来源")
    assert all(name in REVIEW_QUALITY_POLICY for name in required)
    assert "抽样相符不等于严格证明" in REVIEW_QUALITY_POLICY
    assert "待人工核验" in REVIEW_QUALITY_POLICY
    course_specific = ("4-2", "三相", "变压器", "电动机", "9550", "sqrt3")
    assert not any(name in REVIEW_QUALITY_POLICY for name in course_specific)


def test_transcript_projection_is_lossless_including_tables_math_unicode_and_line_endings():
    import copy

    raw = (
        '# 历史资料\r\n\r\n| 关系 | 条件 |\n|---|---|\n| $a=b$ | 待判断 |\r\n'
        '同一结构分别有局部口径与总口径。\n'
        '| $r=a/b$ | 整体比的连接未给出 |\r\n'
        '收支 $q_{in}=q_{out}+q_{loss}$。\n终行'
    )
    reference = {'label': 'S_NON_REAL', 'source_id': 'NON_REAL', 'text': raw,
                 'sha256': 'NON_REAL_ORIGINAL_RECEIPT', 'needs_review': False}
    before = copy.deepcopy(reference)
    projected = quoted_transcript_evidence(reference)
    assert ''.join(projected['quoted_source_lines']) == raw
    assert formula_inventory(''.join(projected['quoted_source_lines'])) == [
        'a=b', 'r=a/b', 'q_{in}=q_{out}+q_{loss}',
    ]
    assert projected['label'] == reference['label']
    assert projected['sha256'] == reference['sha256']
    assert projected['needs_review'] is False  # processing status is never rewritten as truth
    assert 'text' not in projected and reference == before
    obligations = projected['verification_obligations']
    assert obligations
    # Locate a boundary choice, an undefined connection and a conservation claim
    # without repairing any of their text or claiming that the index proved them.
    indexed = {line for obligation in obligations for line in obligation['line_numbers']}
    assert {6, 7, 8} <= indexed
    for obligation in obligations:
        assert obligation['verdict'] == '尚未核验'
        assert obligation['check']
        assert 'quote' not in obligation  # locate claims instead of repeating/endorsing them
        assert obligation['line_numbers'] == sorted(set(obligation['line_numbers']))
        assert all(1 <= line <= len(projected['quoted_source_lines'])
                   for line in obligation['line_numbers'])
    assert '尚未核验' not in reference['text']  # request-only aid, no source annotation


def test_source_table_catalogue_keeps_nine_topics_without_copying_relations_or_code_tables():
    import json

    text = '# NON_REAL目录\n```md\n| 假表 | 内容 |\n|---|---|\n| 不计入 | 代码 |\n```\n\n'
    for index in range(9):
        text += (
            f'## 方法组{index}\n\n| 方法 | 关系 | 单位 |\n|---|---|---|\n'
            f'| 路径{index}甲 | $q_{{{index}}}=a/b$ | unit |\n'
            f'| 路径{index}乙 | NON_REAL_ANSWER_CELL | unit |\n\n'
        )
    catalogue = source_table_inventory(text)
    assert catalogue['table_count'] == 9
    for index, table in enumerate(catalogue['tables']):
        assert table['heading_path'] == ['NON_REAL目录', f'方法组{index}']
        assert table['header'] == ['方法', '关系', '单位']
        assert table['row_topics'] == [f'路径{index}甲', f'路径{index}乙']
        assert table['row_count'] == 2
    assert catalogue['unparsed_pipe_lines'] == []
    projected = json.dumps(catalogue, ensure_ascii=False)
    assert 'NON_REAL_ANSWER_CELL' not in projected
    assert not formula_inventory(projected)
    assert catalogue['output_coverage_verified'] is False
    assert catalogue['formula_correctness_verified'] is False
    assert len(formula_inventory(text)) == 9  # The source math was not repaired or removed.
    escaped = source_table_inventory('| 名称 | 说明 |\n|---|---|\n| 甲\\|乙 | 不额外拆列 |\n')
    assert escaped['table_count'] == 1
    assert escaped['tables'][0]['row_topics'] == ['甲|乙']


def test_incomplete_table_candidates_are_reported_instead_of_counted_as_complete():
    catalogue = source_table_inventory('## NON_REAL\n| 项 | 值 |\n|---|\n| 甲 | 乙 |\n')
    assert catalogue['table_count'] == 0
    assert catalogue['unparsed_pipe_lines'] == [2, 3, 4]
    assert catalogue['output_coverage_verified'] is False
