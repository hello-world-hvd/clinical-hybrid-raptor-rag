from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def summarize_community(
    *,
    community_id: str,
    entity_ids: Sequence[str],
    entities: Mapping[str, Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    max_entities: int = 12,
    max_relations: int = 12,
) -> dict[str, Any]:
    member_set = set(entity_ids)
    internal = [
        relation
        for relation in relations
        if relation["source"] in member_set and relation["target"] in member_set
    ]
    degrees = Counter()
    for relation in internal:
        degrees[relation["source"]] += int(relation.get("weight") or 1)
        degrees[relation["target"]] += int(relation.get("weight") or 1)
    ranked_ids = sorted(
        entity_ids,
        key=lambda entity_id: (
            -degrees[entity_id],
            str(entities[entity_id].get("name") or ""),
        ),
    )
    names = [str(entities[item].get("name") or item) for item in ranked_ids]
    relation_lines = []
    for relation in sorted(
        internal,
        key=lambda item: -int(item.get("weight") or 1),
    )[:max_relations]:
        source = entities[relation["source"]]["name"]
        target = entities[relation["target"]]["name"]
        description = str(relation.get("description") or "").strip()
        line = f"{source} --{relation['type']}--> {target}"
        relation_lines.append(f"{line}: {description}" if description else line)

    summary_parts = [
        f"Community {community_id} gồm các thực thể chính: "
        + ", ".join(names[:max_entities])
        + "."
    ]
    if relation_lines:
        summary_parts.append("Quan hệ nổi bật: " + "; ".join(relation_lines) + ".")
    return {
        "community_id": community_id,
        "entity_ids": list(ranked_ids),
        "title": ", ".join(names[:3]) or community_id,
        "summary": " ".join(summary_parts),
        "relation_count": len(internal),
    }
