"""The example workflow must stay loadable: link slots and types match, and every pack node's
widgets_values line up positionally with its unlinked (widget) inputs."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("workflow", ["roformer_separation"])
def test_example_workflow(nodes, workflow):
    data = json.loads((ROOT / "example_workflows" / f"{workflow}.json").read_text())
    by_id = {n["id"]: n for n in data["nodes"]}
    links = {link[0]: link for link in data["links"]}
    for node in data["nodes"]:
        for slot, inp in enumerate(node.get("inputs") or []):
            if inp.get("link") is None:
                continue
            link = links[inp["link"]]
            assert link[3] == node["id"] and link[4] == slot
            source = by_id[link[1]]["outputs"][link[2]]
            assert source["type"] == link[5]
            assert link[5] in inp["type"].split(","), (node["type"], slot)  # input may declare a union
            assert inp["link"] in (source["links"] or [])
        for slot, out in enumerate(node.get("outputs") or []):
            for link_id in out.get("links") or []:
                link = links[link_id]
                assert link[1] == node["id"] and link[2] == slot
        cls = nodes.NODE_CLASS_MAPPINGS.get(node["type"])
        if cls is None:
            continue  # a native node (LoadAudio, PreviewAudio, MarkdownNote): links already checked
        schema = cls.define_schema()
        inputs = {i.id: i for i in schema.inputs if not i.optional}
        linked = {inp["name"] for inp in node.get("inputs") or []}
        widgets = [name for name in inputs if name not in linked]
        assert len(node["widgets_values"]) == len(widgets), node["type"]
        for name, value in zip(widgets, node["widgets_values"]):
            options = getattr(inputs[name], "options", None)
            if options is not None:
                assert value in options, (node["type"], name)
