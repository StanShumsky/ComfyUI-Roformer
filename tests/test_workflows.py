"""The example workflow must stay loadable: link slots and types match, and every pack node's
widgets_values line up positionally with its unlinked (widget) inputs."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# the schema's dynamic input types (autogrow & co): one execute parameter, a dict of the
# generated inputs' values - they are link inputs, never widgets_values
DYNAMIC_IO = {"COMFY_AUTOGROW_V3", "COMFY_DYNAMICCOMBO_V3", "COMFY_DYNAMICGROUP_V3"}

# widget inputs by io_type: None on the test stub (no io_type attribute), the real V1 type
# names on the real classes - everything else is a link slot
WIDGET_IO = {None, "INT", "FLOAT", "BOOLEAN", "COMBO", "STRING", "MULTILINE"}


def _schema_link_inputs(nodes, cls):
    """The node's link inputs in declaration order: autogrow inputs expanded in place
    to their (group-nested) template names, widget inputs out. Tuples: ("autogrow",
    input_id, template_names) or ("link", input_id, None)."""
    out = []
    for i in cls.define_schema().inputs:
        if isinstance(i, nodes.io.Autogrow.Input):
            out.append(("autogrow", i.id, i.template.names))
        elif getattr(i, "io_type", None) not in WIDGET_IO:
            out.append(("link", i.id, None))
    return out


@pytest.mark.parametrize("workflow", ["roformer_separation", "roformer_ensemble"])
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
        # dynamic inputs (instance io_type: a property on the real classes, an attribute on
        # the test stub) expand to generated link inputs - never widgets_values
        inputs = {i.id: i for i in schema.inputs
                  if not i.optional
                  and getattr(i, "io_type", None) not in DYNAMIC_IO}
        linked = {inp["name"] for inp in node.get("inputs") or []}
        widgets = [name for name in inputs if name not in linked]
        assert len(node["widgets_values"]) == len(widgets), node["type"]
        # the serialized inputs array is the LIVE slot list the current frontend reads:
        # the autogrow slots that exist (the connected ones in template order plus at
        # most one trailing placeholder, under their group-nested names) in declaration
        # order, then the remaining link inputs. The full template list is NOT written -
        # the frontend renders the live group and maps saved inputs by name/slot, so
        # stale generated slots show up as a duplicated group (or break link restore).
        saved = node.get("inputs") or []
        expected = []
        for kind, input_id, template_names in _schema_link_inputs(nodes, cls):
            if kind == "link":
                expected.append(input_id)
                continue
            nested = [f"{input_id}.{name}" for name in template_names]
            group = [inp for inp in saved if inp["name"] in nested]
            assert [inp["name"] for inp in group] == nested[:len(group)], (
                node["type"], f"autogrow slots must keep template order: {nested[:4]}...")
            unlinked = [inp for inp in group if inp["link"] is None]
            assert group[-len(unlinked):] == unlinked if unlinked else True, (
                node["type"], "unconnected autogrow slots must trail the connected ones")
            assert len(unlinked) <= 1, (
                node["type"], "the live group has at most one trailing free slot")
            assert 1 <= len(group) <= len(nested), node["type"]
            expected.extend(inp["name"] for inp in group)
        assert [inp["name"] for inp in saved] == expected, (
            node["type"], f"inputs must be the live slot list in declaration order: {expected}")
        for name, value in zip(widgets, node["widgets_values"]):
            options = getattr(inputs[name], "options", None)
            if options is not None:
                assert value in options, (node["type"], name)
