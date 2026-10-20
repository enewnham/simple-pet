def activity(tool, name=""):
    tool_name = name.lower().rsplit("__", 1)[-1].rsplit(".", 1)[-1]
    if tool_name in ("read", "read_file", "readfile", "grep", "find", "search", "glob"):
        return "investigating"
    if tool_name in (
        "edit",
        "edit_file",
        "editfile",
        "write",
        "write_file",
        "writefile",
        "apply_patch",
        "multiedit",
    ):
        return "editing"
    text = (name + " " + tool).lower()
    if any(word in text for word in ("pytest", " test", "vitest", "jest", "unittest")):
        return "testing"
    if "build" in text:
        return "building"
    if any(word in text for word in ("compile", "tsc", "lint", "typecheck", " check")):
        return "checking"
    if any(word in text for word in ("edit", "write", "patch")):
        return "editing"
    if any(word in text for word in ("read", "grep", "find", "search", "glob")):
        return "investigating"
    return "command"


def result_summary(category, outcome):
    subject = {
        "testing": "Tests",
        "building": "Build",
        "checking": "Checks",
        "editing": "Edit",
        "investigating": "Inspection",
        "command": "Command",
        "harness": "Agent run",
    }[category]
    result = {
        "success": "succeeded",
        "error": "failed",
        "cancelled": "cancelled",
        "stale": "went stale",
    }[outcome]
    return subject + " " + result
