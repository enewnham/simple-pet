import re
import shlex

from ..runtime import ToolActivity


def mutates_files(command: str) -> bool:
    tokens: list[str] = []
    try:
        lexer = shlex.shlex(command, posix=False, punctuation_chars=";&|()<>\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens.extend(lexer)
    except ValueError:
        pass
    start = True
    for index, token in enumerate(tokens):
        if (
            ">" in token
            and "&" not in token
            and all(char in ";&|()<>\n" for char in token)
        ):
            return True
        if token and all(char in ";&|()<>\n" for char in token):
            start = True
            continue
        if not start:
            continue
        start = False
        executable = token.rsplit("/", 1)[-1]
        arguments = []
        for argument in tokens[index + 1 :]:
            if argument and all(char in ";&|()<>\n" for char in argument):
                break
            arguments.append(argument)
        if executable in {
            "apply_patch",
            "patch",
            "tee",
            "touch",
            "cp",
            "mv",
            "rm",
            "mkdir",
            "rmdir",
            "truncate",
            "install",
            "ln",
        }:
            return True
        if executable == "sed" and any(
            argument == "--in-place" or argument.startswith(("--in-place=", "-i"))
            for argument in arguments
        ):
            return True
        if executable == "perl" and any(
            argument.startswith("-") and "i" in argument[1:] for argument in arguments
        ):
            return True
        if executable == "dd" and any(
            argument.startswith("of=") for argument in arguments
        ):
            return True
    return bool(
        re.search(
            r"\b(?:write_text|write_bytes|writeFile(?:Sync)?|appendFile(?:Sync)?|"
            r"binwrite|file_put_contents)\s*\(",
            command,
        )
        or re.search(r"\bopen\([^\n]*?,\s*['\"][wax][bt+]*['\"]", command)
    )


def activity(tool: str, name: str = "") -> ToolActivity:
    tool_name = (name or tool).lower().rsplit("__", 1)[-1].rsplit(".", 1)[-1]
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
    if mutates_files(tool):
        return "editing"
    text = (name + " " + tool).lower()
    if any(word in text for word in ("pytest", " test", "vitest", "jest", "unittest")):
        return "testing"
    if "build" in text:
        return "building"
    if any(word in text for word in ("compile", "tsc", "lint", "typecheck", " check")):
        return "checking"
    if any(word in text for word in ("read", "grep", "find", "search", "glob")):
        return "investigating"
    return "command"
