type JSONValue = (
    None
    | bool
    | int
    | float
    | str
    | list[JSONValue]
    | tuple[JSONValue, ...]
    | dict[str, JSONValue]
)
type JSONObject = dict[str, JSONValue]
