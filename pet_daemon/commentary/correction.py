class CorrectableOutput(ValueError):
    def __init__(self, issue: str, **context: object) -> None:
        super().__init__(issue)
        self.issue = issue
        self.context = context
