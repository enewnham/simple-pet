from dataclasses import dataclass

from ..repository_identity import RepositoryIdentity, repository_columns
from ..work.models import ReportContext, WorkFacts


def location(value: str) -> str:
    return value.removeprefix("work:observation:").split("#L", 1)[0]


def branch_location(repository: RepositoryIdentity, branch: str) -> str:
    _, identifier = repository_columns(repository)
    return f"{identifier}:{branch}"


@dataclass(frozen=True)
class WorkSelection:
    scopes: tuple[str, ...]
    references: tuple[str, ...] = ()

    def includes_report(self, context: ReportContext) -> bool:
        if context.pull_request and self.includes(context.pull_request):
            return True
        if context.repository is not None and context.branch:
            return self.includes(branch_location(context.repository, context.branch))
        return self.includes(context.checkout)

    def includes(
        self, scope: str, identifier: str = "", links: tuple[str, ...] = ()
    ) -> bool:
        targets = set(self.scopes)
        return (
            scope in targets
            or location(identifier) in targets
            or bool(targets.intersection(map(location, links)))
        )


def select_work(
    scope: str, references: tuple[str, ...], facts: WorkFacts
) -> WorkSelection:
    targets = {location(item) for item in (scope, *references) if item}
    while True:
        previous = targets.copy()
        for checkout in facts.checkouts.values():
            local = checkout.local
            if local is None:
                continue
            branch = branch_location(local.repository, local.branch)
            if checkout.worktree in targets or branch in targets:
                targets.update((checkout.worktree, branch))
        for branch in facts.branches.values():
            identifier = branch_location(branch.repository, branch.branch)
            pull_request = branch.pr.url if branch.pr is not None else ""
            if identifier in targets or pull_request in targets:
                targets.add(identifier)
                if pull_request:
                    targets.add(pull_request)
        for observation in facts.pull_requests.values():
            pull_request = observation.pr
            if pull_request is None:
                continue
            if observation.ref not in targets and pull_request.url not in targets:
                continue
            targets.update((observation.ref, pull_request.url))
            if pull_request.head_repository is not None and pull_request.head_branch:
                branch = branch_location(
                    pull_request.head_repository, pull_request.head_branch
                )
                if any(
                    item.local is not None
                    and item.local.repository == pull_request.head_repository
                    and item.local.branch == pull_request.head_branch
                    and item.local.head == pull_request.head
                    and bool(pull_request.head)
                    for item in facts.checkouts.values()
                ):
                    targets.add(branch)
        if previous == targets:
            return WorkSelection(tuple(sorted(targets)), references)
