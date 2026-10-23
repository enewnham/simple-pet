.PHONY: setup build importer run test check lint format assets clean

SWIFT_FLAGS ?= --disable-sandbox
PYTHON_TEST_WORKERS ?= auto
SWIFT_TESTING_PLUGIN = $(wildcard $(dir $(shell xcrun --find swiftc))../lib/swift/host/plugins/testing/libTestingMacros.dylib)
SWIFT_TEST_FLAGS = $(SWIFT_FLAGS) $(if $(SWIFT_TESTING_PLUGIN),-Xswiftc -load-plugin-library -Xswiftc $(SWIFT_TESTING_PLUGIN))
UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
UV_PYTHON_INSTALL_DIR ?= $(CURDIR)/.cache/uv/python
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

setup:
	uv sync --locked --python 3.15

build:
	CI=true xcrun swift build -c release $(SWIFT_FLAGS)
	# Replace the executable's inode to preserve macOS signature validation.
	install -m 755 "$$(CI=true xcrun swift build -c release $(SWIFT_FLAGS) --show-bin-path)/pet-overlay" .build/pet-overlay.tmp
	mv -f .build/pet-overlay.tmp .build/pet-overlay

importer: .build/debug/AssetImporter

.build/debug/AssetImporter: scripts/import-assets.swift Package.swift
	CI=true xcrun swift build $(SWIFT_TEST_FLAGS) --product AssetImporter

run: setup build
	bin/pet-daemon

test: setup
	@ps -ww -axo pid=,ppid=,lstart=,stat=,command= >/dev/null 2>&1 || { \
		echo "Process tests require ps access. Run make test outside the agent sandbox." >&2; \
		exit 1; \
	}
	@desktop=$$(uv run --locked python -c 'from AppKit import NSApplication, NSScreen; NSApplication.sharedApplication(); print(int(bool(NSScreen.screens())))') || exit $$?; \
	swift_filter=; \
	if [ "$$desktop" = 1 ]; then \
		echo "Desktop access available; running all tests."; \
	else \
		echo "Desktop access unavailable; skipping overlay and desktop tests."; \
		swift_filter="--skip PetOverlayTests"; \
	fi; \
	CI=true xcrun swift test $(SWIFT_TEST_FLAGS) $$swift_filter && \
	PET_DESKTOP_CHECK=$$desktop uv run --locked python -m pytest Tests -n $(PYTHON_TEST_WORKERS) --dist worksteal -v

lint: setup
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked ty check

format: setup
	uv run --locked ruff format .

check: lint test
	bash -n scripts/fetch-assets.sh
	bash -n examples/pet-run
	bash -n bin/pet-daemon

assets:
	scripts/fetch-assets.sh $(ARGS)

clean:
	swift package clean
