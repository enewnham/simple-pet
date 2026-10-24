.PHONY: setup build importer run test test-python test-swift check lint lint-python lint-swift complexity complexity-python complexity-swift format assets clean

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

ifneq ($(strip $(ARGS)),)
test:
	$(error Use make test-python ARGS=... or make test-swift ARGS=... to select tests)
else
test: test-swift test-python
endif

test-swift: setup
	@desktop=$$(uv run --locked python scripts/check-desktop.py) || exit $$?; \
	swift_filter=; \
	if [ "$$desktop" != 1 ]; then \
		echo "Desktop access unavailable; skipping overlay tests."; \
		swift_filter="--skip PetOverlayTests"; \
	fi; \
	CI=true xcrun swift test $(SWIFT_TEST_FLAGS) $$swift_filter $(ARGS)

test-python: setup
	@ps -ww -axo pid=,ppid=,lstart=,stat=,command= >/dev/null 2>&1 || { \
		echo "Process tests require ps access. Run make test outside the agent sandbox." >&2; \
		exit 1; \
	}
	@desktop=$$(uv run --locked python scripts/check-desktop.py) || exit $$?; \
	if [ "$$desktop" = 1 ]; then \
		echo "Desktop access available; running Python tests."; \
	else \
		echo "Desktop access unavailable; skipping desktop tests."; \
	fi; \
	PET_DESKTOP_CHECK=$$desktop uv run --locked python -m pytest $(if $(strip $(ARGS)),$(ARGS),Tests) -n $(PYTHON_TEST_WORKERS) --dist worksteal -v

lint: lint-python lint-swift complexity

lint-python: setup
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked ty check

lint-swift complexity-swift: export DYLD_FRAMEWORK_PATH := $(dir $(shell xcrun --find swiftc))../lib$(if $(DYLD_FRAMEWORK_PATH),:$(DYLD_FRAMEWORK_PATH))

lint-swift:
	swiftlint lint --strict --no-cache --config .swiftlint.yml

complexity: complexity-python complexity-swift

complexity-python: setup
	uv run --locked ruff check pet_daemon --select C901,PLR0915 --output-format concise

complexity-swift:
	swiftlint lint --strict --no-cache --config .swiftlint-complexity.yml

format: setup
	uv run --locked ruff format .

check: lint test
	bash -n scripts/fetch-assets.sh
	bash -n examples/pet-run
	bash -n bin/pet-daemon
	bash -n bin/pet-notify
	bash -n scripts/pet-hook

assets:
	scripts/fetch-assets.sh $(ARGS)

clean:
	swift package clean
