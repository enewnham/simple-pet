.PHONY: setup build run test check lint format assets clean

SWIFT_FLAGS ?= --disable-sandbox
SWIFT_TESTING_PLUGIN = $(wildcard $(dir $(shell xcrun --find swiftc))../lib/swift/host/plugins/testing/libTestingMacros.dylib)
SWIFT_TEST_FLAGS = $(SWIFT_FLAGS) $(if $(SWIFT_TESTING_PLUGIN),-Xswiftc -load-plugin-library -Xswiftc $(SWIFT_TESTING_PLUGIN))
UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
UV_PYTHON_INSTALL_DIR ?= $(CURDIR)/.cache/uv/python
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

setup:
	uv sync --locked --python 3.12

build:
	CI=true xcrun swift build -c release $(SWIFT_FLAGS)
	# Replace the executable's inode to preserve macOS signature validation.
	install -m 755 "$$(CI=true xcrun swift build -c release $(SWIFT_FLAGS) --show-bin-path)/pet-overlay" .build/pet-overlay.tmp
	mv -f .build/pet-overlay.tmp .build/pet-overlay
	install -m 755 "$$(CI=true xcrun swift build -c release $(SWIFT_FLAGS) --show-bin-path)/pet-speech-similarity" .build/pet-speech-similarity.tmp
	mv -f .build/pet-speech-similarity.tmp .build/pet-speech-similarity

run: setup build
	bin/pet-daemon

test: setup build
	CI=true xcrun swift test $(SWIFT_TEST_FLAGS)
	PET_DESKTOP_CHECK=1 uv run --locked python -m unittest discover -s Tests -v

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
