.PHONY: setup build run test check desktop-check assets clean

SWIFT_FLAGS ?= --disable-sandbox
UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
UV_PYTHON_INSTALL_DIR ?= $(CURDIR)/.cache/uv/python
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

setup:
	uv sync --locked --python 3.12

build:
	CI=true swift build -c release $(SWIFT_FLAGS)
	cp "$$(CI=true swift build -c release $(SWIFT_FLAGS) --show-bin-path)/pet-overlay" .build/pet-overlay

run: setup build
	bin/pet-daemon

test: setup
	mkdir -p .build/core-tests
	swiftc -parse-as-library -module-cache-path .build/core-tests/cache Sources/PetCore/*.swift Tests/PetCoreChecks.swift -o .build/core-tests/pet-core-checks
	.build/core-tests/pet-core-checks
	uv run python -m unittest discover -s Tests -v
	mkdir -p .build/asset-tests
	swiftc -O -module-cache-path .build/asset-tests/cache scripts/import-assets.swift -o .build/asset-tests/import-assets
	swiftc -module-cache-path .build/asset-tests/cache Tests/AssetImportTests.swift -o .build/asset-tests/asset-import-tests
	.build/asset-tests/asset-import-tests .build/asset-tests/import-assets

check: test
	bash -n scripts/fetch-assets.sh
	bash -n examples/pet-run
	bash -n bin/pet-daemon

desktop-check: setup build
	PET_DESKTOP_CHECK=1 uv run python -m unittest discover -s Tests -p test_desktop.py -v

assets:
	scripts/fetch-assets.sh

clean:
	swift package clean
