.PHONY: setup build run test check desktop-check assets clean

SWIFT_FLAGS ?= --disable-sandbox
UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
UV_PYTHON_INSTALL_DIR ?= $(CURDIR)/.cache/uv/python
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

setup:
	uv sync --locked --python 3.12

build:
	CI=true swift build -c release $(SWIFT_FLAGS)
	# Replace the executable's inode to preserve macOS signature validation.
	install -m 755 "$$(CI=true swift build -c release $(SWIFT_FLAGS) --show-bin-path)/pet-overlay" .build/pet-overlay.tmp
	mv -f .build/pet-overlay.tmp .build/pet-overlay
	install -m 755 "$$(CI=true swift build -c release $(SWIFT_FLAGS) --show-bin-path)/pet-speech-similarity" .build/pet-speech-similarity.tmp
	mv -f .build/pet-speech-similarity.tmp .build/pet-speech-similarity

run: setup build
	bin/pet-daemon

test: setup
	mkdir -p .build/core-tests
	swiftc -parse-as-library -module-cache-path .build/core-tests/cache Sources/PetCore/*.swift Tests/PetCoreChecks.swift -o .build/core-tests/pet-core-checks
	.build/core-tests/pet-core-checks
	swiftc -O -module-cache-path .build/core-tests/cache Sources/PetSpeechSimilarity/main.swift -o .build/pet-speech-similarity
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
	mkdir -p .build/overlay-tests
	swiftc -parse-as-library -whole-module-optimization -emit-module -emit-object -module-name PetCore -module-cache-path .build/overlay-tests/cache Sources/PetCore/*.swift -emit-module-path .build/overlay-tests/PetCore.swiftmodule -o .build/overlay-tests/PetCore.o
	swiftc -parse-as-library -module-cache-path .build/overlay-tests/cache -I .build/overlay-tests Sources/PetOverlay/PetView.swift Tests/PetOverlayChecks.swift .build/overlay-tests/PetCore.o -o .build/overlay-tests/pet-overlay-checks
	.build/overlay-tests/pet-overlay-checks
	PET_DESKTOP_CHECK=1 uv run python -m unittest discover -s Tests -p test_desktop.py -v

assets:
	scripts/fetch-assets.sh $(ARGS)

clean:
	swift package clean
