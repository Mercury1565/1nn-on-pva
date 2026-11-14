# Build the knn1 operator on this machine and install it on loki.
#
#   make            build for loki (only what changed is recompiled)
#   make deploy     build, package, copy to loki:$(LOKI_DIR), run the self-test there
#   make test-loki  run the full correctness tests (31 cases) on loki, then clean up
#   make clean      remove the build and package folders here
#
# Quick checks only: formal results still need a pva-build-test run (build-test/runN/).

# ---- Settings ------------------------------------------------------------------------------
LOKI      ?= loki                 # ssh host
LOKI_DIR  ?= /home/hermon/pva-knn1
LOKI_TMP  ?= /tmp/hermon-pva-knn1-make
BUCKET    ?= 128                  # ref points per bucket: 128, 256 or 512
# --------------------------------------------------------------------------------------------

LOKI     := $(strip $(LOKI))
BUCKET   := $(strip $(BUCKET))
SRC      := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
BUILD    := $(SRC)/build/l4t
PKG      := $(SRC)/build/package
TESTDATA := $(SRC)/../../testdata/knn1_bucket
SSH      := ssh -o BatchMode=yes
# The PVA build needs the system python (conda's python3 lacks pyelftools).
ENV      := PATH=/usr/bin:$$PATH

# Files that make up the operator; the fingerprint in VERSION is their combined sha256.
SOURCES  := $(shell cd $(SRC) && find CMakeLists.txt include src vpu capi sample -type f \
                -not -name '*.md' -not -path '*/__pycache__/*' | sort)

.PHONY: all build package deploy test-loki clean help

all: build

help:
	@sed -n '3,6p' $(SRC)/Makefile | sed 's/^# //'

# Configure once (or when the bucket size changes), then build incrementally.
$(BUILD)/CMakeCache.txt:
	$(ENV) cmake -S $(SRC) -B $(BUILD) -DPVA_BUILD_MODE=L4T -DPVA_DEFAULT_GEN=GEN3 \
	    -DPVA_ONLY_BUILD_GENS=GEN3 -DCMAKE_BUILD_TYPE=Release -DK1_BUCKET=$(BUCKET)

build: $(BUILD)/CMakeCache.txt
	@grep -q '^K1_BUCKET:STRING=$(BUCKET)$$' $(BUILD)/CMakeCache.txt || \
	    $(ENV) cmake -S $(SRC) -B $(BUILD) -DK1_BUCKET=$(BUCKET)
	$(ENV) cmake --build $(BUILD) -j$$(nproc)

# Same layout as the folder on loki: bin, lib, include, python, examples, Makefile, README, VERSION.
package: build
	@rm -rf $(PKG)
	@mkdir -p $(PKG)/bin $(PKG)/lib $(PKG)/include $(PKG)/python $(PKG)/examples
	@cp $(BUILD)/knn1_bucket_s16_kitti $(BUILD)/knn1_bucket_s16_sample $(PKG)/bin/
	@cp $(BUILD)/libknn1_distance.so $(PKG)/lib/
	@cp $(SRC)/capi/knn1_distance.h $(PKG)/include/
	@cp $(SRC)/capi/knn1_distance.py $(PKG)/python/
	@cp $(SRC)/capi/example.c $(SRC)/capi/example*.py $(PKG)/examples/
	@cp $(SRC)/capi/Makefile $(SRC)/capi/README.md $(PKG)/
	@{ echo "knn1 PVA kernel (operator knn1_bucket_s16), built $$(date -u +%F' '%H:%M) UTC with 'make deploy'"; \
	   echo "Source fingerprint: $$(cd $(SRC) && sha256sum $(SOURCES) | sha256sum | cut -c1-16)"; \
	   echo "Git: $$(git -C $(SRC) describe --always --dirty 2>/dev/null || echo 'no commits')"; \
	   echo "PVA SDK 2.10.1 (NVVPU), L4T, GEN3 (Thor), Release, $(BUCKET)-point buckets, both VPUs."; \
	   echo "Quick build: not a pva-build-test result."; } > $(PKG)/VERSION
	@cat $(PKG)/VERSION

# Copy the package (only changed files are sent; old files are removed) and check it works.
deploy: package
	rsync -az --delete -e "$(SSH)" $(PKG)/ $(LOKI):$(LOKI_DIR)/
	$(SSH) $(LOKI) 'make -s -C $(LOKI_DIR) selftest'

# Full correctness matrix on loki, from a temporary folder that is removed afterwards.
test-loki: build
	$(SSH) $(LOKI) 'rm -rf $(LOKI_TMP) && mkdir -p $(LOKI_TMP)'
	rsync -az -e "$(SSH)" $(BUILD)/knn1_bucket_s16_test $(LOKI):$(LOKI_TMP)/
	rsync -az -e "$(SSH)" $(TESTDATA)/ $(LOKI):$(LOKI_TMP)/testdata/
	$(SSH) $(LOKI) 'cd $(LOKI_TMP) && ./knn1_bucket_s16_test testdata > test.log 2>&1; s=$$?; \
	    grep -E "FAIL|SUMMARY" test.log; cd / && rm -rf $(LOKI_TMP); exit $$s'

clean:
	rm -rf $(BUILD) $(PKG)
