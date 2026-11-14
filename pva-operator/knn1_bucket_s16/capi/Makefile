# knn1 on the PVA: shortcuts for running the kernel.
#
#   make run        cloud distance with the command-line program (+ per-point answers in OUT_DIR)
#   make python     the same through Python (examples/example.py)
#   make c          build and run the C example (examples/example.c)
#   make selftest   built-in self-test on random clouds
#   make clean      remove what these targets created in OUT_DIR and BUILD_DIR
#
# Change the settings below, or override them on the command line, e.g.
#   make run REF_FRAME=0000000010 QUERY_FRAME=0000000011

# ---- Settings ------------------------------------------------------------------------------
# KITTI raw layout: $(KITTI_ROOT)/<date>/<drive>/velodyne_points/data/<frame>.bin
KITTI_ROOT  ?= $(KNN1)/../pva-knn/data/kitti_subset
DATE        ?= 2011_09_26
DRIVE       ?= 2011_09_26_drive_0001_sync
REF_FRAME   ?= 0000000089
QUERY_FRAME ?= 0000000090

# Or give the two scan files directly (any float32 x,y,z,reflectance .bin files):
PC_REF      ?= $(KITTI_ROOT)/$(DATE)/$(DRIVE)/velodyne_points/data/$(REF_FRAME).bin
PC_QUERIES  ?= $(KITTI_ROOT)/$(DATE)/$(DRIVE)/velodyne_points/data/$(QUERY_FRAME).bin

# Where results and the compiled C example go (your own home by default).
OUT_DIR     ?= $(HOME)/knn1_out
BUILD_DIR   ?= $(HOME)/knn1_build
# --------------------------------------------------------------------------------------------

KNN1   := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PYTHON ?= /usr/bin/python3
# Do not write __pycache__ into this (shared, read-only) folder.
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: help run python c selftest clean check-inputs

help:
	@sed -n '3,7p' $(KNN1)/Makefile | sed 's/^# //'

check-inputs:
	@test -r "$(PC_REF)"     || { echo "cannot read PC_REF=$(PC_REF)"; exit 1; }
	@test -r "$(PC_QUERIES)" || { echo "cannot read PC_QUERIES=$(PC_QUERIES)"; exit 1; }

run: check-inputs
	@mkdir -p $(OUT_DIR)
	$(KNN1)/bin/knn1_bucket_s16_kitti $(PC_REF) $(PC_QUERIES) $(OUT_DIR)

python: check-inputs
	$(PYTHON) $(KNN1)/examples/example.py $(PC_REF) $(PC_QUERIES)

$(BUILD_DIR)/knn1_example: $(KNN1)/examples/example.c $(KNN1)/include/knn1_distance.h
	@mkdir -p $(BUILD_DIR)
	gcc -O2 -Wall $< -I $(KNN1)/include -L $(KNN1)/lib -lknn1_distance -lm -Wl,-rpath,$(KNN1)/lib -o $@

c: check-inputs $(BUILD_DIR)/knn1_example
	$(BUILD_DIR)/knn1_example $(PC_REF) $(PC_QUERIES)

selftest:
	$(KNN1)/bin/knn1_bucket_s16_sample

clean:
	rm -f $(OUT_DIR)/dist2.s32 $(OUT_DIR)/idx.s32 $(BUILD_DIR)/knn1_example
	-rmdir $(OUT_DIR) $(BUILD_DIR) 2>/dev/null || true
