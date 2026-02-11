# Makefile for Megakernel
# Supports multi-arch builds for T4 (sm_75), A100 (sm_80), H100 (sm_90)

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 --use_fast_math -lineinfo
SRC_DIR = src
BUILD_DIR = build

# Default: detect GPU and build for it
# Override with: make ARCH=sm_80
ARCH ?= $(shell python3 -c "import torch; p=torch.cuda.get_device_properties(0); print(f'sm_{p.major}{p.minor}')" 2>/dev/null || echo "sm_80")

# Source files
KERNEL_SRC = $(SRC_DIR)/megakernel.cu
HEADERS = $(wildcard $(SRC_DIR)/*.cuh)

# Targets
STANDALONE = megakernel
LIBRARY = megakernel.so

.PHONY: all standalone lib clean info

all: standalone

standalone: $(STANDALONE)

$(STANDALONE): $(KERNEL_SRC) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) -I$(SRC_DIR) $< -o $@

lib: $(LIBRARY)

$(LIBRARY): $(KERNEL_SRC) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) -shared -Xcompiler -fPIC -DMEGAKERNEL_LIBRARY_MODE -I$(SRC_DIR) $< -o $@

# Build for all Colab GPU architectures
all-arch:
	$(NVCC) $(NVCC_FLAGS) -arch=sm_75 -I$(SRC_DIR) $(KERNEL_SRC) -o megakernel_t4
	$(NVCC) $(NVCC_FLAGS) -arch=sm_80 -I$(SRC_DIR) $(KERNEL_SRC) -o megakernel_a100
	$(NVCC) $(NVCC_FLAGS) -arch=sm_90 -I$(SRC_DIR) $(KERNEL_SRC) -o megakernel_h100

clean:
	rm -f $(STANDALONE) $(LIBRARY) megakernel_t4 megakernel_a100 megakernel_h100

info:
	@echo "ARCH: $(ARCH)"
	@echo "NVCC: $(NVCC)"
	@echo "Sources: $(KERNEL_SRC)"
	@echo "Headers: $(HEADERS)"
	@python3 -c "import torch; p=torch.cuda.get_device_properties(0); print(f'GPU: {p.name} (sm_{p.major}{p.minor})')" 2>/dev/null || echo "No GPU detected"
