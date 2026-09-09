NVCC ?= nvcc
CUDA_ARCH ?= 60
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
ARCHFLAGS = -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
LIBS = -lcufft
BIN = bin

.PHONY: all smoke clean

all: $(BIN)/holdout-worker \
	$(BIN)/lstar-anchor-label2 \
	$(BIN)/lstar-anchor-variable \
	$(BIN)/lstar-krylov-label2 \
	$(BIN)/lstar-krylov-label5 \
	$(BIN)/lstar-reconstruct-label2 \
	$(BIN)/lstar-reconstruct-label5

$(BIN):
	mkdir -p $@

$(BIN)/holdout-worker: holdout_cuda_worker.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(ARCHFLAGS) $< -o $@

$(BIN)/lstar-anchor-label2: tii254_lstar_anchor_codec_worker.cu holdout_cuda_worker.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(ARCHFLAGS) $< -o $@

$(BIN)/lstar-anchor-variable: tii254_lstar_anchor_codec_worker_v2.cu tii254_lstar_anchor_codec_worker.cu tii254_lstar_anchor_codec_v2_support.cuh holdout_cuda_worker.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(ARCHFLAGS) $< -o $@

$(BIN)/lstar-krylov-label2: tii254_lstar_rej_krylov_worker.cu tii254_lstar_rej_cufft_worker.cu tii254_lstar_anchor_codec_worker.cu holdout_cuda_worker.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(ARCHFLAGS) \
		-DMCELIECEX_ENABLE_LSTAR_REJ_PRODUCTION=1 $< $(LIBS) -o $@

$(BIN)/lstar-krylov-label5: tii254_lstar_rej_krylov_worker.cu tii254_lstar_rej_cufft_worker.cu tii254_lstar_anchor_codec_worker.cu tii254_lstar_anchor_codec_v2_support.cuh holdout_cuda_worker.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(ARCHFLAGS) \
		-DMCELIECEX_ENABLE_LSTAR_REJ_PRODUCTION=1 \
		-DMCELIECEX_TII254_LSTAR_REJ_ANCHOR5=1 $< $(LIBS) -o $@

$(BIN)/lstar-reconstruct-label2: tii254_lstar_rej_reconstruction_worker.cu tii254_lstar_rej_krylov_worker.cu tii254_lstar_rej_cufft_worker.cu tii254_lstar_anchor_codec_worker.cu holdout_cuda_worker.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(ARCHFLAGS) \
		-DMCELIECEX_ENABLE_LSTAR_REJ_PRODUCTION=1 \
		-DMCELIECEX_ENABLE_LSTAR_REJ_RECONSTRUCTION=1 $< $(LIBS) -o $@

$(BIN)/lstar-reconstruct-label5: tii254_lstar_rej_reconstruction_worker.cu tii254_lstar_rej_krylov_worker.cu tii254_lstar_rej_cufft_worker.cu tii254_lstar_anchor_codec_worker.cu tii254_lstar_anchor_codec_v2_support.cuh holdout_cuda_worker.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(ARCHFLAGS) \
		-DMCELIECEX_ENABLE_LSTAR_REJ_PRODUCTION=1 \
		-DMCELIECEX_ENABLE_LSTAR_REJ_RECONSTRUCTION=1 \
		-DMCELIECEX_TII254_LSTAR_REJ_ANCHOR5=1 $< $(LIBS) -o $@

smoke: $(BIN)/holdout-worker $(BIN)/lstar-reconstruct-label2
	$(BIN)/holdout-worker self-test
	$(BIN)/lstar-reconstruct-label2 self-test

clean:
	rm -rf $(BIN)
