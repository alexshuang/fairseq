#!/bin/bash

# arg 1: gpu device id
# arg 2: output dir
function get_hw_info() {
    output=$2/hw.csv
    clks=(`/opt/rocm/bin/rocm-smi -d $1 --showclkfrq | grep '*' | grep 'GPU' | awk '{print $4}'`)
    dev=`/opt/rocm/bin/rocm-smi -d $1 -i | grep 'GPU' | awk '{print $5}'`
    echo "device,sclk,mclk" > $output
    echo "$dev,${clks[2]},${clks[1]}" >> $output
}

TEXT=/data/newdata/wmt14_en_de_joined_dict

STEPS=${2:-1000}
WARMUP_STEPS=${3:-30}

OUT_DIR=${1:-out}
TMP_DIR=$OUT_DIR/tmp
DEVICE=${4:-1}

#rm -rf $OUT_DIR
mkdir -p $OUT_DIR $TMP_DIR

CMD="python3.6 train.py $TEXT \
    --fp16 \
    --arch transformer_wmt_en_de_big_t2t --seed 1 --num-workers 1 --disable-validation \
    --optimizer adam --adam-betas (0.9,0.997) --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-9 --log-interval 1 \
    --lr 1.94e-3 --min-lr 1e-10 --adam-eps "1e-9" \
    --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 10000 --no-save --disable-validation --warmup-updates 1100 \
    --warmup-update $WARMUP_STEPS"
#--bucket-cap-mb 32 

set -e

HIP_VISIBLE_DEVICE=$DEVICE

get_hw_info $DEVICE $OUT_DIR

# end2end perf
rm -rf checkpoints
$CMD --result-dir $OUT_DIR --max-update $STEPS | tee $TMP_DIR/run.log

# record kernels
export ROCBLAS_LAYER=6
export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv
export ROCBLAS_LOG_PROFILE_PATH=${OUT_DIR}/rocblas_config.json
rm -f ${ROCBLAS_LOG_BENCH_PATH}
rm -f ${ROCBLAS_LOG_PROFILE_PATH}
rm -rf checkpoints
echo "pmc: FetchSize L2CacheHit" > input.txt
/opt/rocm/bin/rocprof -i input.txt --obj-tracking on --timestamp on --stats -o ${TMP_DIR}/kernel_prof.csv \
$CMD --max-update 100
rm $TMP_DIR/*.db $TMP_DIR/*.txt $TMP_DIR/*.json

# split one iteration
NUM_GEMM_PER_ITER=399
tail -$NUM_GEMM_PER_ITER $ROCBLAS_LOG_BENCH_PATH > $TMP_DIR/rb.csv
cp $TMP_DIR/rb.csv $ROCBLAS_LOG_BENCH_PATH
sed -n '/Cijk_A/p' ${TMP_DIR}/kernel_prof.csv > $TMP_DIR/gemm_kernel_prof.csv
tail -$NUM_GEMM_PER_ITER $TMP_DIR/gemm_kernel_prof.csv > $OUT_DIR/kernel_prof.csv

sed "s/$/ -i ${STEPS} -j ${WARMUP_STEPS}/g" $ROCBLAS_LOG_BENCH_PATH > ${TMP_DIR}/_rb.csv

# rocblas-bench
TOOL=/root/rocblas/build/release/clients/staging/rocblas-bench
if [ ! -e rocblas-bench ]; then
	ln -s ${TOOL} .
fi
unset ROCBLAS_LAYER
sh $TMP_DIR/_rb.csv 2>&1 > $TMP_DIR/rb_res.txt | tee $TMP_DIR/rocblas_bench.log
sed -E -n '/(^N,|^T,)/p' $TMP_DIR/rb_res.txt > $OUT_DIR/rocblas_bench_res.csv
echo "File $OUT_DIR/rocblas_bench_res.csv is generated."
