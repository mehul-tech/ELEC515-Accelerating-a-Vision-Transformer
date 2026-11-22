#!/usr/bin/env bash
# Build (when needed) and run the ViT accelerator simulation with Verilator.
#
# Usage: ./run_sim.sh [--int8] [MASK]
#   MASK     head mask to use: a name in head_masks/ (default: keep_all), with or without
#            .txt; a path to a file in head_masks/ works too. The simulation only reads masks
#            from head_masks/, and only those that python/export_rtl_data.py has exported.
#   --int8   use the LINEAR_INT8 build (INT8 linear layers, 4 linear units per layer)
# The script can be started from any directory; it always works in its own folder. It exits
# with status 0 only when the simulation reports RESULT: PASS.
#
# Examples:
#   ./run_sim.sh                    16-bit build, every head kept
#   ./run_sim.sh prune4             16-bit build, head_masks/prune4.txt
#   ./run_sim.sh --int8 prune4      INT8 build, head_masks/prune4.txt
#
# The two builds are compiled into obj_dir/build_16bit and obj_dir/build_int8 (if the path of
# the project folder contains a space, into a folder under the system temporary directory
# instead, because GNU Make cannot build in a path with spaces). A build is compiled the
# first time (about a minute; a few minutes on a slow computer) and again whenever the
# content of rtl/, tb/ or this script changes; switching the head mask needs no rebuild.
set -e

CASE=keep_all
MASK_GIVEN=""
MASK_DIR=""
BUILD=build_16bit
DEFINES=""
for arg in "$@"; do
    case "$arg" in
        --int8) BUILD=build_int8; DEFINES="+define+LINEAR_INT8"; continue ;;
        -h|--help) sed -n '2,/^set -e/p' "$0" | grep '^#'; exit 0 ;;
        -*) echo "unknown option: $arg (see ./run_sim.sh --help)"; exit 1 ;;
    esac
    if [ -n "$MASK_GIVEN" ]; then
        echo "./run_sim.sh runs one mask at a time, but got '$MASK_GIVEN' and '$arg'."
        echo "(If you pasted a command with a '# comment' at the end, leave the comment out.)"
        exit 1
    fi
    MASK_GIVEN="$arg"
    case "$arg" in
        */*)   # a path, relative to the directory the script was started from
            file="$arg"
            if [ ! -f "$file" ] && [ -f "$file.txt" ]; then file="$file.txt"; fi
            if [ ! -f "$file" ]; then echo "no such mask file: $arg"; exit 1; fi
            MASK_DIR="$(cd "$(dirname "$file")" && pwd -P)"
            CASE="$(basename "$file" .txt)" ;;
        *) CASE="$(basename "$arg" .txt)" ;;
    esac
done

cd "$(dirname "$0")"

if [ -n "$MASK_DIR" ] && [ "$MASK_DIR" != "$(pwd -P)/head_masks" ]; then
    echo "The simulation reads head masks only from head_masks/ in the project folder."
    echo "Copy the mask there, export again, and run it by name, from the project folder:"
    echo "  cp <your mask file> head_masks/$CASE.txt"
    echo "  python python/export_rtl_data.py --samples 20"
    echo "  ./run_sim.sh $CASE"
    exit 1
fi
if [ ! -f "head_masks/$CASE.txt" ]; then
    echo "no head mask head_masks/$CASE.txt; the masks there are:"
    ls head_masks
    exit 1
fi

# The exported data the simulation runs on and is checked against (python/export_rtl_data.py).
# The source.txt notes are written when an export has finished.
if [ "$BUILD" = build_int8 ]; then
    GOLDEN=data/int8
    EXPORT="python python/export_rtl_data.py --samples 20 --int8"
    EXPORT_NOTE="(add the name of your scales file after --int8 if it is not scales_ptq.npz)"
else
    GOLDEN=data
    EXPORT="python python/export_rtl_data.py --samples 20"
    EXPORT_NOTE=""
fi
export_again() {
    echo "$1 Export again, from the project folder:"
    echo "  $EXPORT"
    if [ -n "$EXPORT_NOTE" ]; then echo "$EXPORT_NOTE"; fi
    exit 1
}
if [ ! -f data/source.txt ]; then
    export_again "No complete export (the last export is missing or was interrupted)."
fi
if [ ! -f "$GOLDEN/source.txt" ]; then
    export_again "The last export was not made with --int8 (an export without --int8 removes the INT8 data)."
fi
if [ ! -f "$GOLDEN/$CASE/golden_logits.txt" ]; then
    export_again "No exported test data for mask '$CASE' in the $BUILD build ($GOLDEN/$CASE/ is missing)."
fi
if [ "head_masks/$CASE.txt" -nt "$GOLDEN/$CASE/golden_logits.txt" ]; then
    export_again "head_masks/$CASE.txt changed after the last export."
fi
if [ weights_float/source.txt -nt data/source.txt ]; then
    export_again "weights_float/ ($(cat weights_float/source.txt)) changed after the last export."
fi

if ! command -v verilator >/dev/null 2>&1; then
    echo "verilator not found: activate the course environment first (conda activate vit-accel)"
    exit 1
fi

# Where the simulator is compiled. GNU Make (used by Verilator) refuses to build in a path
# that contains a space, so such projects build in the temporary directory, in a folder
# named after a checksum of the project path. The simulation itself always runs here.
OBJ=obj_dir
case "$PWD" in
    *" "*) OBJ="${TMPDIR:-/tmp}"; OBJ="${OBJ%/}/vit_sim_$(printf '%s' "$PWD" | cksum | cut -d ' ' -f 1)"
           case "$OBJ" in *" "*) OBJ="/tmp/vit_sim_$(printf '%s' "$PWD" | cksum | cut -d ' ' -f 1)" ;; esac ;;
esac

# Rebuild when the content of the sources, the build options or the Verilator version
# differ from the last build (file times are not enough: unzip keeps the original ones).
VERSION="$(verilator --version)"
SOURCES="$( (cat run_sim.sh rtl/* tb/*; echo "$DEFINES $VERSION") | cksum | cut -d ' ' -f 1)"
BIN="$OBJ/$BUILD/Vtb_vit"
STAMP="$OBJ/$BUILD/sources.cksum"
if [ ! -x "$BIN" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$SOURCES" ]; then
    echo "Compiling the $BUILD simulator with Verilator (about a minute)..."
    rm -rf "$OBJ/$BUILD"
    mkdir -p "$OBJ"
    if [ "$OBJ" != obj_dir ]; then
        echo "(the project path contains a space; building in $OBJ)"
    fi
    if ! verilator --binary --timing -j 4 -Wno-fatal \
            -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-WIDTHCONCAT -Wno-UNUSEDSIGNAL \
            -Wno-PINCONNECTEMPTY -Wno-TIMESCALEMOD \
            +incdir+./rtl $DEFINES --top-module tb_vit -Mdir "$OBJ/$BUILD" \
            ./rtl/norm_stage_50x64.v \
            ./rtl/residual_add_50x64.v \
            ./rtl/gelu_stage_50x128.v \
            ./rtl/classifier_head_64x10.v \
            ./rtl/linear_row_unit.sv \
            ./rtl/vit_weight_memory.sv \
            ./rtl/vit_quant_memory.sv \
            ./rtl/transformer_block_head_serial.sv \
            ./rtl/final_stage.sv \
            ./rtl/vit_top.sv \
            ./tb/tb_vit.sv > "$OBJ/$BUILD.log" 2>&1; then
        tail -n 30 "$OBJ/$BUILD.log"
        echo "Compilation failed; the full log is $OBJ/$BUILD.log"
        exit 1
    fi
    echo "$SOURCES" > "$STAMP"
fi

# +verilator+quiet (no run-time report at the end) exists from Verilator 5.024 on.
QUIET=""
case "$VERSION" in
    "Verilator 5.0"[0-1]*|"Verilator 5.02"[0-3]*) ;;
    *) QUIET="+verilator+quiet" ;;
esac

echo "exported data: $(cat "$GOLDEN/source.txt")"
LOG="$(mktemp "${TMPDIR:-/tmp}/vit_run.XXXXXX")"
trap 'rm -f "$LOG"' EXIT
"$BIN" +CASE="$CASE" $QUIET | tee "$LOG"
grep -q "^RESULT: PASS" "$LOG"
